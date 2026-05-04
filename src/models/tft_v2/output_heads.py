# -*- coding: utf-8 -*-
"""
tft_v2/output_heads.py — Çıktı katmanları ve kayıp fonksiyonları
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A4 — Quantile Crossing Giderme

Üç bileşen:

  QuantileLoss         : Orijinal pinball kaybı (referans / v1 uyumluluğu)
  SortedQuantileOutput : Softplus delta parametrizasyonu ile P10 ≤ P50 ≤ P90 garantisi
  SortedQuantileLoss   : Pinball + crossing penalty birleşimi

Tasarım kararı — neden SortedQuantileOutput?
  Alternatif 1: Isotonic regression post-processing → eğitim sinyali vermiyor
  Alternatif 2: Sorted loss penalty → ağırlık büyük seçilirse optimizasyon dengesizleşir
  Alternatif 3: Bu implementasyon → çıktı katmanı delta tahmini yapar,
                Softplus negatif delta'yı engeller, kümülatif toplam sırayı garanti eder.
                Kayıp fonksiyonu crossing durumunda ekstra sinyal verir (küçük penalty).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# QuantileLoss — orijinal pinball (v1 uyumluluğu için korunur)
# ─────────────────────────────────────────────────────────────────────────────
class QuantileLoss(nn.Module):
    """
    Pinball / Quantile Loss (v1 referansı).
    q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)
    Kuantil sıralaması garanti edilmez.
    """

    def __init__(self, quantiles: List[float] = (0.1, 0.5, 0.9)) -> None:
        super().__init__()
        self.quantiles = list(quantiles)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.view(-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target - preds[:, i]
            losses.append(torch.max(q * err, (q - 1) * err))
        return torch.stack(losses, dim=1).sum(dim=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# SortedQuantileOutput — monoton çıktı garantisi
# ─────────────────────────────────────────────────────────────────────────────
class SortedQuantileOutput(nn.Module):
    """
    P10 ≤ P50 ≤ P90 garantili çıktı katmanı.

    Parametrizasyon:
        w_0 → P10               (serbest — log-return negatif olabilir)
        w_1 → Softplus(w_1)    = Δ₁ = P50 - P10  ≥ 0
        w_2 → Softplus(w_2)    = Δ₂ = P90 - P50  ≥ 0

    Kümülatif toplam:
        P10 = w_0
        P50 = P10 + Δ₁
        P90 = P50 + Δ₂

    Bu sayede gradient sıfırlanmadan Δ'lar her zaman ≥ 0 kalır.
    SortedQuantileLoss ile birlikte kullanıldığında crossing oranı < %1.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Tek projeksiyon: [d_model → 3] (w_0, w_1, w_2)
        self.proj     = nn.Linear(d_model, 3)
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x    : (batch, d_model)
        çıktı: (batch, 3)  sütunlar [P10, P50, P90]
        """
        raw = self.proj(x)                          # (batch, 3)

        p10 = raw[:, 0:1]                           # serbest
        d1  = self.softplus(raw[:, 1:2])            # Δ₁ = P50 - P10 ≥ 0
        d2  = self.softplus(raw[:, 2:3])            # Δ₂ = P90 - P50 ≥ 0

        p50 = p10 + d1
        p90 = p50 + d2

        return torch.cat([p10, p50, p90], dim=-1)   # (batch, 3)


# ─────────────────────────────────────────────────────────────────────────────
# SortedQuantileLoss — pinball + crossing penalty
# ─────────────────────────────────────────────────────────────────────────────
class SortedQuantileLoss(nn.Module):
    """
    Pinball kaybı + kuantil sıralama ihlali penaltısı.

    SortedQuantileOutput ile kullanıldığında crossing_penalty yalnızca
    sayısal hassasiyet sorunlarını yakalayan küçük bir güvenlik ağıdır.
    Düz QuantileLoss yerine doğrudan bu sınıf kullanılmalıdır.

    Args:
        quantiles       : Tahmin kuantilleri, varsayılan [0.1, 0.5, 0.9]
        crossing_penalty: Sıralama ihlali ağırlığı (0 → saf pinball)
    """

    def __init__(
        self,
        quantiles: List[float] = (0.1, 0.5, 0.9),
        crossing_penalty: float = 0.5,
    ) -> None:
        super().__init__()
        self.quantiles        = list(quantiles)
        self.crossing_penalty = crossing_penalty

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds  : (batch, num_quantiles)   — P10, P50, P90
        target : (batch,) veya (batch, 1)
        """
        target = target.view(-1)

        # ── Pinball kaybı ─────────────────────────────────────────────────
        pinball_losses = []
        for i, q in enumerate(self.quantiles):
            err = target - preds[:, i]
            pinball_losses.append(torch.max(q * err, (q - 1) * err))
        pinball = torch.stack(pinball_losses, dim=1).sum(dim=1).mean()

        # ── Crossing penalty ──────────────────────────────────────────────
        # SortedQuantileOutput ile bu terim neredeyse sıfır olur.
        # Farklı output katmanı kullanılırsa burada ek düzeltici sinyal üretilir.
        crossing = (
            torch.relu(preds[:, 0] - preds[:, 1]).mean()   # P10 > P50 ihlali
            + torch.relu(preds[:, 1] - preds[:, 2]).mean() # P50 > P90 ihlali
        )

        return pinball + self.crossing_penalty * crossing

    # ── Tanılama yardımcısı ───────────────────────────────────────────────
    @staticmethod
    def crossing_rate(preds: torch.Tensor) -> float:
        """
        Verilen tahmin tensöründe kuantil sıralaması ihlal eden örnek oranını döndürür.
        Metrik raporlama için kullanılır, backprop'a dahil değildir.

        preds: (N, 3) — sütunlar [P10, P50, P90]
        """
        with torch.no_grad():
            violation = (
                (preds[:, 0] > preds[:, 1])   # P10 > P50
                | (preds[:, 1] > preds[:, 2]) # P50 > P90
            )
        return violation.float().mean().item()

# ─────────────────────────────────────────────────────────────────────────────
# MultiHorizonHead — H ufuk için bağımsız SortedQuantileOutput başlıkları
# ─────────────────────────────────────────────────────────────────────────────
class MultiHorizonHead(nn.Module):
    """
    H tahmin ufku için bağımsız SortedQuantileOutput katmanları.

    Her ufuk (T+1, T+5, T+10, T+21) kendi projeksiyon ağırlıklarını öğrenir.
    Bu sayede decoder'ın H farklı tokeni ufuğa özgü belirsizlik profillerini
    yakalar.

    Args:
        d_model  : Model gizli boyutu
        horizons : Tahmin ufukları listesi, örn. [1, 5, 10, 21]

    Forward:
        x : (batch, H, d_model)  — H decoder token çıktısı
        -> (batch, H, 3)         — her ufuk için [P10, P50, P90]
    """

    def __init__(self, d_model: int, horizons: List[int]) -> None:
        super().__init__()
        self.horizons = list(horizons)
        self.H        = len(horizons)
        # Her ufuk için ayrı SortedQuantileOutput
        self.heads = nn.ModuleList([
            SortedQuantileOutput(d_model) for _ in horizons
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, H, d_model)
        out : (batch, H, 3)  — P10 <= P50 <= P90 her ufukta garanti
        """
        outs = [head(x[:, h, :]) for h, head in enumerate(self.heads)]
        return torch.stack(outs, dim=1)   # (batch, H, 3)
