# -*- coding: utf-8 -*-
"""Peer rank -> absolute trend tendency (yukarı/yatay/aşağı) — E2 Faz 7 ürünü.

Faz 7 ölçümü (tools/e2_faz7_confidence_diracc.py + mutlak-yön çalışması) cross-
sectional peer rank'in MUTLAK yön (h=5 getiri işareti) için MÜTEVAZI ama GERÇEK
ve MONOTON bir eğilim taşıdığını gösterdi (full-evren, 217k OOS satır):

    quintile  P(yukarı)   ort. 5g getiri
    Q1        0.435       -0.0066
    Q2        0.487       +0.0032
    Q3        0.509       +0.0050
    Q4        0.519       +0.0062
    Q5        0.541       +0.0090
    base      0.498        (nominal drift yok)

Bu modül peer_percentile'ı bu kalibrasyonla MUTLAK eğilim etiketine çevirir:
    label : yukarı | yatay | aşağı | belirsiz
    prob_up : kalibre P(h-gün getiri > 0)  (garanti DEĞİL — olasılıksal eğilim)
    expected_return : kalibre ort. h-gün log-getiri

Önemli dürüstlük notları:
- Etiket GÖRELİ rank'tan türetilir; "yukarı" = evrene göre üst dilim + tarihsel
  olarak hafif pozitif mutlak eğilim. Kesinlik değil, eğilim.
- prob_up ~0.54 (Q5) => yazı-turadan biraz iyi. Confidence (segment_ICIR x
  tradability) ne kadar güvenileceğini AYRI yönetir; düşük güvende eğilim zayıf.
- Kalibrasyon sabitleri OOS tarihsel ortalamalardır; rejim değişiminde kayabilir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrendCalibration:
    lo_pct: float = 30.0       # bu percentilin altı -> aşağı eğilim
    hi_pct: float = 70.0       # bu percentilin üstü -> yukarı eğilim
    min_names: int = 15        # bundan az evren -> belirsiz
    # Faz 7 mutlak-yön kalibrasyonu (full-evren OOS, quintile Q1..Q5).
    quintile_prob_up: tuple = (0.435, 0.487, 0.509, 0.519, 0.541)
    quintile_expected_return: tuple = (-0.0066, 0.0032, 0.0050, 0.0062, 0.0090)


@dataclass(frozen=True)
class TrendTendency:
    label: str                                  # yukarı | yatay | aşağı | belirsiz
    prob_up: Optional[float] = None             # kalibre P(getiri>0)
    expected_return: Optional[float] = None     # kalibre ort. h-gün log-getiri
    basis: str = ""                             # kısa açıklama
    reasons: list = field(default_factory=list)


def _quintile_index(pct: float) -> int:
    """peer_percentile (0..100) -> quintile index 0..4 (Q1..Q5)."""
    q = int(pct // 20.0)
    return 0 if q < 0 else (4 if q > 4 else q)


def trend_from_peer(
    peer_percentile: Optional[float],
    universe_size: Optional[int],
    cfg: Optional[TrendCalibration] = None,
) -> TrendTendency:
    """peer_percentile -> mutlak trend eğilimi + kalibre olasılık/beklenen getiri.

    universe_size < min_names veya percentile NaN -> 'belirsiz' (kalibrasyon yok).
    Etiket lo/hi percentil bantlarından; prob_up/expected_return quintile
    kalibrasyonundan (daha ince çözünürlük).
    """
    cfg = cfg or TrendCalibration()
    n = int(universe_size) if universe_size is not None else 0
    pct = float(peer_percentile) if peer_percentile is not None else float("nan")

    if n < cfg.min_names or not np.isfinite(pct):
        return TrendTendency(
            "belirsiz", None, None,
            basis="Cross-sectional evren yetersiz veya skor yok.",
            reasons=["Belirsiz: evren < min_names ya da percentile yok."])

    qi = _quintile_index(pct)
    prob_up = float(cfg.quintile_prob_up[qi])
    exp_ret = float(cfg.quintile_expected_return[qi])

    if pct >= cfg.hi_pct:
        label = "yukarı"
    elif pct <= cfg.lo_pct:
        label = "aşağı"
    else:
        label = "yatay"

    reasons = [
        f"Peer percentil {pct:.0f} (Q{qi+1}); tarihsel P(yukarı)≈{prob_up:.2f}, "
        f"beklenen ~{exp_ret:+.2%} (h-gün). Olasılıksal eğilim, garanti değil."
    ]
    return TrendTendency(label, prob_up, exp_ret,
                         basis="Faz 7 cross-sectional rank kalibrasyonu (mutlak yön).",
                         reasons=reasons)
