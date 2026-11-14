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

Bu modül peer_percentile'ı bu OOS frekans tablosuyla MUTLAK eğilim etiketine çevirir:
    label : yukarı | yatay | aşağı | belirsiz
    prob_up : OOS frekans tablosu (tek dönem, tarih-kümelenmiş belirsizlik
              raporlanmadı) — P(h-gün getiri > 0); garanti DEĞİL
    expected_return : aynı tablodan ort. h-gün log-getiri
    probability_source : "oos_quintile_frequency"

Etiket, olasılıkla AYNI quintile'dan türetilir: Q1→aşağı, Q5→yukarı, Q2-Q4→yatay
(eski 30/70 percentil kesimi quintile sınırlarıyla tutarsızdı: Q2 pct 20-30 'aşağı'
etiketlenirken tablo Q2'ye P(yukarı)=0.487 ve +%0.32 veriyordu).

Önemli dürüstlük notları:
- Etiket GÖRELİ rank'tan türetilir; "yukarı" = evrene göre üst dilim + tarihsel
  olarak hafif pozitif mutlak eğilim. Kesinlik değil, eğilim.
- prob_up ~0.54 (Q5) => yazı-turadan biraz iyi. Confidence (segment_ICIR x
  tradability) ne kadar güvenileceğini AYRI yönetir; düşük güvende eğilim zayıf.
- Tablo sabitleri model tasarımında kullanılan AYNI OOS dönemde tahmin edildi
  (kalibre edilmiş olasılık değil, in-sample frekans); rejim değişiminde kayabilir.
- return_std quintile başına sabittir; `symbol_vol_ratio` (sembol trailing vol /
  evren medyan vol) verilirse sembole göre ölçeklenir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrendCalibration:
    # DEPRECATED: etiket artık quintile'dan türetilir (Q1 aşağı, Q5 yukarı);
    # lo_pct/hi_pct etiketi etkilemez, yalnız geriye uyum için tutulur.
    lo_pct: float = 30.0
    hi_pct: float = 70.0
    min_names: int = 15        # bundan az evren -> belirsiz
    # Faz 7 mutlak-yön OOS frekans tablosu (full-evren OOS, quintile Q1..Q5).
    quintile_prob_up: tuple = (0.435, 0.487, 0.509, 0.519, 0.541)
    quintile_expected_return: tuple = (-0.0066, 0.0032, 0.0050, 0.0062, 0.0090)
    # Faz 7b: her quintile'da gerçekleşen h=5 log-getiri std'si (pooled OOS).
    # Pooled fiyat bandı (kolb_price_low/high) için dispersion; mean ile aynı
    # OOS çalışmasından türetilir (tools/e2_faz7b_quintile_return_std.py).
    quintile_return_std: tuple = (0.045, 0.040, 0.038, 0.037, 0.039)


@dataclass(frozen=True)
class TrendTendency:
    label: str                                  # yukarı | yatay | aşağı | belirsiz
    prob_up: Optional[float] = None             # OOS frekans tablosu P(getiri>0)
    expected_return: Optional[float] = None     # OOS tablosu ort. h-gün log-getiri
    return_std: Optional[float] = None          # h-gün log-getiri std (band; vol-ölçekli)
    basis: str = ""                             # kısa açıklama
    reasons: list = field(default_factory=list)
    probability_source: Optional[str] = None    # "oos_quintile_frequency" (belirsiz -> None)


PROBABILITY_SOURCE = "oos_quintile_frequency"


def _quintile_index(pct: float) -> int:
    """peer_percentile (0..100) -> quintile index 0..4 (Q1..Q5)."""
    q = int(pct // 20.0)
    return 0 if q < 0 else (4 if q > 4 else q)


def trend_from_peer(
    peer_percentile: Optional[float],
    universe_size: Optional[int],
    cfg: Optional[TrendCalibration] = None,
    symbol_vol_ratio: Optional[float] = None,
) -> TrendTendency:
    """peer_percentile -> mutlak trend eğilimi + OOS-frekans olasılığı/beklenen getiri.

    universe_size < min_names veya percentile NaN -> 'belirsiz' (tablo yok).
    Etiket ve prob_up/expected_return AYNI quintile'dan: Q1 aşağı, Q5 yukarı,
    Q2-Q4 yatay. symbol_vol_ratio (sembol trailing vol / evren referans vol;
    None/geçersiz -> 1.0) return_std'yi ölçekler.
    """
    cfg = cfg or TrendCalibration()
    n = int(universe_size) if universe_size is not None else 0
    pct = float(peer_percentile) if peer_percentile is not None else float("nan")

    if n < cfg.min_names or not np.isfinite(pct):
        return TrendTendency(
            "belirsiz", None, None, None,
            basis="Cross-sectional evren yetersiz veya skor yok.",
            reasons=["Belirsiz: evren < min_names ya da percentile yok."])

    qi = _quintile_index(pct)
    prob_up = float(cfg.quintile_prob_up[qi])
    exp_ret = float(cfg.quintile_expected_return[qi])
    ret_std = (
        float(cfg.quintile_return_std[qi])
        if getattr(cfg, "quintile_return_std", None)
        and qi < len(cfg.quintile_return_std)
        else None
    )

    if ret_std is not None:
        vr = float(symbol_vol_ratio) if symbol_vol_ratio is not None else 1.0
        if not np.isfinite(vr) or vr <= 0:
            vr = 1.0
        ret_std *= vr

    # etiket olasılıkla aynı quintile'dan (tutarlılık)
    if qi == 4:
        label = "yukarı"
    elif qi == 0:
        label = "aşağı"
    else:
        label = "yatay"

    reasons = [
        f"Peer percentil {pct:.0f} (Q{qi+1}); OOS frekans tablosu P(yukarı)≈{prob_up:.2f}, "
        f"beklenen ~{exp_ret:+.2%} (h-gün). Olasılıksal eğilim, garanti değil."
    ]
    return TrendTendency(label, prob_up, exp_ret, ret_std,
                         basis=("Faz 7 OOS frekans tablosu (tek dönem, tarih-kümelenmiş "
                                "belirsizlik raporlanmadı; mutlak yön)."),
                         reasons=reasons, probability_source=PROBABILITY_SOURCE)
