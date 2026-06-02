# -*- coding: utf-8 -*-
"""Cross-sectional (within-date) target transform — E2 Faz 3.5 (alpha kaldiraci).

Mutlak getiri hedefi (`target = log(close[t+h]/close[t])`) BIST'te h=5'te
base-rate'i gecmiyor (Faz 3 benchmark): tum hisseler ayni piyasa rejimine
maruz, model "yon" yerine ortak trende uyuyor. Cross-sectional hedef bunu
kirar: HER tarih icin o gun islem goren sembollerin ileri getirisini KENDI
ICINDE siralar -> model "hangi hisse digerlerine GORE daha iyi" ogrenir
(market-neutral). base-rate ~%50 (dengeli) -> edge gercek goreli beceriyi olcer.

Leakage: siralama tek-tarih icidir; o tarihteki tum sembollerin target_date'i
ayni (d+h). pooled_cv purge'u (`target_date < a_k`) bu satirlari blok olarak
ayni tarafta tutar -> capraz-sembol/horizon sizmasi yok.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_sectional_target(
    panel: pd.DataFrame,
    raw_target: str = "target",
    out_col: str = "target_cs",
    date_col: str = "Date",
    method: str = "rank",      # "rank" (merkezli pct) | "zscore"
    min_names: int = 5,        # bu kadardan az sembollu tarihler atilir
) -> pd.DataFrame:
    """Panele tarih-ici cross-sectional hedef kolonu ekler.

    method="rank":  out = 2*pct_rank - 1  -> ~[-1,1], medyan ~0.
    method="zscore": out = (x - mean_d) / std_d (tarih ici).
    < min_names sembollu tarihler dusulur (anlamli siralama yok).
    """
    if raw_target not in panel.columns:
        raise ValueError(f"panel'de '{raw_target}' kolonu yok")
    if date_col not in panel.columns:
        raise ValueError(f"panel'de '{date_col}' kolonu yok")

    df = panel.copy()
    grp = df.groupby(date_col)[raw_target]
    counts = grp.transform("size")

    if method == "rank":
        # simetrik pct: (rank-0.5)/n -> ortalama 0.5; merkezle [-1,1], medyan ~0
        r = grp.rank(method="average")
        pct = (r - 0.5) / counts
        out = 2.0 * pct - 1.0
    elif method == "zscore":
        mean = grp.transform("mean")
        std = grp.transform("std")
        out = (df[raw_target] - mean) / std.replace(0.0, np.nan)
    else:
        raise ValueError(f"bilinmeyen method: {method}")

    df[out_col] = out
    # yetersiz-sembollu tarihler + NaN hedef (tek-sembol gun std=NaN) at
    df = df[counts >= int(min_names)].copy()
    df = df.dropna(subset=[out_col]).reset_index(drop=True)
    return df
