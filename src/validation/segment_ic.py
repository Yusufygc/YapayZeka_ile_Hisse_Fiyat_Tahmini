# -*- coding: utf-8 -*-
"""Stratified (per-segment) cross-sectional IC — E2 Faz 6.

Cross-sectional alpha evren genelinde guclu (ICIR ~1.55) ama TUM segmentlerde
esit degil. Serving guven skoru "bu sembol HANGI segmentte ve o segmentte sinyal
ne kadar guvenilir" sorusuna dayanir. Bu modul OOS tahminleri segmentlere ayirip
(likidite kovasi, volatilite kovasi, sektor) her segment icin gunluk
cross-sectional IC dagilimini cikarir.

Notlar:
- IC cross-sectional'dir (tarih ici, semboller-arasi). Segment IC = tarih ici,
  O SEGMENTIN sembolleri arasi corr(pred,true) -> mean/std/ICIR.
- Segment etiketleri (`symbol_segments`) per-symbol medyan liq_log/vol ile
  STATIK kova; tam-gecmis medyan kullanir -> betimsel ANALIZ icin uygun (model
  girdisi DEGIL). Serving'de kova ataması trailing (causal) yapilmali.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.pooled_oos import _spearman


def symbol_segments(
    panel: pd.DataFrame, n_buckets: int = 5,
    liq_col: str = "liq_log", vol_col: str = "vol", sector_col: str = "sector",
) -> pd.DataFrame:
    """Per-symbol statik segment etiketleri (likidite/volatilite kovasi + sektor).

    Returns DataFrame[symbol, liq_bucket, vol_bucket, sector].
    Kova: per-symbol medyan -> evren-capinda quantile (Q1=en dusuk).
    """
    agg = panel.groupby("symbol").agg(
        liq=(liq_col, "median"), vol=(vol_col, "median"),
        sector=(sector_col, "first"),
    ).reset_index()

    def _bucket(s: pd.Series, n: int) -> pd.Series:
        try:
            q = pd.qcut(s.rank(method="first"), n, labels=[f"Q{i+1}" for i in range(n)])
        except ValueError:
            q = pd.Series(["Q1"] * len(s), index=s.index)
        return q.astype(str)

    agg["liq_bucket"] = _bucket(agg["liq"], n_buckets)
    agg["vol_bucket"] = _bucket(agg["vol"], n_buckets)
    agg["sector"] = agg["sector"].fillna("Unknown").astype(str)
    return agg[["symbol", "liq_bucket", "vol_bucket", "sector"]]


def per_date_bucket(
    panel: pd.DataFrame, col: str, n_buckets: int = 5, date_col: str = "Date",
) -> pd.Series:
    """Her TARIH icinde `col` degerine gore Q1..Qn kovasi (Q1 = en dusuk).

    `col` nedensel (kayan pencere) bir kolon ise (`liq_log`, `vol`) kova ataması
    yalnizca o tarihte bilinen bilgiye dayanir -> statik tam-orneklem medyanindaki
    ileriye bakis yok (K5). Seri `panel` indeksine hizalidir; NaN -> "nan".
    """
    pct = panel.groupby(date_col)[col].rank(method="first", pct=True)
    idx = np.ceil(pct * n_buckets).clip(1, n_buckets)
    out = idx.map(lambda v: f"Q{int(v)}" if np.isfinite(v) else "nan")
    return out.astype(str)


def segment_cross_sectional_ic(
    predictions: pd.DataFrame,
    group_col: str,
    date_col: str = "Date",
    min_names: int = 10,
) -> pd.DataFrame:
    """Her segment icin gunluk cross-sectional IC ozeti.

    predictions: symbol, Date, y_true, y_pred + `group_col` (segment etiketi).
    Bir segment-tarihte >= min_names sembol varsa o gunun IC'si hesaplanir.
    Returns DataFrame[segment, ic_mean, ic_std, icir, pct_positive, n_days, n_symbols].
    """
    recs = []
    for seg, gseg in predictions.groupby(group_col, sort=True):
        ics = []
        for _, g in gseg.groupby(date_col):
            if g["symbol"].nunique() < min_names:
                continue
            ic = _spearman(g["y_pred"].to_numpy(), g["y_true"].to_numpy())
            if np.isfinite(ic):
                ics.append(ic)
        if ics:
            arr = np.array(ics, dtype=float)
            mean, std = float(arr.mean()), float(arr.std())
            recs.append({
                "segment": str(seg),
                "ic_mean": mean,
                "ic_std": std,
                "icir": (mean / std) if std > 0 else float("nan"),
                "pct_positive": float((arr > 0).mean()),
                "n_days": len(arr),
                "n_symbols": int(gseg["symbol"].nunique()),
            })
        else:
            recs.append({
                "segment": str(seg), "ic_mean": float("nan"), "ic_std": float("nan"),
                "icir": float("nan"), "pct_positive": float("nan"),
                "n_days": 0, "n_symbols": int(gseg["symbol"].nunique()),
            })
    return pd.DataFrame(recs).sort_values("segment").reset_index(drop=True)


def _date_bucket_frame(panel: pd.DataFrame, n_buckets: int) -> pd.DataFrame:
    p = panel[["symbol", "Date", "liq_log", "vol", "sector"]].drop_duplicates(["symbol", "Date"]).copy()
    p["liq_bucket"] = per_date_bucket(p, "liq_log", n_buckets)
    p["vol_bucket"] = per_date_bucket(p, "vol", n_buckets)
    p["sector"] = p["sector"].fillna("Unknown").astype(str)
    for c in ("liq_bucket", "vol_bucket"):
        p[c] = p[c].replace("nan", np.nan)
    return p[["symbol", "Date", "liq_bucket", "vol_bucket", "sector"]]


def attach_date_segments(
    predictions: pd.DataFrame, panel: pd.DataFrame, n_buckets: int = 5,
) -> pd.DataFrame:
    """`attach_segments`in NEDENSEL karsiligi (K5): kovalar her tarihte o tarihin
    `liq_log`/`vol` degerlerinden (kayan-pencere, nedensel) evren-capinda atanir.

    Statik `symbol_segments` tum-orneklem medyani kullandigi icin OOS donemini de
    gorur; burada bir satirin kovasi yalniz o gunun bilgisine bagli. Cikti kolonlari
    `attach_segments` ile ayni (liq_bucket, vol_bucket, sector) -> dogrudan degistirilebilir.
    """
    return predictions.merge(
        _date_bucket_frame(panel, n_buckets), on=["symbol", "Date"], how="left")


def latest_date_segments(latest_rows: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """Tek (en guncel) tarihin satirlarindan serving segment tablosu.

    Returns DataFrame[symbol, liq_bucket, vol_bucket, sector] — `symbol_segments`
    ile ayni bicim, ama kovalar yalniz skorlama gunune kadar bilinen bilgiden.
    """
    out = _date_bucket_frame(latest_rows, n_buckets)
    return out[["symbol", "liq_bucket", "vol_bucket", "sector"]].reset_index(drop=True)


def attach_segments(predictions: pd.DataFrame, seg_table: pd.DataFrame) -> pd.DataFrame:
    """predictions'a symbol uzerinden segment kolonlarini ekler."""
    return predictions.merge(seg_table, on="symbol", how="left")
