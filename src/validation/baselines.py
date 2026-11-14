# -*- coding: utf-8 -*-
"""Kol-B degerlendirme yardimcilari: kural-tabanli kiyas + tradable alt kume IC (K2).

Amac: havuzlu modelin baslik IC'sinin (i) en az likit hisselerde yogunlasip
yogunlasmadigini ve (ii) basit nedensel bir faktorun (5 gunluk ters-momentum)
ayni deseni uretip uretmedigini OLCMEK. Model egitmez.

Tum fonksiyonlar `y_pred / y_true / symbol / Date` bicimli tahmin tablosu uretir
veya bekler; `daily_cross_sectional_ic` ve `segment_cross_sectional_ic` ile uyumludur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.pooled_oos import daily_cross_sectional_ic
from src.validation.segment_ic import per_date_bucket, segment_cross_sectional_ic


def liqlog_floor(min_turnover_tl: float) -> float:
    """TL/gun ciro tabanini loader `liq_log` birimine cevir (log1p). 0/negatif -> 0.

    `src.serving.nightly_scoring.liqlog_floor_from_turnover` ile ayni formul
    (validation katmani serving'e bagimli olmasin diye burada tekrar; testle kilitli).
    """
    return float(np.log1p(max(0.0, float(min_turnover_tl))))


def reversal_factor_predictions(
    panel: pd.DataFrame,
    window: int = 5,
    target_col: str = "target_cs",
    price_col: str = "Close",
    symbol_col: str = "symbol",
    date_col: str = "Date",
) -> pd.DataFrame:
    """Kural-tabanli kiyas: `-log(Close_t / Close_{t-window})` (nedensel, sembol-ici).

    Faktor yalniz t ve oncesindeki fiyatlardan hesaplanir (`shift(window)`); `target_col`
    gelecek getiriyi (y_true) tasir. Donus: symbol, Date, y_pred, y_true (+ `liq_log`
    panelde varsa) — NaN faktor/hedef satirlari atilir.
    """
    if target_col not in panel.columns:
        target_col = "target"
    df = panel.sort_values([symbol_col, date_col])
    close = pd.to_numeric(df[price_col], errors="coerce")
    past = np.log(close / close.groupby(df[symbol_col]).shift(int(window)))
    out = pd.DataFrame({
        "symbol": df[symbol_col].to_numpy(),
        "Date": df[date_col].to_numpy(),
        "y_pred": (-past).to_numpy(),
        "y_true": df[target_col].to_numpy(),
    })
    if "liq_log" in df.columns:
        out["liq_log"] = df["liq_log"].to_numpy()
    return out.dropna(subset=["y_pred", "y_true"]).reset_index(drop=True)


def attach_liquidity(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Tahmin tablosuna panelden `liq_log` ekler (symbol+Date eslesmesi)."""
    if "liq_log" in predictions.columns:
        return predictions
    keys = panel[["symbol", "Date", "liq_log"]].drop_duplicates(["symbol", "Date"])
    return predictions.merge(keys, on=["symbol", "Date"], how="left")


def tradable_subset(predictions: pd.DataFrame, min_turnover_tl: float) -> pd.DataFrame:
    """Nedensel `liq_log >= log1p(esik)` olan satirlar (gece betigindeki kapi ile ayni birim).

    `liq_log` yoksa ValueError (sessiz tum-evren donmez).
    """
    if "liq_log" not in predictions.columns:
        raise ValueError("predictions'da 'liq_log' yok; once attach_liquidity cagirin")
    floor = liqlog_floor(min_turnover_tl)
    return predictions[predictions["liq_log"] >= floor].reset_index(drop=True)


def liquidity_quantile_ic(
    predictions: pd.DataFrame, n_buckets: int = 5, min_names: int = 10,
) -> pd.DataFrame:
    """Tarih-bazli likidite kantili (Q1 = en az likit) basina gunluk cross-sectional IC."""
    if "liq_log" not in predictions.columns:
        raise ValueError("predictions'da 'liq_log' yok; once attach_liquidity cagirin")
    df = predictions.copy()
    df["liq_q"] = per_date_bucket(df, "liq_log", n_buckets)
    df = df[df["liq_q"] != "nan"]
    return segment_cross_sectional_ic(df, group_col="liq_q", min_names=min_names)


def ic_report(
    predictions: pd.DataFrame, panel: pd.DataFrame | None = None,
    min_turnover_tl: float = 3_000_000.0, gap_days: int = 5, min_names: int = 10,
) -> dict:
    """Tek tahmin tablosu icin: tum evren / tradable alt kume / likidite kantili IC.

    Her ozet `daily_cross_sectional_ic` sozlugu (ic_mean, ic_std, icir, pct_positive,
    n_days); `*_nonoverlap` `gap_days` takvim gunu arali orneklem.
    """
    preds = attach_liquidity(predictions, panel) if panel is not None else predictions
    out = {
        "all": daily_cross_sectional_ic(preds, min_names=min_names),
        "all_nonoverlap": daily_cross_sectional_ic(
            preds, min_names=min_names, sample_gap_days=gap_days),
    }
    if "liq_log" in preds.columns:
        trad = tradable_subset(preds, min_turnover_tl)
        out["tradable"] = daily_cross_sectional_ic(trad, min_names=min_names)
        out["tradable_nonoverlap"] = daily_cross_sectional_ic(
            trad, min_names=min_names, sample_gap_days=gap_days)
        out["liquidity_quantiles"] = liquidity_quantile_ic(
            preds, min_names=min_names).to_dict("records")
        out["min_turnover_tl"] = float(min_turnover_tl)
    return out
