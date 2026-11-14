# -*- coding: utf-8 -*-
"""Pure macro feature transformation helpers."""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import pandas as pd


def macro_date_window(start_date: str, end_date: str) -> dict[str, pd.Timestamp | str]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    buf_daily = start - timedelta(days=90)
    buf_monthly = start - timedelta(days=395)
    return {
        "start": start,
        "end": end,
        "buf_daily": buf_daily,
        "buf_monthly": buf_monthly,
        "buf_daily_str": buf_daily.strftime("%Y-%m-%d"),
        "buf_monthly_str": buf_monthly.strftime("%Y-%m-%d"),
    }


def filter_macro_frame(
    df: Optional[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def lag_monthly_features(
    features: pd.DataFrame,
    *,
    lag_days: int,
    raw_date_column: str,
    anchor_to_month_end: bool = True,
) -> pd.DataFrame:
    """Aylik feature'lari yayim tarihine tasir (look-ahead onleme).

    Aylik kaynak dosyalar (INTEREST_RATE.csv, CPI.csv) degeri ayin 1'i ile
    damgalar, ancak deger ayin SONUNDAKI seviyeyi / o aya ait gerceklesmeyi
    temsil eder (ornek: 2023-06-01=15.00, TCMB artisi 2023-06-22'de oldu).
    Damgayi dogrudan 1'inden gecikmeli saymak ayin son ~3 haftasini sizdirir.
    Bu yuzden once damga ay sonuna (``MonthEnd(0)``) tasinir, sonra
    ``lag_days`` eklenir. ``features["Date"]`` ay basi damgasi varsayilir;
    ``anchor_to_month_end=False`` eski (1'inden say) davranisi geri getirir.

    Ham damga ``raw_date_column`` icinde korunur.
    """
    lagged = features.copy()
    lagged[raw_date_column] = lagged["Date"]
    release = pd.to_datetime(lagged["Date"]).dt.normalize()
    if anchor_to_month_end:
        release = release + pd.offsets.MonthEnd(0)
    lagged["Date"] = release + pd.to_timedelta(lag_days, unit="D")
    return lagged


def build_base_daily_macro(
    usdtry_df: pd.DataFrame,
    bist100_df: pd.DataFrame,
) -> pd.DataFrame:
    macro = pd.merge(usdtry_df, bist100_df, on="Date", how="outer")
    macro.sort_values("Date", inplace=True)
    macro.ffill(inplace=True)
    return macro


def merge_global_daily_frames(
    macro: pd.DataFrame,
    global_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    for key, df in global_dfs.items():
        try:
            macro = pd.merge(macro, df, on="Date", how="left")
            close_col = df.columns[-1]
            if close_col in macro.columns:
                macro[close_col] = macro[close_col].ffill()
        except Exception as exc:
            print(f"  [MACRO] {key} merge atlandı: {exc}")
    return macro


def merge_monthly_feature_frames(
    macro: pd.DataFrame,
    *,
    monthly_rate_feats: Optional[pd.DataFrame],
    monthly_cpi_feats: Optional[pd.DataFrame],
) -> pd.DataFrame:
    macro = merge_ffill_monthly_features(
        macro,
        monthly_rate_feats,
        columns=["Rate_Level", "Rate_Change"],
    )
    macro = merge_ffill_monthly_features(
        macro,
        monthly_cpi_feats,
        columns=["CPI_MoM", "CPI_YoY"],
    )
    if "Rate_Level" in macro.columns and "CPI_YoY" in macro.columns:
        macro["Real_Rate"] = macro["Rate_Level"] - macro["CPI_YoY"]
    raw_date_cols = [c for c in macro.columns if c.endswith("_Raw_Date")]
    if raw_date_cols:
        macro.drop(columns=raw_date_cols, inplace=True)
    macro.reset_index(drop=True, inplace=True)
    return macro


def merge_ffill_monthly_features(
    macro: pd.DataFrame,
    monthly_feats: Optional[pd.DataFrame],
    *,
    columns: list[str],
) -> pd.DataFrame:
    """Aylik feature'lari gunluk takvime ``merge_asof(backward)`` ile tasir.

    Eski ``merge(on="Date", how="left")`` + ffill, yayim tarihi (ay sonu + lag)
    gunluk takvimde is gunu degilse (hafta sonu/tatil) o ayi sessizce
    dusuruyordu. merge_asof, yayim tarihinden SONRAKI ilk gunluk satira
    degeri tasir; yayim tarihinden onceki satirlar NaN kalir.
    """
    if monthly_feats is None or monthly_feats.empty:
        return macro
    left = macro.copy()
    right = monthly_feats.copy()
    left["Date"] = pd.to_datetime(left["Date"]).astype("datetime64[ns]")
    right["Date"] = pd.to_datetime(right["Date"]).astype("datetime64[ns]")
    # Ayni yayim tarihine dusen birden fazla kayitta sonuncu gecerli.
    right = right.drop_duplicates("Date", keep="last")
    left = left.sort_values("Date", kind="stable")
    right = right.sort_values("Date", kind="stable")
    return pd.merge_asof(left, right, on="Date", direction="backward")
