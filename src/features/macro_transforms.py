# -*- coding: utf-8 -*-
"""Pure macro transformation helpers.

``lag_monthly_features`` risk-free faiz serisi (src/utils/risk_free_rate.py)
tarafindan kullanilir; model feature'i uretmez.
"""

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

    Aylik kaynak dosya (INTEREST_RATE.csv) degeri ayin 1'i ile
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
