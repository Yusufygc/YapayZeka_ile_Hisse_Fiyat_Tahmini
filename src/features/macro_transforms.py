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
) -> pd.DataFrame:
    lagged = features.copy()
    lagged[raw_date_column] = lagged["Date"]
    lagged["Date"] = lagged["Date"] + pd.to_timedelta(lag_days, unit="D")
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
    if monthly_feats is None:
        return macro
    macro = pd.merge(macro, monthly_feats, on="Date", how="left")
    for col in columns:
        if col in macro.columns:
            macro[col] = macro[col].ffill()
    return macro
