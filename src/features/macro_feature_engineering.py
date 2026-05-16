# -*- coding: utf-8 -*-
"""Pure macro feature engineering helpers."""

from __future__ import annotations

import pandas as pd


def engineer_monthly_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    rate_col = "INTEREST_RATE" if "INTEREST_RATE" in df.columns else "Rate"
    df["Rate_Level"] = df[rate_col]
    df["Rate_Change"] = df[rate_col].diff()
    df.drop(columns=[rate_col], inplace=True)
    return df[["Date", "Rate_Level", "Rate_Change"]]


def engineer_monthly_cpi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["CPI_MoM"] = df["CPI"].pct_change(periods=1) * 100
    df["CPI_YoY"] = df["CPI"].pct_change(periods=12) * 100
    df.drop(columns=["CPI"], inplace=True)
    return df[["Date", "CPI_MoM", "CPI_YoY"]]


def engineer_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "USDTRY" in df.columns:
        df["USDTRY_Return"] = df["USDTRY"].pct_change()
        df["USDTRY_MA7"] = df["USDTRY"].rolling(7).mean()
        df["USDTRY_Volatility7"] = df["USDTRY_Return"].rolling(7).std()
        df.drop(columns=["USDTRY"], inplace=True)

    if "BIST100" in df.columns:
        first_bist = df["BIST100"].iloc[0] if df["BIST100"].iloc[0] != 0 else 1.0
        df["BIST100_Norm"] = df["BIST100"] / first_bist
        df["BIST100_Return"] = df["BIST100"].pct_change()
        df["BIST100_MA7"] = df["BIST100_Norm"].rolling(7).mean()
        df.drop(columns=["BIST100"], inplace=True)

    if "EURTRY" in df.columns:
        df["EURTRY_Return"] = df["EURTRY"].pct_change()
        df["EURTRY_Volatility7"] = df["EURTRY_Return"].rolling(7).std()
        df.drop(columns=["EURTRY"], inplace=True)

    if "VIX" in df.columns:
        df["VIX_Level"] = df["VIX"]
        df["VIX_Change"] = df["VIX"].diff()
        df.drop(columns=["VIX"], inplace=True)

    if "GOLD_USD" in df.columns:
        df["Gold_USD_Return"] = df["GOLD_USD"].pct_change()
        if "USDTRY_Return" in df.columns:
            df["Gold_TRY_Return"] = (
                (1 + df["Gold_USD_Return"]) * (1 + df["USDTRY_Return"]) - 1
            )
        df.drop(columns=["GOLD_USD"], inplace=True)

    if "OIL_USD" in df.columns:
        df["Oil_USD_Return"] = df["OIL_USD"].pct_change()
        df.drop(columns=["OIL_USD"], inplace=True)

    if "DXY" in df.columns:
        df["DXY_Return"] = df["DXY"].pct_change()
        df["DXY_Volatility7"] = df["DXY_Return"].rolling(7).std()
        df.drop(columns=["DXY"], inplace=True)

    if "US10Y" in df.columns:
        df["US10Y_Level"] = df["US10Y"]
        df["US10Y_Change"] = df["US10Y"].diff()
        df.drop(columns=["US10Y"], inplace=True)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
