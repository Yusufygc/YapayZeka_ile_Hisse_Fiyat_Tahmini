# -*- coding: utf-8 -*-
"""Pure macro feature engineering helpers (kurate set: USDTRY + BIST100)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_daily(df: pd.DataFrame) -> pd.DataFrame:
    """USDTRY / BIST100 seviyelerinden stasyoner makro ozellikleri turetir.

    Uretilen: USDTRY_Return, USDTRY_Volatility7, BIST100_Return, BIST100_MA7.
    Her iki seri de BIST seansi sirasinda gozlenebildigi icin gecikmesizdir.
    (EURTRY/VIX/Altin/Petrol/DXY/US10Y/sektor endeksleri ve ABD-seansi gecikme
    mantigi IC denetimi sonrasi kaldirildi.)
    """
    df = df.copy()

    if "USDTRY" in df.columns:
        df["USDTRY_Return"] = df["USDTRY"].pct_change()
        df["USDTRY_Volatility7"] = df["USDTRY_Return"].rolling(7).std()
        df.drop(columns=["USDTRY"], inplace=True)

    if "BIST100" in df.columns:
        df["BIST100_Return"] = df["BIST100"].pct_change()
        # BIST100_MA7 stasyonerleştirildi: Fiyatın 7 günlük ortalamasından göreli sapması
        bist_ma = df["BIST100"].rolling(7).mean()
        df["BIST100_MA7"] = df["BIST100"] / bist_ma.replace(0, np.nan) - 1.0
        df.drop(columns=["BIST100"], inplace=True)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
