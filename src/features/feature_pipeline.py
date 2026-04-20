# -*- coding: utf-8 -*-
"""
feature_pipeline.py — Modular Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Teknik göstergeler + isteğe bağlı makroekonomik özellikler üretir.

Adımlar:
  1. Getiri hesaplamaları (Return, Log_Return)
  2. Hareketli ortalamalar (SMA/EMA 7/14/21/50)
  3. Volatilite & Bollinger Bantları
  4. Momentum göstergeleri (RSI, MACD)
  5. Makro bağlam (USD/TRY, BIST100) — opsiyonel, macro_df verilirse eklenir
     + Göreli güç (Relative_Strength = hisse getirisi − BIST100 getirisi)
"""

import pandas as pd
import numpy as np
import ta
from typing import Optional


class FeaturePipeline:
    """
    Ham OHLCV verisini model-ready özellik matrisine dönüştürür.

    Args:
        close_col / open_col / … : Ham CSV'deki sütun adları.
    """

    def __init__(
        self,
        close_col:  str = "Close",
        open_col:   str = "Open",
        high_col:   str = "High",
        low_col:    str = "Low",
        volume_col: str = "Volume",
    ):
        self.close_col  = close_col
        self.open_col   = open_col
        self.high_col   = high_col
        self.low_col    = low_col
        self.volume_col = volume_col
        self.feature_names: list = []

    # ── Ana Metod ─────────────────────────────────────────────────────────────
    def engineer_features(
        self,
        df:        pd.DataFrame,
        macro_df:  Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Tüm özellik üretim adımlarını sırayla uygular.

        Args:
            df       : Tarih + OHLCV sütunlarını içeren ham DataFrame.
            macro_df : MacroPipeline'dan gelen makro özellik DataFrame'i
                       (Date sütunu + makro sütunlar).
                       None ise makro katman atlanır.

        Returns:
            Model eğitimine hazır, NaN içermeyen DataFrame.
        """
        df = df.copy()

        # 1. Getiriler
        df = self._add_returns(df)

        # 2. Hareketli Ortalamalar
        df = self._add_moving_averages(df)

        # 3. Volatilite & Bollinger
        df = self._add_volatility(df)

        # 4. Momentum
        df = self._add_momentum_indicators(df)

        # 5. Makro bağlam (opsiyonel)
        if macro_df is not None and not macro_df.empty:
            df = self._merge_macro(df, macro_df)

        # NaN temizle
        df = df.dropna().reset_index(drop=True)

        self.feature_names = [c for c in df.columns if c not in ["Date", self.close_col]]
        return df

    # ── Teknik Göstergeler ────────────────────────────────────────────────────
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Return"]     = df[self.close_col].pct_change()
        df["Log_Return"] = np.log(df[self.close_col] / df[self.close_col].shift(1))
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in [7, 14, 21, 50]:
            df[f"SMA_{w}"] = ta.trend.sma_indicator(df[self.close_col], window=w)
            df[f"EMA_{w}"] = ta.trend.ema_indicator(df[self.close_col], window=w)
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in [14, 21]:
            df[f"Rolling_Std_{w}"] = df[self.close_col].rolling(window=w).std()
            bb = ta.volatility.BollingerBands(close=df[self.close_col], window=w, window_dev=2)
            df[f"BB_Width_{w}"] = bb.bollinger_wband()
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["RSI_14"] = ta.momentum.rsi(close=df[self.close_col], window=14)
        macd = ta.trend.MACD(close=df[self.close_col])
        df["MACD"]        = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Diff"]   = macd.macd_diff()
        return df

    # ── Makro Birleştirme ─────────────────────────────────────────────────────
    def _merge_macro(
        self,
        df:       pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Makro DataFrame'i hisse DataFrame'iyle tarihe göre birleştirir.
        Eksik günler forward-fill ile doldurulur.

        Ek özellik:
          Relative_Strength = hisse Return − BIST100_Return
          (Hissenin piyasaya göre ayrışan hareketi)
        """
        # Date sütunu normalize
        df_dates       = pd.to_datetime(df["Date"]).dt.normalize()
        macro_dates    = pd.to_datetime(macro_df["Date"]).dt.normalize()

        df       = df.copy()
        macro_cp = macro_df.copy()
        df["Date"]       = df_dates
        macro_cp["Date"] = macro_dates

        # Sol birleşim: hisse veri setinin tarih kümesini koru
        merged = pd.merge(df, macro_cp, on="Date", how="left")

        # Makro verinin tatil farklılıklarını forward-fill ile kapat
        macro_cols = [c for c in macro_cp.columns if c != "Date"]
        merged[macro_cols] = merged[macro_cols].ffill()

        # Göreli güç: hissenin piyasayla ayrışması
        if "Return" in merged.columns and "BIST100_Return" in merged.columns:
            merged["Relative_Strength"] = (
                merged["Return"].fillna(0) - merged["BIST100_Return"].fillna(0)
            )

        return merged
