# -*- coding: utf-8 -*-
"""
feature_pipeline.py — Modular Feature Engineering
Generates financial indicators dynamically while avoiding future data leakage.
"""

import pandas as pd
import numpy as np
import ta

class FeaturePipeline:
    """
    Applies technical indicators and statistical transforms to the raw financial data.
    """
    
    def __init__(self, close_col: str = "Close", open_col: str = "Open", 
                 high_col: str = "High", low_col: str = "Low", volume_col: str = "Volume"):
        self.close_col = close_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col
        self.feature_names = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the entire feature generation pipeline sequentially.
        """
        df = df.copy()
        
        # 1. Returns
        df = self._add_returns(df)
        
        # 2. Moving Averages
        df = self._add_moving_averages(df)
        
        # 3. Volatility & Statistical
        df = self._add_volatility(df)
        
        # 4. Momentum Indicators
        df = self._add_momentum_indicators(df)
        
        # Drop rows with NaNs resulting from rolling windows/shifts
        df = df.dropna().reset_index(drop=True)
        
        self.feature_names = [c for c in df.columns if c not in ["Date", self.close_col]]
        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Simple Return
        df["Return"] = df[self.close_col].pct_change()
        # Log Return
        df["Log_Return"] = np.log(df[self.close_col] / df[self.close_col].shift(1))
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        windows = [7, 14, 21, 50]
        for w in windows:
            df[f"SMA_{w}"] = ta.trend.sma_indicator(df[self.close_col], window=w)
            df[f"EMA_{w}"] = ta.trend.ema_indicator(df[self.close_col], window=w)
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        windows = [14, 21]
        for w in windows:
            # Rolling std
            df[f"Rolling_Std_{w}"] = df[self.close_col].rolling(window=w).std()
            
            # Bollinger Bands width
            band_indicator = ta.volatility.BollingerBands(close=df[self.close_col], window=w, window_dev=2)
            df[f"BB_Width_{w}"] = band_indicator.bollinger_wband()
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI
        df["RSI_14"] = ta.momentum.rsi(close=df[self.close_col], window=14)
        
        # MACD
        macd_indicator = ta.trend.MACD(close=df[self.close_col])
        df["MACD"] = macd_indicator.macd()
        df["MACD_Signal"] = macd_indicator.macd_signal()
        df["MACD_Diff"] = macd_indicator.macd_diff()
        
        return df
