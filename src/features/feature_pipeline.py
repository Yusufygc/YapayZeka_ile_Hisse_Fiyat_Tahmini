# -*- coding: utf-8 -*-
"""
feature_pipeline.py — Modular Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Teknik göstergeler + isteğe bağlı makroekonomik özellikler üretir.

Adımlar:
  1. Getiri hesaplamaları (Return, Log_Return)
  2. Hareketli ortalamalar → fiyata göre göreli oran (SMA_N_rel, EMA_N_rel)
  3. Volatilite → fiyata göre normalize (RollStd_N_norm, BB_Width)
  4. Momentum göstergeleri (RSI — zaten 0-100, MACD → fiyata göre normalize)
  5. Makro bağlam (USD/TRY, BIST100, faiz, CPI) — opsiyonel
     + Göreli güç (Relative_Strength = hisse getirisi − BIST100 getirisi)

v2 (H2 düzeltmesi):
  Tüm fiyat-birimi özellikler (SMA, EMA, RollingStd, MACD) **stasyonerleştirildi**
  (Close'a göre oran alınarak). Böylece trend piyasasında bu özellikler 0'ın
  etrafında dalgalanır ve test döneminde ölçekleyici dışına fırlayıp OOD
  üretmez. Eskiden SMA_50 train-max'ının 2-3 katı olunca MinMaxScaler
  [0,1] aralığı dışına ekstrapole ediyor → model eğitim ortalamasına çöküyordu.
"""

import pandas as pd
import numpy as np
import ta
from typing import Optional

from src.features.correlation_pruning import prune_correlated_features
from src.features.sector_mapping import (
    DEFAULT_SECTOR_INDEX,
    SectorMapping,
    sector_return_column,
)
from src.xai.feature_dictionary import feature_group


class FeaturePipeline:
    """
    Ham OHLCV verisini model-ready özellik matrisine dönüştürür.

    Args:
        close_col / open_col / … : Ham CSV'deki sütun adları.
    """

    def __init__(
        self,
        close_col: str = "Close",
        open_col: str = "Open",
        high_col: str = "High",
        low_col: str = "Low",
        volume_col: str = "Volume",
        feature_mode: str = "stationary_features",
        prune_correlated_features: bool = False,
        correlation_threshold: float = 0.88,
        lag_feature_count: int = 5,
    ):
        self.close_col = close_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col
        self.feature_mode = feature_mode
        self.feature_names: list = []
        self.prune_correlated_features = prune_correlated_features
        self.correlation_threshold = correlation_threshold
        self.lag_feature_count = max(0, int(lag_feature_count))
        self.feature_groups: dict[str, str] = {}
        self.sector_mapping_report: dict = {
            "status": "not_evaluated",
            "feature_created": False,
        }
        self.pruning_report: dict = {
            "enabled": prune_correlated_features,
            "threshold": correlation_threshold,
            "dropped_features": [],
        }

    # ── Ana Metod ─────────────────────────────────────────────────────────────
    def engineer_features(
        self,
        df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        sector_mapping: Optional[SectorMapping | dict] = None,
    ) -> pd.DataFrame:
        """
        Tüm özellik üretim adımlarını sırayla uygular.

        Args:
            df       : Tarih + OHLCV sütunlarını içeren ham DataFrame.
            macro_df : MacroPipeline'dan gelen makro özellik DataFrame'i
                       (Date sütunu + makro sütunlar).
                       None ise makro katman atlanır.
            symbol   : Sektörel eşleştirme için opsiyonel hisse sembolü.

        Returns:
            Model eğitimine hazır, NaN içermeyen DataFrame.
        """
        df = df.copy()

        # 1. Getiriler
        df = self._add_returns(df)

        # 2. Hareketli Ortalamalar
        df = self._add_moving_averages(df)
        df = self._add_market_regime(df)

        # 3. Volatilite & Bollinger
        df = self._add_volatility(df)

        # 4. Momentum
        df = self._add_momentum_indicators(df)

        # 5. Volume and stationary lag features
        df = self._add_volume_features(df)
        df = self._add_lag_features(df)

        # 5. Makro bağlam (opsiyonel)
        if macro_df is not None and not macro_df.empty:
            df = self._merge_macro(
                df,
                macro_df,
                symbol=symbol,
                sector_mapping=sector_mapping,
            )
        else:
            self.sector_mapping_report = {
                **self._sector_mapping_dict(symbol, sector_mapping),
                "status": "skipped",
                "reason": "macro_data_missing",
                "feature_created": False,
            }

        # NaN temizle
        df = df.dropna().reset_index(drop=True)

        candidate_features = [c for c in df.columns if c not in ["Date", self.close_col]]
        if self.prune_correlated_features:
            df, candidate_features = self._prune_correlated(df, candidate_features)

        self.feature_names = candidate_features
        self.feature_groups = {name: feature_group(name) for name in self.feature_names}
        return df

    def _prune_correlated(
        self, df: pd.DataFrame, feature_names: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        df, feature_names, self.pruning_report = prune_correlated_features(
            df,
            feature_names,
            threshold=self.correlation_threshold,
        )
        drop_names = [item["feature"] for item in self.pruning_report["dropped_features"]]
        if drop_names:
            print(
                "  [FEATURE] Korelasyon pruning uygulandi: "
                f"{len(drop_names)} feature dusuruldu (threshold={self.correlation_threshold})."
            )
        return df, feature_names

    # ── Teknik Göstergeler ────────────────────────────────────────────────────
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Return"] = df[self.close_col].pct_change()
        df["Log_Return"] = np.log(df[self.close_col] / df[self.close_col].shift(1))
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        v2: Mutlak seviye yerine **Close'a göre göreli sapma** saklanır.
            SMA_N_rel = Close / SMA_N − 1
            → %0 üstü: fiyat ortalamanın üzerinde (yukarı momentum)
            → %0 altı: fiyat ortalamanın altında (mean-reversion adayı)
        Bu dönüşüm serinin stasyoner olmasını sağlar: trend piyasasında
        oranlar uzun vadede 0 civarında kalır, MinMax/Robust skalanın
        dışına fırlamaz.
        """
        close = df[self.close_col]
        for w in [7, 14, 21, 50]:
            sma = ta.trend.sma_indicator(close, window=w)
            ema = ta.trend.ema_indicator(close, window=w)
            if self.feature_mode in {"stationary_features", "hybrid"}:
                df[f"SMA_{w}_rel"] = close / sma.replace(0, np.nan) - 1.0
                df[f"EMA_{w}_rel"] = close / ema.replace(0, np.nan) - 1.0
            if self.feature_mode in {"legacy_price_features", "hybrid"}:
                df[f"SMA_{w}"] = sma
                df[f"EMA_{w}"] = ema
        return df

    def _add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df[self.close_col]
        sma_200 = close.rolling(window=200, min_periods=200).mean()
        regime = np.where(close > sma_200, 1, -1)
        regime = np.where(sma_200.isna(), 0, regime)
        df["Market_Regime_SMA200"] = regime.astype(int)
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        v2: Rolling_Std fiyat birimiydi (₺). Fiyat arttıkça büyüyor →
            non-stationary. RollStd_N_norm = Rolling_Std_N / Close
            (rolling coefficient of variation) ile stasyoner yapılır.
            BB_Width zaten oran (%); olduğu gibi kalır.
        """
        close = df[self.close_col]
        for w in [14, 21]:
            rolling_std = close.rolling(window=w).std()
            bb = ta.volatility.BollingerBands(close=close, window=w, window_dev=2)
            if self.feature_mode in {"stationary_features", "hybrid"}:
                df[f"RollStd_{w}_norm"] = rolling_std / close.replace(0, np.nan)
                df[f"BB_Width_{w}"] = bb.bollinger_wband()
            if self.feature_mode in {"legacy_price_features", "hybrid"}:
                df[f"Rolling_Std_{w}"] = rolling_std
                df[f"BB_Upper_{w}"] = bb.bollinger_hband()
                df[f"BB_Lower_{w}"] = bb.bollinger_lband()

        # NATR (Normalized Average True Range)
        if self.feature_mode in {"stationary_features", "hybrid"}:
            high = df[self.high_col]
            low = df[self.low_col]
            atr_ind = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
            df["NATR_14"] = atr_ind.average_true_range() / close.replace(0, np.nan)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        v2: RSI zaten 0-100 arası → stasyoner, korunur.
            MACD/MACD_Signal/MACD_Diff fiyat birimiydi → Close'a bölünür.
        """
        close = df[self.close_col]
        df["RSI_14"] = ta.momentum.rsi(close=close, window=14)

        macd = ta.trend.MACD(close=close)
        close_safe = close.replace(0, np.nan)
        if self.feature_mode in {"stationary_features", "hybrid"}:
            df["MACD_norm"] = macd.macd() / close_safe
            df["MACD_Signal_norm"] = macd.macd_signal() / close_safe
            df["MACD_Diff_norm"] = macd.macd_diff() / close_safe

            # ADX, MFI, CMF stasyoner indikatörleri
            high = df[self.high_col]
            low = df[self.low_col]
            volume = df[self.volume_col].astype(float)

            mfi_ind = ta.volume.MFIIndicator(
                high=high, low=low, close=close, volume=volume, window=14
            )
            df["MFI_14"] = mfi_ind.money_flow_index()

            adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            df["ADX_14"] = adx_ind.adx()

            cmf_ind = ta.volume.ChaikinMoneyFlowIndicator(
                high=high, low=low, close=close, volume=volume, window=20
            )
            df["CMF_20"] = cmf_ind.chaikin_money_flow()

        if self.feature_mode in {"legacy_price_features", "hybrid"}:
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["MACD_Diff"] = macd.macd_diff()
        return df

    # ── Makro Birleştirme ─────────────────────────────────────────────────────
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df[self.close_col]
        high = df[self.high_col]
        low = df[self.low_col]
        volume = df[self.volume_col].astype(float)
        volume_sum_20 = volume.rolling(20).sum().replace(0, np.nan)

        direction = np.sign(close.diff()).fillna(0.0)
        obv_flow = direction * volume
        df["OBV_Norm_20"] = obv_flow.rolling(20).sum() / volume_sum_20

        typical_price = (high + low + close) / 3.0
        vwap_20 = (typical_price * volume).rolling(20).sum() / volume_sum_20
        df["VWAP_20_rel"] = close / vwap_20.replace(0, np.nan) - 1.0
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.lag_feature_count <= 0:
            return df
        log_ret = (
            df["Log_Return"]
            if "Log_Return" in df.columns
            else np.log(df[self.close_col] / df[self.close_col].shift(1))
        )
        for i in range(1, self.lag_feature_count + 1):
            df[f"LogRet_Lag_{i}"] = log_ret.shift(i)
        return df

    # ── Sprint 5 (2026-05-25) Plan A5.1 ──────────────────────────────────────
    def recompute_close_dependent(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Recursive forecast sonrasi close sutunundan turetilen tum teknik
        gostergeleri yeniden hesaplar. Macro/sector/lag sutunlari KORUNUR
        (lag_features Sprint 5 oncesi pattern'i takip eder; macro Sprint 5
        A5.2'de ayri MacroForwardProjector ile guncellenir).

        Mevcut feature_mode'a uygun (stationary/legacy/hybrid) tum
        close-bagimli sutunlari tekrar uretir:
          - Return / Log_Return
          - SMA_*_rel / EMA_*_rel (+ legacy SMA/EMA mutlak)
          - Market_Regime_SMA200
          - RollStd_*_norm / BB_Width_* / NATR_14 (+ legacy abs)
          - RSI_14 / MACD_norm / MACD_Signal_norm / MACD_Diff_norm
            (+ legacy MACD/MACD_Signal/MACD_Diff)
          - MFI_14 / ADX_14 / CMF_20
          - OBV_Norm_20 / VWAP_20_rel

        Args:
            frame: En son recursive satir eklenmiş DataFrame.

        Returns:
            Ayni DataFrame, close-bagimli sutunlari yeniden hesaplanmis.
            Macro + sector + lag + diger ek sutunlar dokunulmaz.
        """
        out = frame.copy()
        # Sirali zincir — orjinal engineer_features sirasini koru.
        out = self._add_returns(out)
        out = self._add_moving_averages(out)
        out = self._add_market_regime(out)
        out = self._add_volatility(out)
        out = self._add_momentum_indicators(out)
        out = self._add_volume_features(out)
        return out

    def _merge_macro(
        self,
        df: pd.DataFrame,
        macro_df: pd.DataFrame,
        symbol: Optional[str] = None,
        sector_mapping: Optional[SectorMapping | dict] = None,
    ) -> pd.DataFrame:
        """
        Makro DataFrame'i hisse DataFrame'iyle tarihe göre birleştirir.
        Eksik günler forward-fill ile doldurulur.

        Ek özellikler:
          Relative_Strength = hisse Return − BIST100_Return
          Sector_Relative_Strength = hisse Return − sektörel getiri (veya BIST100 fallback)
        """
        # Date sütunu normalize
        df_dates = pd.to_datetime(df["Date"]).dt.normalize()
        macro_dates = pd.to_datetime(macro_df["Date"]).dt.normalize()

        df = df.copy()
        macro_cp = macro_df.copy()
        df["Date"] = df_dates
        macro_cp["Date"] = macro_dates

        # Sol birleşim: hisse veri setinin tarih kümesini koru
        merged = pd.merge(df, macro_cp, on="Date", how="left")

        # Makro verinin tatil farklılıklarını forward-fill ile kapat
        macro_cols = [c for c in macro_cp.columns if c != "Date"]
        merged[macro_cols] = merged[macro_cols].ffill()

        # Göreli güç: hissenin piyasayla ayrışması
        if "Return" in merged.columns and "BIST100_Return" in merged.columns:
            merged["Relative_Strength"] = merged["Return"].fillna(0) - merged[
                "BIST100_Return"
            ].fillna(0)

        # Sektörel Göreli Güç
        if "Return" in merged.columns:
            sector_report = self._sector_mapping_dict(symbol, sector_mapping)
            sector_col = sector_return_column(sector_report.get("sector_index"))
            if sector_col in merged.columns:
                merged["Sector_Relative_Strength"] = merged["Return"].fillna(0) - merged[
                    sector_col
                ].fillna(0)
                self.sector_mapping_report = {
                    **sector_report,
                    "feature_created": True,
                    "feature_column": "Sector_Relative_Strength",
                    "return_column": sector_col,
                }
            elif "BIST100_Return" in merged.columns:
                merged["Sector_Relative_Strength"] = merged["Return"].fillna(0) - merged[
                    "BIST100_Return"
                ].fillna(0)
                self.sector_mapping_report = {
                    **sector_report,
                    "sector_index": DEFAULT_SECTOR_INDEX,
                    "status": "fallback",
                    "reason": "sector_return_missing",
                    "feature_created": True,
                    "feature_column": "Sector_Relative_Strength",
                    "return_column": "BIST100_Return",
                    "requested_return_column": sector_col,
                }
            else:
                self.sector_mapping_report = {
                    **sector_report,
                    "status": "skipped",
                    "reason": "bist100_return_missing",
                    "feature_created": False,
                    "requested_return_column": sector_col,
                }

        # Drop non-stationary features if they exist
        for col in ["BIST100_Norm", "USDTRY_MA7"]:
            if col in merged.columns:
                merged.drop(columns=[col], inplace=True)

        return merged

    @staticmethod
    def _sector_mapping_dict(
        symbol: Optional[str],
        sector_mapping: Optional[SectorMapping | dict],
    ) -> dict:
        if isinstance(sector_mapping, SectorMapping):
            return sector_mapping.to_dict()
        if isinstance(sector_mapping, dict):
            defaults = FeaturePipeline._default_sector_mapping(symbol)
            defaults.update(dict(sector_mapping))
            return defaults
        return FeaturePipeline._default_sector_mapping(symbol)

    @staticmethod
    def _default_sector_mapping(symbol: Optional[str]) -> dict:
        return {
            "symbol": "" if symbol is None else str(symbol).split(".")[0].upper(),
            "sector_index": DEFAULT_SECTOR_INDEX,
            "sector": None,
            "status": "fallback",
            "reason": "mapping_not_provided",
            "source": "fallback",
            "universe_file": None,
        }
