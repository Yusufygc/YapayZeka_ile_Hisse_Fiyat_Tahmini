# -*- coding: utf-8 -*-
"""
feature_pipeline.py — Modular Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ham OHLCV'den teknik göstergeler üretir (havuzlu Kol-B yükleyicisi, sembol başına).

2026-11 IC denetimi sonrasi KURATE ozellik seti (stationary_features, takvim
yukleyicide kapali): Return, SMA_{7,21,50}_rel, RollStd_{14,21}_norm, NATR_14,
RSI_14, MACD_Diff_norm, LogRet_Lag_1 (+ istege bagli day_of_week). Makro katman
(MacroPipeline, BIST100/USDTRY, momentum_60d, Relative_Strength) 2026-11'de Kol-A ile
birlikte kaldirildi; havuzlu panelde piyasa-goreli ozellikler `src/data/cross_sectional.py`
icinde tarih-ici sembol-arasi hesaplanir.

Denetimde tam/yakin kopya (rank corr > 0.9) veya gurultu cikan ozellikler
(EMA_*, SMA_14, BB_Width, MACD_norm/Signal, ADX/MFI/CMF, OBV/VWAP, takvim
detaylari, FOMC, sektor endeksi, EURTRY/VIX/Altin/Petrol/DXY/US10Y, faiz/CPI)
KALDIRILDI. Detay: docs/wiki/data-pipeline.md (ayri adimda guncellenir).

Notlar:
  * ``Log_Return`` sutunu artik uretilmez (``Return`` ile rank-ozdes). Lag
    ozellikleri log-getiriyi dogrudan Close'dan hesaplar.
  * ``Market_Regime_SMA200`` model OZELLIGI DEGILDIR; sinyal/backtest kapilari
    (Kol-A backtest kapisi) icin frame'de hesaplanmaya devam eder (panel_columns ile dislanir)
    ve ``feature_names``/model girdisinden dislanir (src/data/panel_columns.py).
  * Tum fiyat-birimi ozellikler Close'a gore stasyonerlestirilir (oran).
"""

import pandas as pd
import numpy as np
import ta
from typing import Optional

from src.data.panel_columns import PIPELINE_AUX_COLS
from src.features.correlation_pruning import prune_correlated_features
from src.features.sector_mapping import SECTOR_FEATURES_DISABLED_REPORT
from src.xai.feature_dictionary import feature_group

# Stationary mod pencereleri (kurate set). Legacy/hybrid mutlak SMA/EMA eski
# pencerelerle (14 dahil) uretilmeye devam eder.
_STATIONARY_SMA_WINDOWS = (7, 21, 50)
_LEGACY_MA_WINDOWS = (7, 14, 21, 50)
_VOL_WINDOWS = (14, 21)



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
        lag_feature_count: int = 1,
        enable_calendar_features: bool = True,
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
        self.enable_calendar_features = bool(enable_calendar_features)
        self.feature_groups: dict[str, str] = {}
        # Sektor endeksi ozellikleri kaldirildi (IC denetimi: sektor endeksi serileri
        # hic feature matrisine ulasmiyordu; "sektor" kolonlari BIST100 kopyasiydi).
        # Rapor anahtari tuketiciler (metadata/manifest) icin dogru durumla korunur.
        self.sector_mapping_report: dict = dict(SECTOR_FEATURES_DISABLED_REPORT)
        self.pruning_report: dict = {
            "enabled": prune_correlated_features,
            "threshold": correlation_threshold,
            "dropped_features": [],
        }

    # ── Ana Metod ─────────────────────────────────────────────────────────────
    def engineer_features(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        prune_fit_tail: int = 0,
    ) -> pd.DataFrame:
        """
        Tüm özellik üretim adımlarını sırayla uygular.

        Args:
            df       : Tarih + OHLCV sütunlarını içeren ham DataFrame.
            symbol   : Kullanilmaz (yukleyici cagrisiyla uyum; test sahtelerinin kancasi).
            prune_fit_tail : > 0 ise korelasyon fit'inden son N satir (holdout) haric.

        Returns:
            Model eğitimine hazır, NaN içermeyen DataFrame.
        """
        df = df.copy()

        # 1. Getiriler
        df = self._add_returns(df)

        # 2. Hareketli Ortalamalar (+ sinyal kapilari icin piyasa rejimi; OZELLIK DEGIL)
        df = self._add_moving_averages(df)
        df = self._add_market_regime(df)

        # 3. Volatilite
        df = self._add_volatility(df)

        # 4. Momentum
        df = self._add_momentum_indicators(df)

        # 5. Stasyoner lag ozellikleri
        df = self._add_lag_features(df)

        # 5b. Takvim feature'lari — Date'e bagli, OHLCV'den bagimsiz.
        if self.enable_calendar_features:
            df = self._add_calendar_features(df)

        # NaN temizle
        df = df.dropna().reset_index(drop=True)

        candidate_features = [
            c for c in df.columns
            if c not in ["Date", self.close_col] and c not in PIPELINE_AUX_COLS
        ]
        if self.prune_correlated_features:
            df, candidate_features = self._prune_correlated(
                df, candidate_features, prune_fit_tail=prune_fit_tail
            )

        self.feature_names = candidate_features
        self.feature_groups = {name: feature_group(name) for name in self.feature_names}
        return df

    def _prune_correlated(
        self, df: pd.DataFrame, feature_names: list[str], prune_fit_tail: int = 0
    ) -> tuple[pd.DataFrame, list[str]]:
        # Leakage onlemi: korelasyon yalnizca egitim diliminde hesaplanir.
        # prune_fit_tail > 0 ise son N satir (final holdout) corr fit'inden
        # haric tutulur; dusurme karari tum frame'e uygulanir.
        fit_df = None
        if prune_fit_tail and prune_fit_tail > 0 and len(df) > prune_fit_tail:
            fit_df = df.iloc[:-prune_fit_tail]
        df, feature_names, self.pruning_report = prune_correlated_features(
            df,
            feature_names,
            threshold=self.correlation_threshold,
            fit_df=fit_df,
        )
        self.pruning_report["corr_fit_excluded_tail"] = int(prune_fit_tail or 0)
        drop_names = [item["feature"] for item in self.pruning_report["dropped_features"]]
        if drop_names:
            print(
                "  [FEATURE] Korelasyon pruning uygulandi: "
                f"{len(drop_names)} feature dusuruldu (threshold={self.correlation_threshold})."
            )
        return df, feature_names

    # ── Teknik Göstergeler ────────────────────────────────────────────────────
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Log_Return kaldirildi: Return ile rank-ozdes (lag ozellikleri log-getiriyi
        # _add_lag_features icinde Close'dan hesaplar).
        df["Return"] = df[self.close_col].pct_change()
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

        Kurate set: yalniz SMA_{7,21,50}_rel (EMA_*_rel ve SMA_14_rel kopya/gurultu).
        Legacy/hybrid modda mutlak SMA/EMA eski pencerelerle uretilir.
        """
        close = df[self.close_col]
        if self.feature_mode in {"stationary_features", "hybrid"}:
            for w in _STATIONARY_SMA_WINDOWS:
                sma = ta.trend.sma_indicator(close, window=w)
                df[f"SMA_{w}_rel"] = close / sma.replace(0, np.nan) - 1.0
        if self.feature_mode in {"legacy_price_features", "hybrid"}:
            for w in _LEGACY_MA_WINDOWS:
                df[f"SMA_{w}"] = ta.trend.sma_indicator(close, window=w)
                df[f"EMA_{w}"] = ta.trend.ema_indicator(close, window=w)
        return df

    def _add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMA-200 rejimi (+1/-1/0). Sinyal/backtest kapisi girdisi; model OZELLIGI DEGIL."""
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
        BB_Width_* kaldirildi (RollStd_*_norm ile rank corr > 0.9).
        """
        close = df[self.close_col]
        for w in _VOL_WINDOWS:
            rolling_std = close.rolling(window=w).std()
            if self.feature_mode in {"stationary_features", "hybrid"}:
                df[f"RollStd_{w}_norm"] = rolling_std / close.replace(0, np.nan)
            if self.feature_mode in {"legacy_price_features", "hybrid"}:
                bb = ta.volatility.BollingerBands(close=close, window=w, window_dev=2)
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
            MACD_Diff fiyat birimiydi → Close'a bölünür (MACD_norm/MACD_Signal_norm
            ve ADX/MFI/CMF kurate sette yok).
        """
        close = df[self.close_col]
        df["RSI_14"] = ta.momentum.rsi(close=close, window=14)

        macd = ta.trend.MACD(close=close)
        close_safe = close.replace(0, np.nan)
        if self.feature_mode in {"stationary_features", "hybrid"}:
            df["MACD_Diff_norm"] = macd.macd_diff() / close_safe

        if self.feature_mode in {"legacy_price_features", "hybrid"}:
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["MACD_Diff"] = macd.macd_diff()
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.lag_feature_count <= 0:
            return df
        close = df[self.close_col]
        log_ret = np.log(close / close.shift(1))
        for i in range(1, self.lag_feature_count + 1):
            df[f"LogRet_Lag_{i}"] = log_ret.shift(i)
        return df

    # ── Takvim Feature'lari ───────────────────────────────────────────────────
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Date sutununa dayali stasyoner takvim feature'i.

        Uretilen sutun: day_of_week (0=Pazartesi ... 4=Cuma).
        day_of_month / days_to_month_end / days_to_quarter_end /
        is_quarter_end_week / days_to_next_fomc kaldirildi (IC denetimi: gurultu).
        """
        if "Date" not in df.columns:
            return df
        dates = pd.to_datetime(df["Date"], errors="coerce")
        df["day_of_week"] = dates.dt.dayofweek.astype("Int64").astype(float)
        return df
