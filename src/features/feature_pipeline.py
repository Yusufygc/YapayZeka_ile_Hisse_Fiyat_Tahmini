# -*- coding: utf-8 -*-
"""
feature_pipeline.py — Modular Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Teknik göstergeler + isteğe bağlı makroekonomik özellikler üretir.

2026-11 IC denetimi sonrasi KURATE ozellik seti (stationary_features, makro var,
takvim acik) — 19 model ozelligi:
  Return, SMA_{7,21,50}_rel, RollStd_{14,21}_norm, NATR_14, RSI_14,
  MACD_Diff_norm, LogRet_Lag_1, day_of_week,
  USDTRY_Return, USDTRY_Volatility7, BIST100_Return, BIST100_MA7,
  Relative_Strength, momentum_60d, market_momentum_60d, relative_to_market_60d.

Denetimde tam/yakin kopya (rank corr > 0.9) veya gurultu cikan ozellikler
(EMA_*, SMA_14, BB_Width, MACD_norm/Signal, ADX/MFI/CMF, OBV/VWAP, takvim
detaylari, FOMC, sektor endeksi, EURTRY/VIX/Altin/Petrol/DXY/US10Y, faiz/CPI)
KALDIRILDI. Detay: docs/wiki/data-pipeline.md (ayri adimda guncellenir).

Notlar:
  * ``Log_Return`` sutunu artik uretilmez (``Return`` ile rank-ozdes). Lag
    ozellikleri log-getiriyi dogrudan Close'dan hesaplar.
  * ``Market_Regime_SMA200`` model OZELLIGI DEGILDIR; sinyal/backtest kapilari
    (data_services -> backtesting) okudugu icin frame'de hesaplanmaya devam eder
    ve ``feature_names``/model girdisinden dislanir (src/data/panel_columns.py).
  * Tum fiyat-birimi ozellikler Close'a gore stasyonerlestirilir (oran).
"""

import logging
import pandas as pd
import numpy as np
import ta
from typing import Optional

from src.data.panel_columns import PIPELINE_AUX_COLS
from src.features.correlation_pruning import prune_correlated_features
from src.features.sector_mapping import SECTOR_FEATURES_DISABLED_REPORT
from src.xai.feature_dictionary import feature_group

# Sprint 7 A7.2: cross-sectional momentum lookback (is gunu).
_MOMENTUM_LOOKBACK_DAYS = 60

# Stationary mod pencereleri (kurate set). Legacy/hybrid mutlak SMA/EMA eski
# pencerelerle (14 dahil) uretilmeye devam eder.
_STATIONARY_SMA_WINDOWS = (7, 21, 50)
_LEGACY_MA_WINDOWS = (7, 14, 21, 50)
_VOL_WINDOWS = (14, 21)


logger = logging.getLogger(__name__)


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
        enable_cross_sectional_momentum: bool = True,
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
        self.enable_cross_sectional_momentum = bool(enable_cross_sectional_momentum)
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
        macro_df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        prune_fit_tail: int = 0,
    ) -> pd.DataFrame:
        """
        Tüm özellik üretim adımlarını sırayla uygular.

        Args:
            df       : Tarih + OHLCV sütunlarını içeren ham DataFrame.
            macro_df : MacroPipeline'dan gelen makro özellik DataFrame'i
                       (Date sütunu + makro sütunlar).
                       None ise makro katman atlanır.
            symbol   : Geriye uyumluluk icin korunur (kullanilmaz).

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

        # 6. Makro bağlam (opsiyonel)
        if macro_df is not None and not macro_df.empty:
            df = self._merge_macro(df, macro_df)

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

    # ── Sprint 5 (2026-05-25) Plan A5.1 ──────────────────────────────────────
    def recompute_close_dependent(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Recursive forecast sonrasi close sutunundan turetilen tum teknik
        gostergeleri yeniden hesaplar. Macro/lag sutunlari KORUNUR
        (lag _append_recursive_row'da ele alinir; macro Sprint 5 A5.2'de ayri
        MacroForwardProjector ile guncellenir; cross-sectional momentum
        ``recompute_momentum_features`` ile makro projeksiyondan SONRA tazelenir).

        Mevcut feature_mode'a uygun (stationary/legacy/hybrid) tum
        close-bagimli sutunlari tekrar uretir:
          - Return
          - SMA_{7,21,50}_rel (+ legacy SMA/EMA mutlak)
          - Market_Regime_SMA200 (model ozelligi degil; sinyal kapisi girdisi)
          - RollStd_*_norm / NATR_14 (+ legacy abs)
          - RSI_14 / MACD_Diff_norm (+ legacy MACD/MACD_Signal/MACD_Diff)

        Args:
            frame: En son recursive satir eklenmiş DataFrame.

        Returns:
            Ayni DataFrame, close-bagimli sutunlari yeniden hesaplanmis.
            Macro + lag + diger ek sutunlar dokunulmaz.
        """
        out = frame.copy()
        # Sirali zincir — orjinal engineer_features sirasini koru.
        out = self._add_returns(out)
        out = self._add_moving_averages(out)
        out = self._add_market_regime(out)
        out = self._add_volatility(out)
        out = self._add_momentum_indicators(out)
        return out

    def recompute_momentum_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Recursive adim sonrasi momentum_60d / market_momentum_60d / relative_to_market_60d'yi tazeler.

        Makro projeksiyondan (BIST100_Return -> 0) SONRA cagrilmali. Frame'de
        ``momentum_60d`` yoksa (momentum uretilmemisse) dokunmaz.
        """
        if "momentum_60d" not in frame.columns:
            return frame
        out = frame.copy()
        if "market_momentum_60d" in out.columns and "BIST100_Return" in out.columns:
            # Gecmis satirlarin piyasa momentumu makro takviminde (tarihe gore) hesaplandi
            # ve KORUNUR; yalniz yeni (sentetik) son satir, projekte BIST100_Return ile
            # tazelenir. (Satir-bazli yeniden hesap eksik gunlu hissede kayik pencere olur.)
            history = out["market_momentum_60d"].copy()
            out = self._add_cross_sectional_momentum(out.drop(columns=["market_momentum_60d"]))
            fresh_last = out["market_momentum_60d"].iloc[-1]
            out["market_momentum_60d"] = history
            if pd.notna(fresh_last):
                out.loc[out.index[-1], "market_momentum_60d"] = fresh_last
            out["relative_to_market_60d"] = out["momentum_60d"] - out["market_momentum_60d"]
            return out
        return self._add_cross_sectional_momentum(out)

    # ── Makro Birleştirme ─────────────────────────────────────────────────────
    def _merge_macro(self, df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Makro DataFrame'i hisse DataFrame'iyle tarihe göre birleştirir.
        Eksik günler forward-fill ile doldurulur.

        Ek özellikler:
          Relative_Strength = hisse Return − BIST100_Return
          momentum_60d / market_momentum_60d / relative_to_market_60d
        (Sector_Relative_Strength ve sektor momentumu kaldirildi.)
        """
        # Date sütunu normalize
        df_dates = pd.to_datetime(df["Date"]).dt.normalize()
        macro_dates = pd.to_datetime(macro_df["Date"]).dt.normalize()

        df = df.copy()
        macro_cp = macro_df.copy()
        df["Date"] = df_dates
        macro_cp["Date"] = macro_dates

        # Piyasa 60g momentumu MAKRO TAKVIMINDE hesaplanir (hisse satirlarinda degil):
        # eksik gunu olan hissede satir-bazli rolling(60) kayik pencere uretirdi; boylece
        # ayni tarihte tum hisseler icin ayni deger olur. Seviye kolonu gibi ffill edilir.
        if self.enable_cross_sectional_momentum and "BIST100_Return" in macro_cp.columns:
            macro_cp = macro_cp.sort_values("Date").reset_index(drop=True)
            macro_cp["market_momentum_60d"] = self._market_momentum(macro_cp["BIST100_Return"])

        # Sol birleşim: hisse veri setinin tarih kümesini koru
        merged = pd.merge(df, macro_cp, on="Date", how="left")

        # Makro kapsam kontrolu: makro son tarihi hisse son tarihinden >5 gun
        # geride ise getiri kolonlari 0'a, seviye kolonlari ffill'e gider.
        if len(macro_cp) and len(df):
            macro_max = macro_cp["Date"].max()
            stock_max = df["Date"].max()
            if pd.notna(macro_max) and pd.notna(stock_max) and (
                stock_max - macro_max
            ) > pd.Timedelta(days=5):
                logger.warning(
                    "Macro verisi hisse verisinin gerisinde: macro_max=%s stock_max=%s "
                    "(>5 gun). Getiri/degisim kolonlari 0, seviyeler ffill.",
                    macro_max.date(),
                    stock_max.date(),
                )

        # Makro verinin tatil farklılıklarını forward-fill ile kapat.
        # P1-15: getiri/degisim kolonlari ffill edilirse son getiri aylarca
        # tekrarlanir; bunlar icin eksik deger 0 (degisim yok) olur.
        macro_cols = [c for c in macro_cp.columns if c != "Date"]
        flow_cols = [
            c for c in macro_cols
            if c.endswith("_Return") or c.endswith("_Change") or "Return" in c
        ]
        level_cols = [c for c in macro_cols if c not in flow_cols]
        if level_cols:
            merged[level_cols] = merged[level_cols].ffill()
        if flow_cols and len(macro_cp):
            # Sadece makro takviminde OLMAYAN (kapsam disi / tatil) hisse gunleri
            # 0 olur; makro baslangicindan onceki satirlar NaN kalir (eski davranis:
            # ffill lider NaN'i doldurmaz) ve sonraki dropna ile elenir.
            gap_mask = ~merged["Date"].isin(macro_cp["Date"]) & (
                merged["Date"] >= macro_cp["Date"].min()
            )
            merged.loc[gap_mask, flow_cols] = merged.loc[gap_mask, flow_cols].fillna(0.0)

        # Göreli güç: hissenin piyasayla ayrışması
        if "Return" in merged.columns and "BIST100_Return" in merged.columns:
            merged["Relative_Strength"] = merged["Return"].fillna(0) - merged[
                "BIST100_Return"
            ].fillna(0)

        # Sprint 7 A7.2 — Cross-sectional momentum (hisse 60g vs BIST100).
        if self.enable_cross_sectional_momentum:
            merged = self._add_cross_sectional_momentum(merged)

        # Drop non-stationary features if they exist
        for col in ["BIST100_Norm", "USDTRY_MA7"]:
            if col in merged.columns:
                merged.drop(columns=[col], inplace=True)

        return merged

    def _add_cross_sectional_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hisse 60g momentumu + BIST100 goreli momentum sutunlari.

        Sutunlar:
          - momentum_60d            (hisse Close.pct_change(60))
          - market_momentum_60d     (BIST100_Return cumprod, varsa)
          - relative_to_market_60d  (hisse - market; ikisi de varsa)
        (sector_momentum_60d / relative_momentum_60d kaldirildi: sektor endeksi
        serileri hic ulasmiyor, BIST100 kopyasi uretiyordu.)
        """
        out = df
        close = out.get(self.close_col)
        lookback = _MOMENTUM_LOOKBACK_DAYS
        if close is None or len(out) < lookback + 1:
            return out

        out["momentum_60d"] = close.pct_change(lookback)

        if "BIST100_Return" in out.columns:
            if "market_momentum_60d" not in out.columns:
                # Makro takviminde onceden hesaplanmamis (dogrudan cagri): satir-bazli yedek.
                out["market_momentum_60d"] = self._market_momentum(out["BIST100_Return"])
            out["relative_to_market_60d"] = (
                out["momentum_60d"] - out["market_momentum_60d"]
            )

        return out

    @staticmethod
    def _market_momentum(bist100_return: pd.Series) -> pd.Series:
        """BIST100 getirilerinin 60 satirlik kumulatif getirisi (log-toplam -> expm1)."""
        log_ret = np.log1p(bist100_return.fillna(0.0))
        return np.expm1(log_ret.rolling(_MOMENTUM_LOOKBACK_DAYS).sum())
