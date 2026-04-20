# -*- coding: utf-8 -*-
"""
data_manager.py — Veri Hazırlama Orkestratörü
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SRP: Veri indirme, özellik mühendisliği (teknik + makro),
train/test ayırma ve tensör hazırlama işlemlerinden sorumludur.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd

from src.data_updater import DataUpdater
from src.data_loader import load_data
from src.preprocessor import scale_data, create_sequences
from src.utils.data_splitter import TimeSeriesSplitter
from src.features.feature_pipeline import FeaturePipeline
from src.features.macro_pipeline import MacroPipeline


class DataManager:
    """
    Args:
        data_file      : Hisse senedi CSV dosyasının tam yolu.
        test_ratio     : Test seti oranı (0-1).
        time_steps     : Sequence uzunluğu (LSTM/TFT için).
        models_dir     : Scaler objelerinin kaydedileceği dizin.
        use_macro      : True ise USD/TRY + BIST100 makro özellikler eklenir.
        macro_cache_dir: Makro CSV cache dizini.
    """

    def __init__(
        self,
        data_file:       str,
        test_ratio:      float,
        time_steps:      int,
        models_dir:      str,
        use_macro:       bool = True,
        macro_cache_dir: str  = None,
        target_mode:     str  = "log_return",
        feature_mode:    str  = "stationary_features",
        scaling_mode:    str  = "robust_x_standard_y_clip",
    ):
        self.data_file  = data_file
        self.test_ratio = test_ratio
        self.time_steps = time_steps
        self.models_dir = models_dir
        self.use_macro  = use_macro

        self.stock_symbol = os.path.splitext(os.path.basename(self.data_file))[0]

        # Makro cache dizini: varsayılan olarak proje kökü altında data/macro/
        if macro_cache_dir is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            macro_cache_dir = os.path.join(project_root, "data", "macro")
        self.macro_cache_dir = macro_cache_dir

        # State variables
        self.df:            pd.DataFrame = None
        self.feature_names: list         = []
        self.tensors:       dict         = {}
        self.wf_splits:     list         = []
        self.target_mode:   str          = target_mode
        self.feature_mode:  str          = feature_mode
        self.scaling_mode:  str          = scaling_mode
        self.dataset_metadata: dict      = {}
        self.dataset_hash:  str          = "N/A"

    # ── Veri Yükleme & Özellik Mühendisliği ──────────────────────────────────
    def ingest_and_engineer(self) -> None:
        print("\n" + "=" * 60)
        print("  ADIM 1 | Veri Yükleme & Özellik Mühendisliği (DataManager)")
        print("=" * 60)

        # Ham hisse verisi
        DataUpdater.check_and_update(self.data_file, self.stock_symbol)
        raw_df = load_data(self.data_file)

        # Makro veri (isteğe bağlı)
        macro_df = None
        if self.use_macro:
            macro_df = self._fetch_macro(raw_df)

        # Teknik + makro özellikler
        feature_pipeline  = FeaturePipeline(feature_mode=self.feature_mode)
        self.df           = feature_pipeline.engineer_features(raw_df, macro_df=macro_df)
        self.feature_names = feature_pipeline.feature_names

        # Özet
        has_rel_str  = "Relative_Strength" in self.feature_names
        macro_base   = len(MacroPipeline.macro_feature_names(include_rates=True))
        macro_count  = macro_base + (1 if has_rel_str else 0)
        tech_count   = len(self.feature_names) - (macro_count if self.use_macro and macro_df is not None and not macro_df.empty else 0)

        print(f"  Veri boyutu      : {self.df.shape[0]} satır × {self.df.shape[1]} sütun")
        print(f"  Teknik özellikler: {tech_count}")
        if self.use_macro and macro_df is not None and not macro_df.empty:
            print(f"  Makro özellikler : {macro_count}  "
                  f"(USDTRY_Return, USDTRY_MA7, USDTRY_Volatility7, "
                  f"BIST100_Norm, BIST100_Return, BIST100_MA7, "
                  f"Rate_Level, Rate_Change, CPI_YoY, CPI_MoM, Real_Rate, "
                  f"Relative_Strength)")
        print(f"  Toplam özellik   : {len(self.feature_names)}")
        self._refresh_dataset_metadata()

    def _fetch_macro(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """MacroPipeline'ı çağırır; başarısız olursa boş DataFrame döner."""
        try:
            dates = pd.to_datetime(raw_df["Date"])
            start = dates.min().strftime("%Y-%m-%d")
            end   = dates.max().strftime("%Y-%m-%d")

            mp = MacroPipeline(cache_dir=self.macro_cache_dir)
            macro_df = mp.get_macro_features(start_date=start, end_date=end)

            if macro_df.empty:
                print("  [MACRO] Makro veri boş döndü, teknik özellikler tek başına kullanılacak.")
            return macro_df

        except Exception as exc:
            print(f"  [MACRO] Makro veri alınamadı ({exc}), devam ediliyor.")
            return pd.DataFrame()

    def _refresh_dataset_metadata(self) -> None:
        if self.df is None or self.df.empty:
            self.dataset_metadata = {}
            self.dataset_hash = "N/A"
            return

        date_start = pd.to_datetime(self.df["Date"].iloc[0]).strftime("%Y-%m-%d")
        date_end = pd.to_datetime(self.df["Date"].iloc[-1]).strftime("%Y-%m-%d")
        self.dataset_metadata = {
            "stock_symbol": self.stock_symbol,
            "target_mode": self.target_mode,
            "feature_mode": self.feature_mode,
            "scaling_mode": self.scaling_mode,
            "date_range": f"{date_start}:{date_end}",
            "features_count": len(self.feature_names),
            "features": self.feature_names,
        }
        payload = json.dumps(self.dataset_metadata, ensure_ascii=False, sort_keys=True)
        self.dataset_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def build_run_metadata(self, validation_mode: str) -> tuple[dict, str]:
        run_metadata = dict(self.dataset_metadata)
        run_metadata["validation_mode"] = validation_mode
        payload = json.dumps(run_metadata, ensure_ascii=False, sort_keys=True)
        run_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return run_metadata, run_hash

    # ── Tensör Hazırlama ──────────────────────────────────────────────────────
    def _build_target_series(
        self,
        close_values: np.ndarray,
    ) -> np.ndarray:
        if self.target_mode == "log_return":
            return np.log(close_values[1:] / close_values[:-1])
        if self.target_mode == "return":
            return (close_values[1:] / close_values[:-1]) - 1.0
        if self.target_mode == "price":
            return close_values[1:]
        raise ValueError(
            f"Desteklenmeyen target_mode: {self.target_mode}. "
            "Beklenen: price, return, log_return"
        )

    def prepare_tensors(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """
        Train/test DataFrame'lerini model eğitimine uygun tensörlere çevirir.
        Scaler yalnızca train verisi üzerine fit edilir.

        Semantik (v3):
          X[t] = t gününün sonunda bilinen özellikler
          y[t] = t+1 gününün log-getirisi

        Böylece aynı gün bilgisinden aynı gün hedef üretilmez. Tree ve sequence
        modeller aynı tahmin problemi üzerinde çalışır.
        """
        exclude = {"Date", "Close"}
        features = [c for c in train_df.columns if c not in exclude]

        train_close = train_df["Close"].values.astype(float)
        test_close = test_df["Close"].values.astype(float)

        if len(train_df) < 2 or len(test_df) < 2:
            raise ValueError("t+1 hedefi için train ve test setlerinde en az 2 satır gerekir.")

        # X[t] -> y[t] = hedef(Close[t+1], Close[t])
        X_train = train_df[features].iloc[:-1].values
        X_test = test_df[features].iloc[:-1].values

        train_target = self._build_target_series(train_close)
        test_target = self._build_target_series(test_close)
        test_prev_close = test_close[:-1]

        y_train = train_target.reshape(-1, 1)
        y_test = test_target.reshape(-1, 1)

        X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = scale_data(
            X_train, X_test, y_train, y_test,
            save_dir=self.models_dir,
            scaling_mode=self.scaling_mode,
        )

        X_train_seq, y_train_seq = create_sequences(
            X_train_s, y_train_s, time_steps=self.time_steps
        )

        # İlk test örneği için son time_steps-1 train günü + ilk test günü kullanılır.
        prefix_len = max(0, self.time_steps - 1)
        X_test_input = np.vstack((X_train_s[-prefix_len:], X_test_s))
        y_test_input = np.vstack((y_train_s[-prefix_len:], y_test_s))
        X_test_seq, y_test_seq = create_sequences(
            X_test_input, y_test_input, time_steps=self.time_steps
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_train_s": X_train_s,
            "y_train_s": y_train_s,
            "X_test_s": X_test_s,
            "y_test_s": y_test_s,
            "X_train_seq": X_train_seq,
            "y_train_seq": y_train_seq,
            "X_test_seq": X_test_seq,
            "y_test_seq": y_test_seq,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "original_y_test_aligned": test_close[1:],
            "prev_close_test": test_prev_close,
            "train_close_last": float(train_close[-1]),
            "target_mode": self.target_mode,
            "dates_train": train_df["Date"].iloc[1:] if "Date" in train_df.columns else None,
            "dates_test": test_df["Date"].iloc[1:] if "Date" in test_df.columns else None,
        }

    # ── Train/Test Bölme ──────────────────────────────────────────────────────
    def split_data(self, validation_mode: str) -> None:
        print("\n" + "=" * 60)
        print("  ADIM 2 | Train/Test Split (DataManager)")
        print("=" * 60)

        if validation_mode == "single_split":
            train_df, test_df, _, _ = TimeSeriesSplitter.single_split(
                self.df, test_ratio=self.test_ratio
            )
            self.tensors = self.prepare_tensors(train_df, test_df)

        elif validation_mode == "walk_forward":
            self.wf_splits = TimeSeriesSplitter.walk_forward_splits(
                self.df,
                n_splits=3,
                min_train_size=150,
                test_size=30,
                # Sliding window: yalnızca son ~2.7 yıl (~700 iş günü) kullanılır.
                # Durağan olmayan (trend gösteren) BIST hisselerinde MinMaxScaler'ın
                # neden olduğu sistematik küçük tahmini engeller.
                max_train_size=700,
            )
            print(f"  [INFO] Walk-Forward splitleri oluşturuldu ({len(self.wf_splits)} adet, sliding window max=700).")
