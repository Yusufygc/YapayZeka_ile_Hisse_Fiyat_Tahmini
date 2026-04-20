# -*- coding: utf-8 -*-
"""
data_manager.py — Veri Hazırlama Orkestratörü
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SRP: Veri indirme, özellik mühendisliği (teknik + makro),
train/test ayırma ve tensör hazırlama işlemlerinden sorumludur.
"""

import os
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
        feature_pipeline  = FeaturePipeline()
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

    # ── Tensör Hazırlama ──────────────────────────────────────────────────────
    def prepare_tensors(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """
        Train/test DataFrame'lerini model eğitimine uygun tensörlere çevirir.
        Scaler yalnızca train verisi üzerine fit edilir (data leakage yok).
        """
        exclude  = {"Date", "Close"}
        features = [c for c in train_df.columns if c not in exclude]

        X_train = train_df[features].values
        X_test  = test_df[features].values
        y_train = train_df["Close"].values.reshape(-1, 1)
        y_test  = test_df["Close"].values.reshape(-1, 1)

        # Ölçekleme
        X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = scale_data(
            X_train, X_test, y_train, y_test, save_dir=self.models_dir
        )

        # Sequence oluşturma
        X_train_seq, y_train_seq = create_sequences(X_train_s, y_train_s, time_steps=self.time_steps)

        # Test sekansı: son `time_steps` train satırını önek olarak ekle
        X_test_input = np.vstack((X_train_s[-self.time_steps:], X_test_s))
        y_test_input = np.vstack((y_train_s[-self.time_steps:], y_test_s))
        X_test_seq, y_test_seq = create_sequences(X_test_input, y_test_input, time_steps=self.time_steps)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_test":  X_test,  "y_test":  y_test,
            "X_train_s": X_train_s, "y_train_s": y_train_s,
            "X_test_s":  X_test_s,  "y_test_s":  y_test_s,
            "X_train_seq": X_train_seq, "y_train_seq": y_train_seq,
            "X_test_seq":  X_test_seq,  "y_test_seq":  y_test_seq,
            "scaler_X": scaler_X, "scaler_y": scaler_y,
            "original_y_test_aligned": test_df["Close"].values,
            "dates_train": train_df["Date"] if "Date" in train_df.columns else None,
            "dates_test":  test_df["Date"]  if "Date" in test_df.columns  else None,
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
