# -*- coding: utf-8 -*-
"""
data_manager.py — Veri Hazırlama Orkestratörü
SRP (Single Responsibility Principle): Sadece veri indirme, özellik mühendisliği (FeaturePipeline),
train/test ayırma ve tensör ölçekleme/windowing işlemlerinden sorumludur.
"""

import os
from src.data_updater import DataUpdater
from src.data_loader import load_data
from src.preprocessor import scale_data, create_sequences
from src.utils.data_splitter import TimeSeriesSplitter
from src.features.feature_pipeline import FeaturePipeline

class DataManager:
    def __init__(self, data_file: str, test_ratio: float, time_steps: int, models_dir: str):
        self.data_file = data_file
        self.test_ratio = test_ratio
        self.time_steps = time_steps
        self.models_dir = models_dir
        
        self.stock_symbol = os.path.splitext(os.path.basename(self.data_file))[0]
        
        # State variables
        self.df = None
        self.feature_names = []
        self.tensors = {}
        self.wf_splits = []

    def ingest_and_engineer(self):
        print("\n" + "=" * 60)
        print("  ADIM 1 | Veri Yükleme & Özellik Mühendisliği (DataManager)")
        print("=" * 60)
        
        DataUpdater.check_and_update(self.data_file, self.stock_symbol)
        raw_df = load_data(self.data_file)
        
        feature_pipeline = FeaturePipeline()
        self.df = feature_pipeline.engineer_features(raw_df)
        self.feature_names = feature_pipeline.feature_names
        
        print(f"  Veri boyutu: {self.df.shape[0]} satır × {self.df.shape[1]} sütun")
        print(f"  Özellik Sayısı: {len(self.feature_names)}")

    def prepare_tensors(self, train_df, test_df):
        """df'leri model eğitimine uygun tensörlere çevirir ve scale eder."""
        exclude = {"Date", "Close"}
        features = [c for c in train_df.columns if c not in exclude]
        
        X_train = train_df[features].values
        X_test = test_df[features].values
        y_train = train_df["Close"].values.reshape(-1, 1)
        y_test = test_df["Close"].values.reshape(-1, 1)
        
        # Scale (Fit only on train)
        X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = scale_data(
            X_train, X_test, y_train, y_test, save_dir=self.models_dir
        )
        
        # Sequences
        import numpy as np
        X_train_seq, y_train_seq = create_sequences(X_train_s, y_train_s, time_steps=self.time_steps)
        
        # To predict the first test sample, the model needs the previous `time_steps` samples from train_df
        X_test_input = np.vstack((X_train_s[-self.time_steps:], X_test_s))
        y_test_input = np.vstack((y_train_s[-self.time_steps:], y_test_s))
        
        X_test_seq, y_test_seq = create_sequences(X_test_input, y_test_input, time_steps=self.time_steps)
        
        # Now every item in test_df has exactly 1 sequence and 1 prediction
        original_y_test_aligned = test_df["Close"].values
        
        return {
            "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
            "X_train_s": X_train_s, "y_train_s": y_train_s, "X_test_s": X_test_s, "y_test_s": y_test_s,
            "X_train_seq": X_train_seq, "y_train_seq": y_train_seq, "X_test_seq": X_test_seq, "y_test_seq": y_test_seq,
            "scaler_X": scaler_X, "scaler_y": scaler_y, "original_y_test_aligned": original_y_test_aligned,
            "dates_train": train_df["Date"] if "Date" in train_df.columns else None,
            "dates_test": test_df["Date"] if "Date" in test_df.columns else None
        }

    def split_data(self, validation_mode: str):
        print("\n" + "=" * 60)
        print("  ADIM 2 | Train/Test Split (DataManager)")
        print("=" * 60)
        
        if validation_mode == "single_split":
            train_df, test_df, _, _ = TimeSeriesSplitter.single_split(self.df, test_ratio=self.test_ratio)
            self.tensors = self.prepare_tensors(train_df, test_df)
            
        elif validation_mode == "walk_forward":
            self.wf_splits = TimeSeriesSplitter.walk_forward_splits(
                self.df, n_splits=3, min_train_size=150, test_size=30
            )
            print(f"  [INFO] Walk-Forward splitleri oluşturuldu ({len(self.wf_splits)} adet).")
