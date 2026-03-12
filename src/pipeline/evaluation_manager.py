# -*- coding: utf-8 -*-
"""
evaluation_manager.py — Değerlendirme ve Kayıt Orkestratörü
SRP: Sadece modellerin tahminlerini üretmek, metriklerini hesaplamak, görselleştirmek ve izleme kaydına atmaktan sorumludur.
"""

import os
import pandas as pd
from typing import Dict, Any

from src.evaluator import compute_metrics, plot_comparison, save_metrics_report
from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry

class EvaluationManager:
    def __init__(self, stock_symbol: str, outputs_dir: str, models_dir: str, tracker: ExperimentTracker, registry: ModelRegistry, feature_names: list):
        self.stock_symbol = stock_symbol
        self.outputs_dir = outputs_dir
        self.models_dir = models_dir
        self.tracker = tracker
        self.registry = registry
        self.feature_names = feature_names
        
        self.predictions = {}
        self.y_true_aligned = None

    def generate_predictions(self, trained_models: dict, tensors: dict):
        print("\n" + "=" * 60)
        print("  ADIM 5 | Tahmin Üretimi & Inverse Transform (EvaluationManager)")
        print("=" * 60)

        preds_prophet = trained_models["Prophet"].predict(tensors["X_test"], dates_test=tensors["dates_test"])
        
        preds_xgb_s = trained_models["XGBoost"].predict(tensors["X_test_s"])
        preds_xgb = tensors["scaler_y"].inverse_transform(preds_xgb_s.reshape(-1, 1)).ravel()
        
        preds_rf_s = trained_models["Random Forest"].predict(tensors["X_test_s"])
        preds_rf = tensors["scaler_y"].inverse_transform(preds_rf_s.reshape(-1, 1)).ravel()

        preds_lstm_s = trained_models["LSTM"].predict(tensors["X_test_seq"])
        preds_lstm = tensors["scaler_y"].inverse_transform(preds_lstm_s.reshape(-1, 1)).ravel()
        
        preds_tft_s = trained_models["TFT"].predict(tensors["X_test_seq"])
        preds_tft = tensors["scaler_y"].inverse_transform(preds_tft_s.reshape(-1, 1)).ravel()

        y_test_original = tensors["y_test"].ravel()
        min_len = min(len(preds_prophet), len(preds_xgb), len(preds_rf), len(preds_lstm), len(preds_tft))

        self.predictions = {
            "Prophet": preds_prophet[-min_len:],
            "XGBoost": preds_xgb[-min_len:],
            "Random Forest": preds_rf[-min_len:],
            "LSTM": preds_lstm[-min_len:],
            "TFT": preds_tft[-min_len:]
        }
        self.y_true_aligned = y_test_original[-min_len:]

    def evaluate_single_split(self, trained_models: dict):
        print("\n" + "=" * 60)
        print("  ADIM 7 | Değerlendirme & Registry (EvaluationManager)")
        print("=" * 60)

        metrics = {}
        for name, preds in self.predictions.items():
            metrics[name] = compute_metrics(self.y_true_aligned, preds)
            
            # 1. Evaluate & Extract tracking
            dataset_hash = str(hash(self.stock_symbol))
            self.tracker.log_run(name, {"validation": "single"}, metrics[name], self.feature_names, dataset_hash)
            
            # 2. Save physical model weights/binaries
            model_ext = ".keras" if name in ["LSTM", "TFT"] else ".pkl"
            model_filename = f"{name.replace(' ', '_').lower()}_model{model_ext}"
            model_path = os.path.join(self.models_dir, model_filename)
            
            original_model = trained_models[name]
            original_model.save(model_path)
            
            # 3. Register model into JSON manifest
            self.registry.register(name, "v4", self.feature_names, metrics[name], model_path, dataset_hash)

        report_latest = os.path.join(self.outputs_dir, "metrics_report_v4_latest.csv")
        save_metrics_report(metrics, report_latest)

        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_latest.png")
        title_str = f"{self.stock_symbol} — Model Kıyaslama (Gerçek vs Tahmin)"
        plot_comparison(self.y_true_aligned, self.predictions, save_path=plot_latest, title=title_str)

    def evaluate_walk_forward(self, wf_results: dict):
        print("\n" + "=" * 60)
        print("  ADIM 7 | Değerlendirme Gösterimi (Walk-Forward)")
        print("=" * 60)
        
        df_wf = pd.DataFrame(wf_results).T
        print("\nWalk-Forward Ortalama Metrikleri:")
        print(df_wf)
