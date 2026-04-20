# -*- coding: utf-8 -*-
"""
evaluation_manager.py — Değerlendirme ve Kayıt Orkestratörü
SRP: Sadece modellerin tahminlerini üretmek, metriklerini hesaplamak, görselleştirmek ve izleme kaydına atmaktan sorumludur.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional

from src.evaluator import compute_metrics, plot_comparison, save_metrics_report
from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry
from src.database.stock_model_db import StockModelDB

class EvaluationManager:
    def __init__(
        self,
        stock_symbol:  str,
        outputs_dir:   str,
        models_dir:    str,
        tracker:       ExperimentTracker,
        registry:      ModelRegistry,
        feature_names: list,
        stock_db:      Optional[StockModelDB] = None,
    ):
        self.stock_symbol = stock_symbol
        self.outputs_dir = outputs_dir
        self.models_dir = models_dir
        self.tracker = tracker
        self.registry = registry
        self.feature_names = feature_names
        self.stock_db = stock_db   # None ise DB kaydı atlanır

        self.predictions = {}
        self.y_true_aligned = None

    def generate_predictions(self, trained_models: dict, tensors: dict):
        print("\n" + "=" * 60)
        print("  ADIM 5 | Tahmin Üretimi & Inverse Transform (EvaluationManager)")
        print("=" * 60)

        # Her model türü için hangi tensor formatının kullanılacağı
        # Prophet → ham (ölçeksiz) X_test + tarihler
        # XGBoost / Random Forest → ölçekli X_test_s (düz matris)
        # LSTM / TFT → 3-boyutlu diziler X_test_seq
        SEQ_MODELS  = {"LSTM", "TFT", "AttentionLSTM"}
        TREE_MODELS = {"XGBoost", "Random Forest"}

        raw_preds = {}   # her model → inverse-transform edilmiş 1-D dizi

        for name, model in trained_models.items():
            try:
                if name == "Prophet":
                    preds = model.predict(tensors["X_test"], dates_test=tensors["dates_test"])
                    raw_preds[name] = preds

                elif name in TREE_MODELS:
                    preds_s = model.predict(tensors["X_test_s"])
                    raw_preds[name] = tensors["scaler_y"].inverse_transform(
                        preds_s.reshape(-1, 1)
                    ).ravel()

                elif name in SEQ_MODELS:
                    preds_s = model.predict(tensors["X_test_seq"])
                    raw_preds[name] = tensors["scaler_y"].inverse_transform(
                        preds_s.reshape(-1, 1)
                    ).ravel()

                else:
                    # Bilinmeyen model: önce dizi, sonra düz matris dene
                    try:
                        preds_s = model.predict(tensors["X_test_seq"])
                    except Exception:
                        preds_s = model.predict(tensors["X_test_s"])
                    raw_preds[name] = tensors["scaler_y"].inverse_transform(
                        preds_s.reshape(-1, 1)
                    ).ravel()

                print(f"  [OK] {name} tahmini üretildi — {len(raw_preds[name])} adım")

            except Exception as exc:
                print(f"  [WARN] {name} tahmini başarısız, atlanıyor: {exc}")

        if not raw_preds:
            raise RuntimeError("Hiçbir model tahmin üretemedi. Eğitim adımını kontrol edin.")

        y_test_original = tensors["y_test"].ravel()
        min_len = min(len(v) for v in raw_preds.values())

        self.predictions   = {name: preds[-min_len:] for name, preds in raw_preds.items()}
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
            # TFT artık PyTorch (.pt), LSTM Keras (.keras), diğerleri .pkl
            if name == "TFT":
                model_ext = ".pt"
            elif name == "LSTM":
                model_ext = ".keras"
            else:
                model_ext = ".pkl"
            model_filename = f"{name.replace(' ', '_').lower()}_model{model_ext}"
            model_path = os.path.join(self.models_dir, model_filename)
            
            original_model = trained_models.get(name)
            if original_model is None:
                print(f"  [WARN] {name} için kayıtlı model bulunamadı, dosya kaydı atlanıyor.")
                model_path = ""
            else:
                original_model.save(model_path)
            
            # 3. Register model into JSON manifest
            self.registry.register(name, "v4", self.feature_names, metrics[name], model_path, dataset_hash)

            # 4. SQLite DB'ye kaydet (mevcut JSON/CSV korunur, DB ek katman)
            if self.stock_db is not None:
                self.stock_db.log_experiment(
                    stock_symbol    = self.stock_symbol,
                    model_name      = name,
                    metrics         = metrics[name],
                    model_path      = model_path,
                    features        = self.feature_names,
                    dataset_hash    = dataset_hash,
                    validation_mode = "single_split",
                )

        report_latest = os.path.join(self.outputs_dir, "metrics_report_v4_latest.csv")
        save_metrics_report(metrics, report_latest)

        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_latest.png")
        title_str = f"{self.stock_symbol} — Model Kıyaslama (Gerçek vs Tahmin)"
        plot_comparison(self.y_true_aligned, self.predictions, save_path=plot_latest, title=title_str)

    def evaluate_walk_forward(self, wf_results: dict, wf_predictions: dict, wf_y_true: Any):
        print("\n" + "=" * 60)
        print("  ADIM 7 | Değerlendirme Gösterimi (Walk-Forward)")
        print("=" * 60)
        
        df_wf = pd.DataFrame(wf_results).T
        print("\nWalk-Forward Ortalama Metrikleri:")
        print(df_wf)

        # Walk-Forward sonuçlarını DB'ye kaydet
        if self.stock_db is not None:
            dataset_hash = str(hash(self.stock_symbol))
            for model_name, avg_metrics in wf_results.items():
                self.stock_db.log_experiment(
                    stock_symbol    = self.stock_symbol,
                    model_name      = model_name,
                    metrics         = avg_metrics,
                    model_path      = "",   # walk-forward'da model dosyası kaydedilmez
                    features        = self.feature_names,
                    dataset_hash    = dataset_hash,
                    validation_mode = "walk_forward",
                )

        # Grafik oluşturma
        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_wf.png")
        title_str = f"{self.stock_symbol} — Model Kıyaslama (Gerçek vs Tahmin) [Walk-Forward]"
        plot_comparison(wf_y_true, wf_predictions, save_path=plot_latest, title=title_str)
        print(f"[OK] Walk-Forward karşılaştırma grafiği kaydedildi -> {plot_latest}")
