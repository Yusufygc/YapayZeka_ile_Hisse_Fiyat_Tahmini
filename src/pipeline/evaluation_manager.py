# -*- coding: utf-8 -*-
"""
evaluation_manager.py — Değerlendirme ve Kayıt Orkestratörü
SRP: Sadece modellerin tahminlerini üretmek, metriklerini hesaplamak, görselleştirmek ve izleme kaydına atmaktan sorumludur.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.evaluator import (
    compute_metrics,
    enrich_with_benchmark_metrics,
    plot_comparison,
    plot_prediction_interval,
    save_metrics_report,
)
from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry
from src.database.stock_model_db import StockModelDB, compute_composite_score
from src.preprocessor import reconstruct_prices_from_logret, reconstruct_prices_from_return

class EvaluationManager:
    def __init__(
        self,
        stock_symbol:  str,
        outputs_dir:   str,
        models_dir:    str,
        tracker:       ExperimentTracker,
        registry:      ModelRegistry,
        feature_names: list,
        dataset_hash:  str,
        dataset_metadata: Dict[str, Any],
        registry_version: str = "v5",
        stock_db:      Optional[StockModelDB] = None,
    ):
        self.stock_symbol = stock_symbol
        self.outputs_dir = outputs_dir
        self.models_dir = models_dir
        self.tracker = tracker
        self.registry = registry
        self.feature_names = feature_names
        self.dataset_hash = dataset_hash
        self.dataset_metadata = dataset_metadata
        self.registry_version = registry_version
        self.stock_db = stock_db   # None ise DB kaydı atlanır

        self.predictions = {}
        self.quantile_predictions = {}
        self.y_true_aligned = None

    @staticmethod
    def _attach_composite_scores(
        metrics_dict: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        enriched = enrich_with_benchmark_metrics(metrics_dict)
        for model_name, model_metrics in enriched.items():
            model_metrics["Composite_Score"] = compute_composite_score(model_metrics)
        return enriched

    def _target_to_price(
        self,
        preds_target: np.ndarray,
        prev_close: np.ndarray,
    ) -> np.ndarray:
        target_mode = self.dataset_metadata.get("target_mode", "log_return")
        preds_target = np.asarray(preds_target).ravel()
        prev_close = np.asarray(prev_close).ravel()

        if target_mode == "log_return":
            return reconstruct_prices_from_logret(preds_target, prev_close)
        if target_mode == "return":
            return reconstruct_prices_from_return(preds_target, prev_close)
        if target_mode == "price":
            return preds_target
        raise ValueError(f"Desteklenmeyen target_mode: {target_mode}")

    def generate_predictions(self, trained_models: dict, tensors: dict):
        """
        v2 (H1 düzeltmesi):
          Modellerin çıktısı artık LOG-GETİRİ uzayında (ölçekli). Pipeline iki
          aşamalı ters dönüşüm uygular:
              scaled log-getiri → log-getiri → fiyat
          Fiyat inşası: price[t] = prev_close[t] × exp(log_ret_pred[t])

          Prophet istisnai: ölçeksiz hedefle eğitildi (y_train = log-getiri,
          unscaled). Doğrudan fiyat inşasına gider.
        """
        print("\n" + "=" * 60)
        print("  ADIM 5 | Tahmin Üretimi & Inverse Transform (EvaluationManager)")
        print("=" * 60)

        # Her model türü için hangi tensor formatının kullanılacağı
        # Prophet → ham (ölçeksiz) X_test + tarihler, hedef: raw log-getiri
        # XGBoost / Random Forest → ölçekli X_test_s (düz matris)
        # LSTM / TFT → 3-boyutlu diziler X_test_seq
        SEQ_MODELS  = {"LSTM", "TFT", "AttentionLSTM"}
        TREE_MODELS = {"XGBoost", "Random Forest"}

        # Fiyat inşası için her test satırına karşılık gelen t-1 gerçek kapanışı
        prev_close_test = tensors["prev_close_test"]   # shape (len(test),)

        raw_preds = {}   # her model → fiyat-uzayında 1-D dizi
        raw_quantiles = {}

        for name, model in trained_models.items():
            try:
                if name == "Prophet":
                    # Prophet log-getiriyi doğrudan öğrendi (ölçeksiz).
                    preds_target = model.predict(
                        tensors["X_test"], dates_test=tensors["dates_test"]
                    )

                elif name in TREE_MODELS:
                    preds_s = model.predict(tensors["X_test_s"])
                    preds_target = tensors["scaler_y"].inverse_transform(
                        preds_s.reshape(-1, 1)
                    ).ravel()

                elif name in SEQ_MODELS:
                    if hasattr(model, "predict_quantiles"):
                        quantile_scaled = model.predict_quantiles(tensors["X_test_seq"])
                        quantile_target = np.column_stack([
                            tensors["scaler_y"].inverse_transform(
                                quantile_scaled[:, idx].reshape(-1, 1)
                            ).ravel()
                            for idx in range(quantile_scaled.shape[1])
                        ])
                        preds_target = quantile_target[:, quantile_scaled.shape[1] // 2]
                        raw_quantiles[name] = quantile_target
                    else:
                        preds_s = model.predict(tensors["X_test_seq"])
                        preds_target = tensors["scaler_y"].inverse_transform(
                            preds_s.reshape(-1, 1)
                        ).ravel()

                else:
                    # Bilinmeyen model: önce dizi, sonra düz matris dene
                    try:
                        preds_s = model.predict(tensors["X_test_seq"])
                    except Exception:
                        preds_s = model.predict(tensors["X_test_s"])
                    preds_target = tensors["scaler_y"].inverse_transform(
                        preds_s.reshape(-1, 1)
                    ).ravel()

                # --- Log-getiriden fiyat inşası ---
                # Uzunluk uyuşmazlığı: bazı modeller (TFT/LSTM seq) test_seq
                # boyutunda çıktı veriyor = len(test). prev_close_test de aynı
                # uzunlukta. Yine de hizalama için trailing-align uygula.
                preds_target = np.asarray(preds_target).ravel()
                k = min(len(preds_target), len(prev_close_test))
                aligned_preds_target = preds_target[-k:]
                aligned_prev_close   = prev_close_test[-k:]
                raw_preds[name] = self._target_to_price(aligned_preds_target, aligned_prev_close)

                if name in raw_quantiles:
                    aligned_quantiles = raw_quantiles[name][-k:]
                    quantile_prices = np.column_stack([
                        self._target_to_price(aligned_quantiles[:, idx], aligned_prev_close)
                        for idx in range(aligned_quantiles.shape[1])
                    ])
                    raw_quantiles[name] = quantile_prices

                print(f"  [OK] {name} tahmini üretildi — {len(raw_preds[name])} adım")

            except Exception as exc:
                print(f"  [WARN] {name} tahmini başarısız, atlanıyor: {exc}")

        if not raw_preds:
            raise RuntimeError("Hiçbir model tahmin üretemedi. Eğitim adımını kontrol edin.")

        # v2: kıyaslama hedefi = gerçek kapanış fiyatları (log-getiri değil!)
        y_test_price = tensors["original_y_test_aligned"]
        min_len = min(len(v) for v in raw_preds.values())

        self.predictions    = {name: preds[-min_len:] for name, preds in raw_preds.items()}
        self.quantile_predictions = {
            name: preds[-min_len:]
            for name, preds in raw_quantiles.items()
        }
        self.y_true_aligned = y_test_price[-min_len:]

    def evaluate_single_split(self, trained_models: dict):
        print("\n" + "=" * 60)
        print("  ADIM 7 | Değerlendirme & Registry (EvaluationManager)")
        print("=" * 60)

        metrics = {}
        for name, preds in self.predictions.items():
            metrics[name] = compute_metrics(self.y_true_aligned, preds)

        metrics = self._attach_composite_scores(metrics)

        for name, model_metrics in metrics.items():
            # 1. Evaluate & Extract tracking
            self.tracker.log_run(
                name,
                {"validation": "single"},
                model_metrics,
                self.feature_names,
                self.dataset_hash,
                self.dataset_metadata,
            )
            
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
            self.registry.register(
                name,
                self.registry_version,
                self.feature_names,
                model_metrics,
                model_path,
                self.dataset_hash,
                self.dataset_metadata,
            )

            # 4. SQLite DB'ye kaydet (mevcut JSON/CSV korunur, DB ek katman)
            if self.stock_db is not None:
                self.stock_db.log_experiment(
                    stock_symbol    = self.stock_symbol,
                    model_name      = name,
                    metrics         = model_metrics,
                    model_path      = model_path,
                    features        = self.feature_names,
                    dataset_hash    = self.dataset_hash,
                    validation_mode = "single_split",
                    dataset_metadata = self.dataset_metadata,
                )

        report_latest = os.path.join(self.outputs_dir, "metrics_report_v4_latest.csv")
        save_metrics_report(metrics, report_latest)

        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_latest.png")
        title_str = f"{self.stock_symbol} — Model Kıyaslama (Gerçek vs Tahmin)"
        plot_comparison(self.y_true_aligned, self.predictions, save_path=plot_latest, title=title_str)

        if "TFT" in self.quantile_predictions:
            tft_quantiles = self.quantile_predictions["TFT"]
            quantile_labels = [
                f"Q{idx + 1}" for idx in range(tft_quantiles.shape[1])
            ]
            if tft_quantiles.shape[1] == 3:
                quantile_labels = ["P10", "P50", "P90"]
            quantile_df = pd.DataFrame(
                tft_quantiles,
                columns=quantile_labels,
            )
            quantile_df.insert(0, "Actual", self.y_true_aligned[-len(quantile_df):])
            quantile_csv = os.path.join(self.outputs_dir, "tft_quantiles_v5_latest.csv")
            quantile_df.to_csv(quantile_csv, sep=";", index=False)
            print(f"[OK] TFT quantile raporu kaydedildi -> {quantile_csv}")

            if tft_quantiles.shape[1] >= 3:
                interval_plot = os.path.join(self.outputs_dir, "tft_prediction_interval_v5_latest.png")
                plot_prediction_interval(
                    self.y_true_aligned[-len(tft_quantiles):],
                    tft_quantiles[:, 1],
                    tft_quantiles[:, 0],
                    tft_quantiles[:, 2],
                    save_path=interval_plot,
                    title=f"{self.stock_symbol} — TFT Tahmin Aralığı",
                )

    def evaluate_walk_forward(self, wf_results: dict, wf_predictions: dict, wf_y_true: Any):
        print("\n" + "=" * 60)
        print("  ADIM 7 | Değerlendirme Gösterimi (Walk-Forward)")
        print("=" * 60)
        
        wf_results = self._attach_composite_scores(wf_results)
        df_wf = pd.DataFrame(wf_results).T
        if "Composite_Score" in df_wf.columns:
            df_wf = df_wf.sort_values(by=["Composite_Score", "RMSE"], ascending=[False, True])
        print("\nWalk-Forward Ortalama Metrikleri:")
        print(df_wf)

        # Walk-Forward sonuçlarını DB'ye kaydet
        if self.stock_db is not None:
            for model_name, avg_metrics in wf_results.items():
                self.stock_db.log_experiment(
                    stock_symbol    = self.stock_symbol,
                    model_name      = model_name,
                    metrics         = avg_metrics,
                    model_path      = "",   # walk-forward'da model dosyası kaydedilmez
                    features        = self.feature_names,
                    dataset_hash    = self.dataset_hash,
                    validation_mode = "walk_forward",
                    dataset_metadata = self.dataset_metadata,
                )

        report_latest = os.path.join(self.outputs_dir, "metrics_report_v4_wf.csv")
        save_metrics_report(wf_results, report_latest)

        if wf_y_true is None or len(wf_predictions) == 0:
            print("  [WARN] Walk-forward sonucu yok — grafik oluşturulamadı.")
            return

        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_wf.png")
        title_str = f"{self.stock_symbol} — Model Kıyaslama (Gerçek vs Tahmin) [Walk-Forward]"
        plot_comparison(wf_y_true, wf_predictions, save_path=plot_latest, title=title_str)
        print(f"[OK] Walk-Forward karşılaştırma grafiği kaydedildi -> {plot_latest}")
