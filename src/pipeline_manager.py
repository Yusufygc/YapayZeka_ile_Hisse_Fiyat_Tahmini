# -*- coding: utf-8 -*-
"""
pipeline_manager.py — Pipeline Orkestrasyon Sınıfı
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~──────────────────
Bu modül, tüm model yükleme, eğitim, tahmin ve raporlama süreçlerini
nesne yönelimli (OOP) mantıkla sarmalar. 

Sorumluluklar:
 - Klasör ve sistem ayarları (models, outputs)
 - Veri yükleme ve doğrulama akışı (data_updater, data_loader)
 - Train/Test dizgilerinin yaratılması (preprocessor)
 - Modellerin Optuna veya klasik yollarla eğitimi
 - Tahmin ve Metrik oluşturup kaydetme (evaluator)
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, Any

import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data_updater import DataUpdater
from src.data_loader import load_data
from src.preprocessor import split_data, scale_data, create_sequences
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import AttentionLSTMModel
from src.models.random_forest_model import RandomForestModel
from src.ensemble import EnsembleModel
from src.evaluator import compute_metrics, plot_comparison, save_metrics_report


class ForecastingPipeline:
    def __init__(self, data_file: str, test_ratio: float = 0.20, time_steps: int = 30):
        """
        ForecastingPipeline Nesnesini başlatır.

        Parameters
        ----------
        data_file : str
            Tahmin edilecek CSV dosyasının mutlak (veya proje köküne göre) yolu.
        test_ratio : float
            Verinin yüzde kaçı test kümesine ayrılacak?
        time_steps : int
            LSTM mimarisi için geçmiş pencere uzunluğu.
        """
        self.data_file = data_file
        self.test_ratio = test_ratio
        self.time_steps = time_steps
        
        # Meta veriler
        self.stock_symbol = os.path.splitext(os.path.basename(self.data_file))[0]
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models", self.stock_symbol)
        self.outputs_dir = os.path.join(self.project_root, "outputs", self.stock_symbol)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Runtime Durumu
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.dates_train = self.dates_test = None
        self.X_train_s = self.X_test_s = self.y_train_s = self.y_test_s = None
        self.scaler_X = self.scaler_y = None
        self.X_train_seq = self.X_test_seq = self.y_train_seq = self.y_test_seq = None
        
        self.models = {}
        self.predictions = {}
        
        # Ekstra eğitim ayarları kaydedebilmek için meta
        self.meta_params = {"status": "disabled", "timestamp": self.timestamp}

    def setup_environment(self) -> None:
        """Kayıt dizinlerini oluşturur."""
        for d in [self.models_dir, self.outputs_dir]:
            os.makedirs(d, exist_ok=True)
            
        print(f"\n  [INFO] Hisse Sembolü: {self.stock_symbol}")
        print(f"  [INFO] Model Klasörü: {self.models_dir}")
        print(f"  [INFO] Çıktı Klasörü: {self.outputs_dir}")

    def ingest_data(self) -> None:
        """Verinin güncelliğini kontrol eder, yükler ve DataFrame atamasını sağlar."""
        print("\n" + "=" * 60)
        print("  ADIM 1 | Veri Yükleme & Özellik Mühendisliği (v2)")
        print("=" * 60)
        
        DataUpdater.check_and_update(self.data_file, self.stock_symbol)
        
        self.df = load_data(self.data_file)
        print(f"  Veri boyutu: {self.df.shape[0]} satır × {self.df.shape[1]} sütun")
        feature_cols = [c for c in self.df.columns if c != "Date"]
        print(f"  Özellikler ({len(feature_cols)}): {feature_cols}")

    def preprocess_data(self) -> None:
        """Train/Test dizilerini ayırır, scale işlemlerini tamamlar ve tensörleri üretir."""
        # 2a. Train/Test Split
        print("\n" + "=" * 60)
        print("  ADIM 2 | Train/Test Split & MinMax Scaling")
        print("=" * 60)
        
        (self.X_train, self.X_test, 
         self.y_train, self.y_test, 
         self.dates_train, self.dates_test) = split_data(
            self.df, target_col="Close", test_ratio=self.test_ratio
        )
        print(f"  Train: {self.X_train.shape[0]} örnek  |  Test: {self.X_test.shape[0]} örnek")

        (self.X_train_s, self.X_test_s, 
         self.y_train_s, self.y_test_s, 
         self.scaler_X, self.scaler_y) = scale_data(
            self.X_train, self.X_test, 
            self.y_train, self.y_test, 
            save_dir=self.models_dir
        )

        # 2b. LSTM Windowing (3D Tensor)
        print("\n" + "=" * 60)
        print("  ADIM 3 | LSTM Windowing (3D Tensör Oluşturma)")
        print("=" * 60)
        
        self.X_train_seq, self.y_train_seq = create_sequences(self.X_train_s, self.y_train_s, time_steps=self.time_steps)
        self.X_test_seq, self.y_test_seq = create_sequences(self.X_test_s, self.y_test_s, time_steps=self.time_steps)
        
        print(f"  LSTM Train tensörü : {self.X_train_seq.shape}")
        print(f"  LSTM Test tensörü  : {self.X_test_seq.shape}")

    def train_models(self) -> None:
        """Sisteme tanımlı modelleri başlatır (Prophet, XGB, RF, LSTM), hiperparametre aramalarını yapar ve eğitir."""
        print("\n" + "=" * 60)
        print("  ADIM 4 | Model Eğitimi (v2 — Geliştirilmiş)")
        print("=" * 60)

        # ── Prophet ──
        print("\n--- Prophet ---")
        prophet = ProphetModel(yearly_seasonality=True, weekly_seasonality=True)
        prophet.train(self.X_train, self.y_train, dates_train=self.dates_train)
        prophet.save(os.path.join(self.models_dir, "prophet_model.pkl"))
        self.models["Prophet"] = prophet

        # ── XGBoost ──
        print("\n--- XGBoost (Optuna Tuning) ---")
        xgb = XGBoostModel()
        best_xgb_params = xgb.tune_and_train(self.X_train_s, self.y_train_s, n_trials=50, n_splits=5)
        xgb.save(os.path.join(self.models_dir, "xgboost_model.pkl"))
        self.models["XGBoost"] = xgb
        self.meta_params["xgboost_best_params"] = best_xgb_params

        # ── Random Forest ──
        print("\n--- Random Forest (Optuna Tuning) ---")
        rf = RandomForestModel()
        best_rf_params = rf.tune_and_train(self.X_train_s, self.y_train_s, n_trials=30, n_splits=5)
        rf.save(os.path.join(self.models_dir, "random_forest_model.pkl"))
        self.models["Random Forest"] = rf
        self.meta_params["random_forest_best_params"] = best_rf_params

        # ── Attention LSTM ──
        print("\n--- Attention LSTM (Bidirectional + Attention) ---")
        lstm = AttentionLSTMModel(
            units_1=128, units_2=64, dropout_rate=0.2,
            epochs=80, batch_size=32, learning_rate=0.001,
        )
        lstm.train(self.X_train_seq, self.y_train_seq)
        lstm.save(os.path.join(self.models_dir, "attention_lstm_model.keras"))
        self.models["LSTM"] = lstm

    def generate_predictions(self) -> None:
        """Eğitilen modellerle test kümesinde tahminler üretip, sonuçları orijinal (scale edilmemiş) fiyat ölçeğine çevirir."""
        print("\n" + "=" * 60)
        print("  ADIM 5 | Tahmin Üretimi & Inverse Transform")
        print("=" * 60)

        preds_prophet = self.models["Prophet"].predict(self.X_test, dates_test=self.dates_test)
        
        preds_xgb_s = self.models["XGBoost"].predict(self.X_test_s)
        preds_xgb = self.scaler_y.inverse_transform(preds_xgb_s.reshape(-1, 1)).ravel()
        
        preds_rf_s = self.models["Random Forest"].predict(self.X_test_s)
        preds_rf = self.scaler_y.inverse_transform(preds_rf_s.reshape(-1, 1)).ravel()

        preds_lstm_s = self.models["LSTM"].predict(self.X_test_seq)
        preds_lstm = self.scaler_y.inverse_transform(preds_lstm_s.reshape(-1, 1)).ravel()

        y_test_original = self.y_test.ravel()

        # LSTM due to windowing generates 'TIME_STEPS' fewer predictions.
        # Align arrays to the shortest length.
        min_len = min(len(preds_prophet), len(preds_xgb), len(preds_rf), len(preds_lstm))

        self.predictions = {
            "Prophet": preds_prophet[-min_len:],
            "XGBoost": preds_xgb[-min_len:],
            "Random Forest": preds_rf[-min_len:],
            "LSTM": preds_lstm[-min_len:],
        }
        self.y_true_aligned = y_test_original[-min_len:]

        print("\n" + "=" * 60)
        print("  ADIM 6 | Ensemble Ağırlık Optimizasyonu (KAPALI)")
        print("=" * 60)
        print(f"  Ortak test uzunluğu: {min_len}")
        print("  [!] Kullanıcının isteği üzerine ensemble çalıştırılmayacak.")

    def evaluate(self) -> None:
        """Modellerin tahminlerini kıyaslayarak metrik raporlarını .csv'ye, kıyas grafiğini .png'ye döker."""
        print("\n" + "=" * 60)
        print("  ADIM 7 | Değerlendirme & Çıktı Üretimi (v3)")
        print("=" * 60)

        metrics = {}
        for name, preds in self.predictions.items():
            metrics[name] = compute_metrics(self.y_true_aligned, preds)

        # ── CSV Rapor ──
        report_versioned = os.path.join(self.outputs_dir, f"metrics_report_v2_{self.timestamp}.csv")
        report_latest = os.path.join(self.outputs_dir, "metrics_report_v2_latest.csv")
        save_metrics_report(metrics, report_versioned)
        save_metrics_report(metrics, report_latest)

        # ── Grafik ──
        plot_path = os.path.join(self.outputs_dir, f"benchmark_comparison_v2_{self.timestamp}.png")
        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v2_latest.png")
        title_str = f"{self.stock_symbol} — Model Kıyaslama v2 (Gerçek vs Tahmin)"
        
        plot_comparison(self.y_true_aligned, self.predictions, save_path=plot_path, title=title_str)
        plot_comparison(self.y_true_aligned, self.predictions, save_path=plot_latest, title=title_str)

        # ── Meta / Ensemble Parameters ──
        weights_path = os.path.join(self.outputs_dir, "ensemble_weights.json")
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(self.meta_params, f, indent=2, ensure_ascii=False)
            
        print(f"[OK] Model parametre detayları kaydedildi -> {weights_path}")

        print("\n" + "=" * 60)
        print("  [OK] Pipeline v3 başarıyla tamamlandı!")
        print(f"  [INFO] Hisse       -> {self.stock_symbol}")
        print(f"  [INFO] Modeller    -> {self.models_dir}")
        print(f"  [INFO] Çıktılar    -> {self.outputs_dir}")
        print(f"  [INFO] Zaman Damgası -> {self.timestamp}")
        print("=" * 60 + "\n")

    def run_all(self) -> None:
        """Bütün pipeline'ı orkestre eder."""
        self.setup_environment()
        self.ingest_data()
        self.preprocess_data()
        self.train_models()
        self.generate_predictions()
        self.evaluate()
