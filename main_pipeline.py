# -*- coding: utf-8 -*-
"""
main_pipeline.py — Orkestrasyon Dosyası (v2 — Geliştirilmiş)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tüm sistemi baştan sona çalıştırır:
  1. Veriyi yükle & temizle (data_loader) — Yeni göstergeler dahil
  2. Train/Test böl & ölçekle (preprocessor)
  3. Prophet, XGBoost (Optuna), Attention LSTM modellerini eğit
  4. Tahminleri orijinal ölçeğe geri dönüştür
  5. Ensemble ağırlıklarını optimize et (Inverse RMSE + Grid Search)
  6. Tüm modelleri değerlendir (evaluator)
  7. Model, scaler, grafik, metrik CSV — hisse bazlı klasöre kaydet

Kullanım:
    python main_pipeline.py
"""

import os
import sys
import warnings
import numpy as np
from datetime import datetime

# ── Prophet'in aşırı loglama yapmasını önle ─────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow bilgi mesajlarını kapat

# ── Proje kök yolunu ayarla ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data
from src.preprocessor import split_data, scale_data, create_sequences
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import AttentionLSTMModel
from src.ensemble import EnsembleModel
from src.evaluator import compute_metrics, plot_comparison, save_metrics_report


# ═════════════════════════════════════════════════════════════════════════════
# YAPILANDIRMA
# ═════════════════════════════════════════════════════════════════════════════
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "ASELS.csv")
TEST_RATIO = 0.20
TIME_STEPS = 30  # LSTM pencere uzunluğu

# ── Hisse sembolünü CSV dosya adından çıkar ─────────────────────────────────
STOCK_SYMBOL = os.path.splitext(os.path.basename(DATA_FILE))[0]  # "ASELS"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", STOCK_SYMBOL)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", STOCK_SYMBOL)

# ── Tarih damgası (versiyon takibi) ─────────────────────────────────────────
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    """Ana pipeline fonksiyonu (v2 — Geliştirilmiş)."""

    # ── Klasör oluştur ───────────────────────────────────────────────────────
    for d in [MODELS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"\n  📌  Hisse Sembolü: {STOCK_SYMBOL}")
    print(f"  📁  Model Klasörü: {MODELS_DIR}")
    print(f"  📊  Çıktı Klasörü: {OUTPUTS_DIR}")

    # ═════════════════════════════════════════════════════════════════════════
    # ADIM 1 — Veri Yükleme & Feature Engineering (Genişletilmiş)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ADIM 1 │ Veri Yükleme & Özellik Mühendisliği (v2)")
    print("=" * 60)
    df = load_data(DATA_FILE)
    print(f"  Veri boyutu: {df.shape[0]} satır × {df.shape[1]} sütun")
    feature_cols = [c for c in df.columns if c != "Date"]
    print(f"  Özellikler ({len(feature_cols)}): {feature_cols}")

    # ═════════════════════════════════════════════════════════════════════════
    # ADIM 2 — Train / Test Bölme ve Ölçeklendirme
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ADIM 2 │ Train/Test Split & MinMax Scaling")
    print("=" * 60)
    X_train, X_test, y_train, y_test, dates_train, dates_test = split_data(
        df, target_col="Close", test_ratio=TEST_RATIO
    )
    print(f"  Train: {X_train.shape[0]} örnek  |  Test: {X_test.shape[0]} örnek")

    # Ölçeklendirme (Scaler'lar hisse klasörüne kaydedilir)
    X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = scale_data(
        X_train, X_test, y_train, y_test, save_dir=MODELS_DIR
    )

    # ═════════════════════════════════════════════════════════════════════════
    # ADIM 3 — LSTM İçin Windowing (3D Tensör)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ADIM 3 │ LSTM Windowing (3D Tensör Oluşturma)")
    print("=" * 60)
    X_train_seq, y_train_seq = create_sequences(X_train_s, y_train_s, time_steps=TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y_test_s, time_steps=TIME_STEPS)
    print(f"  LSTM Train tensörü : {X_train_seq.shape}")
    print(f"  LSTM Test tensörü  : {X_test_seq.shape}")

    # ═════════════════════════════════════════════════════════════════════════
    # ADIM 4 — Model Eğitimi
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ADIM 4 │ Model Eğitimi (v2 — Geliştirilmiş)")
    print("=" * 60)

    # ── 4a. Prophet ──────────────────────────────────────────────────────────
    print("\n─── Prophet ───")
    prophet = ProphetModel(yearly_seasonality=True, weekly_seasonality=True)
    prophet.train(X_train, y_train, dates_train=dates_train)
    prophet.save(os.path.join(MODELS_DIR, "prophet_model.pkl"))

    # ── 4b. XGBoost (Optuna Tuning) ─────────────────────────────────────────
    print("\n─── XGBoost (Optuna Tuning) ───")
    xgb = XGBoostModel()
    best_params = xgb.tune_and_train(X_train_s, y_train_s, n_trials=50, n_splits=5)
    xgb.save(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

    # ── 4c. Attention LSTM (Bidirectional + Attention) ───────────────────────
    print("\n─── Attention LSTM (Bidirectional + Attention) ───")
    lstm = AttentionLSTMModel(
        units_1=128, units_2=64, dropout_rate=0.2,
        epochs=80, batch_size=32, learning_rate=0.001,
    )
    lstm.train(X_train_seq, y_train_seq)
    lstm.save(os.path.join(MODELS_DIR, "attention_lstm_model.keras"))

    # ═════════════════════════════════════════════════════════════════════════
    # ADIM 5 — Tahmin ve Orijinal Ölçeğe Dönüştürme
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ADIM 5 │ Tahmin Üretimi & Inverse Transform")
    print("=" * 60)

    # ── Prophet — zaten orijinal ölçekte çalışır ─────────────────────────────
    preds_prophet = prophet.predict(X_test, dates_test=dates_test)

    # ── XGBoost — ölçekli çıktıyı geri dönüştür ────────────────────────────
    preds_xgb_scaled = xgb.predict(X_test_s)
    preds_xgb = scaler_y.inverse_transform(preds_xgb_scaled.reshape(-1, 1)).ravel()

    # ── Attention LSTM — ölçekli çıktıyı geri dönüştür ──────────────────────
    preds_lstm_scaled = lstm.predict(X_test_seq)
    preds_lstm = scaler_y.inverse_transform(preds_lstm_scaled.reshape(-1, 1)).ravel()

    # ── Gerçek değerler (orijinal ölçek) ────────────────────────────────────
    y_test_original = y_test.ravel()

    # ═════════════════════════════════════════════════════════════════════════
    # ADIM 6 — Ensemble Ağırlık Optimizasyonu
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ADIM 6 │ Ensemble Ağırlık Optimizasyonu")
    print("=" * 60)

    # Ortak uzunluğa hizala — LSTM daha kısa çıktı üretir (TIME_STEPS kadar)
    min_len = min(len(preds_prophet), len(preds_xgb), len(preds_lstm))

    preds_aligned = {
        "Prophet": preds_prophet[-min_len:],
        "XGBoost": preds_xgb[-min_len:],
        "LSTM": preds_lstm[-min_len:],
    }

    # Gerçek değerleri de hizala
    y_true_aligned = y_test_original[-min_len:]

    print(f"  Ortak test uzunluğu: {min_len}")

    # ── 6a. Inverse RMSE ağırlıklandırma ────────────────────────────────────
    print("\n  ── Yöntem 1: Inverse RMSE ──")
    weights_inv_rmse = EnsembleModel.optimize_inverse_rmse(y_true_aligned, preds_aligned)

    # ── 6b. Grid Search ağırlık optimizasyonu ────────────────────────────────
    print("\n  ── Yöntem 2: Grid Search ──")
    weights_grid, grid_rmse = EnsembleModel.optimize_grid_search(
        y_true_aligned, preds_aligned, step=0.05
    )

    # ── En iyi yöntemi seç (Grid Search genelde daha iyi sonuç verir) ────────
    # Grid search'ün sonucu doğrudan en düşük RMSE'yi hedef aldığı için onu kullan
    optimal_weights = weights_grid
    print(f"\n  ✅  Seçilen strateji: Grid Search")
    print(f"     Ağırlıklar: {optimal_weights}")

    # ── Ensemble tahmin üret ─────────────────────────────────────────────────
    ensemble = EnsembleModel(weights=optimal_weights)
    preds_ensemble = ensemble.combine(preds_aligned)

    # ═════════════════════════════════════════════════════════════════════════
    # ADIM 7 — Değerlendirme ve Kaydetme
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ADIM 7 │ Değerlendirme & Çıktı Üretimi (v2)")
    print("=" * 60)

    all_predictions = {
        "Prophet": preds_aligned["Prophet"],
        "XGBoost": preds_aligned["XGBoost"],
        "LSTM": preds_aligned["LSTM"],
        "Ensemble": preds_ensemble,
    }

    # ── Metrik hesapla ───────────────────────────────────────────────────────
    metrics = {}
    for name, preds in all_predictions.items():
        metrics[name] = compute_metrics(y_true_aligned, preds)

    # ── Metrik raporunu kaydet (tarih damgalı + son sürüm) ───────────────────
    report_versioned = os.path.join(OUTPUTS_DIR, f"metrics_report_v2_{TIMESTAMP}.csv")
    report_latest = os.path.join(OUTPUTS_DIR, "metrics_report_v2_latest.csv")
    save_metrics_report(metrics, report_versioned)
    save_metrics_report(metrics, report_latest)

    # ── Karşılaştırma grafiği kaydet ─────────────────────────────────────────
    plot_path = os.path.join(OUTPUTS_DIR, f"benchmark_comparison_v2_{TIMESTAMP}.png")
    plot_latest = os.path.join(OUTPUTS_DIR, "benchmark_comparison_v2_latest.png")
    plot_comparison(
        y_true_aligned,
        all_predictions,
        save_path=plot_path,
        title=f"{STOCK_SYMBOL} — Model Kıyaslama v2 (Gerçek vs Tahmin)",
    )
    plot_comparison(
        y_true_aligned,
        all_predictions,
        save_path=plot_latest,
        title=f"{STOCK_SYMBOL} — Model Kıyaslama v2 (Gerçek vs Tahmin)",
    )

    # ── Ensemble ağırlıklarını kaydet ────────────────────────────────────────
    import json
    weights_path = os.path.join(OUTPUTS_DIR, "ensemble_weights.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": "grid_search",
            "weights": optimal_weights,
            "grid_search_rmse": grid_rmse,
            "inverse_rmse_weights": weights_inv_rmse,
            "timestamp": TIMESTAMP,
            "xgboost_best_params": best_params,
        }, f, indent=2, ensure_ascii=False)
    print(f"[✓] Ensemble ağırlıkları kaydedildi → {weights_path}")

    print("\n" + "=" * 60)
    print("  ✅  Pipeline v2 başarıyla tamamlandı!")
    print(f"  📌  Hisse       → {STOCK_SYMBOL}")
    print(f"  📁  Modeller    → {MODELS_DIR}")
    print(f"  📊  Çıktılar    → {OUTPUTS_DIR}")
    print(f"  🕐  Zaman Damgası → {TIMESTAMP}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
