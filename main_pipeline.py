# -*- coding: utf-8 -*-
"""
main_pipeline.py — Orkestrasyon Dosyası (v3 — OOP & Clean Code Geliştirmesi)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tüm sistemi baştan sona nesne yönelimli (OOP) mimariyle çalıştırır.
Katmanlar ForecastingPipeline içinde kapsüllenmiştir.

Adımlar:
  1. Veri yükle & temizle (yfinance ile eksik gün tamamlama özelliği dahil)
  2. Train/Test böl & ölçekle
  3. Prophet, XGBoost, Random Forest ve Attention LSTM modellerini eğit (Optuna dahil)
  4. Tahminleri orijinal ölçeğe dönüştür (Inverse Transform)
  5. Tüm modelleri değerlendir ve metrikleri CSV/PNG olarak kaydet

Kullanım:
    python main_pipeline.py
"""

import os
from src.pipeline_manager import ForecastingPipeline

# ═════════════════════════════════════════════════════════════════════════════
# YAPILANDIRMA
# ═════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "AKSA.csv")


def main() -> None:
    """Temiz ve modüler ana fonksiyon."""
    pipeline = ForecastingPipeline(
        data_file=DATA_FILE,
        test_ratio=0.20,
        time_steps=30
    )
    pipeline.run_all()


if __name__ == "__main__":
    main()
