# -*- coding: utf-8 -*-
"""
evaluator.py — Değerlendirme ve Görselleştirme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tahmin sonuçlarını RMSE, MAE, MAPE metrikleri ile değerlendirir,
tüm modellerin karşılaştırmalı grafiğini çizer ve sonuçları
CSV / PNG olarak diske kaydeder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict


# ── Metrik hesaplama ─────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    RMSE, MAE ve MAPE hesaplar. Tüm değerler **orijinal ölçekte** olmalıdır
    (inverse_transform yapılmış).

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    dict  {'RMSE': float, 'MAE': float, 'MAPE': float}
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # MAPE — sıfır bölme korumalı
    mask = y_true != 0
    if mask.sum() == 0:
        mape = float("inf")
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)}


# ── Karşılaştırma grafiği ────────────────────────────────────────────────────

def plot_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: str,
    title: str = "Model Kıyaslama — Gerçek vs Tahmin",
) -> None:
    """
    Tek bir grafik üzerinde gerçek fiyatı ve tüm model tahminlerini çizer.

    Parameters
    ----------
    y_true : np.ndarray       Gerçek fiyat dizisi (ortak uzunlukta).
    predictions : dict        Model adı → tahmin dizisi.
    save_path : str           PNG dosyasının kaydedileceği tam yol.
    title : str               Grafik başlığı.
    """
    plt.figure(figsize=(16, 7))
    plt.plot(y_true, label="Gerçek Fiyat", color="black", linewidth=2.0, alpha=0.85)

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    for idx, (name, preds) in enumerate(predictions.items()):
        color = colors[idx % len(colors)]
        plt.plot(preds, label=name, color=color, linewidth=1.4, alpha=0.80)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Test Seti İndeksi")
    plt.ylabel("Kapanış Fiyatı (₺)")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Karşılaştırma grafiği kaydedildi → {save_path}")


# ── Metrik raporu (CSV) ──────────────────────────────────────────────────────

def save_metrics_report(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: str,
) -> pd.DataFrame:
    """
    Tüm modellerin metriklerini CSV olarak kaydeder.

    Parameters
    ----------
    metrics_dict : dict
        Model adı → {RMSE, MAE, MAPE} sözlüğü.
    save_path : str
        CSV dosyasının yolu.

    Returns
    -------
    pd.DataFrame  Metrik tablosu.
    """
    df = pd.DataFrame(metrics_dict).T
    df.index.name = "Model"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"[✓] Metrik raporu kaydedildi → {save_path}")
    print("\n" + "=" * 55)
    print("  📊  MODEL KIYaSLAMA SONUÇLARI")
    print("=" * 55)
    print(df.to_string())
    print("=" * 55 + "\n")
    return df
