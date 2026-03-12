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
    predictions : dict        Model adı -> tahmin dizisi.
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
    print(f"[OK] Karşılaştırma grafiği kaydedildi -> {save_path}")


# ── Metrik raporu (CSV) ──────────────────────────────────────────────────────

def save_metrics_report(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: str,
) -> pd.DataFrame:
    """
    Tüm modellerin metriklerini karşılaştırmalı ve detaylı olarak kaydeder.
    Klasik 3 metrik (RMSE, MAE, MAPE) yerine modelleri sıralar,
    zayıf modellerin en iyi modele göre ne kadar sapma gösterdiğini (% ve mutlak) ekler.

    Parameters
    ----------
    metrics_dict : dict
        Model adı -> {RMSE, MAE, MAPE} sözlüğü.
    save_path : str
        CSV dosyasının yolu.

    Returns
    -------
    pd.DataFrame  Gelişmiş metrik tablosu.
    """
    df = pd.DataFrame(metrics_dict).T
    df.index.name = "Model"

    # ── Gelişmiş Raporlama (Rank & Farklar) ─────────────────────────────────
    # RMSE'ye göre sırala (Düşük her zaman daha iyidir)
    df.sort_values(by="RMSE", inplace=True)

    # Sıralama kolonunu ekle (En iyi 1. Seçim, vb.)
    df.insert(0, "Sıra", [f"{i}." for i in range(1, len(df) + 1)])

    # En iyi (hedef alınan) modelin skoru
    best_rmse = df.iloc[0]["RMSE"]
    best_model_name = df.index[0]

    # Fark hesaplamaları
    df["RMSE_Fark_Delta"] = df["RMSE"] - best_rmse
    df["RMSE_Fark_Yüzde"] = ((df["RMSE"] / best_rmse) - 1.0) * 100

    # Formatlama — Virgülden sonra 2/4 hane, % işaretleri (okunabilirlik)
    df["RMSE_Fark_Delta"] = df["RMSE_Fark_Delta"].apply(lambda x: f"+{x:.4f}")
    df["RMSE_Fark_Yüzde"] = df["RMSE_Fark_Yüzde"].apply(lambda x: f"+%{x:.2f}")

    # İlk satırın (en iyi) fark hanelerini temizle daha kalıcı görünsün
    df.loc[df.index[0], "RMSE_Fark_Delta"] = "-"
    df.loc[df.index[0], "RMSE_Fark_Yüzde"] = "Referans"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Raporu Excel Türkçe standartlarına da daha uygun hale getirelim
    df.to_csv(save_path, sep=";")
    
    # ── Raporu Markdown Tablosu Olarak da Kaydet ────────────────────────────
    md_save_path = save_path.replace(".csv", ".md")
    with open(md_save_path, "w", encoding="utf-8") as f:
        f.write(f"## {best_model_name} Modeli Liderliğinde Performans Raporu\n\n")
        f.write(df.to_markdown(index=True))
        f.write(f"\n\n**Analiz Sonucu:**  Bu veri seti dinamiklerinde **{best_model_name}** `{best_rmse:.4f}` skor ile en düşük RMSE'yi üreterek zirvede yer alıyor.")
    
    print(f"[OK] Gelişmiş metrik raporu kaydedildi -> {save_path}")
    print(f"[OK] Markdown çıktı tablosu kaydedildi -> {md_save_path}")
    print("\n" + "=" * 70)
    print("  [INFO]  MODEL KARŞILAŞTIRMA VE PERFORMANS TABLOSU (v3)")
    print("=" * 70)
    print(df.to_string())
    print("-" * 70)
    print(f"  [INFO] En başarılı model: {best_model_name} (RMSE: {best_rmse:.4f})")
    print("=" * 70 + "\n")
    
    return df
