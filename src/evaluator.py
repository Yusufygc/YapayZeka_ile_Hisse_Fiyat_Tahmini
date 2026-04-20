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
from typing import Dict

_BENCHMARK_REFERENCE_ORDER = (
    "Naive Last Value",
    "Naive Drift",
    "Naive Zero Return",
)


# ── Metrik hesaplama ─────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    RMSE, MAE, MAPE ve gelişmiş finansal metrikleri hesaplar.
    """
    from src.evaluation.financial_metrics import compute_financial_metrics
    
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # compute_financial_metrics returns MAE, RMSE, MAPE, Dir_Acc, Sharpe, Hit_Rate
    metrics = compute_financial_metrics(y_true, y_pred)
    
    # Formatlama
    return {
        "RMSE": round(metrics["RMSE"], 4),
        "MAE": round(metrics["MAE"], 4),
        "MAPE": round(metrics["MAPE"], 4),
        "Dir_Acc": round(metrics["Dir_Acc"], 2),
        "Sharpe": round(metrics["Sharpe"], 4),
        "Hit_Rate": round(metrics["Hit_Rate"], 2),
        "Neutral_Rate": round(metrics["Neutral_Rate"], 2),
        "BuyHold_Sharpe": round(metrics["BuyHold_Sharpe"], 4),
    }


def select_benchmark_model(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """
    Relative metrikler için kullanılacak temel naive benchmark modelini seçer.
    """
    for benchmark_name in _BENCHMARK_REFERENCE_ORDER:
        if benchmark_name in metrics_dict:
            return benchmark_name

    if not metrics_dict:
        raise ValueError("Benchmark seçimi için metrics_dict boş olamaz.")

    return next(iter(metrics_dict))


def enrich_with_benchmark_metrics(
    metrics_dict: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Her model için naive benchmark ve buy-hold referansına göre relative metrikler ekler.
    """
    if not metrics_dict:
        return metrics_dict

    benchmark_name = select_benchmark_model(metrics_dict)
    benchmark_metrics = metrics_dict[benchmark_name]
    naive_rmse = max(float(benchmark_metrics.get("RMSE", 0.0)), 1e-8)
    naive_dir_acc = float(benchmark_metrics.get("Dir_Acc", 0.0))

    enriched = {}
    for model_name, model_metrics in metrics_dict.items():
        row = dict(model_metrics)
        rmse = float(row.get("RMSE", 0.0))
        dir_acc = float(row.get("Dir_Acc", 0.0))
        sharpe = float(row.get("Sharpe", 0.0))
        buy_hold_sharpe = float(row.get("BuyHold_Sharpe", 0.0))

        row["Benchmark_Model"] = benchmark_name
        row["RMSE_vs_naive"] = round(rmse / naive_rmse, 4)
        row["DirAcc_vs_naive"] = round(dir_acc - naive_dir_acc, 2)
        row["Sharpe_excess_vs_buy_hold"] = round(sharpe - buy_hold_sharpe, 4)
        row["Beats_Naive_RMSE"] = rmse <= naive_rmse
        enriched[model_name] = row

    return enriched


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


def plot_prediction_interval(
    y_true: np.ndarray,
    median_pred: np.ndarray,
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
    save_path: str,
    title: str = "Tahmin Aralığı",
) -> None:
    """
    Tek model için alt-üst quantile bandını görselleştirir.
    """
    plt.figure(figsize=(16, 7))
    x_axis = np.arange(len(y_true))
    plt.plot(x_axis, y_true, label="Gerçek Fiyat", color="black", linewidth=2.0, alpha=0.9)
    plt.plot(x_axis, median_pred, label="P50 Tahmin", color="#c0392b", linewidth=1.8, alpha=0.9)
    plt.fill_between(
        x_axis,
        lower_pred,
        upper_pred,
        color="#f5b7b1",
        alpha=0.35,
        label="P10-P90 Bandı",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Test Seti İndeksi")
    plt.ylabel("Kapanış Fiyatı (₺)")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Tahmin aralığı grafiği kaydedildi -> {save_path}")


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
    if "Composite_Score" in df.columns:
        df.sort_values(by=["Composite_Score", "RMSE"], ascending=[False, True], inplace=True)
    else:
        df.sort_values(by="RMSE", inplace=True)

    # Sıralama kolonunu ekle (En iyi 1. Seçim, vb.)
    df.insert(0, "Sıra", [f"{i}." for i in range(1, len(df) + 1)])

    # En iyi (hedef alınan) modelin skoru
    best_rmse = df.iloc[0]["RMSE"]
    best_model_name = df.index[0]
    benchmark_name = df.iloc[0]["Benchmark_Model"] if "Benchmark_Model" in df.columns else "-"

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
        if "Composite_Score" in df.columns:
            best_composite = df.iloc[0]["Composite_Score"]
            f.write(
                f"\n\n**Analiz Sonucu:** Referans benchmark `{benchmark_name}` iken "
                f"**{best_model_name}** modeli `{best_composite:.4f}` composite skor ile ilk sırada."
            )
        else:
            f.write(f"\n\n**Analiz Sonucu:**  Bu veri seti dinamiklerinde **{best_model_name}** `{best_rmse:.4f}` skor ile en düşük RMSE'yi üreterek zirvede yer alıyor.")
    
    print(f"[OK] Gelişmiş metrik raporu kaydedildi -> {save_path}")
    print(f"[OK] Markdown çıktı tablosu kaydedildi -> {md_save_path}")
    print("\n" + "=" * 70)
    print("  [INFO]  MODEL KARŞILAŞTIRMA VE PERFORMANS TABLOSU (v3)")
    print("=" * 70)
    print(df.to_string())
    print("-" * 70)
    if "Composite_Score" in df.columns:
        print(
            f"  [INFO] En başarılı model: {best_model_name} "
            f"(Composite: {df.iloc[0]['Composite_Score']:.4f}, Benchmark: {benchmark_name})"
        )
    else:
        print(f"  [INFO] En başarılı model: {best_model_name} (RMSE: {best_rmse:.4f})")
    print("=" * 70 + "\n")
    
    return df
