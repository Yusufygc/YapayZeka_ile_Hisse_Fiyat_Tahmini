# -*- coding: utf-8 -*-
"""
evaluator.py - Evaluation and visualization helpers.
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.reporting_utils import bullet_list, prepare_csv_dataframe, section_table, write_csv_and_aligned_view

_BASELINE_CANDIDATES = (
    "ARIMA",
    "Naive Zero Return",
    "Naive Drift",
    "Naive Last Value",
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute core forecast and financial metrics.
    """
    from src.evaluation.financial_metrics import compute_financial_metrics

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    metrics = compute_financial_metrics(y_true, y_pred)

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
    Pick the strongest baseline benchmark model for relative metrics.
    """
    if not metrics_dict:
        raise ValueError("Benchmark secimi icin metrics_dict bos olamaz.")

    available_candidates = [
        benchmark_name
        for benchmark_name in _BASELINE_CANDIDATES
        if benchmark_name in metrics_dict
    ]
    if available_candidates:
        return min(
            available_candidates,
            key=lambda benchmark_name: (
                float(metrics_dict[benchmark_name].get("RMSE", float("inf"))),
                float(metrics_dict[benchmark_name].get("MAE", float("inf"))),
                -float(metrics_dict[benchmark_name].get("Dir_Acc", float("-inf"))),
                _BASELINE_CANDIDATES.index(benchmark_name),
            ),
        )

    return next(iter(metrics_dict))


def enrich_with_benchmark_metrics(
    metrics_dict: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Add benchmark-relative metrics for each model.
    """
    if not metrics_dict:
        return metrics_dict

    benchmark_name = select_benchmark_model(metrics_dict)
    benchmark_metrics = metrics_dict[benchmark_name]
    benchmark_rmse = max(float(benchmark_metrics.get("RMSE", 0.0)), 1e-8)
    benchmark_dir_acc = float(benchmark_metrics.get("Dir_Acc", 0.0))

    enriched = {}
    for model_name, model_metrics in metrics_dict.items():
        row = dict(model_metrics)
        rmse = float(row.get("RMSE", 0.0))
        dir_acc = float(row.get("Dir_Acc", 0.0))
        sharpe = float(row.get("Sharpe", 0.0))
        buy_hold_sharpe = float(row.get("BuyHold_Sharpe", 0.0))

        row["Benchmark_Model"] = benchmark_name
        row["Benchmark_Source"] = "best_baseline_by_rmse"
        row["RMSE_vs_benchmark"] = round(rmse / benchmark_rmse, 4)
        row["DirAcc_vs_benchmark"] = round(dir_acc - benchmark_dir_acc, 2)
        row["Sharpe_excess_vs_buy_hold"] = round(sharpe - buy_hold_sharpe, 4)
        row["Beats_Benchmark_RMSE"] = rmse <= benchmark_rmse
        enriched[model_name] = row

    return enriched


def plot_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: str,
    title: str = "Model Kiyaslama - Gercek vs Tahmin",
) -> None:
    """
    Plot actual prices and model predictions on one chart.
    """
    plt.figure(figsize=(16, 7))
    plt.plot(y_true, label="Gercek Fiyat", color="black", linewidth=2.0, alpha=0.85)

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    for idx, (name, preds) in enumerate(predictions.items()):
        color = colors[idx % len(colors)]
        plt.plot(preds, label=name, color=color, linewidth=1.4, alpha=0.80)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Test Seti Indeksi")
    plt.ylabel("Kapanis Fiyati")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Karsilastirma grafigi kaydedildi -> {save_path}")


def plot_prediction_interval(
    y_true: np.ndarray,
    median_pred: np.ndarray,
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
    save_path: str,
    title: str = "Tahmin Araligi",
) -> None:
    """
    Plot lower/upper quantile bands for a single model.
    """
    plt.figure(figsize=(16, 7))
    x_axis = np.arange(len(y_true))
    plt.plot(x_axis, y_true, label="Gercek Fiyat", color="black", linewidth=2.0, alpha=0.9)
    plt.plot(x_axis, median_pred, label="P50 Tahmin", color="#c0392b", linewidth=1.8, alpha=0.9)
    plt.fill_between(
        x_axis,
        lower_pred,
        upper_pred,
        color="#f5b7b1",
        alpha=0.35,
        label="P10-P90 Bandi",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Test Seti Indeksi")
    plt.ylabel("Kapanis Fiyati")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Tahmin araligi grafigi kaydedildi -> {save_path}")


def save_metrics_report(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: str,
) -> pd.DataFrame:
    """
    Save model metrics with cleaner CSV, aligned text, and structured Markdown output.
    """
    df = pd.DataFrame(metrics_dict).T
    df.index.name = "Model"

    if "Composite_Score" in df.columns:
        df.sort_values(by=["Composite_Score", "RMSE"], ascending=[False, True], inplace=True)
    else:
        df.sort_values(by="RMSE", inplace=True)

    df.insert(0, "Sira", [f"{i}." for i in range(1, len(df) + 1)])

    best_rmse = float(df.iloc[0]["RMSE"])
    best_model_name = df.index[0]
    benchmark_name = df.iloc[0]["Benchmark_Model"] if "Benchmark_Model" in df.columns else "-"

    df["RMSE_Fark_Delta"] = df["RMSE"] - best_rmse
    df["RMSE_Fark_Yuzde"] = ((df["RMSE"] / best_rmse) - 1.0) * 100

    df["RMSE_Fark_Delta"] = df["RMSE_Fark_Delta"].apply(lambda value: f"+{value:.4f}")
    df["RMSE_Fark_Yuzde"] = df["RMSE_Fark_Yuzde"].apply(lambda value: f"+%{value:.2f}")

    df.loc[df.index[0], "RMSE_Fark_Delta"] = "-"
    df.loc[df.index[0], "RMSE_Fark_Yuzde"] = "Referans"

    csv_df = df.reset_index()
    write_csv_and_aligned_view(csv_df, save_path)

    md_save_path = save_path.replace(".csv", ".md")
    display_df = prepare_csv_dataframe(csv_df)
    with open(md_save_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {best_model_name} Liderliginde Performans Raporu\n\n")
        handle.write("## Ozet\n\n")

        if "Composite_Score" in df.columns:
            best_composite = float(df.iloc[0]["Composite_Score"])
            best_beats_benchmark = bool(df.iloc[0].get("Beats_Benchmark_RMSE", False))
            benchmark_verdict = "benchmark'i geciyor" if best_beats_benchmark else "benchmark'i gecemiyor"
            handle.write(
                bullet_list(
                    [
                        f"Lider model: `{best_model_name}`",
                        f"Composite score: `{best_composite:.4f}`",
                        f"Referans benchmark: `{benchmark_name}`",
                        f"RMSE durumu: `{benchmark_verdict}`",
                    ]
                )
            )
        else:
            handle.write(
                bullet_list(
                    [
                        f"Lider model: `{best_model_name}`",
                        f"En iyi RMSE: `{best_rmse:.4f}`",
                    ]
                )
            )

        handle.write("\n\n## Siralama Ozeti\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "Sira", "RMSE", "MAE", "MAPE", "Composite_Score", "RMSE_Fark_Delta", "RMSE_Fark_Yuzde"],
            )
        )

        handle.write("\n\n## Yon ve Risk Metrikleri\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "Dir_Acc", "Hit_Rate", "Neutral_Rate", "Sharpe", "BuyHold_Sharpe", "Sharpe_excess_vs_buy_hold"],
            )
        )

        handle.write("\n\n## Benchmark Karsilastirmasi\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "Benchmark_Model", "Benchmark_Source", "RMSE_vs_benchmark", "DirAcc_vs_benchmark", "Beats_Benchmark_RMSE"],
            )
        )

    print(f"[OK] Gelismis metrik raporu kaydedildi -> {save_path}")
    print(f"[OK] Markdown cikti tablosu kaydedildi -> {md_save_path}")
    print("\n" + "=" * 70)
    print("  [INFO]  MODEL KARSILASTIRMA VE PERFORMANS TABLOSU")
    print("=" * 70)
    print(prepare_csv_dataframe(csv_df).to_string(index=False))
    print("-" * 70)
    if "Composite_Score" in df.columns:
        print(
            f"  [INFO] En basarili model: {best_model_name} "
            f"(Composite: {df.iloc[0]['Composite_Score']:.4f}, Benchmark: {benchmark_name})"
        )
    else:
        print(f"  [INFO] En basarili model: {best_model_name} (RMSE: {best_rmse:.4f})")
    print("=" * 70 + "\n")

    return df
