# -*- coding: utf-8 -*-
"""
evaluator.py - Evaluation and visualization helpers.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is skipped in minimal/headless runtimes
    plt = None

from src.utils.reporting_utils import (
    bullet_list,
    compact_columns,
    prepare_csv_dataframe,
    route_output_path,
    section_table,
    with_output_extension,
    write_csv_and_aligned_view,
)

_BASELINE_CANDIDATES = (
    "ARIMA",
    "Ridge Return",
    "ElasticNet Return",
    "LightGBM Return",
    "DLinear",
    "NLinear",
    "Naive Zero Return",
    "Naive Drift",
    "Naive Last Value",
)

METRICS_REPORT_COLUMNS = [
    "Model",
    "Sira",
    "Model_Family",
    "Score_Type",
    "RMSE",
    "MAE",
    "MAPE",
    "Return_RMSE",
    "Return_MAE",
    "Dir_Acc",
    "Hit_Rate",
    "Neutral_Rate",
    "Sharpe",
    "BuyHold_Sharpe",
    "Sharpe_excess_vs_buy_hold",
    "Composite_Score",
    "Benchmark_Model",
    "RMSE_vs_benchmark",
    "RMSE_vs_zero_return",
    "DirAcc_vs_benchmark",
    "Beats_Benchmark_RMSE",
    "Beats_Zero_Return_RMSE",
    "Eligible_For_Leader",
    "RMSE_Fark_Delta",
    "RMSE_Fark_Yuzde",
    "Pinball_Loss",
    "P10_P90_Coverage",
    "Avg_Interval_Width",
    "Winkler_Score",
]

METRICS_AUDIT_COLUMNS = [
    "Model",
    "Target_Semantics",
    "Execution_Lag",
    "Transaction_Costs",
    "Validation_Protocol",
    "Threshold_Config",
]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_true_target: np.ndarray | None = None,
    y_pred_target: np.ndarray | None = None,
    prev_close: np.ndarray | None = None,
    target_mode: str = "price",
) -> Dict[str, float]:
    """
    Compute core forecast and financial metrics.
    """
    from src.evaluation.financial_metrics import compute_financial_metrics

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    metrics = compute_financial_metrics(
        y_true,
        y_pred,
        y_true_target=y_true_target,
        y_pred_target=y_pred_target,
        prev_close=prev_close,
        target_mode=target_mode,
    )

    return {
        "RMSE": round(metrics["RMSE"], 4),
        "MAE": round(metrics["MAE"], 4),
        "MAPE": round(metrics["MAPE"], 4),
        "Return_RMSE": round(metrics["Return_RMSE"], 6),
        "Return_MAE": round(metrics["Return_MAE"], 6),
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
    zero_return_metrics = metrics_dict.get("Naive Zero Return", {})
    zero_return_rmse = float(zero_return_metrics.get("RMSE", np.nan))

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
        row["Mandatory_Zero_Return_RMSE"] = round(zero_return_rmse, 4) if np.isfinite(zero_return_rmse) else np.nan
        row["RMSE_vs_zero_return"] = round(rmse / max(zero_return_rmse, 1e-8), 4) if np.isfinite(zero_return_rmse) else np.nan
        row["DirAcc_vs_benchmark"] = round(dir_acc - benchmark_dir_acc, 2)
        row["Sharpe_excess_vs_buy_hold"] = round(sharpe - buy_hold_sharpe, 4)
        row["Beats_Benchmark_RMSE"] = rmse <= benchmark_rmse
        row["Beats_Zero_Return_RMSE"] = bool(np.isfinite(zero_return_rmse) and rmse <= zero_return_rmse)
        row["Beats_BuyHold_Sharpe"] = sharpe >= buy_hold_sharpe
        row["Eligible_For_Leader"] = bool(row["Beats_Benchmark_RMSE"])
        enriched[model_name] = row

    return enriched


def _leader_sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sort_df = df.copy()
    eligible = sort_df["Eligible_For_Leader"] if "Eligible_For_Leader" in sort_df.columns else True
    sort_df["_Leader_Eligible"] = eligible.astype(bool) if hasattr(eligible, "astype") else eligible
    if "Composite_Score" in sort_df.columns:
        sort_df.sort_values(
            by=["_Leader_Eligible", "Composite_Score", "RMSE"],
            ascending=[False, False, True],
            inplace=True,
        )
    else:
        sort_df.sort_values(
            by=["_Leader_Eligible", "RMSE"],
            ascending=[False, True],
            inplace=True,
        )
    return sort_df.drop(columns=["_Leader_Eligible"])


def plot_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: str,
    title: str = "Model Kiyaslama - Gercek vs Tahmin",
) -> None:
    """
    Plot actual prices and model predictions on one chart.
    """
    if plt is None:
        print("[WARN] matplotlib yok; karsilastirma grafigi atlandi.")
        return
    save_path = route_output_path(save_path)
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
    if plt is None:
        print("[WARN] matplotlib yok; tahmin araligi grafigi atlandi.")
        return
    save_path = route_output_path(save_path)
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
    Save model metrics as a compact CSV plus concise Markdown summary.

    The returned DataFrame keeps the full metric set for callers, but the
    persisted CSV intentionally excludes long protocol/audit blobs. Those fields
    remain in the Markdown audit section with truncated display text.
    """
    df = pd.DataFrame(metrics_dict).T
    df.index.name = "Model"

    df = _leader_sort_dataframe(df)

    df.insert(0, "Sira", [f"{i}." for i in range(1, len(df) + 1)])

    best_rmse = float(df.iloc[0]["RMSE"])
    best_model_name = df.index[0]
    benchmark_name = df.iloc[0]["Benchmark_Model"] if "Benchmark_Model" in df.columns else "-"
    score_type = "final_holdout_score" if "final_holdout" in os.path.basename(save_path) else "research_score"
    df["Score_Type"] = score_type

    df["RMSE_Fark_Delta"] = df["RMSE"] - best_rmse
    df["RMSE_Fark_Yuzde"] = ((df["RMSE"] / best_rmse) - 1.0) * 100

    df["RMSE_Fark_Delta"] = df["RMSE_Fark_Delta"].apply(lambda value: f"+{value:.4f}")
    df["RMSE_Fark_Yuzde"] = df["RMSE_Fark_Yuzde"].apply(lambda value: f"+%{value:.2f}")

    df.loc[df.index[0], "RMSE_Fark_Delta"] = "-"
    df.loc[df.index[0], "RMSE_Fark_Yuzde"] = "Referans"

    csv_df = df.reset_index()
    output_paths = write_csv_and_aligned_view(csv_df, save_path, columns=METRICS_REPORT_COLUMNS)

    md_save_path = with_output_extension(save_path, ".md")
    os.makedirs(os.path.dirname(md_save_path), exist_ok=True)
    display_df = prepare_csv_dataframe(compact_columns(csv_df, METRICS_REPORT_COLUMNS))
    audit_df = prepare_csv_dataframe(compact_columns(csv_df, METRICS_AUDIT_COLUMNS))
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
                        f"Score type: `{score_type}`",
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
                ["Model", "Model_Family", "Sira", "RMSE", "MAE", "MAPE", "Return_RMSE", "Return_MAE", "Composite_Score", "RMSE_Fark_Delta", "RMSE_Fark_Yuzde"],
            )
        )

        handle.write("\n\n## Forecast Metrikleri\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "RMSE", "MAE", "MAPE", "Return_RMSE", "Return_MAE"],
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
                [
                    "Model",
                    "Benchmark_Model",
                    "Benchmark_Source",
                    "RMSE_vs_benchmark",
                    "Mandatory_Zero_Return_RMSE",
                    "RMSE_vs_zero_return",
                    "DirAcc_vs_benchmark",
                    "Beats_Benchmark_RMSE",
                    "Beats_Zero_Return_RMSE",
                    "Beats_BuyHold_Sharpe",
                    "Eligible_For_Leader",
                    "Score_Type",
                ],
            )
        )

        probabilistic_table = section_table(
            display_df,
            ["Model", "Pinball_Loss", "P10_P90_Coverage", "Avg_Interval_Width", "Winkler_Score"],
        )
        if probabilistic_table:
            handle.write("\n\n## Probabilistic Metrikler\n\n")
            handle.write(probabilistic_table)

        handle.write("\n\n## Leakage Guard\n\n")
        handle.write(
            section_table(
                audit_df,
                METRICS_AUDIT_COLUMNS,
            )
        )

    print(f"[OK] Gelismis metrik raporu kaydedildi -> {output_paths['csv']}")
    print(f"[OK] Markdown cikti tablosu kaydedildi -> {md_save_path}")
    print("\n" + "=" * 70)
    print("  [INFO]  MODEL KARSILASTIRMA VE PERFORMANS TABLOSU")
    print("=" * 70)
    print(display_df.to_string(index=False))
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
