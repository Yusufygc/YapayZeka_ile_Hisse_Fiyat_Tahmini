# -*- coding: utf-8 -*-
"""
evaluator.py - Evaluation and visualization helpers.
"""

import os
from typing import Dict

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is skipped in minimal/headless runtimes
    plt = None

from src.utils.reporting_utils import route_output_path

_BASELINE_CANDIDATES = (
    "Ridge Return",
    "LightGBM Return",
    "Naive Zero Return",
    "Naive Drift",
    "Naive Last Value",
)


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

    def _round_or_nan(value: float, digits: int) -> float:
        # Sprint 1 A1.1: NaN korunmali ki risk_free_unavailable durumu metric'lere yansisin.
        if value is None or not np.isfinite(value):
            return float("nan")
        return round(value, digits)

    return {
        "RMSE": _round_or_nan(metrics["RMSE"], 4),
        "MAE": _round_or_nan(metrics["MAE"], 4),
        "MAPE": _round_or_nan(metrics["MAPE"], 4),
        "Return_RMSE": _round_or_nan(metrics["Return_RMSE"], 6),
        "Return_MAE": _round_or_nan(metrics["Return_MAE"], 6),
        "Dir_Acc": _round_or_nan(metrics["Dir_Acc"], 2),
        "Sharpe": _round_or_nan(metrics["Sharpe"], 4),
        "Hit_Rate": _round_or_nan(metrics["Hit_Rate"], 2),
        "Neutral_Rate": _round_or_nan(metrics["Neutral_Rate"], 2),
        "BuyHold_Sharpe": _round_or_nan(metrics["BuyHold_Sharpe"], 4),
        # Sprint 1 A1.1: rf yoksa metric raporlarda gorulebilsin.
        "Risk_Free_Unavailable": bool(metrics.get("Risk_Free_Unavailable", False)),
        "Risk_Free_Annual_Used": metrics.get("Risk_Free_Annual_Used"),
        "Sharpe_Warning": metrics.get("Sharpe_Warning", ""),
    }


def select_benchmark_model(metrics_dict: Dict[str, Dict[str, float]]) -> str | None:
    """
    Pick the strongest baseline benchmark model for relative metrics.

    Returns ``None`` when no baseline/benchmark model is present in
    ``metrics_dict``. A non-benchmark model is NEVER returned as its own
    benchmark (that would force RMSE_vs_benchmark=1.0 / DirAcc_vs_benchmark=0
    and silently mark the model as leader-eligible).
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

    return None


def enrich_with_benchmark_metrics(
    metrics_dict: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Add benchmark-relative metrics for each model.
    """
    if not metrics_dict:
        return metrics_dict

    benchmark_name = select_benchmark_model(metrics_dict)
    if benchmark_name is None:
        benchmark_rmse = float("nan")
        benchmark_dir_acc = float("nan")
    else:
        benchmark_metrics = metrics_dict[benchmark_name]
        benchmark_rmse = max(float(benchmark_metrics.get("RMSE", 0.0)), 1e-8)
        benchmark_dir_acc = float(benchmark_metrics.get("Dir_Acc", 0.0))
    benchmark_available = benchmark_name is not None
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
        row["Benchmark_Source"] = (
            "best_baseline_by_rmse" if benchmark_available else "no_benchmark_available"
        )
        # Benchmark yoksa goreli metrikler NaN: model kendi kendisinin benchmark'i
        # olamaz ve lider kapisini gecemez (Eligible_For_Leader=False).
        row["RMSE_vs_benchmark"] = round(rmse / benchmark_rmse, 4) if benchmark_available else np.nan
        row["Mandatory_Zero_Return_RMSE"] = round(zero_return_rmse, 4) if np.isfinite(zero_return_rmse) else np.nan
        row["RMSE_vs_zero_return"] = round(rmse / max(zero_return_rmse, 1e-8), 4) if np.isfinite(zero_return_rmse) else np.nan
        # Bilgi amaclidir: Composite_Score formulune ve lider kapisina GIRMEZ
        # (benchmark Naive Zero Return iken ~Dir_Acc sabit bonusa doyuyordu).
        row["DirAcc_vs_benchmark"] = round(dir_acc - benchmark_dir_acc, 2) if benchmark_available else np.nan
        row["Sharpe_excess_vs_buy_hold"] = round(sharpe - buy_hold_sharpe, 4)
        row["Beats_Benchmark_RMSE"] = bool(benchmark_available and rmse <= benchmark_rmse)
        row["Beats_Zero_Return_RMSE"] = bool(np.isfinite(zero_return_rmse) and rmse <= zero_return_rmse)
        row["Beats_BuyHold_Sharpe"] = sharpe >= buy_hold_sharpe
        row["Eligible_For_Leader"] = bool(row["Beats_Benchmark_RMSE"])
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
