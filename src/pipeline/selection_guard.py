# -*- coding: utf-8 -*-
"""Production best-model eligibility guards."""
from __future__ import annotations

from typing import Any, Dict, Literal

EligibilityStatus = Literal[
    "eligible",
    "naive_low_trades",
    "insufficient_trades",
    "no_candidate",
    "benchmark_failed",
]

MIN_TRADES_DEFAULT = 6
PRODUCTION_ENSEMBLE_METHODS = {"Inverse RMSE", "Cash-Gated"}


def compute_eligibility(
    *,
    model_name: str,
    is_production_candidate: bool,
    is_baseline: bool,
    total_trade_count: int,
    rmse_vs_benchmark: float | None = None,
    min_trades: int = MIN_TRADES_DEFAULT,
) -> tuple[EligibilityStatus, str]:
    if not is_production_candidate:
        return ("no_candidate", "Model production candidate degil.")
    if rmse_vs_benchmark is not None and rmse_vs_benchmark > 1.0:
        return (
            "benchmark_failed",
            f"Model benchmark'i gecemedi (RMSE_vs_benchmark={rmse_vs_benchmark:.3f}).",
        )
    if is_baseline and total_trade_count < min_trades:
        return (
            "naive_low_trades",
            f"Naive model lider, ancak trade sayisi ({total_trade_count}) minimum esigin ({min_trades}) altinda.",
        )
    if total_trade_count < min_trades:
        return (
            "insufficient_trades",
            f"Trade sayisi ({total_trade_count}) minimum esigin ({min_trades}) altinda.",
        )
    return ("eligible", "")


def evaluate_best_model_eligibility(
    experiment_row: Dict[str, Any],
    *,
    min_trades: int = MIN_TRADES_DEFAULT,
) -> tuple[EligibilityStatus, str]:
    from src.pipeline.model_scope import is_benchmark_model

    model_name = str(experiment_row.get("model_name", ""))
    is_prod = bool(experiment_row.get("is_production_candidate", 0))

    if model_name.startswith("Ensemble "):
        metadata = experiment_row.get("ensemble_metadata") or {}
        method = (
            str(metadata.get("method", model_name.replace("Ensemble ", "")))
            if isinstance(metadata, dict)
            else model_name.replace("Ensemble ", "")
        )
        if method not in PRODUCTION_ENSEMBLE_METHODS:
            is_prod = False

    is_base = bool(experiment_row.get("is_baseline", is_benchmark_model(model_name)))

    try:
        trade_count = int(float(experiment_row.get("Trade_Count", 0) or 0))
    except (TypeError, ValueError):
        trade_count = 0

    try:
        raw_rmse_ratio = experiment_row.get("RMSE_vs_benchmark")
        rmse_vs_benchmark = None if raw_rmse_ratio in (None, "") else float(raw_rmse_ratio)
    except (TypeError, ValueError):
        rmse_vs_benchmark = None

    return compute_eligibility(
        model_name=model_name,
        is_production_candidate=is_prod,
        is_baseline=is_base,
        total_trade_count=trade_count,
        rmse_vs_benchmark=rmse_vs_benchmark,
        min_trades=min_trades,
    )
