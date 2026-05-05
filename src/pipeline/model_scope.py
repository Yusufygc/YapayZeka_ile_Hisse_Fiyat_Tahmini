# -*- coding: utf-8 -*-
"""Shared model-scope rules for training, reporting and production selection."""

from __future__ import annotations

from typing import Iterable


BENCHMARK_MODELS: tuple[str, ...] = (
    "Naive Last Value",
    "Naive Zero Return",
    "Naive Drift",
)

CANDIDATE_MODELS: tuple[str, ...] = (
    "Prophet",
    "ARIMA",
    "Ridge Return",
    "ElasticNet Return",
    "LightGBM Return",
    "DLinear",
    "NLinear",
    "XGBoost",
    "Random Forest",
    "LSTM",
    "TFT",
)

DEFAULT_CANDIDATE_MODELS: tuple[str, ...] = (
    "XGBoost",
    "LSTM",
    "TFT",
    "DLinear",
    "NLinear",
)


def normalize_candidate_models(selected_models: Iterable[str] | None) -> set[str]:
    """Return the production candidate model set for a run."""
    if selected_models:
        return {str(model) for model in selected_models if str(model) in CANDIDATE_MODELS}
    return set(DEFAULT_CANDIDATE_MODELS)


def is_benchmark_model(model_name: str) -> bool:
    return str(model_name) in BENCHMARK_MODELS


def is_selection_candidate(model_name: str, candidate_models: Iterable[str] | None) -> bool:
    return str(model_name) in set(candidate_models or [])


def report_group(model_name: str, candidate_models: Iterable[str] | None) -> str:
    name = str(model_name)
    if is_selection_candidate(name, candidate_models):
        return "candidate"
    if is_benchmark_model(name):
        return "benchmark"
    if name.startswith("Ensemble "):
        return "ensemble"
    return "comparison"


def reportable_model_names(model_names: Iterable[str], candidate_models: Iterable[str] | None) -> set[str]:
    """Main reports include only selected candidates and cheap naive benchmarks."""
    candidates = set(candidate_models or [])
    return {
        str(name)
        for name in model_names
        if str(name) in candidates or is_benchmark_model(str(name))
    }
