# -*- coding: utf-8 -*-
"""Faz 2 — model_scope sabitleri registry'den türetiliyor.

Eski tuple isimleri (`BENCHMARK_MODELS`, `CANDIDATE_MODELS`,
`DEFAULT_CANDIDATE_MODELS`) modül-seviyesi `__getattr__` ile çözülmeli;
içerik orijinal tuple ile birebir aynı olmalı.
"""
from __future__ import annotations

from src.pipeline.model_scope import (
    BENCHMARK_MODELS,
    CANDIDATE_MODELS,
    DEFAULT_CANDIDATE_MODELS,
    benchmark_models,
    candidate_models,
    default_candidate_models,
    is_benchmark_model,
    normalize_candidate_models,
)


def test_benchmark_models_content():
    assert set(BENCHMARK_MODELS) == {
        "Naive Last Value", "Naive Zero Return", "Naive Drift",
    }


def test_candidate_models_content():
    expected = {
        "Prophet", "ARIMA", "Ridge Return", "ElasticNet Return", "LightGBM Return",
        "DLinear", "NLinear", "XGBoost", "Random Forest", "LSTM",
    }
    assert set(CANDIDATE_MODELS) == expected


def test_default_candidate_models_content():
    assert set(DEFAULT_CANDIDATE_MODELS) == {
        "XGBoost", "LSTM", "DLinear", "NLinear",
    }


def test_canonical_ordering_preserved():
    """Tarihi tuple sıralaması bozulmamış."""
    assert BENCHMARK_MODELS == ("Naive Last Value", "Naive Zero Return", "Naive Drift")
    # CANDIDATE_MODELS sıralaması da preserve edilmeli.
    expected_order = (
        "Prophet", "ARIMA", "Ridge Return", "ElasticNet Return", "LightGBM Return",
        "DLinear", "NLinear", "XGBoost", "Random Forest", "LSTM",
    )
    assert CANDIDATE_MODELS == expected_order


def test_module_getattr_unknown_raises():
    import src.pipeline.model_scope as scope
    import pytest
    with pytest.raises(AttributeError):
        scope.NOT_A_REAL_CONSTANT  # noqa: B018


def test_function_form_matches_tuple_form():
    assert benchmark_models() == BENCHMARK_MODELS
    assert candidate_models() == CANDIDATE_MODELS
    assert default_candidate_models() == DEFAULT_CANDIDATE_MODELS


def test_normalize_with_unknown_filters_out():
    result = normalize_candidate_models(["XGBoost", "__BogusModel__"])
    assert "XGBoost" in result
    assert "__BogusModel__" not in result


def test_normalize_empty_returns_defaults():
    result = normalize_candidate_models(None)
    assert result == set(DEFAULT_CANDIDATE_MODELS)


def test_is_benchmark_model():
    assert is_benchmark_model("Naive Zero Return") is True
    assert is_benchmark_model("Random Forest") is False
