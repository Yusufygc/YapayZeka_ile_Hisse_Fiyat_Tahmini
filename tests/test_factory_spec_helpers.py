# -*- coding: utf-8 -*-
"""Faz 3 — spec helper'ları ve dinamik set sabitleri registry'den türetiliyor."""
from __future__ import annotations

from src.pipeline import model_factory


def test_benchmark_specs_return_modes():
    """Return ve log_return için Zero Return dahil; sıralama: Last, Zero, Drift."""
    for mode in ("return", "log_return"):
        specs = model_factory.benchmark_specs(mode)
        names = [name for name, _ in specs]
        assert names == ["Naive Last Value", "Naive Zero Return", "Naive Drift"]


def test_benchmark_specs_price_mode_excludes_zero_return():
    """Price hedefinde Naive Zero Return target_modes uyuşmaz, hariç tutulur."""
    specs = model_factory.benchmark_specs("price")
    names = [name for name, _ in specs]
    assert "Naive Zero Return" not in names
    assert "Naive Last Value" in names
    assert "Naive Drift" in names


def test_linear_baseline_specs_contents():
    specs = model_factory.linear_baseline_specs()
    names = {name for name, _ in specs}
    assert names == {"Ridge Return", "ElasticNet Return"}


def test_boosting_baseline_specs_contents():
    specs = model_factory.boosting_baseline_specs()
    names = [name for name, _ in specs]
    assert names == ["LightGBM Return"]


def test_sequence_baseline_specs_ordering():
    """DLinear önce, NLinear sonra (tarihi sıralama)."""
    specs = model_factory.sequence_baseline_specs()
    names = [name for name, _ in specs]
    assert names[:2] == ["DLinear", "NLinear"]


def test_spec_helpers_return_callable_factories():
    """Her spec helper (name, callable) tuple döner — caller `cls()` çağırabilmeli."""
    import numpy as np
    rng = np.random.default_rng(0)

    for name, cls in model_factory.linear_baseline_specs():
        instance = cls()
        instance.train(rng.normal(size=(20, 4)), rng.normal(size=20))
        out = instance.predict(rng.normal(size=(3, 4)))
        assert len(out) == 3


def test_tree_models_dynamic():
    tree = model_factory.TREE_MODELS
    assert isinstance(tree, set)
    assert {"XGBoost", "Random Forest", "Ridge Return", "ElasticNet Return", "LightGBM Return"}.issubset(tree)


def test_seq_models_dynamic():
    seq = model_factory.SEQ_MODELS
    assert isinstance(seq, set)
    assert {"LSTM", "TFT", "DLinear", "NLinear"}.issubset(seq)


def test_benchmark_model_set_dynamic():
    bench = model_factory.BENCHMARK_MODEL_SET
    assert bench == {"Naive Last Value", "Naive Zero Return", "Naive Drift"}


def test_all_models_dynamic():
    all_models = model_factory.ALL_MODELS
    assert isinstance(all_models, list)
    # Default candidate kümesi (Faz 2 sözleşmesi).
    assert set(all_models) == {"XGBoost", "LSTM", "TFT", "DLinear", "NLinear"}


def test_factory_getattr_unknown_raises():
    import pytest
    with pytest.raises(AttributeError):
        model_factory.NOT_A_REAL_CONSTANT  # noqa: B018
