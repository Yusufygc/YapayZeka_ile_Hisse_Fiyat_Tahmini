# -*- coding: utf-8 -*-
"""Faz 2 — model_factory.model_class_for_name registry'e devredildi.

Davranış korunmalı: eski hardcode mapping ile aynı sonuçları verir.
"""
from __future__ import annotations

import pytest

from src.pipeline import model_factory


@pytest.mark.parametrize("name", [
    "Naive Last Value",
    "Naive Zero Return",
    "Naive Drift",
    "Ridge Return",
])
def test_known_models_resolve_tabular(name):
    """Tabular modeller için sıfır-arg cls() train/predict çağrılabilmeli."""
    cls = model_factory.model_class_for_name(name, {}, "log_return")
    instance = cls()
    assert instance is not None
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20)
    instance.train(X, y)
    out = instance.predict(rng.normal(size=(5, 4)))
    assert len(out) == 5


def test_unknown_model_raises_keyerror():
    with pytest.raises(KeyError) as exc:
        model_factory.model_class_for_name("__NoSuchModel__", {}, "log_return")
    assert "Bilinmeyen model adi" in str(exc.value)


@pytest.mark.parametrize("name", ["LSTM", "LSTM Lite", "AttentionLSTM v2", "DLinear", "NLinear", "Prophet", "ARIMA", "ElasticNet Return", "Random Forest"])
def test_removed_models_are_unknown_to_factory(name):
    with pytest.raises(KeyError):
        model_factory.model_class_for_name(name, {}, "log_return")


def test_naive_zero_return_rejects_price_target():
    """Naive Zero Return target_modes = (return, log_return) — price desteklemez."""
    with pytest.raises(KeyError):
        model_factory.model_class_for_name("Naive Zero Return", {}, "price")


def test_model_class_returns_callable():
    """model_class_for_name daima sıfır-arg callable döner — eski davranış."""
    cls = model_factory.model_class_for_name("Ridge Return", {}, "log_return")
    assert callable(cls)
    # `cls()` çağrılabilir olmalı, sınıf değil çağrılabilir bekleniyor.
    inst = cls()
    assert inst.__class__.__name__ == "RidgeReturnModel"
