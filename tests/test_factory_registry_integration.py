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
    "ElasticNet Return",
    "Random Forest",
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


@pytest.mark.parametrize("name", ["DLinear", "NLinear"])
def test_known_models_resolve_sequence(name):
    """DLinear/NLinear 3D sequence tensor bekler."""
    cls = model_factory.model_class_for_name(name, {}, "log_return")
    instance = cls()
    assert instance is not None
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8, 4))  # (batch, seq_len, features)
    y = rng.normal(size=20)
    instance.train(X, y)
    out = instance.predict(rng.normal(size=(5, 8, 4)))
    assert len(out) == 5


def test_unknown_model_raises_keyerror():
    with pytest.raises(KeyError) as exc:
        model_factory.model_class_for_name("__NoSuchModel__", {}, "log_return")
    assert "Bilinmeyen model adi" in str(exc.value)


def test_naive_zero_return_rejects_price_target():
    """Naive Zero Return target_modes = (return, log_return) — price desteklemez."""
    with pytest.raises(KeyError):
        model_factory.model_class_for_name("Naive Zero Return", {}, "price")


def test_arima_factory_uses_config():
    """ARIMA için `arima` alt-sözlüğü unpack edilmeli (config-aware factory)."""
    pytest.importorskip("statsmodels")
    cfg = {"arima": {"order": (2, 1, 1), "auto_order": False}}
    cls = model_factory.model_class_for_name("ARIMA", cfg, "log_return")
    instance = cls()
    assert tuple(instance.order) == (2, 1, 1)
    assert instance.auto_order is False


def test_arima_factory_default_order():
    pytest.importorskip("statsmodels")
    cls = model_factory.model_class_for_name("ARIMA", {}, "log_return")
    instance = cls()
    assert tuple(instance.order) == (1, 0, 0)


def test_model_class_returns_callable():
    """model_class_for_name daima sıfır-arg callable döner — eski davranış."""
    cls = model_factory.model_class_for_name("Random Forest", {}, "log_return")
    assert callable(cls)
    # `cls()` çağrılabilir olmalı, sınıf değil çağrılabilir bekleniyor.
    inst = cls()
    assert inst.__class__.__name__ == "RandomForestModel"
