# -*- coding: utf-8 -*-
"""
Sprint 4 (2026-05-25) — QuantileLightGBM testleri.

Plan A4.1: 3 quantile pred sirali (p10 <= p50 <= p90); predict() median doner.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# lightgbm yoksa modul atlanir. conftest.py MagicMock stub yapiyor -> real
# import kontrolu icin tip kontrolu sart.
lightgbm = pytest.importorskip("lightgbm")
from unittest.mock import MagicMock as _MagicMock
if isinstance(getattr(lightgbm, "LGBMRegressor", None), type) and issubclass(
    lightgbm.LGBMRegressor, _MagicMock
):
    pytest.skip("lightgbm conftest stub (MagicMock) - production dep gerekli", allow_module_level=True)

from src.models.quantile_lightgbm_model import QuantileLightGBMModel


def _toy_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 4))
    y = X @ np.array([0.5, -0.3, 0.2, 0.1]) + rng.normal(0, 0.1, size=n)
    return X, y


def test_constructor_validates_empty_quantiles():
    with pytest.raises(ValueError):
        QuantileLightGBMModel(quantiles=[])


def test_constructor_validates_quantile_range():
    with pytest.raises(ValueError):
        QuantileLightGBMModel(quantiles=[0.0, 0.5])
    with pytest.raises(ValueError):
        QuantileLightGBMModel(quantiles=[0.5, 1.0])


def test_constructor_default_quantiles():
    m = QuantileLightGBMModel()
    assert m.quantiles == (0.1, 0.5, 0.9)


def test_train_predict_quantiles_shape():
    X, y = _toy_dataset()
    m = QuantileLightGBMModel(n_estimators=30)
    m.train(X, y)
    preds = m.predict_quantiles(X)
    assert preds.shape == (200, 3)


def test_quantiles_sorted_per_row():
    X, y = _toy_dataset()
    m = QuantileLightGBMModel(n_estimators=30)
    m.train(X, y)
    preds = m.predict_quantiles(X)
    # row-wise monotonic
    assert np.all(preds[:, 0] <= preds[:, 1])
    assert np.all(preds[:, 1] <= preds[:, 2])


def test_predict_returns_median():
    X, y = _toy_dataset()
    m = QuantileLightGBMModel(n_estimators=30)
    m.train(X, y)
    median = m.predict(X)
    quants = m.predict_quantiles(X)
    np.testing.assert_allclose(median, quants[:, 1])


def test_predict_before_train_raises():
    m = QuantileLightGBMModel(n_estimators=30)
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((1, 4)))
    with pytest.raises(RuntimeError):
        m.predict_quantiles(np.zeros((1, 4)))


def test_save_load_roundtrip():
    X, y = _toy_dataset()
    m = QuantileLightGBMModel(n_estimators=30)
    m.train(X, y)
    p_before = m.predict_quantiles(X)
    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "qlgbm.pkl")
        m.save(pth)
        m2 = QuantileLightGBMModel()
        m2.load(pth)
        p_after = m2.predict_quantiles(X)
    np.testing.assert_allclose(p_before, p_after)
