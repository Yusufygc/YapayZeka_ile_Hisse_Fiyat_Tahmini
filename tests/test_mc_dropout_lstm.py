# -*- coding: utf-8 -*-
"""
Sprint 4 (2026-05-25) — LSTM Lite MC Dropout testleri.

Plan A4.2: training=True dropout aktif inference; n_samples MC sample
empirical posterior; quantile cikti.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from unittest.mock import MagicMock as _MagicMock
# conftest.py tensorflow'u MagicMock stub yapiyor -> tip kontrolu sart.
if isinstance(tf, _MagicMock):
    pytest.skip("tensorflow conftest stub (MagicMock) - production dep gerekli", allow_module_level=True)

from src.models.lstm_lite_model import LSTMLiteModel


def _toy_seq(n=80, time_steps=10, features=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, time_steps, features)).astype(np.float32)
    y = X.sum(axis=(1, 2)) * 0.01 + rng.normal(0, 0.05, size=n).astype(np.float32)
    return X, y


def test_predict_quantiles_validates_empty_quantiles():
    m = LSTMLiteModel(epochs=1)
    X, y = _toy_seq()
    m.train(X, y)
    with pytest.raises(ValueError):
        m.predict_quantiles(X, quantiles=())


def test_predict_quantiles_validates_range():
    m = LSTMLiteModel(epochs=1)
    X, y = _toy_seq()
    m.train(X, y)
    with pytest.raises(ValueError):
        m.predict_quantiles(X, quantiles=(0.0, 0.5))


def test_predict_quantiles_shape():
    m = LSTMLiteModel(epochs=1)
    X, y = _toy_seq()
    m.train(X, y)
    out = m.predict_quantiles(X, n_samples=20)
    assert out.shape == (len(X), 3)


def test_predict_quantiles_sorted_per_row():
    m = LSTMLiteModel(epochs=1)
    X, y = _toy_seq()
    m.train(X, y)
    out = m.predict_quantiles(X, n_samples=20)
    assert np.all(out[:, 0] <= out[:, 1])
    assert np.all(out[:, 1] <= out[:, 2])


def test_predict_quantiles_varies_with_dropout():
    """MC dropout aktif -> tekrar call'da farkli sample, varyans > 0."""
    m = LSTMLiteModel(epochs=1, dropout_rate=0.5)
    X, y = _toy_seq(n=40, time_steps=8, features=3)
    m.train(X, y)
    # Iki ayri quantile call'inda median'lar tamamen ozdes degil (stokastik)
    out1 = m.predict_quantiles(X, n_samples=30, seed=None)
    out2 = m.predict_quantiles(X, n_samples=30, seed=None)
    # En azindan bir satirda differ
    assert not np.allclose(out1, out2)


def test_predict_quantiles_before_train_raises():
    m = LSTMLiteModel(epochs=1)
    with pytest.raises(RuntimeError):
        m.predict_quantiles(np.zeros((1, 5, 3)))
