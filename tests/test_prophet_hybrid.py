# -*- coding: utf-8 -*-
"""Test suite for ProphetHybridModel functionality and serialization."""

from __future__ import annotations

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.models.prophet_hybrid_model import ProphetHybridModel


def test_prophet_hybrid_trend_gate():
    X = np.random.randn(30, 10)
    y = np.random.randn(30) * 0.01
    dates = pd.date_range("2026-01-01", periods=30)

    model = ProphetHybridModel(
        base_model_name="XGBoost",
        hybrid_mode="trend_gate",
        base_model_kwargs={"n_estimators": 5, "max_depth": 2},
    )

    model.train(X, y, dates_train=dates)
    assert model.prophet is not None

    X_test = np.random.randn(5, 10)
    dates_test = pd.date_range("2026-01-31", periods=5)
    preds = model.predict(X_test, dates_test=dates_test)
    assert len(preds) == 5
    assert not np.isnan(preds).any()


def test_prophet_hybrid_residual_decomp():
    X = np.random.randn(30, 10)
    y = np.random.randn(30) * 0.01
    dates = pd.date_range("2026-01-01", periods=30)

    model = ProphetHybridModel(
        base_model_name="XGBoost",
        hybrid_mode="residual_decomp",
        base_model_kwargs={"n_estimators": 5, "max_depth": 2},
    )

    model.train(X, y, dates_train=dates)
    assert model.prophet is not None

    X_test = np.random.randn(5, 10)
    dates_test = pd.date_range("2026-01-31", periods=5)
    preds = model.predict(X_test, dates_test=dates_test)
    assert len(preds) == 5
    assert not np.isnan(preds).any()


def test_prophet_hybrid_save_load():
    X = np.random.randn(30, 10)
    y = np.random.randn(30) * 0.01
    dates = pd.date_range("2026-01-01", periods=30)

    model = ProphetHybridModel(
        base_model_name="XGBoost",
        hybrid_mode="trend_gate",
        base_model_kwargs={"n_estimators": 5, "max_depth": 2},
    )
    model.train(X, y, dates_train=dates)

    temp_dir = tempfile.mkdtemp()
    try:
        model.save(temp_dir)

        loaded = ProphetHybridModel(
            base_model_name="XGBoost",
            hybrid_mode="trend_gate",
        )
        loaded.load(temp_dir)

        X_test = np.random.randn(5, 10)
        dates_test = pd.date_range("2026-01-31", periods=5)
        preds_orig = model.predict(X_test, dates_test=dates_test)
        preds_loaded = loaded.predict(X_test, dates_test=dates_test)

        np.testing.assert_array_almost_equal(preds_orig, preds_loaded)
    finally:
        shutil.rmtree(temp_dir)
