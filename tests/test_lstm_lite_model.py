# -*- coding: utf-8 -*-
"""LSTM Lite model registration and lightweight training coverage."""

from __future__ import annotations

import numpy as np
import pytest


def test_lstm_lite_factory_config_defaults():
    from src.pipeline import model_factory

    deep_config = model_factory.build_deep_config({})
    model = model_factory.make_lstm_lite(deep_config, "single")

    assert model.units == 32
    assert model.dense_units == 16
    assert model.dropout_rate == 0.25
    assert model.learning_rate == 0.0003
    assert model.batch_size == 32
    assert model.tune_on_fit is False
    assert deep_config["lstm_lite_min_sequence_samples"] == 252


def test_lstm_lite_factory_config_overrides():
    from src.pipeline import model_factory

    deep_config = model_factory.build_deep_config({
        "lstm_lite_min_sequence_samples": 300,
        "lstm_lite": {
            "units": 16,
            "dense_units": 8,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs_single": 3,
            "batch_size": 16,
            "tune_on_fit": True,
            "tune_n_trials": 2,
        },
    })
    model = model_factory.make_lstm_lite(deep_config, "single")

    assert deep_config["lstm_lite_min_sequence_samples"] == 300
    assert model.units == 16
    assert model.dense_units == 8
    assert model.dropout_rate == 0.1
    assert model.learning_rate == 0.001
    assert model.epochs == 3
    assert model.batch_size == 16
    assert model.tune_on_fit is True
    assert model.tune_n_trials == 2


def test_lstm_lite_trains_on_synthetic_sequences():
    tf = pytest.importorskip("tensorflow")
    from unittest.mock import MagicMock
    if isinstance(tf, MagicMock):
        pytest.skip("tensorflow stub active")
    from src.models.lstm_lite_model import LSTMLiteModel

    rng = np.random.default_rng(42)
    X = rng.normal(size=(36, 5, 3)).astype("float32")
    y = (0.2 * X[:, -1, 0] - 0.1 * X[:, -1, 1]).astype("float32")

    model = LSTMLiteModel(
        units=4,
        dense_units=4,
        dropout_rate=0.0,
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        patience=1,
        lr_patience=1,
        validation_ratio=0.2,
        min_val_samples=4,
    )
    model.train(X, y)
    preds = model.predict(X[:3])

    assert preds.shape == (3,)
    assert np.isfinite(preds).all()
