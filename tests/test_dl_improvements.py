# -*- coding: utf-8 -*-
"""Test suite for Deep Learning model quality improvements (GRU, L2, AdamW)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.attention_lstm_v2_model import AttentionLSTMV2Model
from src.models.lstm_lite_model import LSTMLiteModel


def test_attention_lstm_v2_new_params():
    # Sahte 3D sequence datası (n_samples, time_steps, n_features)
    X = np.random.randn(40, 5, 10)
    y = np.random.randn(40)

    # GRU hücresi, L2 regularizer ve AdamW ile modeli oluştur
    model = AttentionLSTMV2Model(
        cell_type="gru",
        l2_rate=0.01,
        optimizer_type="adamw",
        epochs=2,
        batch_size=16,
        validation_ratio=0.2,
        min_val_samples=5,
    )

    model.train(X, y)
    assert model.model is not None

    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert not np.isnan(preds).any()


def test_lstm_lite_new_params():
    # Sahte 3D sequence datası (n_samples, time_steps, n_features)
    X = np.random.randn(40, 5, 10)
    y = np.random.randn(40)

    # GRU hücresi, L2 regularizer ve AdamW ile modeli oluştur
    model = LSTMLiteModel(
        cell_type="gru",
        l2_rate=0.01,
        optimizer_type="adamw",
        epochs=2,
        batch_size=16,
        validation_ratio=0.2,
        min_val_samples=5,
    )

    model.train(X, y)
    assert model.model is not None

    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert not np.isnan(preds).any()
