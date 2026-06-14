# -*- coding: utf-8 -*-
"""Light dependency contract tests for EnsemblePooledModel.predict."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.ensemble_pooled_model import EnsemblePooledConfig, EnsemblePooledModel


class _FakeLeg:
    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


class _RecordingLeg:
    dtype = None
    contiguous = None

    def predict(self, X):
        self.dtype = X.dtype
        self.contiguous = bool(X.flags["C_CONTIGUOUS"])
        return np.asarray(X, dtype=float)[:, 0]


def _fitted_fake_ensemble() -> EnsemblePooledModel:
    model = EnsemblePooledModel(EnsemblePooledConfig(mlp_seeds=(1,)))
    model.lgb = _FakeLeg()
    model.mlps = [_FakeLeg()]
    return model


def test_predict_rejects_non_2d_input_without_heavy_deps():
    with pytest.raises(ValueError, match="2D X"):
        _fitted_fake_ensemble().predict(np.zeros(4))


def test_predict_rejects_single_row_cross_section_without_heavy_deps():
    with pytest.raises(ValueError, match="en az 2 satir"):
        _fitted_fake_ensemble().predict(np.zeros((1, 4)))


def test_predict_accepts_minimal_cross_section_without_heavy_deps():
    out = _fitted_fake_ensemble().predict(np.array([[1.0, 0.0], [2.0, 0.0]]))
    assert out.shape == (2,)
    assert np.all((0.0 <= out) & (out <= 1.0))


def test_predict_feeds_float32_matrix_to_component_legs_without_heavy_deps():
    model = EnsemblePooledModel(EnsemblePooledConfig(mlp_seeds=(1,)))
    lgb = _RecordingLeg()
    mlp = _RecordingLeg()
    model.lgb = lgb
    model.mlps = [mlp]

    model.predict(np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64))

    assert lgb.dtype == np.float32 and lgb.contiguous is True
    assert mlp.dtype == np.float32 and mlp.contiguous is True
