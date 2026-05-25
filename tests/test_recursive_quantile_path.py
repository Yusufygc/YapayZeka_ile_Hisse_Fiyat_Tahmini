# -*- coding: utf-8 -*-
"""
Sprint 4 (2026-05-25) — Recursive quantile path testleri.

Plan A4.4: LatestTargetPredictionWorkflow.predict_quantiles_target() ve
ForecastPointGenerator.roll_forward_recursive p10/p50/p90 yayinlamasi.
"""

from __future__ import annotations

import numpy as np
import pytest

# Heavy chain — yoksa skip.
try:
    from src.forecasting.workflows import LatestTargetPredictionWorkflow
except ModuleNotFoundError as exc:
    pytest.skip(f"workflows import failed: {exc}", allow_module_level=True)


class _StubScalerY:
    """Identity inverse_transform (model output zaten target scale)."""

    def inverse_transform(self, arr):
        return np.asarray(arr, dtype=float)


class _StubQuantileModel:
    """Model `predict_quantiles` destekler; constant cikti."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9), values=(-0.02, 0.0, 0.02)):
        self.quantiles = quantiles
        self.values = values

    def predict_quantiles(self, X, **kwargs):
        n = len(X) if hasattr(X, "__len__") else int(X.shape[0])
        return np.tile(np.asarray(self.values, dtype=float), (n, 1))

    def predict(self, X, **kwargs):
        n = len(X) if hasattr(X, "__len__") else int(X.shape[0])
        return np.full(n, self.values[len(self.values) // 2], dtype=float)


class _StubPlainModel:
    """Model `predict_quantiles` desteklemiyor."""

    def predict(self, X, **kwargs):
        n = len(X) if hasattr(X, "__len__") else int(X.shape[0])
        return np.full(n, 0.01, dtype=float)


class _StubOwner:
    pass


def _make_workflow():
    return LatestTargetPredictionWorkflow(_StubOwner())


def test_predict_quantiles_target_returns_none_for_plain_model():
    wf = _make_workflow()
    ctx = {"latest_X": np.zeros((1, 3)), "scaler_y": _StubScalerY()}
    out = wf.predict_quantiles_target("XGBoost", _StubPlainModel(), ctx)
    assert out is None


def test_predict_quantiles_target_returns_dict_for_tree_quantile_model():
    wf = _make_workflow()
    ctx = {
        "latest_X_s": np.zeros((1, 3)),
        "latest_X": np.zeros((1, 3)),
        "scaler_y": _StubScalerY(),
    }
    out = wf.predict_quantiles_target("LightGBM Return", _StubQuantileModel(), ctx)
    # 'LightGBM Return' tree models setinde -> latest_X_s yolu
    assert out is not None
    assert set(out.keys()) == {0.1, 0.5, 0.9}
    assert out[0.1] == -0.02 and out[0.5] == 0.0 and out[0.9] == 0.02


def test_predict_quantiles_target_returns_dict_for_seq_quantile_model():
    wf = _make_workflow()
    ctx = {
        "latest_seq": np.zeros((1, 5, 3)),
        "scaler_y": _StubScalerY(),
    }
    out = wf.predict_quantiles_target("LSTM Lite", _StubQuantileModel(), ctx)
    assert out is not None
    assert len(out) == 3


def test_predict_quantiles_target_handles_missing_latest():
    wf = _make_workflow()
    ctx = {"scaler_y": _StubScalerY()}  # no latest_X_s/seq
    out = wf.predict_quantiles_target("LightGBM Return", _StubQuantileModel(), ctx)
    assert out is None


def test_predict_quantiles_target_uses_scaler_inverse():
    wf = _make_workflow()

    class _ScaleByTwo:
        def inverse_transform(self, arr):
            return np.asarray(arr, dtype=float) * 2.0

    ctx = {
        "latest_X_s": np.zeros((1, 3)),
        "scaler_y": _ScaleByTwo(),
    }
    out = wf.predict_quantiles_target("LightGBM Return", _StubQuantileModel(), ctx)
    assert out is not None
    assert out[0.5] == 0.0  # 0 * 2 = 0
    assert out[0.9] == 0.04  # 0.02 * 2 = 0.04
