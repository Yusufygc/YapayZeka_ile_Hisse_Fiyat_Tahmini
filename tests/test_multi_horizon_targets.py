# -*- coding: utf-8 -*-
"""
Sprint 4 (2026-05-25) — Multi-horizon target uretimi testleri.

Plan A4.3: TensorPreparationService.build_multi_horizon_targets() opt-in
multi-horizon target (h=1,3,5,10). Tek-horizon (h=1) build_target_series
ile uyumlu olmali.
"""

from __future__ import annotations

import numpy as np
import pytest

# Heavy chain (joblib/sklearn) gerekiyor — yoksa skip.
try:
    from src.pipeline.data_services import TensorPreparationService
except ModuleNotFoundError as exc:
    pytest.skip(f"data_services import failed: {exc}", allow_module_level=True)


class _StubDataCfg:
    def __init__(self, target_mode="log_return"):
        self.target_mode = target_mode


class _StubOwner:
    def __init__(self, target_mode="log_return"):
        self.data_cfg = _StubDataCfg(target_mode)

    def _ensure_config_objects(self):  # no-op
        return None


def _make_service(target_mode="log_return"):
    return TensorPreparationService(_StubOwner(target_mode))


def test_default_horizons_returns_four_keys():
    svc = _make_service()
    close = np.linspace(100.0, 200.0, 30)
    out = svc.build_multi_horizon_targets(close)
    assert set(out.keys()) == {1, 3, 5, 10}


def test_horizon_one_matches_build_target_series_log_return():
    svc = _make_service("log_return")
    close = np.linspace(100.0, 200.0, 30)
    legacy = svc.build_target_series(close)
    out = svc.build_multi_horizon_targets(close, horizons=[1])
    np.testing.assert_allclose(out[1], legacy)


def test_horizon_lengths():
    svc = _make_service()
    close = np.linspace(100.0, 200.0, 30)
    out = svc.build_multi_horizon_targets(close, horizons=[1, 3, 5, 10])
    assert len(out[1]) == 29
    assert len(out[3]) == 27
    assert len(out[5]) == 25
    assert len(out[10]) == 20


def test_horizon_larger_than_series_returns_empty():
    svc = _make_service()
    close = np.array([100.0, 101.0, 102.0])
    out = svc.build_multi_horizon_targets(close, horizons=[5])
    assert out[5].size == 0


def test_horizon_zero_raises():
    svc = _make_service()
    close = np.linspace(100.0, 200.0, 10)
    with pytest.raises(ValueError):
        svc.build_multi_horizon_targets(close, horizons=[0])


def test_simple_return_mode():
    svc = _make_service("return")
    close = np.array([100.0, 110.0, 121.0])
    out = svc.build_multi_horizon_targets(close, horizons=[1, 2])
    np.testing.assert_allclose(out[1], [0.10, 0.10])
    np.testing.assert_allclose(out[2], [0.21])


def test_price_mode():
    svc = _make_service("price")
    close = np.array([1.0, 2.0, 3.0, 4.0])
    out = svc.build_multi_horizon_targets(close, horizons=[1, 2, 3])
    np.testing.assert_allclose(out[1], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(out[2], [3.0, 4.0])
    np.testing.assert_allclose(out[3], [4.0])


def test_empty_close_returns_empty_per_horizon():
    svc = _make_service()
    out = svc.build_multi_horizon_targets(np.array([]), horizons=[1, 3])
    assert out[1].size == 0 and out[3].size == 0
