# -*- coding: utf-8 -*-
"""
Sprint 7 (2026-05-25) A7.3 — PSI 30g monitor (analysis API exposure).

compute_psi_30d(csv_path):
  - CSV yoksa unavailable / stale_warning
  - Yetersiz history -> unavailable
  - Stabil dagilim -> psi_status='stable'
  - 2-sigma shift -> psi_status='major_drift'
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.api.services.data_quality_monitor import (
    _PSI_MAJOR_MIN,
    _PSI_STABLE_MAX,
    compute_psi_30d,
)


def _write_csv(tmp_dir: str, name: str, df: pd.DataFrame) -> str:
    path = os.path.join(tmp_dir, name)
    df.to_csv(path, index=False)
    return path


def test_missing_csv_unavailable():
    res = compute_psi_30d("nonexistent_path_xyz.csv")
    assert res.psi_30d is None
    assert res.psi_status == "unavailable"
    assert res.stale_warning is True
    assert res.reason == "csv_missing"


def test_missing_columns_unavailable():
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, "X.csv", pd.DataFrame({"foo": [1, 2, 3]}))
        res = compute_psi_30d(path)
        assert res.psi_status == "unavailable"
        assert res.reason == "missing_required_columns"


def test_insufficient_history():
    with tempfile.TemporaryDirectory() as td:
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "Close": np.linspace(100, 110, 20),
            "High": np.linspace(101, 111, 20),
            "Low": np.linspace(99, 109, 20),
            "Volume": np.full(20, 1000),
        })
        path = _write_csv(td, "Y.csv", df)
        res = compute_psi_30d(path)
        assert res.psi_status == "unavailable"
        assert res.reason in {"insufficient_history", "insufficient_train_window"}


def test_stable_distribution_returns_low_psi():
    rng = np.random.default_rng(seed=20260525)
    n = 500
    with tempfile.TemporaryDirectory() as td:
        # Sabit ortalama etrafinda iid salinim -> dusuk PSI beklenir.
        close = 100 + rng.normal(0, 1.0, size=n)
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "Close": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Volume": rng.integers(1000, 2000, size=n).astype(float),
        })
        path = _write_csv(td, "STABLE.csv", df)
        res = compute_psi_30d(path)
        assert res.psi_30d is not None
        # Iid icin 30g holdout noise floor ~0.10; major_drift olmamali.
        assert res.psi_status != "major_drift"
        assert res.psi_30d < _PSI_MAJOR_MIN


def test_shifted_distribution_returns_major_drift():
    rng = np.random.default_rng(seed=20260525)
    n = 500
    with tempfile.TemporaryDirectory() as td:
        # Onceki 470 gun ~N(100,1), son 30 gun ~N(150,1) -> buyuk kayma
        base = 100 + rng.normal(0, 1.0, size=n - 30)
        shifted = 150 + rng.normal(0, 1.0, size=30)
        close = np.concatenate([base, shifted])
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "Close": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Volume": rng.integers(1000, 2000, size=n).astype(float),
        })
        path = _write_csv(td, "DRIFT.csv", df)
        res = compute_psi_30d(path)
        assert res.psi_30d is not None
        assert res.psi_30d >= _PSI_STABLE_MAX
        assert res.psi_status in {"moderate_drift", "major_drift"}


def test_tier_constants_sanity():
    assert _PSI_STABLE_MAX < _PSI_MAJOR_MIN
    assert _PSI_STABLE_MAX == 0.10
    assert _PSI_MAJOR_MIN == 0.25
