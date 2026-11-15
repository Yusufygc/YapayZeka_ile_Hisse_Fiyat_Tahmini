# -*- coding: utf-8 -*-
"""
Sprint 7 (2026-05-25) A7.3 — PSI 30g monitor (analysis API exposure).

compute_psi_30d(csv_path):
  - CSV yoksa unavailable / stale_warning
  - Yetersiz history -> unavailable
  - Stabil dagilim -> psi_status='stable'
  - Buyuk rejim kaymasi -> psi_status='major_drift'

Kalibrasyon (2026-11-15): tahmin edici referans-kantil bin + Laplace; esikler
gercek verideki placebo null dagiliminin q90/q99'undan (0.40 / 0.80).
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
    _psi_smoothed,
    _tier,
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
        # Iid icin tahmin edici gurultu tabani ~0.05-0.15; drift olmamali.
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


def _regime_frame(rng, n, vol_mult=1.0, vol_shift=1.0):
    """Iid getiri + hacim; son 30 gunde volatilite ve hacim rejimi degisir."""
    ret = rng.normal(0, 0.01, size=n)
    ret[-30:] *= vol_mult
    close = 100 * np.exp(np.cumsum(ret))
    rng_pct = np.abs(rng.normal(0.01, 0.003, size=n))
    rng_pct[-30:] *= vol_mult
    vol = np.exp(rng.normal(12, 0.3, size=n))
    vol[-30:] *= vol_shift
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "Close": close,
        "High": close * (1 + rng_pct / 2),
        "Low": close * (1 - rng_pct / 2),
        "Volume": vol,
    })


def test_iid_stable_in_most_seeds():
    # Null altinda moderate+major orani ~%10 hedeflenir; iid'de belirgin dusuk.
    stable = 0
    n_seeds = 30
    with tempfile.TemporaryDirectory() as td:
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            path = _write_csv(td, f"S{seed}.csv", _regime_frame(rng, 400))
            res = compute_psi_30d(path)
            assert res.psi_30d is not None
            stable += res.psi_status == "stable"
    assert stable >= int(0.9 * n_seeds)


def test_volatility_volume_regime_shift_is_major_drift():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(
            td, "REGIME.csv", _regime_frame(rng, 400, vol_mult=4.0, vol_shift=5.0)
        )
        res = compute_psi_30d(path)
    assert res.psi_status == "major_drift"
    assert res.psi_30d >= _PSI_MAJOR_MIN


def test_psi_smoothed_empty_bin_does_not_blow_up():
    # Holdout referansin tamamen disinda: eski eps=1e-4 yaklasimi PSI'yi
    # patlatirdi; yumusatma ile sonlu ve makul (<~5) kalir.
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 252)
    far = rng.normal(10, 0.1, 30)
    psi = _psi_smoothed(ref, far)
    assert np.isfinite(psi)
    assert _PSI_MAJOR_MIN < psi < 5.0


def test_psi_smoothed_identical_and_degenerate_inputs():
    rng = np.random.default_rng(5)
    ref = rng.normal(0, 1, 252)
    assert _psi_smoothed(ref, ref[:30]) < _PSI_STABLE_MAX
    # Sabit referans (ties) ve NaN'ler cokmemeli.
    assert _psi_smoothed(np.zeros(252), np.zeros(30)) < _PSI_STABLE_MAX
    assert _psi_smoothed(np.full(3, np.nan), np.zeros(30)) == 0.0


def test_tier_boundaries():
    assert _tier(_PSI_STABLE_MAX - 1e-9) == "stable"
    assert _tier(_PSI_STABLE_MAX) == "moderate_drift"
    assert _tier(_PSI_MAJOR_MIN - 1e-9) == "moderate_drift"
    assert _tier(_PSI_MAJOR_MIN) == "major_drift"


def test_tier_constants_sanity():
    # Gercek veride placebo null q90 / q99 (bkz. data_quality_monitor.py).
    assert _PSI_STABLE_MAX < _PSI_MAJOR_MIN
    assert _PSI_STABLE_MAX == 0.40
    assert _PSI_MAJOR_MIN == 0.80
