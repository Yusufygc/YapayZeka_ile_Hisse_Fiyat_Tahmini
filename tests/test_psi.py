# -*- coding: utf-8 -*-
"""
Sprint 2 (2026-05-25) Plan A2.4 — PSI (Population Stability Index) testleri.

src/data/quality.py compute_psi() ve _psi_one_feature() davranisi:
  - Ayni dagilim -> PSI ≈ 0
  - Kayik dagilim -> PSI > 0.25 (psi_high tetiklenir)
  - compute_quality_flags() train+holdout verilirse psi_high doldurur

Plus: survivorship_bias_report quality flag'lerine yansir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.quality import (
    _PSI_THRESHOLD,
    _psi_one_feature,
    compute_psi,
    compute_quality_flags,
)


def test_psi_identical_distributions_near_zero():
    rng = np.random.default_rng(seed=42)
    train = rng.normal(0.0, 1.0, size=5000)
    holdout = rng.normal(0.0, 1.0, size=5000)
    psi = _psi_one_feature(train, holdout, n_bins=10)
    assert psi < 0.05  # neredeyse ayni dagilim


def test_psi_shifted_distribution_above_threshold():
    rng = np.random.default_rng(seed=42)
    train = rng.normal(0.0, 1.0, size=5000)
    # 2-sigma kayma -> ciddi dagilim farki
    holdout = rng.normal(2.0, 1.0, size=5000)
    psi = _psi_one_feature(train, holdout, n_bins=10)
    assert psi > _PSI_THRESHOLD


def test_compute_psi_returns_per_column_dict():
    rng = np.random.default_rng(seed=42)
    n = 3000
    train_df = pd.DataFrame({
        "feat_a": rng.normal(0.0, 1.0, size=n),
        "feat_b": rng.normal(0.0, 1.0, size=n),
        "Date": pd.date_range("2024-01-01", periods=n, freq="H"),
        "Symbol": ["A"] * n,
    })
    holdout_df = pd.DataFrame({
        "feat_a": rng.normal(0.0, 1.0, size=n),  # ayni dagilim
        "feat_b": rng.normal(3.0, 1.0, size=n),  # buyuk kayma
        "Date": pd.date_range("2024-06-01", periods=n, freq="H"),
        "Symbol": ["A"] * n,
    })
    psi = compute_psi(train_df, holdout_df)
    assert "feat_a" in psi and "feat_b" in psi
    assert "Date" not in psi and "Symbol" not in psi
    assert psi["feat_a"] < 0.05
    assert psi["feat_b"] > _PSI_THRESHOLD


def test_quality_flags_psi_high_when_drift_present():
    rng = np.random.default_rng(seed=42)
    n = 3000
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "Date": dates,
        "Close": rng.normal(100.0, 5.0, size=n),
        "Volume": rng.integers(100, 1000, size=n),
    })
    train_df = pd.DataFrame({"feat_x": rng.normal(0.0, 1.0, size=n)})
    holdout_df = pd.DataFrame({"feat_x": rng.normal(3.0, 1.0, size=n)})
    flags = compute_quality_flags(
        df, symbol="DRIFT", train_df=train_df, holdout_df=holdout_df,
    )
    assert flags["psi_high"] is True
    assert flags["psi_max"] > _PSI_THRESHOLD


def test_quality_flags_psi_low_when_distributions_match():
    rng = np.random.default_rng(seed=42)
    n = 3000
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "Date": dates,
        "Close": rng.normal(100.0, 5.0, size=n),
        "Volume": rng.integers(100, 1000, size=n),
    })
    train_df = pd.DataFrame({"feat_x": rng.normal(0.0, 1.0, size=n)})
    holdout_df = pd.DataFrame({"feat_x": rng.normal(0.0, 1.0, size=n)})
    flags = compute_quality_flags(
        df, symbol="STABLE", train_df=train_df, holdout_df=holdout_df,
    )
    assert flags["psi_high"] is False
    assert flags["psi_max"] < _PSI_THRESHOLD


def test_quality_flags_survivorship_report_carried():
    """A2.3: survivorship_bias_report quality outputuna yansir."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "Date": dates,
        "Close": [100, 101, 102, 103, 104],
        "Volume": [10, 10, 10, 10, 10],
    })
    df.attrs["survivorship_bias_report"] = {
        "symbol": "XYZ",
        "actual_start": "2024-01-01",
        "actual_end": "2024-01-05",
        "span_days": 4,
        "row_count": 5,
        "max_gap_days": 1,
        "short_history_warning": True,
        "delisted_or_late_listing_warning": False,
        "warning": "short_history",
    }
    flags = compute_quality_flags(df, symbol="XYZ")
    report = flags.get("survivorship_bias_report") or {}
    assert report.get("warning") == "short_history"
    assert report.get("short_history_warning") is True
