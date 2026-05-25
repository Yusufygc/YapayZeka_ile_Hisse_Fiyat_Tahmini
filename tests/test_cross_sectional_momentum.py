# -*- coding: utf-8 -*-
"""
Sprint 7 (2026-05-25) A7.2 — Cross-sectional momentum testleri.

FeaturePipeline._add_cross_sectional_momentum:
  - momentum_60d hisse Close.pct_change(60) verir
  - market_momentum_60d BIST100_Return cumprod (varsa)
  - sector_momentum_60d sektor index Return cumprod (varsa)
  - relative_momentum_60d = momentum - sector
  - relative_to_market_60d = momentum - market
  - Eksik kolon -> ilgili turev sessiz atlanir, exception yok
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from src.features.feature_pipeline import FeaturePipeline
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest.skip(f"feature_pipeline import basarisiz: {exc}", allow_module_level=True)


def _frame(n=120, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    bist_ret = rng.normal(0.0005, 0.01, size=n)
    sector_ret = rng.normal(0.001, 0.012, size=n)
    return pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "BIST100_Return": bist_ret,
            "XBANK_Return": sector_ret,
        }
    )


def test_momentum_60d_matches_pct_change():
    df = _frame(n=80)
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df, sector_return_col="XBANK_Return")
    expected = df["Close"].pct_change(60)
    diff = (out["momentum_60d"] - expected).abs().dropna()
    assert diff.max() < 1e-12


def test_market_momentum_from_bist_returns():
    df = _frame(n=120)
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df, sector_return_col="XBANK_Return")
    assert "market_momentum_60d" in out.columns
    last = out["market_momentum_60d"].dropna().iloc[-1]
    # Cumprod approximation; mantikli range
    assert -0.95 < float(last) < 10.0


def test_sector_relative_columns_when_sector_return_present():
    df = _frame(n=120)
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df, sector_return_col="XBANK_Return")
    assert "sector_momentum_60d" in out.columns
    assert "relative_momentum_60d" in out.columns
    assert "relative_to_market_60d" in out.columns
    # relative = hisse - sektor
    diff = (
        out["relative_momentum_60d"]
        - (out["momentum_60d"] - out["sector_momentum_60d"])
    ).abs().dropna()
    assert diff.max() < 1e-12


def test_sector_falls_back_to_market_when_sector_missing():
    df = _frame(n=120).drop(columns=["XBANK_Return"])
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df, sector_return_col="XBANK_Return")
    # Sektor yok ama market var -> sector_momentum_60d = market_momentum_60d
    assert "sector_momentum_60d" in out.columns
    assert "market_momentum_60d" in out.columns
    diff = (out["sector_momentum_60d"] - out["market_momentum_60d"]).abs().dropna()
    assert diff.max() < 1e-12


def test_no_market_no_sector_only_self_momentum():
    df = _frame(n=120).drop(columns=["BIST100_Return", "XBANK_Return"])
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df, sector_return_col="XBANK_Return")
    assert "momentum_60d" in out.columns
    assert "market_momentum_60d" not in out.columns
    assert "sector_momentum_60d" not in out.columns
    assert "relative_momentum_60d" not in out.columns


def test_short_frame_returns_unchanged():
    df = _frame(n=30)  # < lookback+1
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df, sector_return_col="XBANK_Return")
    assert "momentum_60d" not in out.columns
