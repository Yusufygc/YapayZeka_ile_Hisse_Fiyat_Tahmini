# -*- coding: utf-8 -*-
"""
Cross-sectional momentum testleri (2026-11: sektor momentumu kaldirildi).

FeaturePipeline._add_cross_sectional_momentum:
  - momentum_60d hisse Close.pct_change(60) verir
  - market_momentum_60d BIST100_Return cumprod (varsa)
  - relative_to_market_60d = momentum - market
  - sector_momentum_60d / relative_momentum_60d ARTIK uretilmez (sektor endeksi
    serileri feature matrisine hic ulasmiyordu; BIST100 kopyasiydi)
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
    return pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "BIST100_Return": bist_ret,
        }
    )


def test_momentum_60d_matches_pct_change():
    df = _frame(n=80)
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df)
    expected = df["Close"].pct_change(60)
    diff = (out["momentum_60d"] - expected).abs().dropna()
    assert diff.max() < 1e-12


def test_market_momentum_from_bist_returns():
    df = _frame(n=120)
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df)
    assert "market_momentum_60d" in out.columns
    last = out["market_momentum_60d"].dropna().iloc[-1]
    # Cumprod approximation; mantikli range
    assert -0.95 < float(last) < 10.0
    expected = np.expm1(np.log1p(df["BIST100_Return"]).rolling(60).sum())
    assert (out["market_momentum_60d"] - expected).abs().dropna().max() < 1e-12


def test_relative_to_market_is_stock_minus_market():
    df = _frame(n=120)
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df)
    diff = (
        out["relative_to_market_60d"]
        - (out["momentum_60d"] - out["market_momentum_60d"])
    ).abs().dropna()
    assert diff.max() < 1e-12


def test_sector_momentum_columns_are_gone():
    df = _frame(n=120)
    df["XBANK_Return"] = 0.001  # sektor serisi olsa bile uretilmez
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df)
    assert "sector_momentum_60d" not in out.columns
    assert "relative_momentum_60d" not in out.columns


def test_no_market_only_self_momentum():
    df = _frame(n=120).drop(columns=["BIST100_Return"])
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df)
    assert "momentum_60d" in out.columns
    assert "market_momentum_60d" not in out.columns
    assert "relative_to_market_60d" not in out.columns


def test_short_frame_returns_unchanged():
    df = _frame(n=30)  # < lookback+1
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df)
    assert "momentum_60d" not in out.columns


def test_recompute_momentum_features_refreshes_last_row():
    df = _frame(n=120)
    fp = FeaturePipeline(lag_feature_count=0)
    out = fp._add_cross_sectional_momentum(df.copy())
    stale = out.copy()
    stale.loc[stale.index[-1], "BIST100_Return"] = 0.0  # makro projeksiyonu
    stale.loc[stale.index[-1], "Close"] = float(stale["Close"].iloc[-1]) * 1.05
    fresh = fp.recompute_momentum_features(stale)
    expected = stale["Close"].iloc[-1] / stale["Close"].iloc[-61] - 1.0
    assert fresh["momentum_60d"].iloc[-1] == pytest.approx(expected)
    assert fresh["relative_to_market_60d"].iloc[-1] == pytest.approx(
        fresh["momentum_60d"].iloc[-1] - fresh["market_momentum_60d"].iloc[-1]
    )


def test_recompute_momentum_features_noop_without_momentum_column():
    df = _frame(n=120)
    fp = FeaturePipeline(lag_feature_count=0)
    assert fp.recompute_momentum_features(df) is df
