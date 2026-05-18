# -*- coding: utf-8 -*-
"""Rejim/trend bağlamı testleri (Adim 2.3)."""
import numpy as np
import pytest

from src.pipeline.regime_context import (
    compute_market_regime,
    compute_relative_strength,
    compute_regime_context,
)


def _bull_series(n=250):
    rng = np.random.default_rng(1)
    prices = 1000.0 + np.cumsum(rng.normal(5, 2, n))
    return np.clip(prices, 100, None)


def _bear_series(n=250):
    rng = np.random.default_rng(2)
    prices = 2000.0 - np.cumsum(rng.normal(5, 2, n))
    return np.clip(prices, 100, None)


def _flat_series(n=250, base=1500.0):
    rng = np.random.default_rng(3)
    return base + rng.normal(0, 1, n)


class TestComputeMarketRegime:
    def test_bull_market_detected(self):
        prices = _bull_series(250)
        regime = compute_market_regime(prices)
        assert regime in ("bull", "uncertain"), f"Boğa serisinde beklenen 'bull', alınan: {regime}"

    def test_bear_market_detected(self):
        prices = _bear_series(250)
        regime = compute_market_regime(prices)
        assert regime in ("bear", "uncertain"), f"Ayı serisinde beklenen 'bear', alınan: {regime}"

    def test_uncertain_with_short_series(self):
        prices = np.array([100.0] * 10)
        assert compute_market_regime(prices) == "uncertain"

    def test_returns_valid_label(self):
        prices = _flat_series(250)
        regime = compute_market_regime(prices)
        assert regime in ("bull", "bear", "sideways", "uncertain")


class TestComputeRelativeStrength:
    def test_outperforming_when_stock_rises_index_flat(self):
        index = np.full(100, 1000.0)
        stock = np.linspace(100, 120, 100)
        rs = compute_relative_strength(stock, index)
        assert rs == "outperforming"

    def test_underperforming_when_stock_falls_index_flat(self):
        index = np.full(100, 1000.0)
        stock = np.linspace(100, 80, 100)
        rs = compute_relative_strength(stock, index)
        assert rs == "underperforming"

    def test_inline_when_both_move_same(self):
        prices = np.linspace(100, 110, 100)
        rs = compute_relative_strength(prices, prices)
        assert rs == "inline"

    def test_inline_with_insufficient_data(self):
        stock = np.array([100.0, 101.0])
        index = np.array([1000.0, 1001.0])
        rs = compute_relative_strength(stock, index)
        assert rs == "inline"


class TestComputeRegimeContext:
    def test_full_context_keys(self):
        stock = _bull_series(100)
        ctx = compute_regime_context(stock)
        for key in ["market_regime", "relative_strength", "alignment_with_forecast", "regime_misalignment"]:
            assert key in ctx

    def test_bull_forecast_up_aligned(self):
        index = _bull_series(250)
        stock = _bull_series(100)
        ctx = compute_regime_context(stock, index_close=index, forecast_direction=1.0)
        if ctx["market_regime"] == "bull":
            assert ctx["alignment_with_forecast"] == "aligned"
            assert ctx["regime_misalignment"] is False

    def test_bull_forecast_down_misaligned(self):
        index = _bull_series(250)
        stock = _bull_series(100)
        ctx = compute_regime_context(stock, index_close=index, forecast_direction=-1.0)
        if ctx["market_regime"] == "bull":
            assert ctx["alignment_with_forecast"] == "misaligned"
            assert ctx["regime_misalignment"] is True

    def test_no_index_gives_uncertain_regime(self):
        stock = _flat_series(100)
        ctx = compute_regime_context(stock, index_close=None)
        assert ctx["market_regime"] == "uncertain"

    def test_neutral_alignment_when_no_forecast_direction(self):
        stock = _bull_series(100)
        index = _bull_series(250)
        ctx = compute_regime_context(stock, index_close=index, forecast_direction=None)
        assert ctx["alignment_with_forecast"] == "neutral"
        assert ctx["regime_misalignment"] is False
