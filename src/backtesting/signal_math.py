# -*- coding: utf-8 -*-
"""Pure numerical helpers for signal generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def expected_return(
    pred_target: np.ndarray | None,
    pred_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    if pred_target is not None and target_mode in {"log_return", "return"}:
        return np.asarray(pred_target, dtype=float).ravel()
    if target_mode == "price":
        return (pred_price / np.maximum(prev_close, 1e-12)) - 1.0
    if pred_target is not None:
        return np.asarray(pred_target, dtype=float).ravel()
    return (pred_price / np.maximum(prev_close, 1e-12)) - 1.0


def expected_return_to_simple_return(expected_return: np.ndarray, target_mode: str) -> np.ndarray:
    expected_return = np.asarray(expected_return, dtype=float).ravel()
    if target_mode == "log_return":
        return np.expm1(expected_return)
    if target_mode == "return":
        return expected_return
    return expected_return


def recommendation_from_decision(
    decision: str,
    expected_return: float,
    entry_threshold: float,
) -> tuple[str, str]:
    if decision == "BUY":
        return "BUY", "AL"
    if decision == "EXIT":
        return "SELL", "SAT"
    if decision == "NO_TRADE" and expected_return < -abs(entry_threshold):
        return "SELL", "SAT"
    return "HOLD", "TUT"


def rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=float).ravel()
    if len(returns) == 0:
        return returns

    window = max(2, int(window))
    vol = pd.Series(returns).rolling(window=window, min_periods=2).std()
    fallback = 1e-6
    vol = vol.fillna(fallback).replace(0.0, fallback)
    return vol.to_numpy(dtype=float)


def regime_entry_multiplier(market_regime: np.ndarray, config: Any) -> np.ndarray:
    regime = np.asarray(market_regime, dtype=float).ravel()
    multiplier = np.full(len(regime), config.regime_neutral_entry_multiplier, dtype=float)
    multiplier[regime > 0] = config.regime_bull_entry_multiplier
    multiplier[regime < 0] = config.regime_bear_entry_multiplier
    return multiplier


def volatility_entry_multiplier(rolling_vol: np.ndarray, config: Any) -> tuple[np.ndarray, np.ndarray]:
    vol = np.asarray(rolling_vol, dtype=float).ravel()
    regimes = np.full(len(vol), "normal_vol", dtype=object)
    multipliers = np.full(len(vol), config.volatility_normal_entry_multiplier, dtype=float)
    for idx in range(len(vol)):
        history = vol[: idx + 1]
        history = history[np.isfinite(history)]
        if history.size < max(5, config.volatility_window // 2):
            continue
        low = float(np.quantile(history, config.volatility_low_quantile))
        high = float(np.quantile(history, config.volatility_high_quantile))
        if vol[idx] <= low:
            regimes[idx] = "low_vol"
            multipliers[idx] = config.volatility_low_entry_multiplier
        elif vol[idx] >= high:
            regimes[idx] = "high_vol"
            multipliers[idx] = config.volatility_high_entry_multiplier
    return regimes, multipliers
