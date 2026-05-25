# -*- coding: utf-8 -*-
"""
Sprint 3 (2026-05-25) — Concat-Sharpe + bootstrap CI testleri.

Plan A3.3: WalkForwardValidator fold strateji getirilerini birlestirir
ve tek Sharpe + %95 CI hesaplar.
"""

from __future__ import annotations

import numpy as np
import pytest

# walk_forward import zinciri sklearn'a baglandigi icin yardimcilari
# dogrudan ic koduyla test ediyoruz (smoke test joblib/sklearn yoksa atlanir).
try:
    from src.validation.walk_forward import (
        _bootstrap_sharpe_ci,
        _compute_strategy_returns,
    )
except ModuleNotFoundError as exc:  # joblib/sklearn yoksa
    pytest.skip(f"walk_forward import failed: {exc}", allow_module_level=True)

from src.evaluation.financial_metrics import _annualized_sharpe


def test_compute_strategy_returns_log_return_mode():
    # True log returns: +0.01, -0.01, +0.02
    y_true_target = np.array([0.01, -0.01, 0.02])
    # Pred signs: +, -, + (perfect alignment)
    y_pred_target = np.array([0.005, -0.002, 0.003])
    out = _compute_strategy_returns(
        y_true_target,
        y_pred_target,
        y_true_price=np.array([1.0, 1.0, 1.0]),
        prev_close=np.array([1.0, 1.0, 1.0]),
        target_mode="log_return",
    )
    # Realized simple returns = expm1(log_returns) ≈ [0.01005, -0.00995, 0.0202]
    # All signs match preds → strategy_returns positive everywhere
    assert out.size == 3
    assert np.all(out > 0)


def test_compute_strategy_returns_sign_mismatch_loses():
    y_true_target = np.array([0.01, 0.01, 0.01])
    y_pred_target = np.array([-0.01, -0.01, -0.01])  # all wrong direction
    out = _compute_strategy_returns(
        y_true_target,
        y_pred_target,
        y_true_price=np.array([1.0, 1.0, 1.0]),
        prev_close=np.array([1.0, 1.0, 1.0]),
        target_mode="log_return",
    )
    assert np.all(out < 0)


def test_compute_strategy_returns_empty_input():
    out = _compute_strategy_returns(
        np.array([]),
        np.array([]),
        y_true_price=np.array([]),
        prev_close=np.array([]),
        target_mode="log_return",
    )
    assert out.size == 0


def test_bootstrap_ci_returns_low_high_pair():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, size=500)
    low, high = _bootstrap_sharpe_ci(returns, risk_free_annual=0.05, n_resamples=200)
    assert np.isfinite(low) and np.isfinite(high)
    assert low <= high


def test_bootstrap_ci_returns_nan_when_rf_none():
    returns = np.random.default_rng(0).normal(0.0, 0.01, size=100)
    low, high = _bootstrap_sharpe_ci(returns, risk_free_annual=None)
    assert np.isnan(low) and np.isnan(high)


def test_bootstrap_ci_returns_nan_when_too_few_samples():
    low, high = _bootstrap_sharpe_ci(np.array([0.01]), risk_free_annual=0.05)
    assert np.isnan(low) and np.isnan(high)


def test_concat_sharpe_differs_from_mean_fold_sharpe():
    # 3 fold; fold sharpes farkli ama concat butun farkli istatistik
    rng = np.random.default_rng(7)
    fold_returns = [
        rng.normal(0.001, 0.005, size=50),
        rng.normal(0.002, 0.020, size=50),
        rng.normal(-0.001, 0.010, size=50),
    ]
    rf = 0.05
    fold_sharpes = [_annualized_sharpe(r, rf) for r in fold_returns]
    mean_fold = float(np.mean(fold_sharpes))
    concat = _annualized_sharpe(np.concatenate(fold_returns), rf)
    # Beklenti: cogu zaman esit degil (vol blend)
    assert abs(mean_fold - concat) > 1e-6
