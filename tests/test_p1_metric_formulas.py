# -*- coding: utf-8 -*-
"""P1 metric-formula regression tests (deep audit 2026-11).

Covers: Sortino target semideviation, Bailey/Lopez de Prado PSR/DSR, cash-bar
risk-free (exposure), point-in-time rf series, max-drawdown initial equity,
Calmar/CAGR short-sample NaN, ddof=1, Dir_Acc / Hit_Rate zero handling and the
stationary block bootstrap Sharpe CI.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtesting import metrics as bm
from src.backtesting.metrics import (
    _annualized_sharpe,
    _annualized_sortino,
    _deflated_sharpe,
    _information_ratio,
    _max_drawdown,
    summarize_backtest,
)
from src.evaluation.financial_metrics import compute_financial_metrics
import src.utils.risk_free_rate as rfr
from src.validation.walk_forward import _bootstrap_sharpe_ci


def _curve(net: np.ndarray, position: np.ndarray | None = None, dates=None) -> pd.DataFrame:
    net = np.asarray(net, dtype=float)
    bh = np.full(net.size, 0.001)
    df = pd.DataFrame({
        "Equity": np.cumprod(1.0 + net),
        "BuyHold_Equity": np.cumprod(1.0 + bh),
        "Net_Return": net,
        "Realized_Return": bh,
        "Position": np.ones(net.size) if position is None else np.asarray(position, dtype=float),
        "Signal": np.ones(net.size),
    })
    if dates is not None:
        df["Date"] = dates
    return df


def _result(curve: pd.DataFrame) -> dict:
    return {"model_name": "M", "equity_curve": curve, "trades": pd.DataFrame()}


# --------------------------------------------------------------------- 1
def test_sortino_matches_target_semideviation_over_all_observations():
    r = np.array([0.02, -0.01, 0.015, -0.03, 0.01, 0.005, -0.002, 0.012])
    expected_dd = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    expected = float(np.mean(r) / expected_dd * math.sqrt(252))
    assert _annualized_sortino(r, 0.0) == pytest.approx(expected, rel=1e-12)


def test_sortino_target_is_risk_free():
    r = np.array([0.02, -0.01, 0.015, -0.03, 0.01, 0.005, -0.002, 0.012])
    rf = 0.40
    d = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    ex = r - d
    dd = math.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    assert _annualized_sortino(r, rf) == pytest.approx(np.mean(ex) / dd * math.sqrt(252), rel=1e-12)


def test_sortino_no_loss_is_nan_not_zero():
    r = np.array([0.01, 0.02, 0.005, 0.03])
    assert math.isnan(_annualized_sortino(r, 0.0))
    assert math.isnan(_annualized_sortino(np.array([]), 0.0))


# --------------------------------------------------------------------- 2
def _hand_psr(r: np.ndarray, trials: int):
    n = r.size
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r, bias=True)
    kurt = stats.kurtosis(r, fisher=False, bias=True)
    se = math.sqrt((1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2) / (n - 1))
    g = 0.5772156649015329
    if trials > 1:
        sr0 = se * ((1 - g) * stats.norm.ppf(1 - 1.0 / trials)
                    + g * stats.norm.ppf(1 - 1.0 / (trials * math.e)))
    else:
        sr0 = 0.0
    return (sr - sr0) * math.sqrt(252), float(stats.norm.cdf((sr - sr0) / se) * 100.0), sr, se


def _series(seed=7, n=300, mu=0.0008, sigma=0.01):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


@pytest.mark.parametrize("trials", [1, 5, 50])
def test_deflated_sharpe_matches_hand_computation(trials):
    r = _series()
    sharpe = _annualized_sharpe(r, 0.0)
    dsr, psr = _deflated_sharpe(sharpe, r, trial_count=trials)
    h_dsr, h_psr, _, _ = _hand_psr(r, trials)
    assert dsr == pytest.approx(h_dsr, rel=1e-6, abs=1e-9)
    assert psr == pytest.approx(h_psr, rel=1e-6, abs=1e-9)


def test_deflated_sharpe_standard_error_is_per_period_scale():
    # Annualized SR ~ 1.3, per-period SR ~ 0.08 with N=300: PSR must be a
    # sensible probability (not saturated 100 from a 16x understated SE).
    r = _series(mu=0.0008)
    _, psr, sr, se = _hand_psr(r, 1)
    assert se > 0.04  # ~ sqrt(1/(N-1)) per period, not ~0.003
    _, got = _deflated_sharpe(_annualized_sharpe(r, 0.0), r, 1)
    assert got == pytest.approx(psr, rel=1e-6)
    assert got < 99.9


def test_psr_decreases_with_more_trials():
    r = _series()
    s = _annualized_sharpe(r, 0.0)
    psrs = [_deflated_sharpe(s, r, t)[1] for t in (1, 2, 10, 100, 1000)]
    dsrs = [_deflated_sharpe(s, r, t)[0] for t in (1, 2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(psrs, psrs[1:]))
    assert all(a > b for a, b in zip(dsrs, dsrs[1:]))


# --------------------------------------------------------------------- 3
def test_exposure_cash_bars_earn_rf_in_excess_returns():
    rng = np.random.default_rng(3)
    asset = rng.normal(0.002, 0.012, 400)
    pos = (rng.random(400) < 0.2).astype(float)
    strat = pos * asset
    rf = 0.45
    d = (1.0 + rf) ** (1.0 / 252.0) - 1.0

    naive = _annualized_sharpe(strat, rf)
    adj = _annualized_sharpe(strat, rf, exposure=pos)
    ex = strat - d * pos
    assert adj == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(252), rel=1e-12)
    assert naive < -1.5
    assert adj > naive + 1.5


def test_exposure_none_keeps_legacy_behaviour():
    r = np.array([0.01, -0.005, 0.007, 0.002])
    d = (1.0 + 0.3) ** (1.0 / 252.0) - 1.0
    ex = r - d
    assert _annualized_sharpe(r, 0.3) == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(252))


def test_summarize_uses_position_as_exposure_and_flat_strategy_not_negative():
    n = 60
    flat = summarize_backtest(_result(_curve(np.zeros(n), np.zeros(n))), risk_free_annual=0.45)
    assert flat["Sharpe"] == 0.0 or math.isnan(flat["Sharpe"])
    assert not (flat["Sharpe"] < -0.5)

    rng = np.random.default_rng(11)
    pos = (rng.random(300) < 0.15).astype(float)
    net = pos * rng.normal(0.003, 0.01, 300)
    s = summarize_backtest(_result(_curve(net, pos)), risk_free_annual=0.45)
    d = (1.45) ** (1 / 252.0) - 1.0
    ex = net - d * pos
    assert s["Sharpe"] == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(252), abs=1e-5)
    # Net_Return stays the pure strategy return (no rf yield added to equity).
    assert s["Net_Return"] == pytest.approx(np.prod(1 + net) - 1, abs=1e-6)


# --------------------------------------------------------------------- 4
def test_rf_series_point_in_time_release_convention(tmp_path):
    macro = tmp_path / "macro"
    macro.mkdir()
    pd.DataFrame({
        "Date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "INTEREST_RATE": [40.0, 45.0, 50.0],
    }).to_csv(macro / "INTEREST_RATE.csv", index=False)
    dates = pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01", "2024-03-29", "2024-04-02"])
    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("RISK_FREE_RATE_ANNUAL", None)
        ann = rfr.get_risk_free_rate_series(dates, macro_cache_dir=str(macro), annual=True)
        daily = rfr.get_risk_free_rate_series(dates, macro_cache_dir=str(macro))
    # Jan row released 2024-02-01; Feb row 2024-03-01; Mar row 2024-04-01.
    assert np.isnan(ann.iloc[0]) and np.isnan(ann.iloc[1])
    assert ann.iloc[2] == pytest.approx(0.40)
    assert ann.iloc[3] == pytest.approx(0.40)
    assert ann.iloc[4] == pytest.approx(0.45)
    assert ann.iloc[5] == pytest.approx(0.45)
    assert ann.iloc[6] == pytest.approx(0.50)
    assert daily.iloc[2] == pytest.approx(1.40 ** (1 / 252.0) - 1.0)


def test_summarize_wires_point_in_time_series_when_dates_and_auto_rf():
    n = 40
    dates = pd.bdate_range("2024-03-01", periods=n)
    rng = np.random.default_rng(5)
    net = rng.normal(0.001, 0.01, n)
    curve = _curve(net, dates=dates)
    daily = np.where(np.arange(n) < 20, 0.0005, 0.0015)
    series = pd.Series(daily, index=dates)
    with patch.object(bm, "_get_rf", return_value=0.40), \
            patch.object(bm, "_get_rf_series", return_value=series):
        s = summarize_backtest(_result(curve))
    ex = net - daily
    assert s["Sharpe"] == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(252), abs=1e-5)


# --------------------------------------------------------------------- 5
def test_max_drawdown_counts_first_bar_loss():
    assert _max_drawdown(np.array([0.90, 0.95, 1.0])) == pytest.approx(-0.10)
    assert _max_drawdown(np.array([1.0, 1.1, 0.99])) == pytest.approx(0.99 / 1.1 - 1.0)


def test_calmar_and_cagr_nan_for_short_samples():
    rng = np.random.default_rng(1)
    short = summarize_backtest(_result(_curve(rng.normal(0.001, 0.01, 30))), risk_free_annual=0.0)
    assert math.isnan(short["Calmar"])
    assert math.isnan(short["CAGR"]) and math.isnan(short["Annualized_Return"])
    assert np.isfinite(short["Net_Return"])


def test_calmar_nan_when_no_drawdown_and_finite_for_long_samples():
    up = summarize_backtest(_result(_curve(np.full(200, 0.001))), risk_free_annual=0.0)
    assert up["Max_Drawdown"] == 0.0
    assert math.isnan(up["Calmar"])
    assert np.isfinite(up["CAGR"])

    rng = np.random.default_rng(2)
    long_ = summarize_backtest(_result(_curve(rng.normal(0.001, 0.01, 200))), risk_free_annual=0.0)
    assert np.isfinite(long_["Calmar"]) and np.isfinite(long_["CAGR"])


# --------------------------------------------------------------------- 6
def test_sharpe_volatility_and_ir_use_ddof1():
    r = np.array([0.01, -0.02, 0.015, 0.005, -0.004, 0.012])
    assert _annualized_sharpe(r, 0.0) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    bh = np.full(r.size, 0.001)
    active = r - bh
    assert _information_ratio(r, bh) == pytest.approx(
        active.mean() * 252 / (active.std(ddof=1) * math.sqrt(252))
    )
    s = summarize_backtest(_result(_curve(r)), risk_free_annual=0.0)
    assert s["Volatility"] == pytest.approx(r.std(ddof=1) * math.sqrt(252), abs=1e-5)


# --------------------------------------------------------------------- 7
def _fm(true_t, pred_t, rf=0.0):
    true_t = np.asarray(true_t, dtype=float)
    pred_t = np.asarray(pred_t, dtype=float)
    prev = np.full(true_t.size, 100.0)
    return compute_financial_metrics(
        prev * (1 + true_t), prev * (1 + pred_t),
        y_true_target=true_t, y_pred_target=pred_t,
        prev_close=prev, target_mode="return", risk_free_annual=rf,
    )


def test_dir_acc_excludes_zero_true_returns_and_reports_n_and_pvalue():
    m = _fm([0.01, 0.0, -0.01, 0.02], [0.01, 0.02, -0.02, -0.01])
    assert m["Dir_Acc_N"] == 3
    assert m["Dir_Acc"] == pytest.approx(2 / 3 * 100)
    assert m["Dir_Acc_PValue"] == pytest.approx(stats.binomtest(2, 3, 0.5, alternative="greater").pvalue)


def test_dir_acc_zero_pred_zero_true_not_a_hit():
    m = _fm([0.0, 0.0, 0.01, -0.01], [0.0, 0.0, 0.02, 0.01])
    assert m["Dir_Acc_N"] == 2
    assert m["Dir_Acc"] == pytest.approx(50.0)


def test_hit_rate_excludes_zero_return_bars():
    m = _fm([0.01, 0.0, -0.01, 0.02], [0.01, 0.02, -0.02, -0.01])
    # strategy returns sign(pred)*true = [.01, 0, .01, -.02] -> 2 wins / 3 non-zero
    assert m["Hit_Rate"] == pytest.approx(2 / 3 * 100)


def test_long_short_sharpe_is_not_rf_adjusted():
    t = [0.01, -0.012, 0.008, 0.004, -0.006, 0.011]
    p = [0.01, -0.01, 0.01, -0.01, -0.01, 0.01]
    assert _fm(t, p, rf=0.45)["Sharpe"] == pytest.approx(_fm(t, p, rf=0.0)["Sharpe"])


# --------------------------------------------------------------------- 8
def _ar1(n=600, phi=0.7, seed=4):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x + 0.0005


def _iid_ci(returns, n_resamples=1000, seed=1):
    rng = np.random.default_rng(seed)
    n = returns.size
    s = np.array([_annualized_sharpe(returns[rng.integers(0, n, n)], 0.0) for _ in range(n_resamples)])
    return np.quantile(s, 0.025), np.quantile(s, 0.975)


def test_block_bootstrap_ci_wider_than_iid_on_autocorrelated_series():
    r = _ar1()
    lo, hi = _bootstrap_sharpe_ci(r, 0.0)
    ilo, ihi = _iid_ci(r)
    assert (hi - lo) > (ihi - ilo) * 1.15


def test_block_bootstrap_is_deterministic_and_keeps_signature():
    r = _ar1(n=200)
    a = _bootstrap_sharpe_ci(r, 0.0, n_resamples=200)
    b = _bootstrap_sharpe_ci(r, 0.0, n_resamples=200)
    assert a == b and a[0] < a[1]
    assert all(math.isnan(v) for v in _bootstrap_sharpe_ci(r, None))
