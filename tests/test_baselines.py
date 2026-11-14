# -*- coding: utf-8 -*-
"""K2: kural-tabanli kiyas + tradable alt kume IC yardimcilari."""

import numpy as np
import pandas as pd
import pytest

from src.serving.nightly_scoring import liqlog_floor_from_turnover
from src.validation.baselines import (
    attach_liquidity,
    ic_report,
    liqlog_floor,
    liquidity_quantile_ic,
    reversal_factor_predictions,
    tradable_subset,
)


def _panel(n_sym=24, n_dates=60, seed=0, h=5, rev_strength=1.0):
    """Close rastgele yuruyus; hedef = -past5 * rev_strength + gurultu (ters-momentum alfa)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rows = []
    for s in range(n_sym):
        close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_dates)))
        lc = np.log(close)
        past = np.full(n_dates, np.nan)
        past[5:] = lc[5:] - lc[:-5]
        for i, d in enumerate(dates):
            tgt = (-rev_strength * past[i] if np.isfinite(past[i]) else 0.0) + rng.normal(0, 0.005)
            rows.append({"symbol": f"S{s:02d}", "Date": d, "Close": close[i],
                         "target_cs": tgt, "liq_log": 10.0 + s * 0.2})
    return pd.DataFrame(rows)


def test_liqlog_floor_matches_nightly_definition():
    for v in (0.0, -5.0, 3_000_000.0, 1e9):
        assert liqlog_floor(v) == pytest.approx(liqlog_floor_from_turnover(v))


def test_reversal_factor_is_causal():
    panel = _panel()
    base = reversal_factor_predictions(panel)
    cut = pd.Timestamp("2021-02-15")
    mutated = panel.copy()
    mutated.loc[mutated["Date"] > cut, "Close"] *= 3.0     # gelecegi boz
    after = reversal_factor_predictions(mutated)
    k = ["symbol", "Date"]
    a = base[base["Date"] <= cut].set_index(k)["y_pred"]
    b = after[after["Date"] <= cut].set_index(k)["y_pred"]
    assert np.allclose(a.sort_index().to_numpy(), b.sort_index().to_numpy())


def test_reversal_factor_ic_positive_when_target_mean_reverts():
    preds = reversal_factor_predictions(_panel(rev_strength=1.0))
    out = ic_report(preds, min_names=10)
    assert out["all"]["ic_mean"] > 0.5 and out["all"]["n_days"] > 20
    # panel verilmediyse (liq_log tahmin tablosunda zaten var) tradable ozet de gelir
    assert "tradable" in out


def test_tradable_subset_filters_by_turnover_floor():
    preds = reversal_factor_predictions(_panel())
    floor_tl = float(np.expm1(10.0 + 12 * 0.2))      # S12 ve ustu gecer
    sub = tradable_subset(preds, floor_tl)
    assert set(sub["symbol"]) == {f"S{s:02d}" for s in range(12, 24)}
    assert tradable_subset(preds, 0.0)["symbol"].nunique() == 24


def test_tradable_subset_requires_liquidity_column():
    preds = reversal_factor_predictions(_panel()).drop(columns=["liq_log"])
    with pytest.raises(ValueError):
        tradable_subset(preds, 1e6)
    panel = _panel()
    assert "liq_log" in attach_liquidity(preds, panel).columns


def test_liquidity_quantile_ic_flags_signal_location():
    """Sinyal yalniz en az likit yarida (S00-S11) -> Q1/Q2 IC > Q4/Q5 IC."""
    panel = _panel(rev_strength=0.0)
    rng = np.random.default_rng(3)
    past = np.log(panel["Close"]) - np.log(panel.groupby("symbol")["Close"].shift(5))
    low_liq = panel["liq_log"] < 12.4
    panel["target_cs"] = np.where(low_liq, -past.fillna(0.0), rng.normal(0, 1, len(panel))) \
        + rng.normal(0, 0.001, len(panel))
    preds = reversal_factor_predictions(panel)
    table = liquidity_quantile_ic(preds, n_buckets=4, min_names=4).set_index("segment")
    assert table.loc["Q1", "ic_mean"] > 0.5
    assert table.loc["Q4", "ic_mean"] < 0.3
