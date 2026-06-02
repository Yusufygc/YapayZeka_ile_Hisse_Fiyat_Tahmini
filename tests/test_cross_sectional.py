# -*- coding: utf-8 -*-
"""Cross-sectional target transform tests — E2 Faz 3.5."""

import numpy as np
import pandas as pd

from src.data.cross_sectional import (
    add_cross_sectional_features,
    add_cross_sectional_target,
)


def _panel(n_dates=30, n_sym=8, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rows = []
    for s in range(n_sym):
        for d in dates:
            rows.append({"symbol": f"S{s}", "Date": d,
                         "target": float(rng.normal(0, 0.03))})
    return pd.DataFrame(rows)


def test_rank_centered_zero_per_date():
    out = add_cross_sectional_target(_panel(), method="rank", min_names=5)
    # her tarih ici ortalama ~0 (merkezli pct rank)
    by_date = out.groupby("Date")["target_cs"].mean()
    assert np.allclose(by_date.to_numpy(), 0.0, atol=1e-9)
    assert out["target_cs"].between(-1.0, 1.0).all()


def test_sign_balanced_near_base_rate_50():
    out = add_cross_sectional_target(_panel(n_sym=10), method="rank", min_names=5)
    pos = float((out["target_cs"] > 0).mean())
    assert 0.40 <= pos <= 0.60   # dengeli -> base-rate ~50


def test_zscore_unit_scale_per_date():
    out = add_cross_sectional_target(_panel(), method="zscore", min_names=5)
    g = out.groupby("Date")["target_cs"]
    assert np.allclose(g.mean().to_numpy(), 0.0, atol=1e-9)
    # std ~1 (tek-tarih, ddof=1)
    assert np.allclose(g.std().dropna().to_numpy(), 1.0, atol=1e-6)


def test_thin_dates_dropped():
    p = _panel(n_dates=5, n_sym=8)
    # bir tarihte sadece 2 sembol birak
    d0 = p["Date"].min()
    p = p[~((p["Date"] == d0) & (~p["symbol"].isin(["S0", "S1"])))]
    out = add_cross_sectional_target(p, method="rank", min_names=5)
    assert d0 not in set(out["Date"])


def test_no_nan_and_deterministic():
    p = _panel()
    a = add_cross_sectional_target(p, method="rank")
    b = add_cross_sectional_target(p, method="rank")
    assert not a["target_cs"].isna().any()
    pd.testing.assert_frame_equal(a, b)


def _feat_panel(n_dates=20, n_sym=10, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rows = []
    for s in range(n_sym):
        for d in dates:
            rows.append({"symbol": f"S{s}", "Date": d,
                         "mom": float(rng.normal(0, 1)),
                         "liq_log": float(rng.normal(5, 1))})
    return pd.DataFrame(rows)


def test_cs_features_added_rank_and_zscore():
    p = _feat_panel()
    out, new = add_cross_sectional_features(p, ["mom", "liq_log"])
    assert set(new) == {"mom_csr", "mom_csz", "liq_log_csr", "liq_log_csz"}
    # rank merkezli [-1,1], tarih ici ortalama ~0
    by_date = out.groupby("Date")["mom_csr"].mean()
    assert np.allclose(by_date.to_numpy(), 0.0, atol=1e-9)
    # zscore tarih ici ortalama ~0, std ~1
    g = out.groupby("Date")["mom_csz"]
    assert np.allclose(g.mean().to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(g.std().dropna().to_numpy(), 1.0, atol=1e-6)


def test_cs_features_no_nan_and_causal_within_date():
    """cs feature degeri sadece o tarihin cross-section'ina baglidir; gelecekteki
    tarihi degistirmek erken tarihin cs feature'ini bozmaz."""
    p = _feat_panel(n_dates=8, n_sym=10, seed=4)
    base, new = add_cross_sectional_features(p, ["mom"])
    assert not base[new].isna().any().any()
    last = p["Date"].max()
    p2 = p.copy()
    p2.loc[p2["Date"] == last, "mom"] += 50.0
    pert, _ = add_cross_sectional_features(p2, ["mom"])
    e1 = base[base["Date"] < last].reset_index(drop=True)
    e2 = pert[pert["Date"] < last].reset_index(drop=True)
    pd.testing.assert_series_equal(e1["mom_csr"], e2["mom_csr"])


def test_within_date_only_no_future_dependency():
    """Bir tarihin target_cs'i sadece o tarihteki ham hedeflere baglidir;
    gelecekteki bir tarihi degistirmek erken tarihin degerini bozmaz."""
    p = _panel(n_dates=10, n_sym=8, seed=3)
    base = add_cross_sectional_target(p, method="rank")
    last = p["Date"].max()
    p2 = p.copy()
    p2.loc[p2["Date"] == last, "target"] += 99.0  # sadece son tarihi boz
    pert = add_cross_sectional_target(p2, method="rank")
    early = base[base["Date"] < last].reset_index(drop=True)
    early2 = pert[pert["Date"] < last].reset_index(drop=True)
    pd.testing.assert_series_equal(early["target_cs"], early2["target_cs"])
