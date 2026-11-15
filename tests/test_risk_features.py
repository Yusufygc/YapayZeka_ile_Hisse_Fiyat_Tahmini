# -*- coding: utf-8 -*-
"""Risk ozellik/hedef nedensellik + purge testleri (sentetik veri)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_synth import make_panel
from src.risk.features import EPS, RISK_FEATURE_COLS, build_risk_frame, labeled_rows

H = 5


@pytest.fixture(scope="module")
def panel():
    return make_panel(n_sym=14, n_days=400, seed=1)


@pytest.fixture(scope="module")
def frame(panel):
    return build_risk_frame(panel)


def _cmp(a: pd.DataFrame, b: pd.DataFrame, cols) -> None:
    m = a.merge(b, on=["symbol", "Date"], suffixes=("_a", "_b"))
    assert len(m) > 0
    for c in cols:
        x, y = m[f"{c}_a"].to_numpy(float), m[f"{c}_b"].to_numpy(float)
        assert np.allclose(x, y, equal_nan=True), c


def test_features_never_use_future(panel, frame):
    """Cercevenin T'den SONRASI kesilirse T ve oncesi ozellikler degismez."""
    cut = pd.Timestamp(sorted(panel["Date"].unique())[250])
    short = build_risk_frame(panel[panel["Date"] <= cut])
    _cmp(frame[frame["Date"] <= cut], short, RISK_FEATURE_COLS)


def test_features_invariant_to_future_price_shock(panel, frame):
    d = sorted(panel["Date"].unique())
    t0, shock_from = pd.Timestamp(d[200]), pd.Timestamp(d[201])
    p2 = panel.copy()
    p2.loc[p2["Date"] >= shock_from, "Close"] *= 3.0
    f2 = build_risk_frame(p2)
    _cmp(frame[frame["Date"] <= t0], f2[f2["Date"] <= t0], RISK_FEATURE_COLS)


def test_target_uses_only_t_plus_1_to_h(panel, frame):
    """y_t, t'den ONCEKI fiyata ve t+h'den SONRAKI fiyata bagli degil; t+k (1..h) degisince degisir."""
    d = [pd.Timestamp(x) for x in sorted(panel["Date"].unique())]
    t = 200
    sym = "S03"

    def y_at(p):
        f = build_risk_frame(p)
        return f[(f["symbol"] == sym) & (f["Date"] == d[t])]["y"].iloc[0]

    base = y_at(panel)
    before = panel.copy()
    before.loc[(before["symbol"] == sym) & (before["Date"] < d[t]), "Close"] *= 1.7
    assert y_at(before) == pytest.approx(base)          # t'den onceki: etkisiz
    after = panel.copy()
    after.loc[(after["symbol"] == sym) & (after["Date"] > d[t + H]), "Close"] *= 1.7
    assert y_at(after) == pytest.approx(base)           # t+h'den sonrasi: etkisiz
    inside = panel.copy()
    inside.loc[(inside["symbol"] == sym) & (inside["Date"] == d[t + 3]), "Close"] *= 1.2
    assert y_at(inside) != pytest.approx(base)          # pencere ici: etkili


def test_target_value_matches_definition(panel, frame):
    sym = "S01"
    s = panel[panel["symbol"] == sym].sort_values("Date").reset_index(drop=True)
    r = np.log(s["Close"]).diff().to_numpy()
    t = 150
    expect = np.log(np.sqrt(np.mean(r[t + 1:t + 1 + H] ** 2)) + EPS)
    row = frame[(frame["symbol"] == sym) & (frame["Date"] == s["Date"][t])].iloc[0]
    assert row["y"] == pytest.approx(expect)
    assert row["fwd_ret"] == pytest.approx(np.log(s["Close"][t + H] / s["Close"][t]))
    assert row["target_date"] == s["Date"][t + H]
    rv21 = np.log(np.sqrt(np.mean(r[t - 20:t + 1] ** 2)) + EPS)
    assert row["rv_21"] == pytest.approx(rv21)
    assert row["rv_1"] == pytest.approx(np.log(abs(r[t]) + EPS))
    assert row["rv_1_down"] == pytest.approx(row["rv_1"] if r[t] < 0 else 0.0)


def test_last_h_days_unlabeled_and_target_date_strictly_after(frame):
    lab = labeled_rows(frame)
    assert (lab["target_date"] > lab["Date"]).all()
    last = frame["Date"].max()
    assert frame[frame["Date"] == last]["y"].isna().all()       # etiketsiz (skorlama) satirlar
    assert lab["Date"].max() < last


def test_missing_day_breaks_returns_not_silently_multiday():
    p = make_panel(n_sym=12, n_days=200, seed=2)
    miss = [sorted(p["Date"].unique())[100]]
    p2 = p[~((p["symbol"] == "S00") & p["Date"].isin(miss))]
    f = build_risk_frame(p2)
    d = [pd.Timestamp(x) for x in sorted(p["Date"].unique())]
    s0 = f[f["symbol"] == "S00"].set_index("Date")
    assert np.isnan(s0.loc[d[101], "rv_1"])                   # eksik gunun ertesi: 2g getiri sayilmaz
    assert np.isnan(s0.loc[d[99], "y"]) and np.isnan(s0.loc[d[96], "y"])  # hedef penceresi gunu kapsiyor
    assert not np.isnan(s0.loc[d[90], "y"])
    other = f[f["symbol"] == "S01"].set_index("Date")
    assert not np.isnan(other.loc[d[101], "rv_1"])            # diger semboller etkilenmez


def test_market_columns_are_cross_sectional_same_date(panel, frame):
    d = frame["Date"].iloc[200 * 14]
    g = frame[frame["Date"] == d]
    assert g["mkt_rv_21"].nunique() == 1 and g["mkt_ret"].nunique() == 1
    assert g["mkt_rv_21"].iloc[0] == pytest.approx(g["rv_21"].median())
    assert g["mkt_ret"].iloc[0] == pytest.approx(
        np.nanmean(panel.pivot(index="Date", columns="symbol", values="Close")
                   .pipe(np.log).diff().loc[d].to_numpy()))


def test_requires_columns():
    with pytest.raises(ValueError):
        build_risk_frame(make_panel(n_sym=3, n_days=50).drop(columns=["NATR_14"]))
