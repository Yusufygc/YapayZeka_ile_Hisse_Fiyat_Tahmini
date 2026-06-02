# -*- coding: utf-8 -*-
"""Stratified per-segment cross-sectional IC tests — E2 Faz 6."""

import numpy as np
import pandas as pd

from src.validation.segment_ic import (
    attach_segments,
    segment_cross_sectional_ic,
    symbol_segments,
)


def _panel(n_sym=20, n_dates=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rows = []
    for s in range(n_sym):
        liq = 3.0 + s * 0.5          # monoton -> kovalar ayrisir
        vol = 0.01 + (s % 5) * 0.01
        sec = "Energy" if s % 2 == 0 else "Financials"
        for d in dates:
            rows.append({"symbol": f"S{s}", "Date": d, "liq_log": liq + rng.normal(0, 0.01),
                         "vol": vol, "sector": sec})
    return pd.DataFrame(rows)


def test_symbol_segments_schema_and_buckets():
    seg = symbol_segments(_panel(), n_buckets=5)
    assert set(seg.columns) == {"symbol", "liq_bucket", "vol_bucket", "sector"}
    assert seg["symbol"].nunique() == 20
    # 20 sembol / 5 kova -> her likidite kovasinda 4 sembol
    assert set(seg["liq_bucket"]) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert (seg["liq_bucket"].value_counts() == 4).all()


def _preds(n_sym=20, n_dates=40, seed=1, strong_segment="Energy"):
    """Energy sembollerinde pred=true (mukemmel IC), Financials'ta gurultu."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rows = []
    for s in range(n_sym):
        sec = "Energy" if s % 2 == 0 else "Financials"
        for d in dates:
            yt = float(rng.normal(0, 1))
            yp = yt if sec == strong_segment else float(rng.normal(0, 1))
            rows.append({"symbol": f"S{s}", "Date": d, "y_true": yt,
                         "y_pred": yp, "sector": sec})
    return pd.DataFrame(rows)


def test_segment_ic_separates_strong_from_noise():
    preds = _preds()
    out = segment_cross_sectional_ic(preds, group_col="sector", min_names=5)
    by = out.set_index("segment")
    # Energy pred=true -> IC ~ +1; Financials gurultu -> IC ~ 0
    assert by.loc["Energy", "ic_mean"] > 0.9
    assert abs(by.loc["Financials", "ic_mean"]) < 0.2
    assert by.loc["Energy", "pct_positive"] == 1.0


def test_segment_ic_respects_min_names():
    preds = _preds(n_sym=6)  # her sektorde 3 sembol
    out = segment_cross_sectional_ic(preds, group_col="sector", min_names=5)
    # 3 < 5 -> hicbir gun sayilmaz
    assert (out["n_days"] == 0).all()


def test_attach_segments_merges_by_symbol():
    preds = _preds(n_sym=10)
    seg = symbol_segments(
        _panel(n_sym=10), n_buckets=2)
    merged = attach_segments(preds.drop(columns=["sector"]), seg)
    assert "liq_bucket" in merged.columns and "sector" in merged.columns
    assert not merged["liq_bucket"].isna().any()


def test_segment_ic_deterministic():
    preds = _preds()
    a = segment_cross_sectional_ic(preds, group_col="sector", min_names=5)
    b = segment_cross_sectional_ic(preds, group_col="sector", min_names=5)
    pd.testing.assert_frame_equal(a, b)
