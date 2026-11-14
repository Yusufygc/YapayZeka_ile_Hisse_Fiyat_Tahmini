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


def test_per_date_bucket_is_causal_and_ordered():
    """K5/K2: tarih-bazli kova ayni tarihin degerlerine bagli; gelecek tarih degisince gecmis degismez."""
    from src.validation.segment_ic import per_date_bucket

    p = _panel(n_sym=20, n_dates=10)
    b0 = per_date_bucket(p, "liq_log", 5)
    assert set(b0) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    d0 = p["Date"].min()
    # ayni tarihte liq_log monoton -> S0 en dusuk (Q1), S19 en yuksek (Q5)
    row0 = p[(p["Date"] == d0)]
    assert b0[row0[row0.symbol == "S0"].index[0]] == "Q1"
    assert b0[row0[row0.symbol == "S19"].index[0]] == "Q5"
    # gelecegi boz -> gecmis kovalar ayni
    q = p.copy()
    q.loc[q["Date"] > p["Date"].iloc[len(p) // 2], "liq_log"] = np.random.default_rng(1).normal(
        size=int((q["Date"] > p["Date"].iloc[len(p) // 2]).sum()))
    b1 = per_date_bucket(q, "liq_log", 5)
    past = p["Date"] <= p["Date"].iloc[len(p) // 2]
    assert (b0[past] == b1[past]).all()


def test_attach_date_segments_matches_schema_and_is_causal():
    """K5: kovalar tarih-bazli; gelecekteki liq_log degisimi gecmis satirlarin kovasini degistirmez."""
    from src.validation.segment_ic import attach_date_segments

    p = _panel(n_sym=20, n_dates=12)
    preds = p[["symbol", "Date"]].assign(y_pred=0.0, y_true=0.0)
    out = attach_date_segments(preds, p, n_buckets=5)
    assert {"liq_bucket", "vol_bucket", "sector"} <= set(out.columns)
    assert len(out) == len(preds)
    assert set(out["liq_bucket"].dropna()) == {"Q1", "Q2", "Q3", "Q4", "Q5"}

    cut = p["Date"].iloc[len(p) // 2]
    q = p.copy()
    q.loc[q["Date"] > cut, "liq_log"] = 99.0 - np.arange(int((q["Date"] > cut).sum())) * 0.001
    out2 = attach_date_segments(preds, q, n_buckets=5)
    past = out["Date"] <= cut
    assert (out.loc[past, "liq_bucket"] == out2.loc[past, "liq_bucket"]).all()


def test_latest_date_segments_single_date_table():
    from src.validation.segment_ic import latest_date_segments

    p = _panel(n_sym=20, n_dates=3)
    latest = p[p["Date"] == p["Date"].max()]
    seg = latest_date_segments(latest, n_buckets=5)
    assert list(seg.columns) == ["symbol", "liq_bucket", "vol_bucket", "sector"]
    assert len(seg) == 20 and seg["symbol"].is_unique
    assert seg.set_index("symbol").loc["S0", "liq_bucket"] == "Q1"
    assert seg.set_index("symbol").loc["S19", "liq_bucket"] == "Q5"


def test_date_segments_feed_segment_ic_like_static_ones():
    """attach_date_segments ciktisi segment_cross_sectional_ic ile dogrudan kullanilabilir."""
    from src.validation.segment_ic import attach_date_segments

    preds = _preds()
    panel = _panel()
    preds = preds[["symbol", "Date", "y_pred", "y_true"]]
    seg = attach_date_segments(preds, panel, n_buckets=5)
    tab = segment_cross_sectional_ic(seg, group_col="liq_bucket", min_names=3)
    assert set(tab["segment"]) <= {"Q1", "Q2", "Q3", "Q4", "Q5"} and len(tab) >= 1
