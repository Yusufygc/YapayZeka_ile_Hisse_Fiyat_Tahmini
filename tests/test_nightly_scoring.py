# -*- coding: utf-8 -*-
"""Nightly scoring orchestration tests — E2 Faz 5."""

import numpy as np
import pandas as pd

from src.serving.nightly_scoring import (
    assemble_peer_table,
    composite_icir,
    liqlog_floor_from_turnover,
    segment_icir_from_table,
)


class _StubModel:
    """pred = ilk ozellik."""

    def predict(self, X):
        return np.asarray(X)[:, 0]


def _latest_panel(n=20):
    return pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)],
        "Date": pd.Timestamp("2026-06-03"),
        "f0": np.arange(n, dtype=float),
    })


def _seg_table(n=20):
    # ilk yari Q1 (en az likit, guclu IC), ikinci yari Q5 (likit, zayif)
    return pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)],
        "liq_bucket": ["Q1" if i < n // 2 else "Q5" for i in range(n)],
        "vol_bucket": ["Q5"] * n,
        "sector": ["Industrials"] * n,
    })


_ICIR = {"Q1": 1.35, "Q5": 0.30}


def test_assemble_schema_and_segment_join():
    out = assemble_peer_table(_StubModel(), _latest_panel(), ["f0"],
                              _seg_table(), _ICIR, scoring_cfg=None)
    for c in ["symbol", "peer_score", "peer_label", "segment_liq",
              "segment_icir", "confidence_label", "confidence_reasons"]:
        assert c in out.columns
    assert out["segment_liq"].isin(["Q1", "Q5"]).all()


def test_segment_icir_drives_confidence():
    out = assemble_peer_table(_StubModel(), _latest_panel(), ["f0"],
                              _seg_table(), _ICIR).set_index("symbol")
    # Q1 segment ICIR 1.35 -> high (tradable varsayilan); Q5 ICIR 0.30 -> low
    assert out.loc["S0", "segment_icir"] == 1.35
    assert out.loc["S0", "confidence_label"] == "high"
    assert out.loc["S19", "confidence_label"] == "low"


def test_tradability_gate_overrides_strong_segment():
    # en az likit (Q1, guclu IC) ama islem yapilamiyor -> low
    out = assemble_peer_table(
        _StubModel(), _latest_panel(), ["f0"], _seg_table(), _ICIR,
        tradable_for=lambda s: False).set_index("symbol")
    assert (out["confidence_label"] == "low").all()


def test_segment_icir_from_table():
    tbl = pd.DataFrame({"segment": ["Q1", "Q5"], "icir": [1.35, float("nan")]})
    m = segment_icir_from_table(tbl)
    assert m["Q1"] == 1.35
    assert m["Q5"] != m["Q5"]  # NaN korunur


def test_liqlog_floor_from_turnover():
    # liq_log = log1p(ciro) -> taban donusumu tersinir
    assert abs(liqlog_floor_from_turnover(3_000_000) - np.log1p(3_000_000)) < 1e-9
    assert liqlog_floor_from_turnover(0) == 0.0
    assert liqlog_floor_from_turnover(-5) == 0.0  # negatif -> kapi etkisiz


def test_turnover_floor_gates_illiquid_in_assemble():
    """Medyan cirosu taban altindaki (az likit) guclu-sinyal hisse -> low."""
    floor = liqlog_floor_from_turnover(3_000_000)  # liq_log ~14.91
    # S0..S9 Q1 (guclu IC) ama ciro dusuk; tradable_for taban altinda -> low
    out = assemble_peer_table(
        _StubModel(), _latest_panel(), ["f0"], _seg_table(), _ICIR,
        tradable_for=lambda s: 10.0 >= floor).set_index("symbol")  # 10 < 14.91 -> hicbiri tradable degil
    assert (out["confidence_label"] == "low").all()


_MAPS = {
    "liq": {"Q1": 1.35, "Q5": 0.39},
    "vol": {"Q5": 1.23, "Q1": 0.64},
    "sector": {"Industrials": 1.20, "Technology": 0.30},
}


def test_composite_icir_weighted_mean():
    # liq=Q5(0.39) vol=Q5(1.23) sector=Industrials(1.20), agirlik 0.5/0.3/0.2
    v = composite_icir("Q5", "Q5", "Industrials", _MAPS)
    expected = 0.5 * 0.39 + 0.3 * 1.23 + 0.2 * 1.20
    assert abs(v - expected) < 1e-9


def test_composite_icir_skips_missing_axis_renormalizes():
    # vol bucket map'te yok -> atlanir, liq+sector agirligi yeniden normalize
    v = composite_icir("Q1", "ZZ", "Industrials", _MAPS)
    expected = (0.5 * 1.35 + 0.2 * 1.20) / (0.5 + 0.2)
    assert abs(v - expected) < 1e-9


def test_composite_icir_all_missing_nan():
    import numpy as np
    assert np.isnan(composite_icir(None, None, None, _MAPS))


def test_blended_path_lifts_liquid_highvol_above_pure_liq():
    """Q5 (likit, liq-IC 0.39) ama yuksek-vol+Industrials -> harman > 0.39,
    saf-liq pathindeki low'dan medium/high'a cikabilir."""
    seg = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(20)],
        "liq_bucket": ["Q5"] * 20, "vol_bucket": ["Q5"] * 20,
        "sector": ["Industrials"] * 20})
    out = assemble_peer_table(_StubModel(), _latest_panel(), ["f0"], seg,
                              icir_maps=_MAPS).set_index("symbol")
    # harman = 0.5*0.39+0.3*1.23+0.2*1.20 = 0.804 -> medium (>=0.5)
    assert abs(out["segment_icir"].iloc[0] - 0.804) < 1e-6
    assert (out["confidence_label"] == "medium").all()


def test_assemble_adds_trend_columns():
    # Test case 1: Close is missing
    out = assemble_peer_table(_StubModel(), _latest_panel(), ["f0"],
                              _seg_table(), _ICIR).set_index("symbol")
    for c in ["trend_label", "trend_prob_up", "trend_expected_return",
              "kolb_price_p50", "kolb_price_low", "kolb_price_high",
              "kolb_horizon_days", "kolb_band_level"]:
        assert c in out.columns
    # pred = f0 = arange -> S19 en yuksek percentile -> yukarı; S0 en dusuk -> aşağı
    assert out.loc["S19", "trend_label"] == "yukarı"
    assert out.loc["S0", "trend_label"] == "aşağı"
    assert out.loc["S19", "trend_prob_up"] > out.loc["S0", "trend_prob_up"]
    assert out["kolb_price_p50"].isna().all()

    # Test case 2: Close is present
    panel_with_close = _latest_panel()
    panel_with_close["Close"] = 100.0
    out2 = assemble_peer_table(_StubModel(), panel_with_close, ["f0"],
                               _seg_table(), _ICIR).set_index("symbol")
    assert not out2["kolb_price_p50"].isna().any()
    assert (out2["kolb_price_low"] < out2["kolb_price_p50"]).all()
    assert (out2["kolb_price_p50"] < out2["kolb_price_high"]).all()
    assert (out2["kolb_horizon_days"] == 5).all()
    assert (out2["kolb_band_level"] == 0.8).all()


def test_empty_panel_safe():
    empty = pd.DataFrame({"symbol": [], "Date": [], "f0": []})
    out = assemble_peer_table(_StubModel(), empty, ["f0"], _seg_table(), _ICIR)
    assert out.empty
