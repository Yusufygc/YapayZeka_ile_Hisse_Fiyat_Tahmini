# -*- coding: utf-8 -*-
"""Nightly scoring orchestration tests — E2 Faz 5."""

import numpy as np
import pandas as pd

from src.serving.nightly_scoring import (
    assemble_peer_table,
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


def test_empty_panel_safe():
    empty = pd.DataFrame({"symbol": [], "Date": [], "f0": []})
    out = assemble_peer_table(_StubModel(), empty, ["f0"], _seg_table(), _ICIR)
    assert out.empty
