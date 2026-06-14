# -*- coding: utf-8 -*-
"""Peer (cross-sectional) scoring tests — E2 Faz 5."""

import numpy as np
import pandas as pd
import pytest

from src.serving.peer_scoring import (
    PeerScoringConfig,
    rank_to_peer_scores,
    score_latest_universe,
)


def _preds(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)],
        "y_pred": rng.normal(0, 1, n),
    })


def test_peer_score_centered_and_percentile_range():
    out = rank_to_peer_scores(_preds(20), as_of_date="2026-06-03")
    assert out["peer_score"].between(-1, 1).all()
    assert out["peer_percentile"].between(0, 100).all()
    # merkezli: ortalama ~0
    assert abs(out["peer_score"].mean()) < 1e-9
    assert out["universe_size"].iloc[0] == 20


def test_highest_pred_gets_top_percentile_outperform():
    df = pd.DataFrame({"symbol": [f"S{i}" for i in range(20)],
                       "y_pred": np.arange(20, dtype=float)})
    out = rank_to_peer_scores(df, as_of_date="2026-06-03").set_index("symbol")
    # S19 en yuksek pred -> en yuksek percentile -> outperform
    assert out.loc["S19", "peer_label"] == "outperform"
    assert out.loc["S0", "peer_label"] == "underperform"
    assert out.loc["S19", "peer_percentile"] > out.loc["S0", "peer_percentile"]


def test_labels_respect_thresholds():
    df = pd.DataFrame({"symbol": [f"S{i}" for i in range(10)],
                       "y_pred": np.arange(10, dtype=float)})
    out = rank_to_peer_scores(
        df, as_of_date="d", cfg=PeerScoringConfig(lo_pct=30, hi_pct=70, min_names=5)).set_index("symbol")
    # 10 isim: percentile = (rank-0.5)/10*100 -> 5,15,...,95
    assert out.loc["S9", "peer_label"] == "outperform"   # 95
    assert out.loc["S5", "peer_label"] == "inline"       # 55
    assert out.loc["S0", "peer_label"] == "underperform" # 5


def test_thin_universe_unknown_label():
    out = rank_to_peer_scores(
        _preds(5), as_of_date="d", cfg=PeerScoringConfig(min_names=15))
    assert (out["peer_label"] == "unknown").all()


def test_empty_predictions_safe():
    out = rank_to_peer_scores(
        pd.DataFrame({"symbol": [], "y_pred": []}), as_of_date="d")
    assert out.empty


class _StubModel:
    """pred = ilk ozellik (deterministik)."""

    def predict(self, X):
        return np.asarray(X)[:, 0]


class _DTypeModel:
    dtype = None
    contiguous = None

    def predict(self, X):
        self.dtype = X.dtype
        self.contiguous = bool(X.flags["C_CONTIGUOUS"])
        return np.asarray(X)[:, 0]


def test_score_latest_universe_rejects_multi_date_by_default():
    dates = pd.to_datetime(["2026-06-01", "2026-06-02"])
    rows = []
    for d in dates:
        for i in range(20):
            rows.append({"symbol": f"S{i}", "Date": d, "f0": float(i if d == dates[1] else 0)})
    panel = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="tek skorlama tarihi"):
        score_latest_universe(_StubModel(), panel, feature_cols=["f0"])


def test_score_latest_universe_can_pick_latest_date_when_explicitly_allowed():
    dates = pd.to_datetime(["2026-06-01", "2026-06-02"])
    rows = []
    for d in dates:
        for i in range(20):
            rows.append({"symbol": f"S{i}", "Date": d, "f0": float(i if d == dates[1] else 0)})
    panel = pd.DataFrame(rows)
    cfg = PeerScoringConfig(strict_single_date=False)

    out = score_latest_universe(_StubModel(), panel, feature_cols=["f0"], cfg=cfg)

    assert out["as_of_date"].iloc[0].startswith("2026-06-02")
    assert out["universe_size"].iloc[0] == 20
    # f0=i en yuksek S19 -> outperform
    assert out.set_index("symbol").loc["S19", "peer_label"] == "outperform"


def test_score_latest_universe_feeds_float32_matrix_to_model():
    panel = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(20)],
        "Date": pd.Timestamp("2026-06-02"),
        "f0": np.arange(20, dtype=float),
        "f1": np.linspace(0.0, 1.0, 20),
    })
    model = _DTypeModel()

    out = score_latest_universe(model, panel, feature_cols=["f1", "f0"])

    assert model.dtype == np.float32
    assert model.contiguous is True
    assert out["universe_size"].iloc[0] == 20
