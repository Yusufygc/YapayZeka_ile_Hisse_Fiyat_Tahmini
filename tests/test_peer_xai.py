# -*- coding: utf-8 -*-
"""E2 Kol-B XAI — pooled cross-sectional attribution tests."""

import numpy as np
import pytest

from src.serving.peer_xai import compute_peer_xai

_FEATURES = ["RSI_14_csr", "vol", "Return_csz", "liq_log", "symbol_id"]
_FACTOR_KEYS = {
    "feature_name", "human_label", "importance", "direction",
    "feature_group", "reason", "method", "contribution", "approximate",
}


def _train_booster(seed: int = 0):
    """Kucuk LightGBM booster (5 feature, sinyal target_0 + target_2)."""
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    n = 400
    X = rng.normal(size=(n, len(_FEATURES)))
    y = 1.5 * X[:, 0] - 1.2 * X[:, 2] + 0.1 * rng.normal(size=n)
    ds = lgb.Dataset(X, label=y)
    return lgb.train(
        {"objective": "regression", "num_leaves": 15, "verbose": -1,
         "min_data_in_leaf": 5},
        ds, num_boost_round=30,
    )


class _Single:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        return self.booster.predict(np.asarray(X, dtype=float))


class _Ensemble:
    """EnsemblePooledModel benzeri: .lgb -> .booster."""
    def __init__(self, booster):
        self.lgb = _Single(booster)

    def predict(self, X):
        return self.lgb.predict(X)


def _X(n=6):
    return np.random.default_rng(1).normal(size=(n, len(_FEATURES)))


def test_single_model_attribution_shape():
    model = _Single(_train_booster())
    syms = [f"S{i}" for i in range(6)]
    out = compute_peer_xai(model, _X(6), _FEATURES, syms, top_k=3)
    assert set(out.keys()) == set(syms)
    for sym in syms:
        rec = out[sym]
        assert rec["caveat"] == ""  # tekil model -> caveat yok
        assert len(rec["top_positive"]) <= 3
        assert len(rec["top_negative"]) <= 3
        assert rec["top_positive"] or rec["top_negative"]
        assert rec["group_summaries"]
        assert all("akran siralamas" in g["reason"] for g in rec["group_summaries"])
        for f in rec["top_positive"] + rec["top_negative"]:
            assert _FACTOR_KEYS <= set(f.keys())
            assert f["importance"] >= 0


def test_direction_matches_sign():
    model = _Single(_train_booster())
    out = compute_peer_xai(model, _X(6), _FEATURES, [f"S{i}" for i in range(6)])
    for rec in out.values():
        for f in rec["top_positive"]:
            assert f["contribution"] > 0 and f["direction"] == "yukarı"
        for f in rec["top_negative"]:
            assert f["contribution"] < 0 and f["direction"] == "aşağı"


def test_ensemble_has_caveat():
    model = _Ensemble(_train_booster())
    out = compute_peer_xai(model, _X(4), _FEATURES, ["A", "B", "C", "D"])
    assert all(r["caveat"] for r in out.values())  # LGB-leg uyarisi dolu
    assert {r["method"] for r in out.values()} == {"ensemble_permutation"}
    assert all(r["diagnostic_method"] == "lgb_leg_shap_available" for r in out.values())


def test_no_booster_is_noop():
    class _Empty:
        pass
    assert compute_peer_xai(_Empty(), _X(3), _FEATURES, ["A", "B", "C"]) == {}


def test_human_label_covers_pooled_features():
    """Cross-sectional/meta feature adlari 'other'/jenerik metne dusmemeli."""
    model = _Single(_train_booster())
    out = compute_peer_xai(model, _X(3), _FEATURES, ["A", "B", "C"])
    labels = {f["feature_name"]: f["human_label"]
              for rec in out.values()
              for f in rec["top_positive"] + rec["top_negative"]}
    for name, label in labels.items():
        assert "okunabilir etiketi olmayan" not in label, name


def test_validation_symbol_length_mismatch():
    model = _Single(_train_booster())
    with pytest.raises(ValueError):
        compute_peer_xai(model, _X(6), _FEATURES, ["A", "B"])  # 6 satir, 2 sembol


def test_validation_feature_count_mismatch():
    model = _Single(_train_booster())
    with pytest.raises(ValueError):
        compute_peer_xai(model, _X(3), _FEATURES[:3], ["A", "B", "C"])
