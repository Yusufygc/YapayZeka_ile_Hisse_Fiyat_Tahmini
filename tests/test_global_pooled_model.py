# -*- coding: utf-8 -*-
"""Global pooled (conditioned) model tests — Faz 3."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from src.models.global_pooled_model import (  # noqa: E402
    GlobalPooledConfig,
    GlobalPooledModel,
    build_pooled_features,
    make_global_model_factory,
)
from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward  # noqa: E402
from src.validation.pooled_oos import PerSymbolOOSConfig, evaluate_per_symbol  # noqa: E402


def _panel(n_dates: int = 220, symbols=("AAA", "BBB", "CCC"), h: int = 5,
           sector_signal: bool = False) -> pd.DataFrame:
    """Sentetik panel. sector_signal=True -> hedef sektore bagli (kosullandirma
    ogrenilebilirligi testi)."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    sec_map = {"AAA": "Energy", "BBB": "Energy", "CCC": "Financial Services"}
    sign_map = {"Energy": 1.0, "Financial Services": -1.0}
    rows = []
    for sid, s in enumerate(symbols):
        sd = dates
        close = 10.0 + np.cumsum(rng.normal(0, 0.2, len(sd)))
        close = np.clip(close, 1.0, None)
        for i, d in enumerate(sd):
            if i + h >= len(sd):
                continue
            tgt = float(np.log(close[i + h] / close[i]))
            sec = sec_map[s]
            if sector_signal:
                feat = sign_map[sec] * abs(tgt) + rng.normal(0, 0.005)
                tgt = sign_map[sec] * abs(tgt)
            else:
                feat = tgt + rng.normal(0, 0.01)
            rows.append({"symbol": s, "Date": d, "feat": feat, "target": tgt,
                         "target_date": sd[i + h], "sector": sec,
                         "symbol_id": sid, "liq_log": 5.0 + sid,
                         "vol": 0.02 + 0.001 * sid})
    return pd.DataFrame(rows).reset_index(drop=True)


def test_build_pooled_features_schema_and_cat_indices():
    panel = _panel()
    aug, feats, cat_idx = build_pooled_features(panel)
    assert "sector_code" in aug.columns
    # kategorikler sonda; cat_indices onlarin pozisyonu
    assert feats[-2:] == ["symbol_id", "sector_code"]
    assert [feats[i] for i in cat_idx] == ["symbol_id", "sector_code"]
    # numerik kosullandirma da iceride
    assert "liq_log" in feats and "vol" in feats and "feat" in feats


def test_sector_code_stable_sorted():
    panel = _panel()
    aug, _, _ = build_pooled_features(panel)
    pairs = aug.drop_duplicates("sector").set_index("sector")["sector_code"].to_dict()
    assert pairs == {s: i for i, s in enumerate(sorted(pairs))}


def test_fit_predict_shape_and_deterministic():
    panel = _panel()
    aug, feats, cat_idx = build_pooled_features(panel)
    X = aug[feats].to_numpy(dtype=float)
    y = aug["target"].to_numpy(dtype=float)
    cfg = GlobalPooledConfig(num_boost_round=60, cat_indices=tuple(cat_idx))
    p1 = GlobalPooledModel(cfg).fit(X, y).predict(X)
    p2 = GlobalPooledModel(cfg).fit(X, y).predict(X)
    assert p1.shape == (len(X),)
    assert np.allclose(p1, p2)  # deterministic=True + tek thread


def test_factory_runs_in_oos_harness():
    panel = _panel()
    aug, feats, cat_idx = build_pooled_features(panel)
    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=5, embargo_buffer=2, window_len=12,
        n_windows=3, min_train_days=20,
    )).split(aug)
    factory = make_global_model_factory(
        cat_idx, GlobalPooledConfig(num_boost_round=40))
    res = evaluate_per_symbol(
        aug, folds, factory, PerSymbolOOSConfig(feature_cols=feats))
    assert res.n_folds_used >= 1
    assert set(res.per_symbol["symbol"]) <= {"AAA", "BBB", "CCC"}
    assert "dir_acc" in res.per_symbol.columns


def test_conditioning_lets_model_learn_sector_signal():
    """sektore bagli isaretli hedef -> model kosullandirmayi kullanip ogrenir."""
    panel = _panel(sector_signal=True)
    aug, feats, cat_idx = build_pooled_features(panel)
    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=5, embargo_buffer=2, window_len=12,
        n_windows=3, min_train_days=20,
    )).split(aug)
    factory = make_global_model_factory(
        cat_idx, GlobalPooledConfig(num_boost_round=120, min_data_in_leaf=5))
    res = evaluate_per_symbol(
        aug, folds, factory, PerSymbolOOSConfig(feature_cols=feats))
    rel = res.per_symbol[res.per_symbol["reliable"]]
    assert len(rel) >= 1
    assert rel["dir_acc"].mean() >= 75.0
