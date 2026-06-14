# -*- coding: utf-8 -*-
"""EnsemblePooledModel testleri (E2 Faz 9)."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightgbm")

from src.models.ensemble_pooled_model import EnsemblePooledConfig, EnsemblePooledModel
from src.models.global_pooled_model import GlobalPooledConfig
from src.models.torch_mlp_model import TorchMLPConfig


def _toy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(n, 3))
    cat = rng.integers(0, 5, size=(n, 1)).astype(float)
    X = np.hstack([num, cat])
    y = 0.7 * num[:, 0] - 0.4 * num[:, 1] + 0.2 * (cat[:, 0] - 2.0)
    return X, y


def _cfg(**kw):
    base = dict(
        mlp_seeds=(42, 7),
        lgb=GlobalPooledConfig(num_boost_round=30),
        mlp=TorchMLPConfig(hidden=(16, 8), epochs=6, batch_size=128),
        cat_indices=(3,), cat_cardinalities=(5,),
    )
    base.update(kw)
    return EnsemblePooledConfig(**base)


def test_fit_predict_blend_range():
    X, y = _toy()
    m = EnsemblePooledModel(_cfg()).fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),)
    # blend = pct-rank agirlikli ortalama -> [0,1] araliginda
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        EnsemblePooledModel(_cfg()).predict(np.zeros((3, 4)))


def test_predict_rejects_non_2d_input_after_fit():
    X, y = _toy()
    m = EnsemblePooledModel(_cfg()).fit(X, y)
    with pytest.raises(ValueError, match="2D X"):
        m.predict(np.zeros(4))


def test_predict_rejects_single_row_cross_section_after_fit():
    X, y = _toy()
    m = EnsemblePooledModel(_cfg()).fit(X, y)
    with pytest.raises(ValueError, match="en az 2 satir"):
        m.predict(X[:1])


def test_invalid_blend_weight_raises():
    with pytest.raises(ValueError):
        EnsemblePooledModel(_cfg(blend_weight_lgb=1.5))


def test_empty_seeds_raises():
    with pytest.raises(ValueError):
        EnsemblePooledModel(_cfg(mlp_seeds=()))


def test_weight_extremes_match_components_rank():
    """w=1 -> sadece LGB rank; w=0 -> sadece MLP rank. Siralama tutmali."""
    X, y = _toy()
    import pandas as pd

    m_lgb = EnsemblePooledModel(_cfg(blend_weight_lgb=1.0)).fit(X, y)
    p = m_lgb.predict(X)
    # w=1 -> ciktilar lgb pct-rank'iyla birebir
    lgb_rank = pd.Series(m_lgb.lgb.predict(X)).rank(method="average", pct=True).to_numpy()
    assert np.allclose(p, lgb_rank)


def test_fit_trains_all_legs():
    X, y = _toy()
    m = EnsemblePooledModel(_cfg(mlp_seeds=(42, 7, 123))).fit(X, y)
    assert m.lgb is not None
    assert len(m.mlps) == 3
