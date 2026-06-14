# -*- coding: utf-8 -*-
"""TorchMLPModel testleri (E2 Faz 9)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from src.models.torch_mlp_model import TorchMLPConfig, TorchMLPModel, make_mlp_factory


def _toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(n, 3))
    cat = rng.integers(0, 4, size=(n, 1)).astype(float)  # 1 kategorik kolon (kard 4)
    X = np.hstack([num, cat])
    # hedef: sayisal + kategoriye bagli sinyal
    y = 0.8 * num[:, 0] - 0.5 * num[:, 1] + 0.3 * (cat[:, 0] - 1.5)
    return X, y


def _cfg(**kw):
    base = dict(hidden=(16, 8), epochs=8, batch_size=64,
                cat_indices=(3,), cat_cardinalities=(4,))
    base.update(kw)
    return TorchMLPConfig(**base)


def test_fit_predict_shape():
    X, y = _toy_data()
    m = TorchMLPModel(_cfg()).fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),)
    assert np.isfinite(p).all()


def test_predict_uses_configured_batch_size_and_returns_single_array():
    X, y = _toy_data(n=37)
    m = TorchMLPModel(_cfg(epochs=3, batch_size=16, predict_batch_size=7)).fit(X, y)
    p = m.predict(X)
    assert p.shape == (37,)
    assert np.isfinite(p).all()


def test_training_skips_singleton_batch_without_crashing():
    X, y = _toy_data(n=65)
    m = TorchMLPModel(_cfg(epochs=3, batch_size=32)).fit(X, y)
    p = m.predict(X[:5])
    assert p.shape == (5,)
    assert np.isfinite(p).all()


def test_fit_keeps_scaler_stats_float32():
    X, y = _toy_data()
    m = TorchMLPModel(_cfg(epochs=1)).fit(X, y)
    assert m.mu.dtype == np.float32
    assert m.sd.dtype == np.float32
    assert m._as_float32_matrix(X).dtype == np.float32


def test_deterministic_same_seed():
    X, y = _toy_data()
    p1 = TorchMLPModel(_cfg(seed=42)).fit(X, y).predict(X)
    p2 = TorchMLPModel(_cfg(seed=42)).fit(X, y).predict(X)
    assert np.allclose(p1, p2)


def test_learns_signal():
    """Egitilmis model hedefle pozitif korelasyon yakalamali (rastgele degil)."""
    X, y = _toy_data(n=800)
    p = TorchMLPModel(_cfg(epochs=25)).fit(X, y).predict(X)
    corr = np.corrcoef(p, y)[0, 1]
    assert corr > 0.5, f"dusuk korelasyon: {corr}"


def test_cat_length_mismatch_raises():
    with pytest.raises(ValueError):
        TorchMLPModel(TorchMLPConfig(cat_indices=(3,), cat_cardinalities=(4, 5)))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        TorchMLPModel(_cfg()).predict(np.zeros((2, 4)))


def test_bad_cat_index_raises():
    X, y = _toy_data()
    m = TorchMLPModel(TorchMLPConfig(cat_indices=(99,), cat_cardinalities=(4,),
                                     hidden=(8,), epochs=1))
    with pytest.raises(ValueError):
        m.fit(X, y)


def test_factory_produces_independent_models():
    X, y = _toy_data()
    f = make_mlp_factory((3,), (4,), _cfg())
    a, b = f(), f()
    assert a is not b
    a.fit(X, y)
    assert b.net is None  # b egitilmedi -> bagimsiz
