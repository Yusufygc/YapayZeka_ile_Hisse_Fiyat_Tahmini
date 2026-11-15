# -*- coding: utf-8 -*-
"""Kol-B serving XAI stratejisi testleri (SHAP tree / permutation fallback)."""

import sys
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.xai.background import XAIBackgroundProvider
from src.xai.strategies import TabularContributionStrategy


def test_tree_strategy_uses_shap_when_available():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.1
    model = RandomForestRegressor(n_estimators=3, random_state=1).fit(X, y)
    strategy = TabularContributionStrategy(["f0", "f1"])

    contribs, method, approximate = strategy.tree_contributions(model, X[:4])

    assert method == "shap_tree"
    assert approximate is False
    assert contribs.shape == (4, 2)


def test_tree_strategy_falls_back_to_permutation_when_shap_fails(monkeypatch):
    class BrokenTreeExplainer:
        def __init__(self, estimator):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(TreeExplainer=BrokenTreeExplainer))
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X[:, 0] * 0.5
    model = RandomForestRegressor(n_estimators=3, random_state=1).fit(X, y)
    strategy = TabularContributionStrategy(["f0", "f1"])

    contribs, method, approximate = strategy.tree_contributions(model, X[:4])

    assert method == "permutation_fallback"
    assert approximate is True
    assert contribs.shape == (4, 2)


def test_background_provider_uses_train_median_for_permutation_mask():
    class _Model:
        def predict(self, X):
            return np.asarray(X)[:, 0]

    X_train = np.array([[10.0, 0.0], [14.0, 0.0], [18.0, 0.0]])
    X_test = np.array([[100.0, 1.0]])
    background = XAIBackgroundProvider.from_arrays(X_train=X_train)
    strategy = TabularContributionStrategy(["f0", "f1"], background_provider=background)

    contribs = strategy.permutation_contributions(_Model(), X_test)

    assert contribs[0, 0] == 86.0  # 100 - train median 14
