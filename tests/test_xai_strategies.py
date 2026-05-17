# -*- coding: utf-8 -*-
"""Strategy-boundary tests for XAI SHAP/LIME Phase 4 decomposition."""

import builtins
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.xai.explainer import XAIExplainer
from src.xai.strategies import SequenceContributionStrategy, TabularContributionStrategy


def test_tree_strategy_uses_shap_when_available():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.1
    model = RandomForestRegressor(n_estimators=3, random_state=1).fit(X, y)
    explainer = XAIExplainer("TEST", ["f0", "f1"], {})

    contribs, method = explainer._tree_contributions(model, X[:4])

    assert method == "shap_tree"
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


def test_linear_strategy_uses_shap_or_coefficients():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = X[:, 0] * 0.2 - X[:, 1] * 0.1
    model = Ridge().fit(X, y)
    explainer = XAIExplainer("TEST", ["f0", "f1", "f2"], {})

    contribs, method = explainer._linear_contributions(model, X[:5])

    assert method in {"shap_linear", "linear_coefficients"}
    assert contribs.shape == (5, 3)


def test_lime_tabular_contributions_when_available():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = X[:, 0] * 0.2 - X[:, 1] * 0.1
    model = Ridge().fit(X, y)
    explainer = XAIExplainer("TEST", ["f0", "f1", "f2"], {}, max_rows=3)

    contribs, method = explainer._lime_tabular_contributions(model, X)

    assert method == "lime_tabular"
    assert contribs.shape == (3, 3)


def test_lime_unavailable_does_not_break(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("lime"):
            raise ImportError("no lime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    strategy = SequenceContributionStrategy(["f0", "f1"], max_rows=2)

    contribs, method, approximate = strategy.lime_sequence_local_contributions(object(), np.ones((2, 3, 2)))

    assert method == "lime_unavailable"
    assert approximate is True
    assert contribs.shape == (0, 2)
