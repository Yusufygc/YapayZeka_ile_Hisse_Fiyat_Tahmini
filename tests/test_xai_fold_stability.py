# -*- coding: utf-8 -*-
"""XAI fold-stabilite skoru testleri (Adim 2.4)."""
import pytest

from src.xai.strategies import compute_feature_stability_scores


class TestComputeFeatureStabilityScores:
    def test_feature_always_top_ratio_is_one(self):
        folds = [
            {"A": 1.0, "B": 0.5, "C": 0.3},
            {"A": 0.9, "B": 0.4, "C": 0.2},
            {"A": 0.8, "B": 0.3, "C": 0.1},
        ]
        scores = compute_feature_stability_scores(folds, top_k=1)
        assert scores["A"] == 1.0

    def test_feature_never_top_has_no_entry(self):
        folds = [
            {"A": 1.0, "B": 0.5},
            {"A": 0.9, "B": 0.4},
        ]
        scores = compute_feature_stability_scores(folds, top_k=1)
        assert "B" not in scores

    def test_ratio_is_fraction_of_folds(self):
        folds = [
            {"A": 1.0, "B": 0.1},
            {"B": 1.0, "A": 0.1},
            {"A": 1.0, "B": 0.1},
        ]
        scores = compute_feature_stability_scores(folds, top_k=1)
        assert abs(scores["A"] - 2 / 3) < 1e-9
        assert abs(scores["B"] - 1 / 3) < 1e-9

    def test_empty_folds_returns_empty(self):
        scores = compute_feature_stability_scores([])
        assert scores == {}

    def test_top_k_larger_than_features(self):
        folds = [{"A": 1.0}, {"A": 0.9}]
        scores = compute_feature_stability_scores(folds, top_k=10)
        assert scores["A"] == 1.0

    def test_uses_abs_importance_for_ranking(self):
        folds = [
            {"A": -5.0, "B": 3.0, "C": 1.0},
        ]
        scores = compute_feature_stability_scores(folds, top_k=1)
        assert "A" in scores
        assert "B" not in scores

    def test_scores_in_range_0_to_1(self):
        folds = [
            {"A": 1.0, "B": 0.5, "C": 0.3, "D": 0.1},
            {"B": 1.0, "C": 0.6, "D": 0.4, "E": 0.2},
            {"C": 1.0, "D": 0.7, "E": 0.3, "F": 0.1},
        ]
        scores = compute_feature_stability_scores(folds, top_k=2)
        for feat, ratio in scores.items():
            assert 0.0 <= ratio <= 1.0, f"{feat}: {ratio}"
