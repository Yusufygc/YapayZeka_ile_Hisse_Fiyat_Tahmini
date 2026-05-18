# -*- coding: utf-8 -*-
"""Rolling-holdout değerlendirme testleri (Adim 2.1)."""
import numpy as np
import pytest

from src.validation.rolling_holdout import rolling_holdout_evaluate


def _always_positive_model(X: np.ndarray) -> np.ndarray:
    return np.ones(len(X))


def _always_negative_model(X: np.ndarray) -> np.ndarray:
    return -np.ones(len(X))


def _perfect_model(X: np.ndarray, y: np.ndarray):
    return lambda x: y[: len(x)]


class TestRollingHoldoutEvaluate:
    def _make_data(self, n: int = 300):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (n, 5))
        y = rng.normal(0, 0.01, n)
        return X, y

    def test_returns_dict_with_required_keys(self):
        X, y = self._make_data()
        result = rolling_holdout_evaluate(lambda x: np.ones(len(x)), X, y)
        for key in ["median_net_return", "positive_window_ratio", "iqr_net_return",
                    "window_returns", "n_windows"]:
            assert key in result

    def test_n_windows_matches_step(self):
        X, y = self._make_data(300)
        result = rolling_holdout_evaluate(lambda x: np.ones(len(x)), X, y,
                                           window_size=60, step_size=20)
        assert result["n_windows"] > 0
        expected = len(range(0, 300 - 60 + 1, 20))
        assert result["n_windows"] == expected

    def test_positive_model_ratio_is_one(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (300, 3))
        y = np.full(300, 0.005)
        result = rolling_holdout_evaluate(_always_positive_model, X, y)
        assert result["positive_window_ratio"] == 1.0

    def test_negative_model_ratio_is_zero(self):
        X, _ = self._make_data(300)
        y = np.full(300, 0.005)
        result = rolling_holdout_evaluate(_always_negative_model, X, y)
        assert result["positive_window_ratio"] == 0.0

    def test_insufficient_data_returns_empty(self):
        X = np.ones((10, 3))
        y = np.ones(10)
        result = rolling_holdout_evaluate(lambda x: np.ones(len(x)), X, y,
                                           window_size=60)
        assert result["n_windows"] == 0
        assert result["median_net_return"] is None

    def test_iqr_non_negative(self):
        X, y = self._make_data(300)
        result = rolling_holdout_evaluate(lambda x: np.ones(len(x)), X, y)
        assert result["iqr_net_return"] >= 0.0

    def test_median_is_float(self):
        X, y = self._make_data(300)
        result = rolling_holdout_evaluate(lambda x: np.ones(len(x)), X, y)
        assert isinstance(result["median_net_return"], float)

    def test_window_returns_list_length_matches_n_windows(self):
        X, y = self._make_data(300)
        result = rolling_holdout_evaluate(lambda x: np.ones(len(x)), X, y)
        assert len(result["window_returns"]) == result["n_windows"]
