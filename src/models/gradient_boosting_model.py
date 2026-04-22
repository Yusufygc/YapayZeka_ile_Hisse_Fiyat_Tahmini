# -*- coding: utf-8 -*-
"""
gradient_boosting_model.py - Optional modern boosting return baseline.

LightGBM is used when installed. The class keeps the dependency optional so the
core pipeline can still run in minimal environments.
"""

from __future__ import annotations

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover - minimal test runtimes
    import pickle

    class _PickleJoblib:
        @staticmethod
        def dump(obj, path):
            with open(path, "wb") as handle:
                pickle.dump(obj, handle)

        @staticmethod
        def load(path):
            with open(path, "rb") as handle:
                return pickle.load(handle)

    joblib = _PickleJoblib()

from .base_model import BaseModel


class LightGBMReturnModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": "regression",
            "verbosity": -1,
        }
        self.model = None

    def _build_model(self):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "LightGBM Return secildi ama lightgbm kurulu degil. "
                "`pip install lightgbm` veya requirements.txt kurulumu gerekir."
            ) from exc
        return LGBMRegressor(**self.params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        self.model = self._build_model()
        self.model.fit(X_train, np.asarray(y_train).ravel())
        print("[OK] LightGBM return baseline egitildi.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM modeli henuz egitilmedi.")
        return np.asarray(self.model.predict(X_test), dtype=float)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Kaydedilecek LightGBM modeli yok.")
        joblib.dump({"params": self.params, "model": self.model}, path)
        print(f"[OK] LightGBM return baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.params = payload["params"]
        self.model = payload["model"]
        print(f"[OK] LightGBM return baseline yuklendi <- {path}")
