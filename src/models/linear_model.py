# -*- coding: utf-8 -*-
"""
linear_model.py - Strong low-variance return regressors.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from .base_model import BaseModel


class RidgeReturnModel(BaseModel):
    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        self.model.fit(X_train, np.asarray(y_train).ravel())
        print("[OK] Ridge return baseline egitildi.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(X_test)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path, compress=3)
        print(f"[OK] Ridge return baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        print(f"[OK] Ridge return baseline yuklendi <- {path}")


# --- Registry tescili (Faz 1) -------------------------------------------
from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="Ridge Return",
    factory=lambda **kw: RidgeReturnModel(**kw),
    category="linear_shrinkage",
    role="candidate",
    ensemble_eligible=True,
    default_candidate=True,
    target_modes=("return", "log_return"),
    description="L2 shrinkage; düşük sinyal kalitesi fakat Holdout'ta sermaye koruma rolü.",
))
