# -*- coding: utf-8 -*-
"""
linear_model.py - Strong low-variance return regressors.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

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
        joblib.dump(self.model, path)
        print(f"[OK] Ridge return baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        print(f"[OK] Ridge return baseline yuklendi <- {path}")


class ElasticNetReturnModel(BaseModel):
    def __init__(self, alpha: float = 0.001, l1_ratio: float = 0.15, random_state: int = 42):
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
            max_iter=10000,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        self.model.fit(X_train, np.asarray(y_train).ravel())
        print("[OK] ElasticNet return baseline egitildi.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(X_test)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
        print(f"[OK] ElasticNet return baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        print(f"[OK] ElasticNet return baseline yuklendi <- {path}")


# --- Registry tescili (Faz 1) -------------------------------------------
from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="Ridge Return",
    factory=lambda **kw: RidgeReturnModel(**kw),
    category="linear_shrinkage",
    role="candidate",
    ensemble_eligible=True,
    target_modes=("return", "log_return"),
    description="L2 shrinkage; düşük sinyal kalitesi fakat Holdout'ta sermaye koruma rolü.",
))

register_model(ModelSpec(
    name="ElasticNet Return",
    factory=lambda **kw: ElasticNetReturnModel(**kw),
    category="linear_shrinkage",
    role="candidate",
    ensemble_eligible=True,
    target_modes=("return", "log_return"),
    description="L1+L2 karmasi; l1_ratio ~ 0 ise Ridge'e dejenere olur.",
))
