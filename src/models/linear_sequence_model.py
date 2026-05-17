# -*- coding: utf-8 -*-
"""
linear_sequence_model.py — DLinear / NLinear hafif sequence baseline'ları.

Kasıtlı olarak düşük parametreli sanity-check baseline'larıdır. LSTM ile
aynı 3D sequence tensörlerini tüketir; hızlı lineer regresörler çalıştırır.

Referans: Zeng et al. (2022) "Are Transformers Effective for Time Series
Forecasting?" — DLinear/NLinear'ın Transformer modellerini aşabildiğini gösterir.

Not: PatchTSTExperimentalModel Faz 6 Optimizasyon kapsamında kaldırıldı.
Gerçek PatchTST bir Transformer mimarisidir (Nie et al., ICLR 2023); Ridge
regresyon tabanlı bir taklit değildir. İsim yanıltıcıydı.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.linear_model import Ridge
except ImportError:  # pragma: no cover - minimal test runtimes
    class Ridge:
        def __init__(self, alpha: float = 1.0):
            self.alpha = alpha
            self.coef_ = None
            self.intercept_ = 0.0

        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float).ravel()
            X_aug = np.column_stack([np.ones(len(X)), X])
            penalty = np.eye(X_aug.shape[1]) * float(self.alpha)
            penalty[0, 0] = 0.0
            params = np.linalg.pinv(X_aug.T @ X_aug + penalty) @ X_aug.T @ y
            self.intercept_ = float(params[0])
            self.coef_ = params[1:]
            return self

        def predict(self, X):
            if self.coef_ is None:
                raise RuntimeError("Ridge fallback henuz egitilmedi.")
            X = np.asarray(X, dtype=float)
            return X @ self.coef_ + self.intercept_

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


class _BaseLinearSequenceModel(BaseModel):
    model_name = "LinearSequence"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.input_shape = None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"{self.model_name} 3D sequence tensor bekler, alinan: {X.ndim}D")
        return X.reshape(X.shape[0], -1)

    def _extra_state(self) -> dict:
        return {}

    def _load_extra_state(self, state: dict) -> None:
        _ = state
        return None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        self.input_shape = tuple(X_train.shape[1:])
        X_flat = self._transform(X_train)
        self.model.fit(X_flat, np.asarray(y_train).ravel())
        print(f"[OK] {self.model_name} sequence baseline egitildi.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        return np.asarray(self.model.predict(self._transform(X_test)), dtype=float)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "alpha": self.alpha,
                "model": self.model,
                "input_shape": self.input_shape,
                "model_name": self.model_name,
                "extra_state": self._extra_state(),
            },
            path,
        )
        print(f"[OK] {self.model_name} sequence baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.alpha = payload["alpha"]
        self.model = payload["model"]
        self.input_shape = payload.get("input_shape")
        self._load_extra_state(payload.get("extra_state", {}))
        print(f"[OK] {self.model_name} sequence baseline yuklendi <- {path}")


class DLinearSequenceModel(_BaseLinearSequenceModel):
    model_name = "DLinear"

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"{self.model_name} 3D sequence tensor bekler, alinan: {X.ndim}D")
        trend = X.mean(axis=1, keepdims=True)
        seasonal = X - trend
        return np.concatenate(
            [
                seasonal.reshape(X.shape[0], -1),
                trend.reshape(X.shape[0], -1),
            ],
            axis=1,
        )


class NLinearSequenceModel(_BaseLinearSequenceModel):
    model_name = "NLinear"

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"{self.model_name} 3D sequence tensor bekler, alinan: {X.ndim}D")
        anchor = X[:, -1:, :]
        normalized = X - anchor
        return np.concatenate(
            [
                normalized.reshape(X.shape[0], -1),
                anchor.reshape(X.shape[0], -1),
            ],
            axis=1,
        )


# PatchTSTExperimentalModel kaldırıldı (Faz 6 Optimizasyon).


# --- Registry tescili (Faz 1) -------------------------------------------
from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="DLinear",
    factory=lambda **kw: DLinearSequenceModel(**kw),
    category="linear_decomp",
    role="candidate",
    ensemble_eligible=True,
    default_candidate=True,
    description="Trend + seasonal decomposition; lineer aile içinde tek pozitif WF.",
))

register_model(ModelSpec(
    name="NLinear",
    factory=lambda **kw: NLinearSequenceModel(**kw),
    category="linear_decomp",
    role="candidate",
    ensemble_eligible=False,
    default_candidate=True,
    description="Last-value normalization; DLinear varken bilgi içermeyen kuzen.",
))
