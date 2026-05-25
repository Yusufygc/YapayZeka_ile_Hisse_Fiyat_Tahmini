# -*- coding: utf-8 -*-
"""
quantile_lightgbm_model.py - Quantile regression with LightGBM.

Sprint 4 (2026-05-25) Plan A4.1:
  Tek-noktasal tahmin yerine p10/p50/p90 dagitim. Advisory confidence
  band (forecast.points icinde p10_close, p50_close, p90_close)
  uretmek icin zorunlu.

Her quantile icin ayri LGBMRegressor egitilir (objective="quantile",
alpha=q). Inference'ta predict_quantiles() N x len(quantiles) dizi
doner. .predict() (BaseModel sozlesmesi) median (p50) doner.
"""

from __future__ import annotations

from typing import Sequence

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


_DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


class QuantileLightGBMModel(BaseModel):
    """LightGBM quantile regressor returning p10/p50/p90 (default)."""

    def __init__(
        self,
        quantiles: Sequence[float] = _DEFAULT_QUANTILES,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        qs = tuple(sorted(float(q) for q in quantiles))
        if not qs:
            raise ValueError("quantiles bos olamaz")
        if any(q <= 0.0 or q >= 1.0 for q in qs):
            raise ValueError(f"quantiles 0<q<1 olmali, alindi: {qs}")
        self.quantiles: tuple[float, ...] = qs
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbosity": -1,
        }
        # quantile -> trained model
        self.models: dict[float, object] = {}

    def _build_one(self, alpha: float):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "Quantile LightGBM secildi ama lightgbm kurulu degil. "
                "`pip install lightgbm` gerekir."
            ) from exc
        return LGBMRegressor(objective="quantile", alpha=alpha, **self.params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        y = np.asarray(y_train).ravel()
        for q in self.quantiles:
            mdl = self._build_one(q)
            mdl.fit(X_train, y)
            self.models[q] = mdl
        print(f"[OK] Quantile LightGBM egitildi (quantiles={list(self.quantiles)}).")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """BaseModel sozlesmesi: median (p50) doner. p50 yoksa en yakin q."""
        if not self.models:
            raise RuntimeError("Quantile LightGBM henuz egitilmedi.")
        median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
        return np.asarray(self.models[median_q].predict(X_test), dtype=float)

    def predict_quantiles(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """N x len(quantiles) matris. Siralanmis: sutun_0 = en kucuk quantile."""
        if not self.models:
            raise RuntimeError("Quantile LightGBM henuz egitilmedi.")
        cols = [self.models[q].predict(X_test) for q in self.quantiles]
        out = np.column_stack(cols).astype(float)
        # Quantile crossing fix: row-wise sort. Saglikli olcum icin sutunlar
        # zaten artan ama LightGBM quantile crossing yapabilir.
        out.sort(axis=1)
        return out

    def save(self, path: str) -> None:
        if not self.models:
            raise RuntimeError("Kaydedilecek Quantile LightGBM yok.")
        joblib.dump(
            {"params": self.params, "quantiles": self.quantiles, "models": self.models},
            path,
        )
        print(f"[OK] Quantile LightGBM kaydedildi -> {path}")

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.params = payload["params"]
        self.quantiles = tuple(payload["quantiles"])
        self.models = payload["models"]
        print(f"[OK] Quantile LightGBM yuklendi <- {path}")


# --- Registry tescili (Sprint 4) ----------------------------------------
from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="LightGBM Quantile",
    factory=lambda **kw: QuantileLightGBMModel(**kw),
    category="tree",
    role="candidate",
    ensemble_eligible=False,  # quantile output icin scalar ensemble uygunsuz
    requires=("lightgbm",),
    target_modes=("return", "log_return"),
    description="LightGBM quantile regressor; p10/p50/p90 dagitim, advisory confidence band uretir.",
))
