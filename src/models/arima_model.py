# -*- coding: utf-8 -*-
"""
arima_model.py - ARIMA return baseline.

The default remains fast ARIMA(1,0,0). Optional auto_order searches a small
AIC-ranked candidate set on the training target only.
"""

from __future__ import annotations

import numpy as np

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # pragma: no cover - optional dependency guard
    ARIMA = None

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


class ARIMAModel(BaseModel):
    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        auto_order: bool = False,
        candidate_orders: list[tuple[int, int, int]] | None = None,
    ):
        self.order = order
        self.auto_order = auto_order
        self.candidate_orders = candidate_orders or [
            (1, 0, 0),
            (0, 0, 1),
            (1, 0, 1),
            (2, 0, 0),
            (0, 0, 2),
        ]
        self.model_fit = None

    def _fit_order(self, y: np.ndarray, order: tuple[int, int, int]):
        if ARIMA is None:
            raise ImportError(
                "ARIMA baseline secildi ama statsmodels kurulu degil. "
                "`pip install statsmodels` veya requirements.txt kurulumu gerekir."
            )
        return ARIMA(y, order=order).fit()

    def _select_order(self, y: np.ndarray) -> tuple[int, int, int]:
        best_order = self.order
        best_aic = float("inf")
        for order in self.candidate_orders:
            try:
                fit = self._fit_order(y, order)
                aic = float(fit.aic)
                if aic < best_aic:
                    best_order = order
                    best_aic = aic
            except Exception:
                continue
        return best_order

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        y = np.asarray(y_train).ravel()
        if self.auto_order:
            self.order = self._select_order(y)
        self.model_fit = self._fit_order(y, self.order)
        print(f"[OK] ARIMA baseline egitildi (order={self.order}, auto_order={self.auto_order}).")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if self.model_fit is None:
            raise RuntimeError("ARIMA modeli henuz egitilmedi.")
        return np.asarray(self.model_fit.forecast(steps=len(X_test)), dtype=float)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "order": self.order,
                "auto_order": self.auto_order,
                "candidate_orders": self.candidate_orders,
                "model_fit": self.model_fit,
            },
            path,
        )
        print(f"[OK] ARIMA baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.order = tuple(payload["order"])
        self.auto_order = bool(payload.get("auto_order", False))
        self.candidate_orders = [tuple(order) for order in payload.get("candidate_orders", [(1, 0, 0)])]
        self.model_fit = payload["model_fit"]
        print(f"[OK] ARIMA baseline yuklendi <- {path}")


# --- Registry tescili (Faz 1 + Faz 2 config-aware) ----------------------
from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402


def _build_arima(arima: dict | None = None, **_unused) -> ARIMAModel:
    """ARIMA için config-aware factory; `arima` alt-sözlüğü unpack edilir.

    Eski `model_factory.make_arima` davranışını birebir korur.
    """
    cfg = arima or {}
    candidate_orders = [tuple(order) for order in cfg.get("candidate_orders", [])] or None
    return ARIMAModel(
        order=tuple(cfg.get("order", (1, 0, 0))),
        auto_order=bool(cfg.get("auto_order", False)),
        candidate_orders=candidate_orders,
    )


register_model(ModelSpec(
    name="ARIMA",
    factory=_build_arima,
    category="stat",
    role="candidate",
    ensemble_eligible=False,
    requires=("statsmodels",),
    needs_config_keys=("arima",),
    description="Klasik ARIMA; candidate listesinde ancak default değil.",
))
