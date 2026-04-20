# -*- coding: utf-8 -*-
"""
arima_model.py — ARIMA baseline modeli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Log-getiri hedefi üzerinde klasik zaman serisi baseline'ı sağlar.
"""

from __future__ import annotations

import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .base_model import BaseModel


class ARIMAModel(BaseModel):
    """
    Varsayılan ARIMA(1,0,0) baseline.

    Log-getiri hedefi stationer kabul edildiği için d=0 kullanılır.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0)):
        self.order = order
        self.model_fit = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        y = np.asarray(y_train).ravel()
        self.model_fit = ARIMA(y, order=self.order).fit()
        print(f"[OK] ARIMA baseline eğitildi (order={self.order}).")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if self.model_fit is None:
            raise RuntimeError("ARIMA modeli henüz eğitilmedi.")
        return np.asarray(self.model_fit.forecast(steps=len(X_test)), dtype=float)

    def save(self, path: str) -> None:
        joblib.dump({"order": self.order, "model_fit": self.model_fit}, path)
        print(f"[OK] ARIMA baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.order = tuple(payload["order"])
        self.model_fit = payload["model_fit"]
        print(f"[OK] ARIMA baseline yüklendi <- {path}")
