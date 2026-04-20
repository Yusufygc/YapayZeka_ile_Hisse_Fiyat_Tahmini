# -*- coding: utf-8 -*-
"""
naive_model.py — Naive baseline modeller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Forecasting çekirdeği için zorunlu referans modeller sağlar.

Tüm baseline'lar mevcut pipeline ile uyumlu biçimde log-getiri hedefi üretir.
"""

from __future__ import annotations

import joblib
import numpy as np

from .base_model import BaseModel


class NaiveLastValueModel(BaseModel):
    """
    Son gözlenen hedef değerini tüm gelecek adımlar için tekrarlar.

    Log-getiri pipeline'ında bu, "yarın da bugünle aynı getiri olacak"
    varsayımıdır.
    """

    def __init__(self):
        self.last_value: float = 0.0

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        self.last_value = float(np.asarray(y_train).ravel()[-1])
        print("[OK] NaiveLastValue baseline hazırlandı.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        return np.full(len(X_test), self.last_value, dtype=float)

    def save(self, path: str) -> None:
        joblib.dump({"last_value": self.last_value}, path)
        print(f"[OK] NaiveLastValue baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.last_value = float(payload["last_value"])
        print(f"[OK] NaiveLastValue baseline yüklendi <- {path}")


class NaiveZeroReturnModel(BaseModel):
    """
    Tüm gelecek adımlar için sıfır log-getiri tahmini yapar.

    Fiyat uzayında bu, "yarın fiyat değişmeyecek" baseline'ıdır.
    """

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        print("[OK] NaiveZeroReturn baseline hazırlandı.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        return np.zeros(len(X_test), dtype=float)

    def save(self, path: str) -> None:
        joblib.dump({"type": "zero_return"}, path)
        print(f"[OK] NaiveZeroReturn baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        joblib.load(path)
        print(f"[OK] NaiveZeroReturn baseline yüklendi <- {path}")


class NaiveDriftModel(BaseModel):
    """
    Eğitim dönemindeki ortalama log-getiriyi geleceğe taşır.

    Bu, zayıf ama faydalı bir trend/drift baseline'ıdır.
    """

    def __init__(self):
        self.mean_return: float = 0.0

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        self.mean_return = float(np.asarray(y_train).ravel().mean())
        print("[OK] NaiveDrift baseline hazırlandı.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        return np.full(len(X_test), self.mean_return, dtype=float)

    def save(self, path: str) -> None:
        joblib.dump({"mean_return": self.mean_return}, path)
        print(f"[OK] NaiveDrift baseline kaydedildi -> {path}")

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.mean_return = float(payload["mean_return"])
        print(f"[OK] NaiveDrift baseline yüklendi <- {path}")
