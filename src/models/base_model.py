# -*- coding: utf-8 -*-
"""
base_model.py — Soyut Temel Model Sınıfı
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tüm tahmin modellerinin (Prophet, XGBoost, LSTM) uygulaması gereken
ortak arayüzü tanımlar: train, predict, save, load.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Her tahmin modelinin uyması gereken soyut sözleşme (contract)."""

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Modeli eğitim verisi üzerinde eğitir.

        Parameters
        ----------
        X_train : np.ndarray  Özellik matrisi.
        y_train : np.ndarray  Hedef vektörü.
        """
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Test verisi üzerinde tahmin üretir.

        Parameters
        ----------
        X_test : np.ndarray  Test özellik matrisi.

        Returns
        -------
        np.ndarray  Tahmin dizisi.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Eğitilmiş modeli diske kaydeder."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Kaydedilmiş modeli diskten yükler."""
        ...
