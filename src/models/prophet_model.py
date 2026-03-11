# -*- coding: utf-8 -*-
"""
prophet_model.py — Facebook Prophet Sarmalayıcı
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prophet, tarih (ds) ve hedef (y) çiftleriyle çalışır.
Bu sınıf, BaseModel arayüzüne uygun bir Prophet sarmalayıcısıdır.
Modelin kaydedilmesi/yüklenmesi joblib ile yapılır.
"""

import numpy as np
import pandas as pd
import joblib
from prophet import Prophet

from .base_model import BaseModel


class ProphetModel(BaseModel):
    """Facebook Prophet sarmalayıcısı."""

    def __init__(self, **prophet_kwargs):
        """
        Parameters
        ----------
        **prophet_kwargs
            Prophet modeline iletilecek ek parametreler
            (örn. yearly_seasonality, weekly_seasonality).
        """
        self.prophet_kwargs = prophet_kwargs
        self.model: Prophet | None = None
        # Eğitim sırasında tarih bilgisi saklanır
        self._train_dates: pd.Series | None = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        dates_train: pd.Series | None = None,
        **kwargs,
    ) -> None:
        """
        Prophet modelini eğitir.

        Parameters
        ----------
        X_train : np.ndarray  (kullanılmaz — Prophet sadece ds+y ister)
        y_train : np.ndarray  Hedef değerler (orijinal ölçekte).
        dates_train : pd.Series  Eğitim tarih dizisi.
        """
        if dates_train is None:
            raise ValueError("Prophet modeli için 'dates_train' parametresi gereklidir.")

        self._train_dates = dates_train.reset_index(drop=True)
        train_df = pd.DataFrame({
            "ds": self._train_dates,
            "y": y_train.ravel(),
        })

        self.model = Prophet(**self.prophet_kwargs)
        self.model.fit(train_df)
        print("[✓] Prophet modeli eğitildi.")

    def predict(self, X_test: np.ndarray, dates_test: pd.Series | None = None) -> np.ndarray:
        """
        Test tarihleri üzerinde tahmin üretir.

        Parameters
        ----------
        X_test : np.ndarray   (kullanılmaz)
        dates_test : pd.Series  Test tarih dizisi.

        Returns
        -------
        np.ndarray  Tahmin dizisi.
        """
        if self.model is None:
            raise RuntimeError("Model henüz eğitilmedi. Önce train() çağrılmalıdır.")
        if dates_test is None:
            raise ValueError("Prophet tahmini için 'dates_test' parametresi gereklidir.")

        future_df = pd.DataFrame({"ds": dates_test.reset_index(drop=True)})
        forecast = self.model.predict(future_df)
        return forecast["yhat"].values

    def save(self, path: str) -> None:
        """Modeli .pkl olarak kaydeder."""
        joblib.dump(self.model, path)
        print(f"[✓] Prophet modeli kaydedildi → {path}")

    def load(self, path: str) -> None:
        """Kaydedilmiş modeli diskten yükler."""
        self.model = joblib.load(path)
        print(f"[✓] Prophet modeli yüklendi ← {path}")
