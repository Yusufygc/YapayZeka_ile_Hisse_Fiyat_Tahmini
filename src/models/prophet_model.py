# -*- coding: utf-8 -*-
"""
prophet_model.py - Prophet wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - optional until save/load is used
    joblib = None

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - optional dependency in minimal runtimes
    Prophet = None

from .base_model import BaseModel

DEFAULT_PROPHET_REGRESSORS = [
    "USDTRY_Return",
    "BIST100_Return",
    "Rate_Level",
    "CPI_YoY",
    "Real_Rate",
    "Relative_Strength",
]


class ProphetModel(BaseModel):
    """BaseModel-compatible Prophet wrapper with optional macro regressors."""

    def __init__(
        self,
        use_regressors: bool = False,
        regressor_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        **prophet_kwargs,
    ):
        self.prophet_kwargs = prophet_kwargs
        self.use_regressors = bool(use_regressors)
        self.regressor_names = regressor_names or DEFAULT_PROPHET_REGRESSORS[:]
        self.feature_names = feature_names
        self.regressors_used: list[str] = []
        self.regressors_missing: list[str] = []
        self.model: object | None = None
        self._train_dates: pd.Series | None = None

    def _feature_frame(self, X: np.ndarray | pd.DataFrame | None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True).copy()
        if X is None:
            return pd.DataFrame()
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if self.feature_names and len(self.feature_names) == array.shape[1]:
            return pd.DataFrame(array, columns=self.feature_names)
        return pd.DataFrame(array)

    def _select_train_regressors(self, X: np.ndarray | pd.DataFrame | None, length: int) -> pd.DataFrame:
        feature_df = self._feature_frame(X).iloc[:length].reset_index(drop=True)
        if feature_df.empty:
            self.regressors_used = []
            self.regressors_missing = self.regressor_names[:]
            return pd.DataFrame(index=range(length))

        self.regressors_used = [name for name in self.regressor_names if name in feature_df.columns]
        self.regressors_missing = [name for name in self.regressor_names if name not in self.regressors_used]
        if not self.regressors_used:
            return pd.DataFrame(index=range(length))
        return feature_df[self.regressors_used].copy()

    def train(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray,
        dates_train: pd.Series | None = None,
        **kwargs,
    ) -> None:
        if Prophet is None:
            raise ImportError("prophet paketi kurulu degil; Prophet modeli atlanacak.")
        if dates_train is None:
            raise ValueError("Prophet modeli icin 'dates_train' parametresi gereklidir.")

        y_values = np.asarray(y_train).ravel()
        dates = pd.Series(dates_train).reset_index(drop=True)
        n = min(len(dates), len(y_values), len(X_train) if X_train is not None else len(y_values))
        self._train_dates = dates.iloc[:n].reset_index(drop=True)
        train_df = pd.DataFrame({
            "ds": self._train_dates,
            "y": y_values[:n],
        })

        if self.use_regressors:
            regressor_df = self._select_train_regressors(X_train, n)
            for col in self.regressors_used:
                train_df[col] = pd.to_numeric(regressor_df[col], errors="coerce").fillna(0.0).values

        self.model = Prophet(**self.prophet_kwargs)
        for col in self.regressors_used:
            self.model.add_regressor(col)
        self.model.fit(train_df)
        if self.use_regressors:
            print(f"[OK] Prophet regressors used: {self.regressors_used or 'none'}")
        print("[OK] Prophet modeli egitildi.")

    def predict(self, X_test: np.ndarray | pd.DataFrame, dates_test: pd.Series | None = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model henuz egitilmedi. Once train() cagrilmalidir.")
        if dates_test is None:
            raise ValueError("Prophet tahmini icin 'dates_test' parametresi gereklidir.")

        dates = pd.Series(dates_test).reset_index(drop=True)
        n = min(len(dates), len(X_test) if X_test is not None else len(dates))
        future_df = pd.DataFrame({"ds": dates.iloc[:n].reset_index(drop=True)})

        if self.regressors_used:
            feature_df = self._feature_frame(X_test).iloc[:n].reset_index(drop=True)
            for col in self.regressors_used:
                if col in feature_df.columns:
                    values = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0).values
                else:
                    values = np.zeros(n, dtype=float)
                future_df[col] = values

        forecast = self.model.predict(future_df)
        return forecast["yhat"].values

    def save(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib paketi kurulu degil; Prophet modeli kaydedilemez.")
        joblib.dump(self.model, path)
        print(f"[OK] Prophet modeli kaydedildi -> {path}")

    def load(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib paketi kurulu degil; Prophet modeli yuklenemez.")
        self.model = joblib.load(path)
        print(f"[OK] Prophet modeli yuklendi <- {path}")
