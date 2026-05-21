# -*- coding: utf-8 -*-
"""Prophet Hybrid Model wrapping Prophet and ML/DL Models.

This model allows combining the long-term trend modeling of Prophet
with the short-term return modeling capabilities of ML/DL models.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

from src.models.base_model import BaseModel
from src.pipeline.model_registry import ensure_loaded, get_spec, has_spec


class ProphetHybridModel(BaseModel):
    """Hybrid model that integrates Prophet trends with a base ML/DL model."""

    def __init__(
        self,
        base_model_name: str = "XGBoost",
        hybrid_mode: str = "trend_gate",
        target_type: str = "log_return",
        base_model_kwargs: dict | None = None,
        prophet_kwargs: dict | None = None,
    ):
        self.base_model_name = base_model_name
        self.hybrid_mode = hybrid_mode
        self.target_type = target_type
        self.base_model_kwargs = base_model_kwargs or {}
        self.prophet_kwargs = prophet_kwargs or {}
        self.prophet: Prophet | None = None
        self.base_model: BaseModel | None = None
        self._last_train_trend: float = 0.0

        # Dinamik olarak base modeli instantiate et
        ensure_loaded()
        if has_spec(self.base_model_name):
            spec = get_spec(self.base_model_name)
            self.base_model = spec.factory(**self.base_model_kwargs)
        else:
            raise ValueError(f"Base model '{self.base_model_name}' registry'de bulunamadi.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        dates_train: pd.Series | None = None,
        **kwargs,
    ) -> None:
        if Prophet is None:
            raise ImportError("prophet paketi kurulu degil; ProphetHybridModel egitilemez.")
        if dates_train is None:
            raise ValueError("ProphetHybridModel egitimi icin 'dates_train' parametresi gereklidir.")

        y_train_arr = np.asarray(y_train).ravel()
        dates = pd.Series(dates_train).reset_index(drop=True)
        n = min(len(dates), len(y_train_arr), len(X_train) if X_train is not None else len(y_train_arr))

        # 1. Getirileri Fiyat Serisine Donustur (Prophet trendi fit etsin)
        if self.target_type == "log_return":
            price_scaled = np.cumsum(y_train_arr[:n])
        else:
            price_scaled = np.cumprod(1.0 + y_train_arr[:n]) - 1.0

        df_prophet = pd.DataFrame({
            "ds": dates.iloc[:n],
            "y": price_scaled,
        })

        # 2. Prophet Modelini Eğit
        self.prophet = Prophet(**self.prophet_kwargs)
        self.prophet.fit(df_prophet)

        # Train seti trend tahminlerini çıkar
        forecast_train = self.prophet.predict(df_prophet)
        trend_train = forecast_train["trend"].values
        self._last_train_trend = float(trend_train[-1])

        # 3. Moda Gore Baz Modeli Eğit
        if self.hybrid_mode == "residual_decomp":
            # Bileşen Ayrıştırma: y_base = y - trend
            residuals_price = price_scaled - trend_train
            if self.target_type == "log_return":
                residuals_return = np.diff(residuals_price, prepend=0.0)
            else:
                residuals_return = np.zeros_like(residuals_price)
                residuals_return[0] = residuals_price[0]
                for i in range(1, len(residuals_price)):
                    prev = residuals_price[i-1] + 1.0
                    if abs(prev) > 1e-8:
                        residuals_return[i] = (residuals_price[i] + 1.0) / prev - 1.0

            self.base_model.train(X_train[:n], residuals_return, dates_train=dates_train[:n], **kwargs)
        else:
            # trend_gate modunda taban model normal getiriler üzerinde eğitilir
            self.base_model.train(X_train[:n], y_train_arr[:n], dates_train=dates_train[:n], **kwargs)

        print(f"[OK] ProphetHybridModel ({self.hybrid_mode}) egitildi.")

    def predict(
        self,
        X_test: np.ndarray,
        dates_test: pd.Series | None = None,
        **kwargs,
    ) -> np.ndarray:
        if self.prophet is None:
            raise RuntimeError("Model egitilmedi. Once train() cagrilmalidir.")
        if dates_test is None:
            raise ValueError("ProphetHybridModel tahmini icin 'dates_test' parametresi gereklidir.")

        dates = pd.Series(dates_test).reset_index(drop=True)
        n = min(len(dates), len(X_test))

        # 1. Prophet Trend Tahmini
        future_df = pd.DataFrame({"ds": dates.iloc[:n]})
        forecast_test = self.prophet.predict(future_df)
        trend_test = forecast_test["trend"].values

        # 2. Baz Model Tahmini (dates_test parametresi esnek olarak gecilir)
        try:
            yhat_base = self.base_model.predict(X_test[:n], dates_test=dates_test[:n], **kwargs)
        except TypeError:
            yhat_base = self.base_model.predict(X_test[:n], **kwargs)

        # 3. Tahmin Birleştirme (Moda Göre)
        if self.hybrid_mode == "trend_gate":
            # Eğim Pozitifse tahmini aynen ver, Negatifse 0 (Nakit)
            slopes = np.zeros(n)
            prev_trend = self._last_train_trend
            for i in range(n):
                slopes[i] = trend_test[i] - prev_trend
                prev_trend = trend_test[i]
            yhat = np.where(slopes > 0.0, yhat_base, 0.0)
            return yhat

        elif self.hybrid_mode == "residual_decomp":
            # Nihai Tahmin = Trend Getirisi + Artık Getirisi (Baz model tahmini)
            trend_returns = np.zeros(n)
            prev_trend = self._last_train_trend
            for i in range(n):
                if self.target_type == "log_return":
                    trend_returns[i] = trend_test[i] - prev_trend
                else:
                    prev_val = prev_trend + 1.0
                    if abs(prev_val) > 1e-8:
                        trend_returns[i] = (trend_test[i] + 1.0) / prev_val - 1.0
                prev_trend = trend_test[i]

            yhat = trend_returns + yhat_base
            return yhat

        return yhat_base

    def save(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib paketi kurulu degil; ProphetHybridModel kaydedilemez.")
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.prophet, os.path.join(path, "prophet.pkl"))

        if hasattr(self.base_model, "save"):
            self.base_model.save(os.path.join(path, "base_model"))

        meta = {
            "base_model_name": self.base_model_name,
            "hybrid_mode": self.hybrid_mode,
            "target_type": self.target_type,
            "_last_train_trend": self._last_train_trend,
        }
        joblib.dump(meta, os.path.join(path, "meta.pkl"))
        print(f"[OK] ProphetHybridModel kaydedildi -> {path}")

    def load(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib paketi kurulu degil; ProphetHybridModel yuklenemez.")
        meta = joblib.load(os.path.join(path, "meta.pkl"))
        self.base_model_name = meta["base_model_name"]
        self.hybrid_mode = meta["hybrid_mode"]
        self.target_type = meta["target_type"]
        self._last_train_trend = meta["_last_train_trend"]

        self.prophet = joblib.load(os.path.join(path, "prophet.pkl"))

        ensure_loaded()
        spec = get_spec(self.base_model_name)
        self.base_model = spec.factory(**self.base_model_kwargs)
        if hasattr(self.base_model, "load"):
            self.base_model.load(os.path.join(path, "base_model"))
        print(f"[OK] ProphetHybridModel yuklendi <- {path}")


# --- Registry kaydı ----------------------------------------------------
from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="Prophet-ML/DL Hybrid",
    factory=lambda **kw: ProphetHybridModel(**kw),
    category="stat",
    role="candidate",
    ensemble_eligible=True,
    requires=("prophet", "tensorflow"),
    needs_config_keys=("prophet_hybrid",),
    description="Prophet trend base model hybrid wrapper (trend_gate / residual_decomp).",
))
