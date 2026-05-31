# -*- coding: utf-8 -*-
"""
random_forest_model.py — Random Forest Regressor Sarmalayıcı (Optuna Tuning Destekli)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RandomForestRegressor'ı BaseModel arayüzüne uygun şekilde sarar.
Eğitim 2-boyutlu (düz) özellik matrisi üzerinde yapılır.
Optuna ile hiperparametre optimizasyonu desteklenir.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest Regressor sarmalayıcısı — opsiyonel Optuna tuning destekli."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: float | str = 1.0,
        random_state: int = 42,
        *,
        tune_on_fit: bool = False,
        tune_n_trials: int = 40,
        tune_n_splits: int = 3,
        **rf_kwargs,
    ):
        self._tune_on_fit = bool(tune_on_fit)
        self._tune_n_trials = int(tune_n_trials)
        self._tune_n_splits = int(tune_n_splits)
        self._random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # Tüm çekirdekleri kullan
            **rf_kwargs,
        )
        self.best_params: dict | None = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Random Forest modelini eğitir.

        tune_on_fit=True ise Optuna ile hiperparametre optimizasyonu çalışır.

        Parameters
        ----------
        X_train : np.ndarray  (samples, features)
        y_train : np.ndarray  (samples,) veya (samples, 1)
        """
        if self._tune_on_fit:
            self.tune_and_train(
                X_train,
                y_train,
                n_trials=self._tune_n_trials,
                n_splits=self._tune_n_splits,
                random_state=self._random_state,
            )
            return
        self.model.fit(X_train, y_train.ravel())
        print("[OK] Random Forest modeli eğitildi.")

    def tune_and_train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 40,
        n_splits: int = 5,
        random_state: int = 42,
        study_storage: str | None = None,
        study_name: str | None = None,
    ) -> dict:
        """
        Optuna ile hiperparametre optimizasyonu yapar ve en iyi model ile eğitir.

        Parameters
        ----------
        X_train : np.ndarray  (samples, features)
        y_train : np.ndarray  (samples,) veya (samples, 1)
        n_trials : int        Optuna deneme sayısı (varsayılan: 40).
        n_splits : int        TimeSeriesSplit katman sayısı.
        random_state : int    Tekrarlanabilirlik için rastgele tohum.

        Returns
        -------
        dict  En iyi hiperparametreler.
        """
        from ._tuning import run_optuna_study, stability_adjusted_cv_objective

        y_flat = y_train.ravel()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
                "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", [1.0, "sqrt", "log2"]),
            }

            def _fit_predict(X_tr, y_tr, X_val, y_val):
                model = RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
                model.fit(X_tr, y_tr)
                return model.predict(X_val)

            return stability_adjusted_cv_objective(X_train, y_flat, tscv, _fit_predict)

        self.best_params, best_obj = run_optuna_study(
            objective,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state,
            study_name=study_name or f"rf_{self.__class__.__name__}",
            log_prefix="Optuna RF",
            study_storage=study_storage,
        )
        print(f"  [Optuna RF] En iyi stability-adj objective: {best_obj:.4f}")
        print(f"  [Optuna RF] En iyi parametreler: {self.best_params}")

        # Nihai modeli en iyi parametrelerle eğit
        self.model = RandomForestRegressor(
            **self.best_params,
            random_state=random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_flat)
        print("[OK] Random Forest modeli (Optuna-tuned) eğitildi.")
        return self.best_params

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        Test verisi üzerinde tahmin üretir.

        Parameters
        ----------
        X_test : np.ndarray  (samples, features)

        Returns
        -------
        np.ndarray  (samples,)
        """
        return self.model.predict(X_test)

    def save(self, path: str) -> None:
        """Modeli .pkl olarak kaydeder."""
        joblib.dump(self.model, path)
        print(f"[OK] Random Forest modeli kaydedildi -> {path}")

    def load(self, path: str) -> None:
        """Kaydedilmiş modeli diskten yükler."""
        self.model = joblib.load(path)
        print(f"[OK] Random Forest modeli yüklendi <- {path}")


# --- Registry tescili (Faz 1) -------------------------------------------
from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="Random Forest",
    factory=lambda **kw: RandomForestModel(**kw),
    category="tree",
    role="candidate",
    ensemble_eligible=True,
    description="RF regressor — SASA WF'sinde en yüksek DSR; savunmacı exposure.",
))
