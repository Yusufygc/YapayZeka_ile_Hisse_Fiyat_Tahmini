# -*- coding: utf-8 -*-
"""
xgboost_model.py — XGBoost Regressor Sarmalayıcı (Optuna Tuning Destekli)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
XGBRegressor'ı BaseModel arayüzüne uygun şekilde sarar.
Eğitim 2-boyutlu (düz) özellik matrisi üzerinde yapılır.
Optuna ile hiperparametre optimizasyonu desteklenir.
Modelin kaydedilmesi/yüklenmesi joblib ile yapılır.
"""

import os

import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost Regressor sarmalayıcısı — opsiyonel Optuna tuning destekli."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        *,
        tune_on_fit: bool = False,
        tune_n_trials: int = 40,
        tune_n_splits: int = 3,
        early_stopping_rounds: int = 50,
        **xgb_kwargs,
    ):
        self._tune_on_fit = bool(tune_on_fit)
        self._tune_n_trials = int(tune_n_trials)
        self._tune_n_splits = int(tune_n_splits)
        self._early_stopping_rounds = int(early_stopping_rounds)
        self._random_state = random_state
        self._init_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
        )
        self._xgb_kwargs = dict(xgb_kwargs)
        self.model = XGBRegressor(
            **self._init_kwargs,
            objective="reg:squarederror",
            **self._xgb_kwargs,
        )
        self.best_params: dict | None = None

    def _fit_with_early_stop(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(y)
        split = max(1, int(n * 0.85))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        if len(y_val) >= 10:
            self.model.set_params(early_stopping_rounds=self._early_stopping_rounds)
            self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X, y, eval_set=[(X, y)], verbose=False)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        XGBoost modelini eğitir.

        tune_on_fit=True ise Optuna ile hiperparametre optimizasyonu çalışır,
        aksi takdirde varsayılan parametrelerle holdout early-stopping uygulanır.

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

        self._fit_with_early_stop(X_train, y_train.ravel())
        print("[OK] XGBoost modeli eğitildi.")

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
        n_trials : int        Optuna deneme sayısı (varsayılan: 50).
        n_splits : int        TimeSeriesSplit katman sayısı (varsayılan: 5).
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
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            }

            def _fit_predict(X_tr, y_tr, X_val, y_val):
                model = XGBRegressor(
                    **params,
                    random_state=random_state,
                    objective="reg:squarederror",
                    early_stopping_rounds=20,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                return model.predict(X_val)

            return stability_adjusted_cv_objective(X_train, y_flat, tscv, _fit_predict)

        self.best_params, best_obj = run_optuna_study(
            objective,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state,
            study_name=study_name or f"xgb_{self.__class__.__name__}",
            log_prefix="Optuna",
            study_storage=study_storage,
        )
        print(f"  [Optuna] En iyi stability-adj objective: {best_obj:.4f}")
        print(f"  [Optuna] En iyi parametreler: {self.best_params}")

        # ── En iyi parametrelerle nihai modeli eğit ──────────────────────────
        self.model = XGBRegressor(
            **self.best_params,
            random_state=random_state,
            objective="reg:squarederror",
        )
        self._fit_with_early_stop(X_train, y_flat)
        best_iter = getattr(self.model, "best_iteration", None)
        if best_iter is not None:
            print(f"[OK] XGBoost modeli (Optuna-tuned) eğitildi. best_iteration={best_iter}")
        else:
            print("[OK] XGBoost modeli (Optuna-tuned) eğitildi.")
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
        print(f"[OK] XGBoost modeli kaydedildi -> {path}")

    def load(self, path: str) -> None:
        """Kaydedilmiş modeli diskten yükler."""
        self.model = joblib.load(path)
        print(f"[OK] XGBoost modeli yüklendi <- {path}")


# --- Registry tescili (Faz 1) -------------------------------------------
from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="XGBoost",
    factory=lambda **kw: XGBoostModel(**kw),
    category="tree",
    role="candidate",
    ensemble_eligible=True,
    requires=("xgboost",),
    default_candidate=True,
    description="Gradient boosting; SASA WF'sinde zayıf, hyperparam tune önerilir.",
))
