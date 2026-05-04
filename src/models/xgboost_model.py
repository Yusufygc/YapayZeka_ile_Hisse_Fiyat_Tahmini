# -*- coding: utf-8 -*-
"""
xgboost_model.py — XGBoost Regressor Sarmalayıcı (Optuna Tuning Destekli)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
XGBRegressor'ı BaseModel arayüzüne uygun şekilde sarar.
Eğitim 2-boyutlu (düz) özellik matrisi üzerinde yapılır.
Optuna ile hiperparametre optimizasyonu desteklenir.
Modelin kaydedilmesi/yüklenmesi joblib ile yapılır.
"""

import numpy as np
import joblib
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
        **xgb_kwargs,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror",
            **xgb_kwargs,
        )
        self.best_params: dict | None = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        XGBoost modelini eğitir.

        Parameters
        ----------
        X_train : np.ndarray  (samples, features)
        y_train : np.ndarray  (samples,) veya (samples, 1)
        """
        self.model.fit(
            X_train,
            y_train.ravel(),
            eval_set=[(X_train, y_train.ravel())],
            verbose=False,
        )
        print("[OK] XGBoost modeli eğitildi.")

    def tune_and_train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 50,
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
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

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

            rmse_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_flat[train_idx], y_flat[val_idx]

                model = XGBRegressor(
                    **params,
                    random_state=random_state,
                    objective="reg:squarederror",
                    early_stopping_rounds=20,
                )
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                preds = model.predict(X_val)
                rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
                rmse_scores.append(rmse)

            return float(np.mean(rmse_scores))

        # ── Optuna çalıştır ──────────────────────────────────────────────────
        print(f"  [Optuna] {n_trials} deneme başlatılıyor ({n_splits}-fold TSCV)...")
        # Warm-start: SQLite backend varsa onceki denemeleri yukle
        _storage = study_storage or "sqlite:///optuna_studies.db"
        _study_name = study_name or f"xgb_{self.__class__.__name__}"
        try:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
                storage=_storage,
                study_name=_study_name,
                load_if_exists=True,
            )
        except Exception:
            # Storage hatasi durumunda hafiza ici fallback
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        best_rmse = study.best_value
        print(f"  [Optuna] En iyi CV RMSE: {best_rmse:.4f}")
        print(f"  [Optuna] En iyi parametreler: {self.best_params}")

        # ── En iyi parametrelerle nihai modeli eğit ──────────────────────────
        self.model = XGBRegressor(
            **self.best_params,
            random_state=random_state,
            objective="reg:squarederror",
        )
        self.model.fit(
            X_train,
            y_flat,
            eval_set=[(X_train, y_flat)],
            verbose=False,
        )
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
