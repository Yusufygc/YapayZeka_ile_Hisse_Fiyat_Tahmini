# -*- coding: utf-8 -*-
"""Lightweight LSTM sequence model for single-symbol daily data."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM  # type: ignore
from tensorflow.keras.models import Model, load_model  # type: ignore

from .base_model import BaseModel


class LSTMLiteModel(BaseModel):
    """Small, regularized LSTM alternative to the attention LSTM."""

    def __init__(
        self,
        units: int = 32,
        dense_units: int = 16,
        dropout_rate: float = 0.25,
        epochs: int = 80,
        batch_size: int = 32,
        learning_rate: float = 0.0003,
        patience: int = 12,
        lr_patience: int = 4,
        validation_ratio: float = 0.1,
        min_val_samples: int = 32,
        tune_on_fit: bool = False,
        tune_n_trials: int = 12,
    ):
        self.units = int(units)
        self.dense_units = int(dense_units)
        self.dropout_rate = float(dropout_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.lr_patience = int(lr_patience)
        self.validation_ratio = float(validation_ratio)
        self.min_val_samples = int(min_val_samples)
        self.tune_on_fit = bool(tune_on_fit)
        self.tune_n_trials = int(tune_n_trials)
        self.model: Model | None = None

    def _chronological_validation_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = len(X)
        if n_samples < 4:
            raise ValueError("LSTM Lite egitimi icin yeterli sequence yok.")

        n_val = max(1, int(n_samples * self.validation_ratio))
        n_val = min(max(self.min_val_samples, n_val), max(1, n_samples - 1))
        n_train = n_samples - n_val
        if n_train <= 0:
            raise ValueError("Chronological validation split sonrasi train ornegi kalmadi.")

        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    def _build_model(
        self,
        input_shape: tuple[int, int],
        *,
        units: int | None = None,
        dense_units: int | None = None,
        dropout_rate: float | None = None,
        learning_rate: float | None = None,
    ) -> Model:
        units = int(self.units if units is None else units)
        dense_units = int(self.dense_units if dense_units is None else dense_units)
        dropout_rate = float(self.dropout_rate if dropout_rate is None else dropout_rate)
        learning_rate = float(self.learning_rate if learning_rate is None else learning_rate)

        inputs = Input(shape=input_shape)
        x = LSTM(units, return_sequences=False)(inputs)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_units, activation="relu")(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=["mae"])
        return model

    def _callbacks(self) -> list:
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.lr_patience,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

    def _tune_hyperparameters(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_shape: tuple[int, int],
    ) -> None:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("LSTM Lite HPO icin optuna gerekli.") from exc

        def objective(trial) -> float:
            units = trial.suggest_categorical("units", [16, 32, 64])
            dense_units = trial.suggest_categorical("dense_units", [8, 16, 32])
            dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
            learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 1e-3])
            batch_size = trial.suggest_categorical("batch_size", [16, 32])

            model = self._build_model(
                input_shape,
                units=units,
                dense_units=dense_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )
            history = model.fit(
                X_tr,
                y_tr,
                epochs=self.epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=self._callbacks(),
                verbose=0,
                shuffle=False,
            )
            best_val_loss = min(history.history.get("val_loss", [float("inf")]))
            best_val_mae = min(history.history.get("val_mae", [0.0]))
            tf.keras.backend.clear_session()
            return float(best_val_loss + 0.1 * best_val_mae)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=max(1, self.tune_n_trials), show_progress_bar=False)
        params = study.best_params
        self.units = int(params.get("units", self.units))
        self.dense_units = int(params.get("dense_units", self.dense_units))
        self.dropout_rate = float(params.get("dropout", self.dropout_rate))
        self.learning_rate = float(params.get("learning_rate", self.learning_rate))
        self.batch_size = int(params.get("batch_size", self.batch_size))
        print(f"  [Optuna LSTM Lite] En iyi validation objective: {study.best_value:.6f}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        if X_train.ndim != 3:
            raise ValueError(
                f"LSTM Lite girdisi 3-boyutlu olmalidir, alinan: {X_train.ndim}D"
            )

        input_shape = (X_train.shape[1], X_train.shape[2])
        X_tr, y_tr, X_val, y_val = self._chronological_validation_split(X_train, y_train)

        if self.tune_on_fit:
            self._tune_hyperparameters(X_tr, y_tr, X_val, y_val, input_shape)

        self.model = self._build_model(input_shape=input_shape)
        self.model.fit(
            X_tr,
            y_tr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=self._callbacks(),
            verbose=1,
            shuffle=False,
        )
        print("[OK] LSTM Lite modeli egitildi.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model henuz egitilmedi.")
        preds = self.model.predict(X_test, verbose=0)
        return preds.ravel()

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Kaydedilecek model yok.")
        self.model.save(path)
        print(f"[OK] LSTM Lite modeli kaydedildi -> {path}")

    def load(self, path: str) -> None:
        self.model = load_model(path)
        print(f"[OK] LSTM Lite modeli yuklendi <- {path}")


from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="LSTM Lite",
    factory=lambda **kw: LSTMLiteModel(**kw),
    category="seq",
    role="candidate",
    ensemble_eligible=True,
    requires=("tensorflow",),
    needs_config_keys=("lstm_lite",),
    default_candidate=False,
    description="Small regularized LSTM for single-symbol sequence experiments.",
))
