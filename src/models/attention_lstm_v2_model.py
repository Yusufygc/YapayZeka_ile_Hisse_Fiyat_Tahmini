# -*- coding: utf-8 -*-
"""Regularized Attention-LSTM v2 model.

This model is intentionally registered as a separate candidate so the existing
``LSTM`` and ``LSTM Lite`` contracts stay unchanged.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM, Layer  # type: ignore
from tensorflow.keras.models import Model, load_model  # type: ignore

from src.models.base_model import BaseModel
from src.xai.feature_dictionary import describe_feature, feature_group


class TemporalAttentionV2(Layer):
    """Temporal attention layer returning context and per-step weights."""

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(int(input_shape[-1]), 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        scores = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(x * weights, axis=1)
        return context, weights

    def get_config(self):
        return super().get_config()


class AttentionLSTMV2Model(BaseModel):
    """Smaller regularized BiLSTM attention model with XAI weight export."""

    def __init__(
        self,
        units_1: int = 64,
        units_2: int = 32,
        dense_units: int = 32,
        dropout_rate: float = 0.30,
        epochs: int = 80,
        batch_size: int = 32,
        learning_rate: float = 0.0005,
        patience: int = 12,
        lr_patience: int = 4,
        validation_ratio: float = 0.1,
        min_val_samples: int = 32,
        tune_on_fit: bool = False,
        tune_n_trials: int = 12,
        loss: str = "huber",
    ):
        self.units_1 = int(units_1)
        self.units_2 = int(units_2)
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
        self.loss = str(loss)
        self.model: Model | None = None
        self.attention_model: Model | None = None

    def _chronological_validation_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = len(X)
        if n_samples < 4:
            raise ValueError("AttentionLSTM v2 egitimi icin yeterli sequence yok.")
        n_val = max(1, int(n_samples * self.validation_ratio))
        n_val = min(max(self.min_val_samples, n_val), max(1, n_samples - 1))
        n_train = n_samples - n_val
        if n_train <= 0:
            raise ValueError("Chronological validation split sonrasi train ornegi kalmadi.")
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    def _loss(self):
        if self.loss.lower() == "mse":
            return "mse"
        return tf.keras.losses.Huber()

    def _build_model(self, input_shape: tuple[int, int]) -> Model:
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(self.units_1, return_sequences=True))(inputs)
        x = Dropout(self.dropout_rate)(x)
        x = Bidirectional(LSTM(self.units_2, return_sequences=True))(x)
        x = Dropout(self.dropout_rate)(x)
        context, attention_weights = TemporalAttentionV2(name="temporal_attention_v2")(x)
        x = Dense(self.dense_units, activation="relu")(context)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        self.attention_model = Model(inputs=inputs, outputs=attention_weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=self._loss(), metrics=["mae"])
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
            raise ImportError("AttentionLSTM v2 HPO icin optuna gerekli.") from exc

        def objective(trial) -> float:
            units_1 = trial.suggest_categorical("units_1", [32, 64, 96])
            units_2 = trial.suggest_categorical("units_2", [16, 32, 48])
            dense_units = trial.suggest_categorical("dense_units", [16, 32, 64])
            dropout_rate = trial.suggest_float("dropout", 0.15, 0.45)
            learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 5e-4, 1e-3])
            batch_size = trial.suggest_categorical("batch_size", [16, 32])

            old = (
                self.units_1,
                self.units_2,
                self.dense_units,
                self.dropout_rate,
                self.learning_rate,
            )
            self.units_1 = int(units_1)
            self.units_2 = int(units_2)
            self.dense_units = int(dense_units)
            self.dropout_rate = float(dropout_rate)
            self.learning_rate = float(learning_rate)
            model = self._build_model(input_shape)
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
            (
                self.units_1,
                self.units_2,
                self.dense_units,
                self.dropout_rate,
                self.learning_rate,
            ) = old
            return float(best_val_loss + 0.1 * best_val_mae)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=max(1, self.tune_n_trials), show_progress_bar=False)
        params = study.best_params
        self.units_1 = int(params.get("units_1", self.units_1))
        self.units_2 = int(params.get("units_2", self.units_2))
        self.dense_units = int(params.get("dense_units", self.dense_units))
        self.dropout_rate = float(params.get("dropout", self.dropout_rate))
        self.learning_rate = float(params.get("learning_rate", self.learning_rate))
        self.batch_size = int(params.get("batch_size", self.batch_size))
        print(f"  [Optuna AttentionLSTM v2] En iyi validation objective: {study.best_value:.6f}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        if X_train.ndim != 3:
            raise ValueError(
                f"AttentionLSTM v2 girdisi 3-boyutlu olmalidir, alinan: {X_train.ndim}D"
            )
        input_shape = (X_train.shape[1], X_train.shape[2])
        X_tr, y_tr, X_val, y_val = self._chronological_validation_split(X_train, y_train)
        if self.tune_on_fit:
            self._tune_hyperparameters(X_tr, y_tr, X_val, y_val, input_shape)
        self.model = self._build_model(input_shape)
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
        print("[OK] AttentionLSTM v2 modeli egitildi.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model henuz egitilmedi.")
        return self.model.predict(X_test, verbose=0).ravel()

    def predict_attention_weights(self, X: np.ndarray) -> np.ndarray:
        if self.attention_model is None:
            if self.model is None:
                raise RuntimeError("Model henuz egitilmedi.")
            attention_layer = self.model.get_layer("temporal_attention_v2")
            self.attention_model = Model(
                inputs=self.model.input,
                outputs=attention_layer.output[1],
            )
        weights = self.attention_model.predict(X, verbose=0)
        return np.asarray(weights).squeeze(-1)

    def export_attention_xai(
        self,
        X_seq: np.ndarray,
        feature_names: Iterable[str],
        output_path: str,
        *,
        model_name: str = "AttentionLSTM v2",
    ) -> None:
        feature_names = list(feature_names)
        if X_seq.size == 0 or not feature_names:
            return
        weights = self.predict_attention_weights(X_seq)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        mean_time = weights.mean(axis=0)
        last_seq = np.asarray(X_seq)[-1]
        if last_seq.ndim != 2:
            return
        weighted_feature_signal = np.average(last_seq, axis=0, weights=mean_time)
        rows = []
        for idx, feature in enumerate(feature_names[: len(weighted_feature_signal)]):
            rows.append({
                "Model": model_name,
                "Feature": feature,
                "Readable_Feature": describe_feature(feature),
                "Feature_Group": feature_group(feature),
                "Importance": float(weighted_feature_signal[idx]),
                "Direction": "positive" if weighted_feature_signal[idx] >= 0 else "negative",
                "Method": "Temporal attention weights",
            })
        rows.sort(key=lambda row: abs(row["Importance"]), reverse=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Kaydedilecek model yok.")
        self.model.save(path)
        print(f"[OK] AttentionLSTM v2 modeli kaydedildi -> {path}")

    def load(self, path: str) -> None:
        self.model = load_model(path, custom_objects={"TemporalAttentionV2": TemporalAttentionV2})
        attention_layer = self.model.get_layer("temporal_attention_v2")
        self.attention_model = Model(inputs=self.model.input, outputs=attention_layer.output[1])
        print(f"[OK] AttentionLSTM v2 modeli yuklendi <- {path}")


from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="AttentionLSTM v2",
    factory=lambda **kw: AttentionLSTMV2Model(**kw),
    category="seq",
    role="candidate",
    ensemble_eligible=True,
    requires=("tensorflow",),
    needs_config_keys=("attention_lstm_v2",),
    default_candidate=False,
    description="Regularized BiLSTM attention v2 with temporal attention XAI export.",
))
