# -*- coding: utf-8 -*-
"""
lstm_model.py — Bidirectional LSTM + Attention Modeli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tek model sınıfı sunar:
  • AttentionLSTMModel — Bidirectional LSTM + Bahdanau Attention + Dense

BaseModel arayüzünü uygular ve .keras formatında kaydedilir.

Not: Vanilla LSTMModel (2×LSTM + Dropout) Faz 6 Optimizasyon kapsamında
kaldırıldı — pipeline tarafından hiçbir zaman örneklenmiyordu (ölü kod).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM, Dense, Dropout, Bidirectional,
    Input, Layer,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

from .base_model import BaseModel


# ═════════════════════════════════════════════════════════════════════════════
# Custom Attention Layer
# ═════════════════════════════════════════════════════════════════════════════

class AttentionLayer(Layer):
    """
    Basit Bahdanau tarzı Attention mekanizması.
    Girdi  : (batch, time_steps, features)
    Çıktı  : (batch, features) — ağırlıklı zaman adımı özeti
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, time_steps, features)
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
        # x: (batch, time_steps, features)
        # Score hesapla: e = tanh(x · W + b)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)  # (batch, time_steps, 1)
        # Softmax ile dikkat ağırlıkları
        alpha = tf.nn.softmax(e, axis=1)  # (batch, time_steps, 1)
        # Ağırlıklı toplam — context vector
        context = tf.reduce_sum(x * alpha, axis=1)  # (batch, features)
        return context

    def get_config(self):
        return super().get_config()


# ═════════════════════════════════════════════════════════════════════════════
# Bidirectional LSTM + Attention Model
# ═════════════════════════════════════════════════════════════════════════════

class AttentionLSTMModel(BaseModel):
    """
    Bidirectional LSTM + Attention mekanizması.

    Mimari:
        Input -> Bidirectional(LSTM(128, return_sequences=True))
              -> Dropout
              -> Bidirectional(LSTM(64, return_sequences=True))
              -> Dropout
              -> AttentionLayer
              -> Dense(64, relu) -> Dropout -> Dense(1)
    """

    def __init__(
        self,
        units_1: int = 128,
        units_2: int = 64,
        dropout_rate: float = 0.2,
        epochs: int = 80,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 15,
        lr_patience: int = 5,
        validation_ratio: float = 0.1,
        min_val_samples: int = 32,
    ):
        self.units_1 = units_1
        self.units_2 = units_2
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.lr_patience = lr_patience
        self.validation_ratio = validation_ratio
        self.min_val_samples = min_val_samples
        self.model: Model | None = None

    def _chronological_validation_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = len(X)
        if n_samples < 4:
            raise ValueError("Attention LSTM eğitimi için yeterli sequence yok.")

        n_val = max(1, int(n_samples * self.validation_ratio))
        n_val = min(max(self.min_val_samples, n_val), max(1, n_samples - 1))
        n_train = n_samples - n_val
        if n_train <= 0:
            raise ValueError("Chronological validation split sonrası train örneği kalmadı.")

        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    def _build_model(self, input_shape: tuple) -> Model:
        """
        Functional API ile Bidirectional LSTM + Attention modeli oluşturur.

        Parameters
        ----------
        input_shape : tuple  (time_steps, features)
        """
        inputs = Input(shape=input_shape)

        # ── Bidirectional LSTM katmanları ─────────────────────────────────────
        x = Bidirectional(LSTM(self.units_1, return_sequences=True))(inputs)
        x = Dropout(self.dropout_rate)(x)

        x = Bidirectional(LSTM(self.units_2, return_sequences=True))(x)
        x = Dropout(self.dropout_rate)(x)

        # ── Attention mekanizması ────────────────────────────────────────────
        x = AttentionLayer()(x)

        # ── Fully connected çıkış ────────────────────────────────────────────
        x = Dense(64, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Bidirectional LSTM + Attention modelini eğitir.

        Parameters
        ----------
        X_train : np.ndarray  (samples, time_steps, features)
        y_train : np.ndarray  (samples,)
        """
        if X_train.ndim != 3:
            raise ValueError(
                f"LSTM girdi tensörü 3-boyutlu olmalıdır, alınan: {X_train.ndim}D"
            )

        self.model = self._build_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        X_tr, y_tr, X_val, y_val = self._chronological_validation_split(X_train, y_train)

        # ── Callbacks ────────────────────────────────────────────────────────
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.lr_patience,
            min_lr=1e-6,
            verbose=1,
        )

        self.model.fit(
            X_tr,
            y_tr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=1,
            shuffle=False,
        )
        print("[OK] Attention LSTM modeli eğitildi.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        Test verisi üzerinde tahmin üretir.

        Parameters
        ----------
        X_test : np.ndarray  (samples, time_steps, features)

        Returns
        -------
        np.ndarray  (samples,)
        """
        if self.model is None:
            raise RuntimeError("Model henüz eğitilmedi.")
        preds = self.model.predict(X_test, verbose=0)
        return preds.ravel()

    def save(self, path: str) -> None:
        """Modeli .keras formatında kaydeder."""
        if self.model is None:
            raise RuntimeError("Kaydedilecek model yok.")
        self.model.save(path)
        print(f"[OK] Attention LSTM modeli kaydedildi -> {path}")

    def load(self, path: str) -> None:
        """Kaydedilmiş Keras modelini yükler."""
        self.model = load_model(path, custom_objects={"AttentionLayer": AttentionLayer})
        print(f"[OK] Attention LSTM modeli yüklendi <- {path}")
