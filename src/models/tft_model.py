# -*- coding: utf-8 -*-
"""
tft_model.py — Temporal Fusion Transformer (Keras Implementation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides a clean Keras-based TFT architecture for time series forecasting.
Inherits from BaseModel to maintain compatibility.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Dense, Dropout, LSTM, LayerNormalization, 
    MultiHeadAttention, Add, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

from .base_model import BaseModel

# ═════════════════════════════════════════════════════════════════════════════
# Gated Linear Unit (GLU)
# ═════════════════════════════════════════════════════════════════════════════
@tf.keras.utils.register_keras_serializable()
class GLU(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_sig = Dense(units, activation="sigmoid")
        self.dense_lin = Dense(units, activation="linear")

    def call(self, inputs):
        sig = self.dense_sig(inputs)
        lin = self.dense_lin(inputs)
        return tf.multiply(sig, lin)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# ═════════════════════════════════════════════════════════════════════════════
# Gated Residual Network (GRN)
# ═════════════════════════════════════════════════════════════════════════════
@tf.keras.utils.register_keras_serializable()
class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense1 = Dense(units, activation="elu")
        self.dense2 = Dense(units, activation="linear")
        self.dropout = Dropout(dropout_rate)
        self.glu = GLU(units)
        self.add = Add()
        self.norm = LayerNormalization()
        # Projection layer if input dimension doesn't match units
        self.project = Dense(units)

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.needs_projection = True
        else:
            self.needs_projection = False
        super().build(input_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.glu(x)
        
        shortcut = self.project(inputs) if self.needs_projection else inputs
        
        x = self.add([x, shortcut])
        return self.norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout_rate": self.dropout_rate})
        return config


# ═════════════════════════════════════════════════════════════════════════════
# TFT Model Class
# ═════════════════════════════════════════════════════════════════════════════
class TFTModel(BaseModel):
    """
    Simplified Temporal Fusion Transformer using Keras Functional API.
    Architecture:
      Input (Time Steps x Features)
      -> GRN (Feature Transformations)
      -> LSTM Encoder (Local Processing)
      -> Multi-Head Attention (Long-term dependencies)
      -> GRN & GLU
      -> Dense (Output)
    """

    def __init__(
        self,
        hidden_units: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        epochs: int = 80,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model: Model | None = None

    def _build_model(self, input_shape: tuple) -> Model:
        inputs = Input(shape=input_shape)
        
        # 1. Feature processing with GRN
        x = GatedResidualNetwork(self.hidden_units, self.dropout_rate)(inputs)
        
        # 2. Sequence processing with LSTM
        # We return sequences to pass to attention
        lstm_out = LSTM(self.hidden_units, return_sequences=True)(x)
        lstm_out = LayerNormalization()(lstm_out)
        
        # 3. Multi-Head Attention
        attention_out = MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.hidden_units, 
            dropout=self.dropout_rate
        )(lstm_out, lstm_out)
        
        # Add & Norm
        x = Add()([lstm_out, attention_out])
        x = LayerNormalization()(x)
        
        # 4. Final Position-wise GRN
        x = GatedResidualNetwork(self.hidden_units, self.dropout_rate)(x)
        
        # 5. Flatten and Output
        # We focus on the last timestep for forecasting or flatten the whole sequence
        x = Flatten()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        if X_train.ndim != 3:
            raise ValueError(f"TFT girdi tensörü 3-boyutlu olmalıdır, alınan: {X_train.ndim}D")

        self.model = self._build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)

        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )
        print("[OK] TFT modeli eğitildi.")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model henüz eğitilmedi.")
        preds = self.model.predict(X_test, verbose=0)
        return preds.ravel()

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Kaydedilecek model yok.")
        self.model.save(path)
        print(f"[OK] TFT modeli kaydedildi -> {path}")

    def load(self, path: str) -> None:
        self.model = load_model(path, custom_objects={
            "GLU": GLU,
            "GatedResidualNetwork": GatedResidualNetwork
        })
        print(f"[OK] TFT modeli yüklendi <- {path}")
