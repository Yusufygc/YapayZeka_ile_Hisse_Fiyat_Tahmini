# -*- coding: utf-8 -*-
"""
preprocessor.py — Veri Ön İşleme (Split, Scale, Windowing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Zaman serisi verisini kronolojik sırayla train/test'e ayırır,
MinMaxScaler ile ölçeklendirir ve LSTM için 3-boyutlu pencereler oluşturur.

⚠ Data Leakage Önlemi: Scaler yalnızca eğitim verisi üzerinde fit edilir.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def split_data(
    df: pd.DataFrame,
    target_col: str = "Close",
    feature_cols: list | None = None,
    test_ratio: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Zaman serisi verisini kronolojik sırayla Train/Test olarak böler.
    **shuffle=False** — veri sızıntısı önlenir.

    Parameters
    ----------
    df : pd.DataFrame
        Özellik mühendisliği uygulanmış DataFrame.
    target_col : str
        Hedef değişken sütun adı (varsayılan: 'Close').
    feature_cols : list | None
        Kullanılacak özellik sütunları. None ise sayısal sütunlardan
        Date ve hedef sütun otomatik çıkarılır.
    test_ratio : float
        Test kümesi oranı (varsayılan: %20).

    Returns
    -------
    X_train, X_test, y_train, y_test, dates_train, dates_test
    """
    if feature_cols is None:
        exclude = {"Date", target_col}
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    split_idx = int(len(df) * (1 - test_ratio))

    X_train = df[feature_cols].iloc[:split_idx].values
    X_test = df[feature_cols].iloc[split_idx:].values
    y_train = df[target_col].iloc[:split_idx].values.reshape(-1, 1)
    y_test = df[target_col].iloc[split_idx:].values.reshape(-1, 1)

    dates_train = df["Date"].iloc[:split_idx]
    dates_test = df["Date"].iloc[split_idx:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def scale_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    save_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    MinMaxScaler(0–1) ile ölçeklendirir.  Scaler yalnızca train üzerinde fit edilir.
    Scaler objeleri disk'e kaydedilir.

    Parameters
    ----------
    X_train, X_test : np.ndarray  Özellik matrisleri.
    y_train, y_test : np.ndarray  Hedef vektörleri.
    save_dir : str                Scaler'ların kaydedileceği klasör yolu.

    Returns
    -------
    X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y
    """
    os.makedirs(save_dir, exist_ok=True)

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    # Scaler'ları kaydet — canlı projede yeniden kullanılacak
    joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y.pkl"))
    print(f"[✓] Scaler objeleri kaydedildi → {save_dir}")

    return X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    time_steps: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LSTM için kayan pencere (sliding window) yöntemiyle 3-boyutlu tensör üretir.

    Girdi  : (N, features) → Çıktı: (N - time_steps, time_steps, features)
    Hedef  : y[time_steps:]  (T+1 tahmini)

    Parameters
    ----------
    X : np.ndarray  (samples, features)
    y : np.ndarray  (samples, 1)
    time_steps : int  Pencere uzunluğu (varsayılan: 30 gün).

    Returns
    -------
    X_seq : np.ndarray  (samples - time_steps, time_steps, features)
    y_seq : np.ndarray  (samples - time_steps,)
    """
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X)):
        X_seq.append(X[i - time_steps : i])
        y_seq.append(y[i, 0])
    return np.array(X_seq), np.array(y_seq)
