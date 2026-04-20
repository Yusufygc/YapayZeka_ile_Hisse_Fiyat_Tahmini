# -*- coding: utf-8 -*-
"""
preprocessor.py — Veri Ön İşleme (Split, Scale, Windowing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Zaman serisi verisini kronolojik sırayla train/test'e ayırır,
RobustScaler ile ölçeklendirir ve LSTM/TFT için 3-boyutlu pencereler oluşturur.

⚠ Data Leakage Önlemi: Scaler yalnızca eğitim verisi üzerinde fit edilir.

Ölçekleme Stratejisi (v2 — H2 & H3 düzeltmesi):
  • X (özellikler)  → RobustScaler (medyan/IQR) + clip([-5, +5])
       Durağan özellikler bile uç değer içerebilir; RobustScaler outlier'lara
       dayanıklıdır. Clip, test döneminde eğitim dağılımı dışına çıkan
       uç noktaların (distribution shift) ağa zarar vermesini engeller.
  • y (log-getiri)  → StandardScaler (z-score)
       Hedef ham fiyat DEĞİL; log-getiri olduğu için dağılımı merkezi
       (μ≈0, σ sabit), StandardScaler mükemmel uyar.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from typing import Tuple


# Güvenlik clip aralığı: RobustScaler sonrası değerlerin güvenli kabul edileceği
# çeyrekler-arası mesafe cinsinden sınır. 5×IQR dışı her şey OOD sayılıp klipsenir.
_CLIP_LOW:  float = -5.0
_CLIP_HIGH: float =  5.0


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
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    save_dir: str,
    scaling_mode: str = "robust_x_standard_y_clip",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    """
    Özellikler ve hedef değişkeni ölçeklendirir.
    Scaler yalnızca train üzerinde fit edilir (data leakage yok).

    v2 (H2 + H3 düzeltmesi):
      • X → RobustScaler + clip(-5, +5)
            MinMaxScaler(0,1) non-stationary özellik seviyelerinde test
            döneminde [0,1] aralığı dışına ekstrapole oluyor ve modeli
            tamamen "görülmemiş" girdilerle besliyordu. RobustScaler
            medyan/IQR kullandığı için outlier'lara dayanıklıdır; clip
            ise uç noktalardaki katastrofik OOD'yi sınırlar.

      • y (log-getiri) → StandardScaler
            Hedef artık ham fiyat değil, log-getiri (stationary).
            μ≈0 etrafında dar dağılımlı olduğu için z-score uyumludur.

    Parameters
    ----------
    X_train, X_test : np.ndarray  Özellik matrisleri.
    y_train, y_test : np.ndarray  Hedef vektörleri (log-getiri, 2D).
    save_dir        : str         Scaler'ların kaydedileceği klasör.

    Returns
    -------
    X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y
    """
    os.makedirs(save_dir, exist_ok=True)

    if scaling_mode == "robust_x_standard_y_clip":
        scaler_X = RobustScaler()
        scaler_y = StandardScaler()
        clip_enabled = True
    elif scaling_mode == "robust":
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
        clip_enabled = True
    elif scaling_mode == "standard":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        clip_enabled = False
    elif scaling_mode == "minmax":
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        clip_enabled = False
    else:
        raise ValueError(
            f"Desteklenmeyen scaling_mode: {scaling_mode}. "
            "Beklenen: robust_x_standard_y_clip, robust, standard, minmax"
        )

    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    if clip_enabled:
        X_train_s = np.clip(X_train_s, _CLIP_LOW, _CLIP_HIGH)
        X_test_s = np.clip(X_test_s, _CLIP_LOW, _CLIP_HIGH)

    # Scaler'ları kaydet (reproducibility + inference)
    joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y.pkl"))
    print(f"[OK] Scaler objeleri kaydedildi -> {save_dir}")

    return X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y


# ── Fiyat Yeniden İnşa Helper'ı ──────────────────────────────────────────────

def reconstruct_prices_from_logret(
    preds_logret: np.ndarray,
    prev_close:   np.ndarray,
) -> np.ndarray:
    """
    Log-getiri tahminlerini, bir önceki günün gerçek kapanışı ile çarparak
    fiyata geri çevirir (one-step-ahead re-forecasting protokolü).

        price_pred[t] = prev_close[t] * exp(log_ret_pred[t])

    Bu standart değerlendirme protokolüdür: modelin her gün yeniden
    tahmin yaptığı, güncel fiyatı besleme imkânı olduğu bir senaryo.
    Autoregressive rollout (hatanın zincirlenmesi) istersen ayrı bir
    yardımcı fonksiyon yazılmalı.

    Parameters
    ----------
    preds_logret : np.ndarray   (N,) tahmin edilen log-getiriler.
    prev_close   : np.ndarray   (N,) her tahmin için t-1 gerçek kapanışı.

    Returns
    -------
    np.ndarray  (N,) ₺ cinsinden fiyat tahminleri.
    """
    preds_logret = np.asarray(preds_logret, dtype=float).ravel()
    prev_close   = np.asarray(prev_close,   dtype=float).ravel()

    if preds_logret.shape != prev_close.shape:
        raise ValueError(
            f"preds_logret ve prev_close aynı uzunlukta olmalı, "
            f"alınan: preds_logret={preds_logret.shape}, prev_close={prev_close.shape}"
        )

    # Taşma (overflow) koruması: log-getiri için güvenli aralık [-1, +1]
    # (ham sınır günlük ±%100 değişim — normalde model asla bu aralıkta olmaz,
    # ama OOD tahmin olursa exp(x) taşmasın diye kırpıyoruz).
    preds_logret = np.clip(preds_logret, -1.0, 1.0)

    return prev_close * np.exp(preds_logret)


def reconstruct_prices_from_return(
    preds_return: np.ndarray,
    prev_close: np.ndarray,
) -> np.ndarray:
    """
    Basit yüzde getiri tahminini fiyata çevirir:
        price_pred[t] = prev_close[t] * (1 + return_pred[t])
    """
    preds_return = np.asarray(preds_return, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()

    if preds_return.shape != prev_close.shape:
        raise ValueError(
            f"preds_return ve prev_close aynı uzunlukta olmalı, "
            f"alınan: preds_return={preds_return.shape}, prev_close={prev_close.shape}"
        )

    preds_return = np.clip(preds_return, -0.95, 5.0)
    return prev_close * (1.0 + preds_return)


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    time_steps: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LSTM için kayan pencere (sliding window) yöntemiyle 3-boyutlu tensör üretir.

    Girdi  : (N, features) -> Çıktı: (N - time_steps + 1, time_steps, features)
    Hedef  : Her pencerenin son gününe ait hedef.

    Semantik:
      X[t] = t gününün sonunda bilinen özellikler
      y[t] = t+1 gününün hedefi
    Bu nedenle pencere [t-time_steps+1 ... t] -> y[t] eşleşmesi kurulur.

    Parameters
    ----------
    X : np.ndarray  (samples, features)
    y : np.ndarray  (samples, 1)
    time_steps : int  Pencere uzunluğu (varsayılan: 30 gün).

    Returns
    -------
    X_seq : np.ndarray  (samples - time_steps + 1, time_steps, features)
    y_seq : np.ndarray  (samples - time_steps + 1,)
    """
    X_seq, y_seq = [], []
    for end_idx in range(time_steps - 1, len(X)):
        start_idx = end_idx - time_steps + 1
        X_seq.append(X[start_idx : end_idx + 1])
        y_seq.append(y[end_idx, 0])
    return np.array(X_seq), np.array(y_seq)
