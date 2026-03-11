# -*- coding: utf-8 -*-
"""
data_loader.py — Veri Yükleme ve Özellik Mühendisliği
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ham OHLCV CSV dosyasını okur, Türkçe sütun adlarını İngilizceye çevirir,
piyasanın kapalı olduğu (hacim=0) günleri eler ve teknik göstergeler
(RSI, MACD, SMA, EMA) ile gecikme (lag) özellikleri ekler.
"""

import pandas as pd
import numpy as np
import ta


# ── Sütun eşleştirme tablosu (TR → EN) ──────────────────────────────────────
_COLUMN_MAP = {
    "Tarih": "Date",
    "Açılış": "Open",
    "Yüksek": "High",
    "Düşük": "Low",
    "Kapanış": "Close",
    "Düzeltilmiş_Kapanış": "Adj_Close",
    "Hacim": "Volume",
}


def load_and_clean(csv_path: str, drop_zero_volume: bool = True) -> pd.DataFrame:
    """
    CSV dosyasını okuyup temel temizleme adımlarını uygular.

    Parameters
    ----------
    csv_path : str
        Ham OHLCV verisinin bulunduğu CSV yolu.
    drop_zero_volume : bool
        True ise hacmi sıfır olan (borsa kapalı) günler çıkarılır.

    Returns
    -------
    pd.DataFrame
        Temizlenmiş, tarih sıralı DataFrame.
    """
    df = pd.read_csv(csv_path)
    df.rename(columns=_COLUMN_MAP, inplace=True)

    # Tarih parse — gün/ay/yıl formatı
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if drop_zero_volume:
        df = df[df["Volume"] > 0].copy()
        df.reset_index(drop=True, inplace=True)

    return df


def add_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Teknik gösterge ve gecikme özelliklerini ekler.

    Eklenen özellikler:
        • RSI (14 periyot)
        • MACD ve MACD Signal çizgisi
        • SMA (20 periyot)
        • EMA (12 periyot)
        • Bollinger Bands (Upper, Lower, Width — 20 periyot, 2 std)
        • ATR (Average True Range — 14 periyot)
        • Stochastic Oscillator (K, D — 14 periyot)
        • Close_Lag_1 … Close_Lag_{lags}

    Parameters
    ----------
    df : pd.DataFrame
        Temizlenmiş OHLCV verisi (en azından Close, High, Low sütunları).
    lags : int
        Gecikme (lag) sayısı.

    Returns
    -------
    pd.DataFrame
        Özellik mühendisliği uygulanmış, NaN satırları düşürülmüş DataFrame.
    """
    df = df.copy()

    # ── Mevcut teknik göstergeler ────────────────────────────────────────────
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)

    # ── Bollinger Bands (20 periyot, 2 standart sapma) ───────────────────────
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()

    # ── ATR — Average True Range (14 periyot) ────────────────────────────────
    df["ATR"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )

    # ── Stochastic Oscillator (14 periyot) ───────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # ── Gecikme (lag) özellikleri ─────────────────────────────────────────────
    for i in range(1, lags + 1):
        df[f"Close_Lag_{i}"] = df["Close"].shift(i)

    # ── NaN satırları temizle ────────────────────────────────────────────────
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Kolaylık fonksiyonu: veriyi oku → temizle → özellik çıkar.

    Parameters
    ----------
    csv_path : str
        Ham CSV yolu.

    Returns
    -------
    pd.DataFrame
        Kullanıma hazır, özellik mühendisliği yapılmış DataFrame.
    """
    df = load_and_clean(csv_path)
    df = add_features(df)
    return df
