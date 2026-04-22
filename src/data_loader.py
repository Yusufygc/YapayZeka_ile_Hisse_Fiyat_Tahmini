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


# ── Sütun eşleştirme tablosu (TR -> EN) ──────────────────────────────────────
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

    # Tarih parse — Karışık formatlara (mixed) karşı esneklik
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    except Exception:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    corporate_action_report = {
        "adj_close_available": False,
        "price_source": "Close",
        "warning": "Adj_Close bulunamadi; corporate action duzeltmesi dogrulanamadi.",
        "max_abs_adj_close_diff_pct": None,
        "mean_abs_adj_close_diff_pct": None,
        "rows_with_adj_close": 0,
    }

    if "Adj_Close" in df.columns and df["Adj_Close"].notna().any():
        raw_close = pd.to_numeric(df["Close"], errors="coerce")
        adj_close = pd.to_numeric(df["Adj_Close"], errors="coerce")
        valid = raw_close.notna() & adj_close.notna() & (raw_close != 0)
        if valid.any():
            diff_ratio = (adj_close[valid] / raw_close[valid]) - 1.0
            max_abs_diff = float(diff_ratio.abs().max())
            mean_abs_diff = float(diff_ratio.abs().mean())
            corporate_action_report = {
                "adj_close_available": True,
                "price_source": "Adj_Close",
                "warning": "",
                "max_abs_adj_close_diff_pct": max_abs_diff * 100.0,
                "mean_abs_adj_close_diff_pct": mean_abs_diff * 100.0,
                "rows_with_adj_close": int(valid.sum()),
            }
            if max_abs_diff > 1e-6:
                print(
                    "  [DATA] Adj_Close bulundu; hedef ve teknik feature hesaplari "
                    f"duzeltilmis kapanis uzerinden yapilacak (max fark={max_abs_diff:.4%})."
                )
            df["Close"] = adj_close.combine_first(raw_close)
        df.drop(columns=["Adj_Close"], inplace=True)
    else:
        print("  [DATA] Adj_Close bulunamadi; corporate action duzeltmesi dogrulanamadi.")

    if drop_zero_volume:
        df = df[df["Volume"] > 0].copy()
        df.reset_index(drop=True, inplace=True)

    df.attrs["corporate_action_report"] = corporate_action_report
    return df


def add_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Teknik gösterge ve gecikme özelliklerini ekler.

    v2 (H2 düzeltmesi) — stasyonerleştirilmiş sürüm:
      Eski sürüm fiyat-birimi seviyeleri (SMA_20, EMA_12, MACD, BB_Upper/Lower,
      ATR, Close_Lag_*) saklıyordu. Trend piyasasında bu kolonlar test
      döneminde MinMax/Robust skalanın dışına fırlıyor, modeli OOD besliyordu.
      Şimdi hepsi "Close'a göre oran" veya "% bant" olarak tutuluyor.

    Eklenen özellikler:
        • RSI (14)                  — zaten 0-100, stasyoner
        • SMA_20_rel, EMA_12_rel    — Close / MA − 1  (mean-reversion sinyali)
        • BB_Upper_rel, BB_Lower_rel, BB_Width_20
        • ATR_norm                  — ATR / Close (volatilite %'i)
        • Stoch_K, Stoch_D          — zaten 0-100
        • LogRet_Lag_1 … LogRet_Lag_{lags}   (Close_Lag yerine stasyoner lag)

    Not: MACD artık FeaturePipeline tarafından MACD_norm / MACD_Signal_norm /
    MACD_Diff_norm olarak sağlanıyor → duplikasyonu önlemek için buradan
    kaldırıldı.

    Parameters
    ----------
    df : pd.DataFrame
        Temizlenmiş OHLCV verisi (Close, High, Low, Open, Volume sütunları).
    lags : int
        Gecikme (lag) sayısı.

    Returns
    -------
    pd.DataFrame
        Özellik mühendisliği uygulanmış, NaN satırları düşürülmüş DataFrame.
    """
    df = df.copy()
    close = df["Close"]
    close_safe = close.replace(0, np.nan)

    # ── Stasyoner teknik göstergeler ─────────────────────────────────────────
    df["RSI"] = ta.momentum.rsi(close, window=14)

    # MA → Close'a göre göreli (0 civarında dalgalanır)
    sma_20 = ta.trend.sma_indicator(close, window=20)
    ema_12 = ta.trend.ema_indicator(close, window=12)
    df["SMA_20_rel"] = close / sma_20.replace(0, np.nan) - 1.0
    df["EMA_12_rel"] = close / ema_12.replace(0, np.nan) - 1.0

    # Bollinger Bantları — bantlara göre göreli konum + bant genişliği
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_up  = bb.bollinger_hband()
    bb_lo  = bb.bollinger_lband()
    df["BB_Upper_rel"] = close / bb_up.replace(0, np.nan) - 1.0
    df["BB_Lower_rel"] = close / bb_lo.replace(0, np.nan) - 1.0
    df["BB_Width_20"]  = bb.bollinger_wband()   # zaten %, stasyoner

    # ATR: volatilite fiyat birimi → Close'a böl → ≈ günlük volatilite katsayısı
    atr = ta.volatility.average_true_range(df["High"], df["Low"], close, window=14)
    df["ATR_norm"] = atr / close_safe

    # Stochastic — zaten 0-100 aralığında, stasyoner
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], close, window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # ── Lag özellikleri: log-getiri lag'leri (Close lag'leri non-stationary'di)
    # LogRet_Lag_i[t] = log(Close[t-i+1] / Close[t-i]) = geçmiş i. günün getirisi
    log_ret = np.log(close / close.shift(1))
    for i in range(1, lags + 1):
        df[f"LogRet_Lag_{i}"] = log_ret.shift(i)   # i gün öncesinin getirisi

    # ── NaN satırları temizle ────────────────────────────────────────────────
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Kolaylık fonksiyonu: veriyi oku ve temizle.

    Parameters
    ----------
    csv_path : str
        Ham CSV yolu.

    Returns
    -------
    pd.DataFrame
        Ham ama temizlenmiş OHLCV DataFrame.
    """
    return load_and_clean(csv_path)
