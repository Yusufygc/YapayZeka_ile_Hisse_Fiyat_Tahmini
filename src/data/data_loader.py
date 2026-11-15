# -*- coding: utf-8 -*-
"""
data_loader.py — Veri Yükleme ve Özellik Mühendisliği
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ham OHLCV CSV dosyasını okur, Türkçe sütun adlarını İngilizceye çevirir,
piyasanın kapalı olduğu (hacim=0) günleri eler ve teknik göstergeler
(RSI, MACD, SMA, EMA) ile gecikme (lag) özellikleri ekler.
"""

import os

import pandas as pd
import numpy as np


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

_ADJ_CLOSE_ANOMALY_THRESHOLD_PCT = 1000.0
_ADJ_CLOSE_TRUST_FULL_DIFF_PCT = _ADJ_CLOSE_ANOMALY_THRESHOLD_PCT


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
        "adjusted_price_trusted": False,
        "adjusted_price_trust_score": 0.0,
        "corporate_action_anomaly": False,
        "anomaly_threshold_pct": _ADJ_CLOSE_ANOMALY_THRESHOLD_PCT,
    }

    if "Adj_Close" in df.columns and df["Adj_Close"].notna().any():
        raw_close = pd.to_numeric(df["Close"], errors="coerce")
        adj_close = pd.to_numeric(df["Adj_Close"], errors="coerce")
        valid = raw_close.notna() & adj_close.notna() & (raw_close != 0)
        if valid.any():
            diff_ratio = (adj_close[valid] / raw_close[valid]) - 1.0
            max_abs_diff = float(diff_ratio.abs().max())
            mean_abs_diff = float(diff_ratio.abs().mean())
            max_abs_diff_pct = max_abs_diff * 100.0
            mean_abs_diff_pct = mean_abs_diff * 100.0
            invalid_adj_values = bool((adj_close[valid] <= 0).any())
            corporate_action_anomaly = (
                invalid_adj_values
                or max_abs_diff_pct > _ADJ_CLOSE_ANOMALY_THRESHOLD_PCT
            )
            trust_score = max(
                0.0,
                1.0 - min(mean_abs_diff_pct, _ADJ_CLOSE_TRUST_FULL_DIFF_PCT)
                / _ADJ_CLOSE_TRUST_FULL_DIFF_PCT,
            )

            corporate_action_report = {
                "adj_close_available": True,
                "price_source": "Close" if corporate_action_anomaly else "Adj_Close",
                "warning": "",
                "max_abs_adj_close_diff_pct": max_abs_diff_pct,
                "mean_abs_adj_close_diff_pct": mean_abs_diff_pct,
                "rows_with_adj_close": int(valid.sum()),
                "adjusted_price_trusted": not corporate_action_anomaly,
                "adjusted_price_trust_score": 0.0 if corporate_action_anomaly else round(trust_score, 6),
                "corporate_action_anomaly": corporate_action_anomaly,
                "anomaly_threshold_pct": _ADJ_CLOSE_ANOMALY_THRESHOLD_PCT,
                "invalid_adj_close_values": invalid_adj_values,
            }
            if corporate_action_anomaly:
                corporate_action_report["warning"] = (
                    "Adj_Close anomalisi tespit edildi; model hedefi ve teknik feature "
                    "hesaplari ham Close uzerinden yapilacak."
                )
                print(
                    "  [DATA] Adj_Close anomalisi tespit edildi; "
                    f"ham Close kullanilacak (max fark={max_abs_diff:.4%})."
                )
            else:
                if max_abs_diff > 1e-6:
                    print(
                        "  [DATA] Adj_Close bulundu; hedef ve teknik feature hesaplari "
                        f"duzeltilmis kapanis uzerinden yapilacak (max fark={max_abs_diff:.4%})."
                    )
                # Close = Adj_Close iken Open/High/Low ham kalirsa OHLC tutarsiz olur
                # (Low > Close vb.). Ayni faktorle (Adj/Close) olceklenir; faktor
                # yalniz her ikisi de sonlu ve > 0 ise uygulanir, aksi halde 1. Hacim olceklenmez.
                factor = (adj_close / raw_close).where(valid & (adj_close > 0), 1.0)
                factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                for ohl_col in ("Open", "High", "Low"):
                    if ohl_col in df.columns:
                        df[ohl_col] = pd.to_numeric(df[ohl_col], errors="coerce") * factor
                df["Close"] = adj_close.combine_first(raw_close)
        else:
            corporate_action_report.update({
                "adj_close_available": True,
                "warning": "Adj_Close bulundu ancak gecerli sayisal deger yok; ham Close kullanilacak.",
                "corporate_action_anomaly": True,
                "invalid_adj_close_values": True,
            })
            print(
                "  [DATA] Adj_Close bulundu ancak gecerli sayisal deger yok; "
                "ham Close kullanilacak."
            )
        df.drop(columns=["Adj_Close"], inplace=True)
    else:
        print("  [DATA] Adj_Close bulunamadi; corporate action duzeltmesi dogrulanamadi.")

    if drop_zero_volume:
        df = df[df["Volume"] > 0].copy()
        df.reset_index(drop=True, inplace=True)

    df.attrs["corporate_action_report"] = corporate_action_report
    # Sprint 2 (2026-05-25) Plan A2.3: survivorship_bias_report.
    df.attrs["survivorship_bias_report"] = _compute_survivorship_report(df, csv_path)
    return df


def _compute_survivorship_report(df: pd.DataFrame, csv_path: str) -> dict:
    """
    Sprint 2 (2026-05-25) Plan A2.3:
      OHLCV serisinden survivorship/listing tabanli risk bayraklarini cikarir.
      `expected_start` mevcut sistemde bilinmiyor (BIST listing-date metadata
      yok); bu yuzden actual_start, span_days, max_gap_days uretilir ve
      kisalik (<2 yil) veya gun-bos boyu (>10 gun) varsa warning eklenir.

    Return shape:
      {
        "symbol": "TUPRS",
        "actual_start": "2015-05-01",
        "actual_end": "2026-05-23",
        "span_days": 4036,
        "row_count": 2715,
        "max_gap_days": 4,
        "short_history_warning": False,
        "delisted_or_late_listing_warning": True,
        "warning": "delisted_or_late_listing" | "" ,
      }
    """
    try:
        symbol = os.path.splitext(os.path.basename(csv_path))[0].upper()
    except Exception:
        symbol = "UNKNOWN"

    report = {
        "symbol": symbol,
        "actual_start": None,
        "actual_end": None,
        "span_days": 0,
        "row_count": int(len(df)),
        "max_gap_days": 0,
        "short_history_warning": False,
        "delisted_or_late_listing_warning": False,
        "warning": "",
    }
    if "Date" not in df.columns or df.empty:
        return report
    try:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna().sort_values()
        if len(dates) < 2:
            return report
        start, end = dates.iloc[0], dates.iloc[-1]
        span_days = int((end - start).days)
        gaps = dates.diff().dt.days.dropna()
        max_gap = int(gaps.max()) if len(gaps) else 0
        short_history = span_days < int(365 * 2)
        late_listing = max_gap > 10

        report.update({
            "actual_start": start.strftime("%Y-%m-%d"),
            "actual_end": end.strftime("%Y-%m-%d"),
            "span_days": span_days,
            "max_gap_days": max_gap,
            "short_history_warning": short_history,
            "delisted_or_late_listing_warning": late_listing,
            "warning": (
                "delisted_or_late_listing" if late_listing
                else ("short_history" if short_history else "")
            ),
        })
    except Exception:
        # Sessizce sukut etme: kalan akis kalite kontrolu yapmadan devam etmesin.
        pass
    return report


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
