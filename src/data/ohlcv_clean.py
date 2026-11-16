# -*- coding: utf-8 -*-
"""Sembol OHLCV CSV'leri icin ORTAK okuma/temizleme (tek dogruluk kaynagi).

`src/data/pooled_loader.py` (panel), `src/data/clean_store.py` ve `/analysis` piyasa-sonu
okumasi ayni adimlari kullanir; boylece yollar arasinda hedef/ozellik sapmasi olmaz:

  1. `read_ohlcv_csv`  : satir basina alan sayisi dogrulanir. Baslikla ayni sayida
     alani olmayan (orn. Adj Close eksik 6 alanli) satir KARANTINAYA alinir ve
     uyari loglanir; kolonlar ASLA kaydirilmaz (Hacim -> Close kontaminasyonu yok).
  2. `dedupe_sort_by_date` : `parse_dates` ile tarih ayristirma, gecersiz tarihleri
     at, tarihe gore sirala, tekrar eden tarihlerde SON satiri tut.
  3. `apply_adj_close_adjustment` : Open/High/Low'u Adj/Close orani ile olcekle,
     Close = Adj_Close (yoksa ham Close).
  4. `drop_non_trading_rows` : Hacim <= 0 / NaN (borsa kapali) satirlarini at.
"""
from __future__ import annotations

import csv
import logging

import numpy as np
import pandas as pd

from src.data.dates import parse_dates

logger = logging.getLogger(__name__)

# TR / alternatif basliklar -> standart (Adj_Close ayri kolon olarak korunur).
COLUMN_MAP = {
    "Tarih": "Date",
    "Açılış": "Open",
    "Yüksek": "High",
    "Düşük": "Low",
    "Kapanış": "Close",
    "Düzeltilmiş_Kapanış": "Adj_Close",
    "Adj Close": "Adj_Close",
    "Hacim": "Volume",
}
NUMERIC_COLS = ("Open", "High", "Low", "Close", "Adj_Close", "Volume")


def read_ohlcv_csv(path: str, label: str | None = None) -> pd.DataFrame:
    """CSV'yi satir-alan-sayisi dogrulamasiyla oku, basliklari standartlastir.

    Alan sayisi basliktan farkli satirlar atilir (karantina); sayi
    `df.attrs["rejected_rows"]` icinde ve uyari logunda raporlanir. Sayisal
    kolonlar `to_numeric(errors="coerce")` ile cevrilir; Date ham metin kalir.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if not rows:
        return pd.DataFrame()
    header = [h.strip() for h in rows[0]]
    good = [r for r in rows[1:] if len(r) == len(header)]
    rejected = len(rows) - 1 - len(good)
    if rejected:
        logger.warning(
            "%s: %d satir alan sayisi basliktan farkli (%d bekleniyor); karantinaya alindi, "
            "kolon kaydirma yapilmadi (tools/repair_symbol_csv.py ile onarin).",
            label or path, rejected, len(header))
    df = pd.DataFrame(good, columns=header).rename(columns=COLUMN_MAP)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.attrs["rejected_rows"] = rejected
    return df


def dedupe_sort_by_date(df: pd.DataFrame, label: str | None = None) -> pd.DataFrame:
    """Date'i `parse_dates` ile ayristir, NaT'leri at, sirala, tekrar tarihlerde SON satiri tut."""
    out = df.copy()
    out["Date"] = parse_dates(out["Date"])
    n_bad = int(out["Date"].isna().sum())
    if n_bad:
        logger.warning("%s: %d satirda gecersiz tarih atildi.", label or "ohlcv", n_bad)
        out = out.dropna(subset=["Date"])
    out = out.sort_values("Date", kind="stable")
    n_before = len(out)
    out = out.drop_duplicates(subset="Date", keep="last")
    if len(out) != n_before:
        logger.warning("%s: %d tekrar eden tarih atildi (son satir tutuldu).",
                       label or "ohlcv", n_before - len(out))
    return out.reset_index(drop=True)


def apply_adj_close_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Open/High/Low'u Adj_Close/Close orani ile olcekle; Close = Adj_Close (yoksa ham).

    Faktor yalniz Adj ve Close sonlu, Close != 0 ve Adj > 0 ise uygulanir, aksi halde 1.
    Hacim olceklenmez. `Adj_Close` kolonu birakilir (cagiran dusurur). `Adj_Close`
    yoksa df aynen doner.
    """
    if "Adj_Close" not in df.columns or "Close" not in df.columns:
        return df
    out = df.copy()
    raw_close = pd.to_numeric(out["Close"], errors="coerce")
    adj_close = pd.to_numeric(out["Adj_Close"], errors="coerce")
    valid = raw_close.notna() & adj_close.notna() & (raw_close != 0)
    factor = (adj_close / raw_close).where(valid & (adj_close > 0), 1.0)
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    for col in ("Open", "High", "Low"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * factor
    out["Close"] = adj_close.combine_first(raw_close)
    return out


def drop_non_trading_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Hacmi 0/NaN (borsa kapali) satirlari at; Volume kolonu yoksa df aynen doner."""
    if "Volume" not in df.columns:
        return df
    return df[pd.to_numeric(df["Volume"], errors="coerce") > 0].reset_index(drop=True)


def clean_ohlcv_csv(path: str, label: str | None = None, drop_zero_volume: bool = True) -> pd.DataFrame:
    """Panel yolu icin tek cagri: oku -> tarih/dedupe -> Adj olcekleme -> sifir-hacim at."""
    df = read_ohlcv_csv(path, label=label)
    if df.empty or "Date" not in df.columns:
        return df
    rejected = df.attrs.get("rejected_rows", 0)
    df = dedupe_sort_by_date(df, label=label)
    df = apply_adj_close_adjustment(df).drop(columns=["Adj_Close"], errors="ignore")
    if drop_zero_volume:
        df = drop_non_trading_rows(df)
    df.attrs["rejected_rows"] = rejected
    return df
