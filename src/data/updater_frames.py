# -*- coding: utf-8 -*-
"""updater_frames.py - DataUpdater icin saf DataFrame/tarih yardimcilari."""

from __future__ import annotations

from typing import Any

import pandas as pd

_OVERLAP_DAYS = 7
_OVERLAP_TOLERANCE = 0.005  # |yeni/eski - 1| > %0.5 -> yeniden olcekleme suphesi


def _prepare_provider_rows(
    new_data: pd.DataFrame, columns: Any, date_col: str, max_date: pd.Timestamp
) -> pd.DataFrame:
    """Saglayici ciktisini CSV semasina cevirir; Date > son tamamlanmis seans satirlari atilir.

    Seans kapanmadan gelen bugunluk (kismi intraday) mum boylece CSV'ye yazilmaz.
    """
    frame = _normalize_yfinance_frame(new_data)
    rows = _map_to_existing_schema(frame, columns, date_col)
    rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=[date_col])
    rows = rows[rows[date_col] <= pd.Timestamp(max_date).normalize()]
    rows = rows.drop_duplicates(subset=[date_col], keep="last").sort_values(date_col)
    return rows.reset_index(drop=True)


def _max_overlap_deviation(
    df_raw: pd.DataFrame,
    rows_all: pd.DataFrame,
    date_col: str,
    compare_col: str | None,
    last_date: pd.Timestamp,
) -> float | None:
    """Ust uste binen tarihlerde max |yeni/eski - 1|; karsilastirilabilir satir yoksa None."""
    if compare_col is None or compare_col not in rows_all.columns:
        return None
    old = pd.DataFrame({
        "d": _parse_date_series(df_raw[date_col]).dt.normalize(),
        "old": pd.to_numeric(df_raw[compare_col], errors="coerce"),
    }).dropna()
    old = old[old["d"] >= last_date - pd.Timedelta(days=_OVERLAP_DAYS)].drop_duplicates("d", keep="last")
    new = pd.DataFrame({
        "d": rows_all[date_col],
        "new": pd.to_numeric(rows_all[compare_col], errors="coerce"),
    }).dropna()
    merged = old.merge(new, on="d")
    merged = merged[(merged["old"] > 0) & (merged["new"] > 0)]
    if merged.empty:
        return None
    return float((merged["new"] / merged["old"] - 1.0).abs().max())


def _find_column(columns: Any, candidates: list[str]) -> str | None:
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def _parse_date_series(values: pd.Series) -> pd.Series:
    """Tarih kolonunu ayristirir (yil-ilk ise dayfirst=False, degilse True); NaT'ler atilir."""
    vals_clean = values.dropna()
    if vals_clean.empty:
        return pd.Series([], dtype="datetime64[ns]")
    sample = str(vals_clean.iloc[0]).strip()
    is_year_first = False
    for separator in ["-", "/", "."]:
        if separator in sample:
            parts = sample.split(separator)
            if parts and len(parts[0]) == 4:
                is_year_first = True
                break
    parsed = pd.to_datetime(vals_clean, errors="coerce", dayfirst=not is_year_first)
    return parsed.dropna()


def _parse_last_date(values: pd.Series) -> pd.Timestamp:
    parsed = _parse_date_series(values)
    return pd.NaT if parsed.empty else pd.Timestamp(parsed.max())


def _normalize_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    data.columns = [str(col).replace("Datetime", "Date") for col in data.columns]
    return data


def _map_to_existing_schema(new_data: pd.DataFrame, existing_columns: Any, date_col: str) -> pd.DataFrame:
    source_map = {
        "date": "Date",
        "tarih": "Date",
        "open": "Open",
        "açılış": "Open",
        "acilis": "Open",
        "high": "High",
        "yüksek": "High",
        "yuksek": "High",
        "low": "Low",
        "düşük": "Low",
        "dusuk": "Low",
        "close": "Close",
        "kapanış": "Close",
        "kapanis": "Close",
        "adj_close": "Adj Close",
        "adj close": "Adj Close",
        "düzeltilmiş_kapanış": "Adj Close",
        "duzeltilmis_kapanis": "Adj Close",
        "volume": "Volume",
        "hacim": "Volume",
    }
    available = {str(col).strip().lower(): str(col) for col in new_data.columns}
    result = pd.DataFrame(index=new_data.index)
    for target_col in existing_columns:
        key = str(target_col).strip().lower()
        source_name = source_map.get(key)
        source_col = available.get(str(source_name).lower()) if source_name else None
        if source_col is None and source_name == "Adj Close":
            source_col = available.get("close")
        if source_col is None:
            result[target_col] = pd.NA
        else:
            result[target_col] = new_data[source_col]
    if date_col in result.columns:
        result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    return result


def _infer_date_format(sample: str) -> str:
    sample = sample.strip()
    if "-" in sample:
        return "%Y-%m-%d" if len(sample.split("-")[0]) == 4 else "%d-%m-%Y"
    if "/" in sample:
        return "%Y/%m/%d" if len(sample.split("/")[0]) == 4 else "%d/%m/%Y"
    if "." in sample:
        return "%Y.%m.%d" if len(sample.split(".")[0]) == 4 else "%d.%m.%Y"
    return "%Y-%m-%d"
