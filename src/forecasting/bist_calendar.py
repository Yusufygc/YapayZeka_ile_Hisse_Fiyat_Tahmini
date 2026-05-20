# -*- coding: utf-8 -*-
"""BIST calendar maintenance helpers.

The project keeps a local CSV calendar because common market-calendar packages
do not consistently ship a ready-made BIST/XIST calendar.  This module provides
a deterministic local calendar with official fixed-date closures and a rolling
range.  Religious holidays can be added to the CSV later without changing the
consumer API.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Iterable

import pandas as pd

try:  # Optional dependency requested by the serving plan.
    import pandas_market_calendars as _pmc  # noqa: F401
except Exception:  # pragma: no cover - dependency may be absent in light envs
    _pmc = None


_FIXED_CLOSURES = {
    (1, 1): "New Year's Day",
    (4, 23): "National Sovereignty and Children's Day",
    (5, 1): "Labour and Solidarity Day",
    (5, 19): "Commemoration of Ataturk Youth and Sports Day",
    (7, 15): "Democracy and National Unity Day",
    (8, 30): "Victory Day",
    (10, 29): "Republic Day",
}


def default_calendar_path(project_root: str) -> str:
    return os.path.join(project_root, "data", "meta", "bist_calendar.csv")


def ensure_bist_calendar(
    calendar_path: str,
    *,
    today: date | None = None,
    years_back: int = 5,
    years_forward: int = 1,
) -> str:
    today = today or date.today()
    start = pd.Timestamp(today).normalize() - pd.DateOffset(years=years_back)
    end = pd.Timestamp(today).normalize() + pd.DateOffset(years=years_forward)
    required_start = start.date()
    required_end = end.date()

    existing = _read_calendar(calendar_path)
    if _covers(existing, required_start, required_end):
        return calendar_path

    generated = _generate_calendar(required_start, required_end)
    if existing is not None and not existing.empty:
        generated = _merge_manual_overrides(generated, existing)

    os.makedirs(os.path.dirname(calendar_path), exist_ok=True)
    generated.to_csv(calendar_path, index=False)
    return calendar_path


def _read_calendar(calendar_path: str) -> pd.DataFrame | None:
    if not os.path.exists(calendar_path):
        return None
    try:
        df = pd.read_csv(calendar_path, parse_dates=["Date"])
    except Exception:
        return None
    required = {"Date", "Is_Trading_Day", "Session_Type", "Note"}
    if df.empty or not required.issubset(df.columns):
        return None
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df.sort_values("Date").reset_index(drop=True)


def _covers(df: pd.DataFrame | None, start: date, end: date) -> bool:
    if df is None or df.empty:
        return False
    return df["Date"].min().date() <= start and df["Date"].max().date() >= end


def _generate_calendar(start: date, end: date) -> pd.DataFrame:
    rows = []
    for ts in pd.date_range(start=start, end=end, freq="D"):
        d = ts.date()
        is_weekday = ts.weekday() < 5
        fixed_note = _FIXED_CLOSURES.get((d.month, d.day))
        is_trading = bool(is_weekday and fixed_note is None)
        rows.append({
            "Date": ts.strftime("%Y-%m-%d"),
            "Is_Trading_Day": is_trading,
            "Session_Type": "full" if is_trading else "closed",
            "Note": "Regular session" if is_trading else (fixed_note or "Weekend"),
        })
    return pd.DataFrame(rows)


def _merge_manual_overrides(generated: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    generated = generated.copy()
    generated["Date"] = pd.to_datetime(generated["Date"]).dt.normalize()
    manual = existing.copy()
    manual["Date"] = pd.to_datetime(manual["Date"]).dt.normalize()
    manual_dates = set(manual["Date"].dt.date)
    generated = generated[~generated["Date"].dt.date.isin(manual_dates)]
    merged = pd.concat([generated, manual], ignore_index=True)
    merged.sort_values("Date", inplace=True)
    merged["Date"] = pd.to_datetime(merged["Date"]).dt.strftime("%Y-%m-%d")
    return merged.reset_index(drop=True)
