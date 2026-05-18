# -*- coding: utf-8 -*-
"""Veri tazelik kontrolü.

BIST takvimini kullanarak son gözlem tarihinin kaç işlem günü geride
kaldığını hesaplar ve ``data_freshness`` status'ünü döner.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import NamedTuple, Optional

import pandas as pd

from src.api.constants import DATA_STALENESS_MAX_TRADING_DAYS

_DEFAULT_CALENDAR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "meta", "bist_calendar.csv"
)


class FreshnessResult(NamedTuple):
    status: str  # "fresh" | "stale_data"
    staleness_days: int


def _load_trading_days(calendar_path: Optional[str]) -> set[date]:
    """CSV takvimden işlem günlerini yükle. Dosya yoksa boş küme döner."""
    path = calendar_path or _DEFAULT_CALENDAR_PATH
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        mask = df.get("Is_Trading_Day", pd.Series(dtype=bool)).astype(bool)
        return {pd.Timestamp(d).date() for d in df.loc[mask, "Date"]}
    except Exception:
        return set()


def _count_trading_days_between(start: date, end: date, trading_days: set[date]) -> int:
    """start'tan sonraki ve end'e kadar (end dahil) olan işlem günü sayısı."""
    if start >= end:
        return 0
    count = 0
    current = start + pd.Timedelta(days=1)
    current = current.to_pydatetime().date() if hasattr(current, "to_pydatetime") else current
    end_date = end
    from datetime import timedelta

    d = current
    while d <= end_date:
        if trading_days:
            if d in trading_days:
                count += 1
        else:
            if d.weekday() < 5:
                count += 1
        d += timedelta(days=1)
    return count


def compute_freshness(
    last_observed_date: str,
    today: Optional[str] = None,
    calendar_path: Optional[str] = None,
    max_trading_days: int = DATA_STALENESS_MAX_TRADING_DAYS,
) -> FreshnessResult:
    """Veri tazelik durumu hesapla.

    Parameters
    ----------
    last_observed_date:
        Son gözlem tarihi (YYYY-MM-DD).
    today:
        Karşılaştırma tarihi; None ise bugün.
    calendar_path:
        BIST takvim CSV dosyası yolu; None ise varsayılan.
    max_trading_days:
        Bu işlem günü sayısının üzerindeyse ``stale_data``.
    """
    try:
        last_date = pd.Timestamp(last_observed_date).date()
    except Exception:
        return FreshnessResult(status="stale_data", staleness_days=-1)

    if today is None:
        today_date = datetime.now().date()
    else:
        try:
            today_date = pd.Timestamp(today).date()
        except Exception:
            today_date = datetime.now().date()

    if last_date >= today_date:
        return FreshnessResult(status="fresh", staleness_days=0)

    trading_days = _load_trading_days(calendar_path)
    staleness = _count_trading_days_between(last_date, today_date, trading_days)

    if staleness <= max_trading_days:
        return FreshnessResult(status="fresh", staleness_days=staleness)
    return FreshnessResult(status="stale_data", staleness_days=staleness)
