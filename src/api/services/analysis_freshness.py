# -*- coding: utf-8 -*-
"""Veri tazelik kontrolu."""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import NamedTuple, Optional

import pandas as pd

from src.api.constants import DATA_STALENESS_MAX_TRADING_DAYS

logger = logging.getLogger(__name__)

_DEFAULT_CALENDAR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "meta", "bist_calendar.csv"
)


_ISTANBUL = timezone(timedelta(hours=3))  # Turkiye sabit UTC+3 (DST yok)


class FreshnessResult(NamedTuple):
    status: str  # "fresh" | "stale_data"
    staleness_days: int
    warning: str = ""


class TradingCalendar(NamedTuple):
    days: set[date]
    min_date: date | None
    max_date: date | None


def _load_trading_calendar(calendar_path: Optional[str]) -> TradingCalendar:
    """CSV takvimden islem gunlerini yukle. Dosya yoksa bos takvim doner."""
    path = calendar_path or _DEFAULT_CALENDAR_PATH
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        mask = df.get("Is_Trading_Day", pd.Series(dtype=bool)).astype(bool)
        all_dates = [pd.Timestamp(d).date() for d in df["Date"]]
        return TradingCalendar(
            days={pd.Timestamp(d).date() for d in df.loc[mask, "Date"]},
            min_date=min(all_dates) if all_dates else None,
            max_date=max(all_dates) if all_dates else None,
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("BIST takvimi okunamadi (%s): %s; hafta-ici yedegi kullanilacak", path, exc)
        return TradingCalendar(days=set(), min_date=None, max_date=None)


def _count_trading_days_between(start: date, end: date, trading_days: set[date]) -> int:
    """start'tan sonraki ve end'e kadar olan islem gunu sayisi."""
    if start >= end:
        return 0
    count = 0
    d = start + timedelta(days=1)
    while d <= end:
        if trading_days:
            if d in trading_days:
                count += 1
        elif d.weekday() < 5:
            count += 1
        d += timedelta(days=1)
    return count


def compute_freshness(
    last_observed_date: str,
    today: Optional[str] = None,
    calendar_path: Optional[str] = None,
    max_trading_days: int = DATA_STALENESS_MAX_TRADING_DAYS,
) -> FreshnessResult:
    """Veri tazelik durumu hesapla."""
    try:
        last_date = pd.Timestamp(last_observed_date).date()
    except Exception:
        return FreshnessResult(status="stale_data", staleness_days=-1)

    if today is None:
        today_date = datetime.now(_ISTANBUL).date()
    else:
        try:
            today_date = pd.Timestamp(today).date()
        except Exception:
            today_date = datetime.now(_ISTANBUL).date()

    if last_date >= today_date:
        return FreshnessResult(status="fresh", staleness_days=0)

    staleness, warning = _staleness_between(last_date, today_date, calendar_path)
    status = "fresh" if staleness <= max_trading_days else "stale_data"
    return FreshnessResult(status=status, staleness_days=staleness, warning=warning)


def _staleness_between(start: date, end: date, calendar_path: Optional[str]) -> tuple[int, str]:
    """(start, end] araligindaki BIST islem gunu sayisi + takvim kapsama uyarisi.

    Takvim araligi kapsamiyorsa / okunamazsa hafta-ici yedegi kullanilir (uyari doner).
    """
    calendar = _load_trading_calendar(calendar_path)
    warning = ""
    trading_days = calendar.days
    if calendar.min_date is not None and calendar.max_date is not None:
        if start < calendar.min_date or end > calendar.max_date:
            warning = (
                "BIST calendar range does not fully cover the freshness window; "
                "weekday fallback was used."
            )
            trading_days = set()
    return _count_trading_days_between(start, end, trading_days), warning


def trading_days_between(
    start: str | date, end: str | date, calendar_path: Optional[str] = None
) -> int:
    """`start` sonrasi ile `end` dahil arasindaki BIST islem gunu sayisi (start >= end -> 0).

    Gecersiz tarih -> ValueError. Gecelik havuz `as_of_date` ile piyasa CSV son tarihi
    arasindaki geriligi olcmek icin kullanilir.
    """
    a = pd.Timestamp(start).date()
    b = pd.Timestamp(end).date()
    return _staleness_between(a, b, calendar_path)[0]
