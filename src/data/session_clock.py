# -*- coding: utf-8 -*-
"""session_clock.py - BIST seans takvimi ve "son tamamlanmis seans" hesabi.

Kaynak: ``data/meta/bist_calendar.csv`` (Is_Trading_Day sutunu). Dosya yoksa,
okunamazsa ya da sorgulanan tarihi kapsamiyorsa hafta ici gunler (Pzt-Cuma)
islem gunu sayilir. Seans, Europe/Istanbul saatiyle 18:10'dan sonra
"tamamlanmis" kabul edilir; o ana kadar bugunun mumu kismi (intraday) sayilir.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")
SESSION_CLOSE_HOUR = 18
SESSION_CLOSE_MINUTE = 10

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_cache: dict[tuple[str, float], set[pd.Timestamp] | None] = {}


def default_calendar_path() -> str:
    return str(_PROJECT_ROOT / "data" / "meta" / "bist_calendar.csv")


def now_istanbul(now: datetime | None = None) -> datetime:
    """Verilen ani (naive ise Istanbul yerel saati varsayilir) Istanbul'a cevirir."""
    if now is None:
        return datetime.now(ISTANBUL_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=ISTANBUL_TZ)
    return now.astimezone(ISTANBUL_TZ)


def _load_calendar(calendar_path: str | None) -> tuple[set[pd.Timestamp], pd.Timestamp, pd.Timestamp] | None:
    """(islem_gunleri, ilk_tarih, son_tarih) dondurur; dosya yok/bozuksa None."""
    path = calendar_path or default_calendar_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (path, mtime)
    if key not in _cache:
        try:
            df = pd.read_csv(path)
            dates = pd.to_datetime(df["Date"]).dt.normalize()
            flags = df["Is_Trading_Day"].astype(str).str.strip().str.lower().isin({"true", "1"})
            _cache[key] = (set(dates[flags]), dates.min(), dates.max())  # type: ignore[assignment]
        except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
            logger.warning("BIST takvimi okunamadi (%s): %s; hafta ici varsayimi kullaniliyor", path, exc)
            _cache[key] = None
    return _cache[key]  # type: ignore[return-value]


def is_trading_day(day: pd.Timestamp, calendar_path: str | None = None) -> bool:
    day = pd.Timestamp(day).normalize()
    cal = _load_calendar(calendar_path)
    if cal is not None:
        days, first, last = cal
        if first <= day <= last:
            return day in days
    return day.weekday() < 5


def latest_completed_session(
    now: datetime | None = None, calendar_path: str | None = None
) -> pd.Timestamp:
    """Seansi bitmis (kapanis sonrasi) en son islem gunu (naive, normalize)."""
    local = now_istanbul(now)
    today = pd.Timestamp(local.date())
    closed_today = (local.hour, local.minute) >= (SESSION_CLOSE_HOUR, SESSION_CLOSE_MINUTE)
    day = today if closed_today else today - timedelta(days=1)
    for _ in range(15):  # uzun tatil bloklari icin guvenli ust sinir
        if is_trading_day(day, calendar_path):
            return pd.Timestamp(day).normalize()
        day = day - timedelta(days=1)
    return pd.Timestamp(day).normalize()
