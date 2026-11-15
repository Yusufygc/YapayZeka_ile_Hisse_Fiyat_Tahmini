# -*- coding: utf-8 -*-
"""Veri tazelik (stale) kontrolu: egitim/serving bayat veriyle sessizce calismasin.

`check_freshness(last_date)` son veri tarihinin `today`'den kac IS GUNU geride
oldugunu hesaplar (hafta sonlari sayilmaz; resmi tatiller sayilir -> kasitli
olarak muhafazakar-gevsek degil, hafif siki: tatil haftasi 1-2 gun fazla gorunebilir,
bu yuzden varsayilan esik 5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MAX_BUSINESS_DAYS = 5
_ISTANBUL = timezone(timedelta(hours=3))  # Turkiye sabit UTC+3 (DST yok)


@dataclass(frozen=True)
class FreshnessResult:
    stale: bool
    lag_business_days: int | None   # None: son tarih bilinmiyor
    message: str
    last_date: pd.Timestamp | None = None


def _today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(_ISTANBUL).date())


def check_freshness(
    last_date: Any,
    today: Any = None,
    max_business_days: int = DEFAULT_MAX_BUSINESS_DAYS,
    label: str = "veri",
) -> FreshnessResult:
    """Son veri tarihi `today`'e gore bayat mi?

    Args:
        last_date: Son gozlem tarihi (Timestamp/date/str). NaT/None -> stale.
        today: Referans gun (varsayilan: Istanbul bugunu).
        max_business_days: Bundan fazla is gunu gerideyse stale.
    """
    last = pd.NaT if last_date is None else pd.Timestamp(last_date)
    if pd.isna(last):
        return FreshnessResult(True, None, f"{label}: son tarih bilinmiyor (bayat kabul edildi)")
    last = last.normalize()
    ref = (_today() if today is None else pd.Timestamp(today)).normalize()
    if last >= ref:
        lag = 0
    else:
        lag = int(np.busday_count(last.date(), ref.date()))
    stale = lag > int(max_business_days)
    state = "BAYAT" if stale else "taze"
    msg = (f"{label}: son tarih {last.date()}, {lag} is gunu geride "
           f"(esik {max_business_days}) -> {state}")
    return FreshnessResult(stale, lag, msg, last)


def stale_symbols(
    last_dates: Mapping[str, Any],
    today: Any = None,
    max_business_days: int = DEFAULT_MAX_BUSINESS_DAYS,
) -> dict[str, FreshnessResult]:
    """{sembol: son_tarih} -> yalniz bayat olanlarin sonuclari."""
    out = {}
    for sym, last in last_dates.items():
        res = check_freshness(last, today, max_business_days, label=str(sym))
        if res.stale:
            out[str(sym)] = res
    return out


def warn_if_stale(
    last_dates: Mapping[str, Any],
    context: str,
    today: Any = None,
    max_business_days: int = DEFAULT_MAX_BUSINESS_DAYS,
) -> dict[str, FreshnessResult]:
    """Bayat sembolleri tek ozet uyari ile logla; bayat sonuclari dondur."""
    stale = stale_symbols(last_dates, today, max_business_days)
    if stale:
        worst = max(stale.values(), key=lambda r: r.lag_business_days or 0)
        sample = ", ".join(sorted(stale)[:5])
        logger.warning(
            "%s: %d/%d sembol bayat (>%d is gunu); en kotu: %s; ornek: %s",
            context, len(stale), len(last_dates), max_business_days, worst.message, sample)
    return stale
