# -*- coding: utf-8 -*-
"""Ortak tarih ayristirici (tek dogruluk kaynagi).

Sembol CSV'lerinde ayni dosyada hem ISO (`2024-01-02`) hem `dd/mm/yyyy` /
`dd.mm.yyyy` tarihleri bulunabilir. `pd.to_datetime(..., dayfirst=True)` ISO
degerleri sessizce takas eder (2024-01-02 -> 2024-02-01); dayfirst=False ise
dd/mm degerlerini bozar. Bu modul her degeri BICIMINE gore ayirir:

  * `YYYY-MM-DD[...]` (ayrac - / .)  -> yil-ilk
  * `dd/mm/yyyy`, `dd.mm.yyyy`, `dd-mm-yyyy` -> gun-ilk
  * Timestamp / datetime / datetime64 -> aynen gecer
  * diger metinler -> pandas varsayilani (belirsiz degilse)

Karisik seriler eleman-gruplari halinde ayristirilir; hicbir deger sessizce
takas edilmez.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

_YEAR_FIRST = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:$|[T\s])")
_DAY_FIRST = re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}(?:$|[T\s])")


def _as_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    if isinstance(values, (pd.Index, list, tuple)) or hasattr(values, "__array__"):
        return pd.Series(list(values) if isinstance(values, (list, tuple)) else values)
    return pd.Series([values])  # skaler


def parse_dates(values: Any, errors: str = "coerce") -> pd.Series:
    """Tarihleri biciminden bagimsiz, takas olmadan ayristirir.

    Args:
        values: Series / Index / liste / skaler (str, Timestamp, datetime).
        errors: "coerce" (gecersiz -> NaT) veya "raise" (ValueError).

    Returns:
        datetime64[ns] Series (girdi indeksi korunur). `pd.DatetimeIndex(result)`
        ile indekse cevrilebilir.
    """
    if errors not in ("coerce", "raise"):
        raise ValueError("errors yalniz 'coerce' veya 'raise' olabilir")
    raw = _as_series(values)
    if pd.api.types.is_datetime64_any_dtype(raw.dtype):
        out = raw.copy()
        return out.dt.tz_localize(None) if getattr(out.dt, "tz", None) is not None else out

    text = raw.where(raw.map(lambda v: isinstance(v, str)), other=None)
    stripped = text.map(lambda v: v.strip() if isinstance(v, str) else None)
    is_str = stripped.notna()
    year_first = stripped.str.match(_YEAR_FIRST, na=False)
    day_first = stripped.str.match(_DAY_FIRST, na=False) & ~year_first
    other = ~(year_first | day_first)

    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    groups = (
        (year_first, {"yearfirst": True, "dayfirst": False}, stripped),
        (day_first, {"dayfirst": True}, stripped),
        (other & is_str, {}, stripped),
        (other & ~is_str & raw.notna(), {}, raw),
    )
    for mask, kwargs, source in groups:
        if not mask.any():
            continue
        part = pd.to_datetime(source[mask].tolist(), errors=errors, format="mixed", **kwargs)
        part = pd.Series(pd.DatetimeIndex(part), index=raw.index[mask])
        if getattr(part.dt, "tz", None) is not None:
            part = part.dt.tz_localize(None)
        parsed.loc[mask] = part.astype("datetime64[ns]")
    parsed.name = raw.name
    return parsed
