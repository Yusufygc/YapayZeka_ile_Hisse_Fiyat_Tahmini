# -*- coding: utf-8 -*-
"""market_providers.py - Piyasa verisi saglayici soyutlamasi (tekli + toplu indirme)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 50
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_S = 2.0
# Ag zaman asimlari: yf.download `timeout` yalniz tekil HTTP istegini sinirlar; thread havuzu /
# DNS / TLS asilmasi tum cagriyi sonsuza dek bekletebilir. Bu yuzden cagri ayrica sert bir
# duvar-saati siniriyla (call_with_deadline) sarilir; asilirsa ProviderTimeout firlatilir.
DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_SINGLE_CALL_DEADLINE_S = 120.0
DEFAULT_BATCH_CALL_DEADLINE_S = 300.0
DEFAULT_REFRESH_DEADLINE_S = 2700.0  # tum toplu tazeleme icin genel sinir (45 dk)
DEADLINE_ERROR = "deadline_exceeded"

_T = TypeVar("_T")


class ProviderTimeout(TimeoutError):
    """Saglayici cagrisi duvar-saati sinirini asti (cagri terk edildi)."""


# Terk edilen (zaman asimina ugrayan) cagri sayaci. yfinance ThreadPoolExecutor thread'leri
# interpreter cikisinda join edilir; asili bir indirme kosu bittikten sonra bile sureci
# canli tutabilir. Gecelik giris noktasi bu sayaca bakip gerekirse os._exit ile sonlanir.
_ABANDONED_LOCK = threading.Lock()
_ABANDONED_CALLS = 0


def abandoned_call_count() -> int:
    """Surecte zaman asimiyla terk edilen saglayici cagrisi sayisi."""
    with _ABANDONED_LOCK:
        return _ABANDONED_CALLS


def reset_abandoned_calls() -> None:
    global _ABANDONED_CALLS
    with _ABANDONED_LOCK:
        _ABANDONED_CALLS = 0


def call_with_deadline(fn: Callable[[], _T], timeout_s: float | None, label: str = "provider") -> _T:
    """``fn()`` en fazla ``timeout_s`` sn bekler; asilirsa ProviderTimeout.

    ``fn`` daemon thread'de kosar; asildiginda thread oldurulemez, terk edilir (daemon oldugu icin
    surec cikisini engellemez). ``timeout_s`` None/<=0 ise sinirsiz dogrudan cagri.
    """
    if timeout_s is None or timeout_s <= 0:
        return fn()
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - cagirana aynen iletilir
            box["error"] = exc

    worker = threading.Thread(target=_run, name=f"market-{label}", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        global _ABANDONED_CALLS
        with _ABANDONED_LOCK:
            _ABANDONED_CALLS += 1
        logger.warning("%s %.0f sn icinde donmedi; cagri terk edildi", label, timeout_s)
        raise ProviderTimeout(f"{label}: {timeout_s:.0f} sn zaman asimi")
    if "error" in box:
        raise box["error"]
    return box["value"]


class MarketDataProvider:
    def download(self, ticker: str, *, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError

    def download_many(self, tickers: Sequence[str], *, start: str, end: str) -> dict[str, pd.DataFrame]:
        """Varsayilan: tek tek indirir. Basarisiz ticker sonuca konmaz (cagiran yeniden dener)."""
        return download_each(self, tickers, start=start, end=end)


def download_each(
    provider: Any, tickers: Sequence[str], *, start: str, end: str
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            out[ticker] = provider.download(ticker, start=start, end=end)
        except Exception as exc:  # saglayici/ag hatasi: ticker atlanir, retry dongusu yeniden dener
            logger.warning("indirme basarisiz %s: %s", ticker, exc)
    return out


class YFinanceProvider(MarketDataProvider):
    """
    Sprint 2 (2026-05-25) Plan A2.1:
      ``auto_adjust=True`` HARD kontrol. yfinance close serisini split ve
      temettu icin otomatik duzeltir. Duzeltilmemis seride 1:2 split
      gunu log_return ≈ -0.69 olur ve tum modeller bozulur. Bu deger
      degistirilmemelidir; testler de bunu denetler. Toplu (batch) yol da
      ayni ``AUTO_ADJUST`` degerini kullanir.
    """

    AUTO_ADJUST: bool = True
    REQUEST_TIMEOUT_S: float = DEFAULT_REQUEST_TIMEOUT_S
    SINGLE_CALL_DEADLINE_S: float = DEFAULT_SINGLE_CALL_DEADLINE_S
    BATCH_CALL_DEADLINE_S: float = DEFAULT_BATCH_CALL_DEADLINE_S

    def download(self, ticker: str, *, start: str, end: str) -> pd.DataFrame:
        # auto_adjust=True ZORUNLU — split/temettu sizintisini onler.
        return call_with_deadline(
            lambda: yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=self.AUTO_ADJUST,
                timeout=self.REQUEST_TIMEOUT_S,
            ),
            self.SINGLE_CALL_DEADLINE_S, f"yf.download({ticker})",
        )

    def download_many(self, tickers: Sequence[str], *, start: str, end: str) -> dict[str, pd.DataFrame]:
        tickers = list(tickers)
        if not tickers:
            return {}
        raw = call_with_deadline(
            lambda: yf.download(
                tickers=tickers,
                start=start,
                end=end,
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=self.AUTO_ADJUST,
                timeout=self.REQUEST_TIMEOUT_S,
            ),
            self.BATCH_CALL_DEADLINE_S, f"yf.download({len(tickers)} ticker)",
        )
        return split_batch_frame(raw, tickers)


def split_batch_frame(raw: pd.DataFrame | None, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    """``group_by='ticker'`` ciktisini ticker->duz OHLCV tablosuna boler (tum-NaN satirlar atilir)."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return {t: pd.DataFrame() for t in tickers}
    multi = isinstance(raw.columns, pd.MultiIndex)
    level0 = set(raw.columns.get_level_values(0)) if multi else set()
    for ticker in tickers:
        if multi and ticker in level0:
            part = raw[ticker]
        elif len(tickers) == 1:
            part = raw.droplevel(-1, axis=1) if multi else raw
        else:
            continue  # yfinance bu ticker'i hic dondurmedi -> basarisiz say
        out[ticker] = part.dropna(how="all")
    return out


def download_with_retries(
    provider: Any,
    tickers: Sequence[str],
    *,
    start: str,
    end: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Ticker'lari chunk'lar halinde indirir; basarisiz/bos olanlari ustel geri cekilmeyle yeniden dener.

    Doner: (frames, errors). ``frames`` icinde bos DataFrame kalan ticker "bos" demektir
    (tum denemelerden sonra); ``errors`` hic yanit alinamayan ticker'larin son hatasidir.
    ``deadline`` (``clock()`` ekseninde mutlak an) asilirsa yeni chunk/deneme baslatilmaz;
    veri alinamayan ticker'lar ``errors[t] = "deadline_exceeded"`` olur.
    """
    many = getattr(provider, "download_many", None)
    fetch = many if callable(many) else (lambda ts, **kw: download_each(provider, ts, **kw))
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    pending = list(dict.fromkeys(tickers))
    for attempt in range(max_retries + 1):
        if not pending:
            break
        if attempt > 0:
            sleep(backoff_s * (2 ** (attempt - 1)))
        next_pending: list[str] = []
        for i in range(0, len(pending), max(1, chunk_size)):
            chunk = pending[i:i + max(1, chunk_size)]
            if deadline is not None and clock() >= deadline:
                for t in pending[i:]:  # bu turda hic denenmeyenler (veri almis olanlar korunur)
                    if t not in frames:
                        errors[t] = DEADLINE_ERROR
                logger.warning("indirme genel siniri asildi; kalan ticker'lar indirilmedi")
                return frames, errors
            try:
                got = fetch(chunk, start=start, end=end)
            except Exception as exc:  # tum chunk dustu (ag/rate-limit): hepsi yeniden denenecek
                logger.warning("toplu indirme hatasi (%d ticker, deneme %d): %s", len(chunk), attempt + 1, exc)
                for t in chunk:
                    errors[t] = str(exc)
                next_pending.extend(chunk)
                continue
            for t in chunk:
                frame = got.get(t)
                if frame is None:
                    errors[t] = "provider_returned_nothing"
                    next_pending.append(t)
                elif frame.empty:
                    frames[t] = frame
                    errors.pop(t, None)
                    next_pending.append(t)
                else:
                    frames[t] = frame
                    errors.pop(t, None)
        pending = next_pending
    return frames, errors
