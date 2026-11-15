# -*- coding: utf-8 -*-
"""
Sprint 9 (2026-05-26) A9.2 — Analysis API response cache.

In-memory TTL cache by `symbol` (horizon implicit — default per pipeline).
Default TTL 1 saat. Her girdi bir "tazelik parmak izi" (fingerprint) tasir:
nightly batch best model / forecast run / serving-pool run'ini degistirirse
parmak izi uyusmaz ve girdi cache-miss sayilir (24 saat eski yanit sorunu).
Ayrica manuel `invalidate(symbol)` desteklenir.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = int(os.getenv("AI_CORE_RESPONSE_CACHE_TTL_SECONDS", "3600"))
_DISABLED = os.getenv("AI_CORE_RESPONSE_CACHE_DISABLED", "0") in {"1", "true", "True"}


@dataclass
class _Entry:
    value: Any
    expires_at: datetime
    fingerprint: Any = None


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _readonly_fetch_one(db_path: Optional[str], sql: str, params: Tuple = ()) -> Optional[Tuple]:
    """Tek satirlik salt-okunur sqlite sorgusu; DB yok/okunamaz ise None."""
    if not db_path or not os.path.exists(db_path):
        return None
    try:
        uri = "file:" + str(db_path).replace("\\", "/") + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2)
        try:
            row = conn.execute(sql, params).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        logger.debug("freshness fingerprint query failed (%s): %s", db_path, exc)
        return None
    return None if row is None else tuple(row)


def default_stock_db_path() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "data", "stock_models.db")


def default_serving_db_path() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "data", "serving_pool.db")


def compute_freshness_fingerprint(
    symbol: str,
    stock_db_path: Optional[str] = None,
    serving_db_path: Optional[str] = None,
) -> Tuple:
    """Sembol icin ucuz tazelik parmak izi (3 kucuk salt-okunur sorgu).

    (best_models.experiment_id + updated_at,
     son forecast_runs.id + run_at,
     serving_pool son global_model_runs.run_id).
    Kaynak DB/tablo yoksa ilgili bilesen None.
    """
    sym = str(symbol).upper().strip()
    stock_db = stock_db_path if stock_db_path is not None else default_stock_db_path()
    serving_db = serving_db_path if serving_db_path is not None else default_serving_db_path()
    best = _readonly_fetch_one(
        stock_db,
        "SELECT experiment_id, updated_at FROM best_models WHERE stock_symbol = ?",
        (sym,),
    )
    forecast = _readonly_fetch_one(
        stock_db,
        "SELECT id, run_at FROM forecast_runs WHERE stock_symbol = ? "
        "ORDER BY run_at DESC, id DESC LIMIT 1",
        (sym,),
    )
    pool = _readonly_fetch_one(
        serving_db, "SELECT run_id FROM global_model_runs ORDER BY run_id DESC LIMIT 1"
    )
    return (best, forecast, None if pool is None else pool[0])


class ResponseCache:
    """
    Simple thread-safe TTL cache. Production'da Redis ile degistirilebilir;
    interface stable.
    """

    def __init__(
        self,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._clock = clock or _utcnow
        self._ttl = max(0, int(ttl_seconds))
        self._store: Dict[str, _Entry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(symbol: str) -> str:
        return str(symbol).upper().strip()

    def get(self, symbol: str, fingerprint: Any = None) -> Optional[Any]:
        """Sembolün önbellekteki değerini döner; süresi geçmiş/yoksa None (lazy evict).

        `fingerprint` verilirse girdinin parmak iziyle birebir eşleşmesi gerekir;
        uyuşmazsa girdi bayat sayılır, silinir ve None döner.
        """
        if _DISABLED or self._ttl <= 0:
            return None
        key = self._key(symbol)
        now = self._clock()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                # expired; lazy evict
                self._store.pop(key, None)
                return None
            if entry.fingerprint != fingerprint:
                # source DBs changed since the entry was cached -> stale
                self._store.pop(key, None)
                return None
            return entry.value

    def set(self, symbol: str, value: Any, fingerprint: Any = None) -> None:
        """Sembol için değeri TTL süresiyle önbelleğe yazar (cache kapalıysa no-op)."""
        if _DISABLED or self._ttl <= 0:
            return
        key = self._key(symbol)
        expires_at = self._clock() + timedelta(seconds=self._ttl)
        with self._lock:
            self._store[key] = _Entry(value=value, expires_at=expires_at, fingerprint=fingerprint)

    def invalidate(self, symbol: str) -> bool:
        """Sembolün önbellek girdisini siler; girdi vardıysa True döner.

        Yeni training/forecast run sonrası çağrılmalı.
        """
        key = self._key(symbol)
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> int:
        """Tüm önbelleği temizler ve silinen girdi sayısını döner."""
        with self._lock:
            n = len(self._store)
            self._store.clear()
            return n

    @property
    def ttl_seconds(self) -> int:
        """Önbellek girdilerinin yaşam süresi (saniye; 0 = devre dışı)."""
        return self._ttl


_default_cache: Optional[ResponseCache] = None


def get_default_cache() -> ResponseCache:
    """Süreç-genelinde paylaşılan tekil ResponseCache örneğini döner (lazy init)."""
    global _default_cache
    if _default_cache is None:
        _default_cache = ResponseCache()
    return _default_cache
