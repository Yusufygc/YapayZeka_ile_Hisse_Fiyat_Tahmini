# -*- coding: utf-8 -*-
"""
Sprint 9 (2026-05-26) A9.2 — Analysis API response cache.

In-memory TTL cache by `symbol` (horizon implicit — default per pipeline).
Default TTL 1 saat; en fazla 1024 girdi (LRU). Her girdi bir "tazelik parmak izi" (fingerprint) tasir:
nightly batch serving-pool run'ini ya da piyasa verisi CSV'sini degistirirse
parmak izi uyusmaz ve girdi cache-miss sayilir (24 saat eski yanit sorunu).
Ayrica manuel `invalidate(symbol)` desteklenir.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional, Tuple

from src.database.connection import open_readonly
from src.serving.paths import default_data_dir, default_serving_db_path

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = int(os.getenv("AI_CORE_RESPONSE_CACHE_TTL_SECONDS", "3600"))
_DEFAULT_MAX_ENTRIES = int(os.getenv("AI_CORE_RESPONSE_CACHE_MAX_ENTRIES", "1024"))
_ISTANBUL = timezone(timedelta(hours=3))  # Turkiye sabit UTC+3 (servis tarihi ile ayni)
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
        conn = open_readonly(str(db_path))
        try:
            row = conn.execute(sql, params).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        logger.debug("freshness fingerprint query failed (%s): %s", db_path, exc)
        return None
    return None if row is None else tuple(row)


def compute_freshness_fingerprint(
    symbol: str,
    serving_db_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    today: Optional[date] = None,
) -> Tuple:
    """Sembol icin ucuz tazelik parmak izi.

    (serving_pool son global_model_runs.run_id,
     sembol piyasa CSV'sinin (mtime_ns, boyut),
     Istanbul takvim gunu).  Kaynak yoksa ilgili bilesen None. Takvim gunu, bayatlik
    durumunun gun donumunde yeniden hesaplanmasini saglar; servis de Istanbul tarihi
    kullandigi icin gun donumu ikisinde ayni anda olur (UTC gunu 3 saat farkli olurdu).
    """
    sym = str(symbol).upper().strip()
    serving_db = serving_db_path if serving_db_path is not None else default_serving_db_path()
    base = data_dir if data_dir is not None else default_data_dir()
    pool = _readonly_fetch_one(
        serving_db, "SELECT run_id FROM global_model_runs ORDER BY run_id DESC LIMIT 1"
    )
    csv_sig: Optional[Tuple[int, int]] = None
    try:
        st = os.stat(os.path.join(base, f"{sym}.csv"))
        csv_sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        csv_sig = None
    day = (today or datetime.now(tz=_ISTANBUL).date()).isoformat()
    return (None if pool is None else pool[0], csv_sig, day)


class ResponseCache:
    """
    Simple thread-safe TTL cache. Production'da Redis ile degistirilebilir;
    interface stable.
    """

    def __init__(
        self,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        clock: Optional[Callable[[], datetime]] = None,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._clock = clock or _utcnow
        self._ttl = max(0, int(ttl_seconds))
        self._max = max(1, int(max_entries))
        self._store: "OrderedDict[str, _Entry]" = OrderedDict()  # LRU: sondaki en yeni
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
            self._store.move_to_end(key)  # LRU: kullanilan girdi en yeniye tasinir
            return entry.value

    def set(self, symbol: str, value: Any, fingerprint: Any = None) -> None:
        """Sembol için değeri TTL süresiyle önbelleğe yazar (cache kapalıysa no-op)."""
        if _DISABLED or self._ttl <= 0:
            return
        key = self._key(symbol)
        expires_at = self._clock() + timedelta(seconds=self._ttl)
        with self._lock:
            self._store[key] = _Entry(value=value, expires_at=expires_at, fingerprint=fingerprint)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)  # en az yakin kullanilani at (sinirli bellek)

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

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

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
