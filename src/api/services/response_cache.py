# -*- coding: utf-8 -*-
"""
Sprint 9 (2026-05-26) A9.2 — Analysis API response cache.

In-memory TTL cache by `symbol` (horizon implicit — default per pipeline).
Default TTL 24 saat. Invalidation manuel veya yeni training/forecast run
sonrasi `invalidate(symbol)` cagrisi.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

_DEFAULT_TTL_SECONDS = int(os.getenv("AI_CORE_RESPONSE_CACHE_TTL_SECONDS", "86400"))
_DISABLED = os.getenv("AI_CORE_RESPONSE_CACHE_DISABLED", "0") in {"1", "true", "True"}


@dataclass
class _Entry:
    value: Any
    expires_at: datetime


class ResponseCache:
    """
    Simple thread-safe TTL cache. Production'da Redis ile degistirilebilir;
    interface stable.
    """

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = max(0, int(ttl_seconds))
        self._store: Dict[str, _Entry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(symbol: str) -> str:
        return str(symbol).upper().strip()

    def get(self, symbol: str) -> Optional[Any]:
        if _DISABLED or self._ttl <= 0:
            return None
        key = self._key(symbol)
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                # expired; lazy evict
                self._store.pop(key, None)
                return None
            return entry.value

    def set(self, symbol: str, value: Any) -> None:
        if _DISABLED or self._ttl <= 0:
            return
        key = self._key(symbol)
        expires_at = datetime.now(tz=timezone.utc) + timedelta(seconds=self._ttl)
        with self._lock:
            self._store[key] = _Entry(value=value, expires_at=expires_at)

    def invalidate(self, symbol: str) -> bool:
        key = self._key(symbol)
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
            return n

    @property
    def ttl_seconds(self) -> int:
        return self._ttl


_default_cache: Optional[ResponseCache] = None


def get_default_cache() -> ResponseCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = ResponseCache()
    return _default_cache
