# -*- coding: utf-8 -*-
"""
Sprint 9 (2026-05-26) A9.3 — IP-based rate limiter middleware.

Production'da slowapi/redis tercih edilebilir; bu modul harici dependency
gerektirmeyen in-memory fallback. Sliding-window degil sabit-window:
dakika basina sayilir, dakika degisince sayac resetlenir.

Env:
    AI_CORE_RATE_LIMIT_PER_MINUTE   (default 60; 0 = disabled)
    AI_CORE_RATE_LIMIT_TRUSTED_IPS  (virgul ayrik allow-list; sayilmaz)
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

_DEFAULT_PER_MINUTE = int(os.getenv("AI_CORE_RATE_LIMIT_PER_MINUTE", "60"))
_TRUSTED_IPS = {
    ip.strip()
    for ip in os.getenv("AI_CORE_RATE_LIMIT_TRUSTED_IPS", "127.0.0.1").split(",")
    if ip.strip()
}


class RateLimiter:
    """Fixed-window IP rate limiter."""

    def __init__(self, per_minute: int = _DEFAULT_PER_MINUTE) -> None:
        self._per_minute = max(0, int(per_minute))
        self._counts: dict[str, dict[str, int]] = defaultdict(dict)
        self._lock = threading.Lock()

    @property
    def per_minute(self) -> int:
        """Dakika başına izin verilen istek limiti (0 = devre dışı)."""
        return self._per_minute

    def enabled(self) -> bool:
        """Limit pozitifse (devredeyse) True döner."""
        return self._per_minute > 0

    @staticmethod
    def _window_key() -> str:
        return datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M")

    def is_allowed(self, ip: str) -> bool:
        """IP için güncel dakika penceresinde limit aşılmadıysa True döner.

        Trusted IP'ler ve limit kapalıysa daima True. Sayaç dakika değişince
        sıfırlanır (sabit-window).
        """
        if not self.enabled():
            return True
        if ip in _TRUSTED_IPS:
            return True
        window = self._window_key()
        with self._lock:
            buckets = self._counts[ip]
            # Sadece guncel window'i tut.
            if window not in buckets:
                buckets.clear()
            buckets[window] = buckets.get(window, 0) + 1
            return buckets[window] <= self._per_minute

    def reset(self) -> None:
        """Tüm IP sayaçlarını sıfırlar (test/manuel kullanım)."""
        with self._lock:
            self._counts.clear()


_default_limiter: Optional[RateLimiter] = None


def get_default_limiter() -> RateLimiter:
    """Süreç-genelinde paylaşılan tekil RateLimiter örneğini döner (lazy init)."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter()
    return _default_limiter


def rate_limit_middleware_factory():
    """
    Return ASGI middleware function for FastAPI/Starlette `add_middleware`.

    Lazy-imports starlette to keep test envs without FastAPI happy.
    """
    try:
        from starlette.responses import JSONResponse
        from starlette.middleware.base import BaseHTTPMiddleware
    except Exception:  # pragma: no cover
        return None

    limiter = get_default_limiter()

    class _RateLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            client = request.client.host if request.client else "unknown"
            if not limiter.is_allowed(client):
                return JSONResponse(
                    {
                        "error": "rate_limited",
                        "detail": (
                            f"Rate limit asildi (>{limiter.per_minute}/dk). "
                            "Lutfen daha sonra tekrar deneyin."
                        ),
                        "retry_after_seconds": 60,
                    },
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
            return await call_next(request)

    return _RateLimitMiddleware
