# -*- coding: utf-8 -*-
"""
Sprint 9 (2026-05-26) A9.3 — rate limit testleri.
"""
from __future__ import annotations

import pytest

from src.api.services.rate_limit import RateLimiter, _TRUSTED_IPS


def test_disabled_when_per_minute_zero():
    rl = RateLimiter(per_minute=0)
    assert rl.enabled() is False
    # Sinirsiz allow
    for _ in range(1000):
        assert rl.is_allowed("1.2.3.4") is True


def test_basic_allows_up_to_limit_then_blocks():
    rl = RateLimiter(per_minute=3)
    ip = "10.0.0.1"
    assert rl.is_allowed(ip) is True
    assert rl.is_allowed(ip) is True
    assert rl.is_allowed(ip) is True
    # 4. istek limit ustu
    assert rl.is_allowed(ip) is False


def test_trusted_ip_never_blocked():
    rl = RateLimiter(per_minute=1)
    trusted = next(iter(_TRUSTED_IPS))
    for _ in range(20):
        assert rl.is_allowed(trusted) is True


def test_independent_per_ip_counts():
    rl = RateLimiter(per_minute=2)
    assert rl.is_allowed("a") is True
    assert rl.is_allowed("a") is True
    assert rl.is_allowed("a") is False
    # Farkli IP sayaci ayri
    assert rl.is_allowed("b") is True


def test_reset_clears_counts():
    rl = RateLimiter(per_minute=1)
    assert rl.is_allowed("x") is True
    assert rl.is_allowed("x") is False
    rl.reset()
    assert rl.is_allowed("x") is True


def test_per_minute_getter():
    rl = RateLimiter(per_minute=60)
    assert rl.per_minute == 60
