# -*- coding: utf-8 -*-
"""
Sprint 9 (2026-05-26) A9.2 — response cache (TTL) testleri.
"""
from __future__ import annotations

import time

import pytest

from src.api.services.response_cache import ResponseCache


def test_cache_hit_returns_stored_value():
    c = ResponseCache(ttl_seconds=60)
    c.set("TUPRS", {"x": 1})
    out = c.get("TUPRS")
    assert out == {"x": 1}


def test_cache_miss_returns_none():
    c = ResponseCache(ttl_seconds=60)
    assert c.get("UNKNOWN") is None


def test_cache_key_normalized_uppercase_and_strip():
    c = ResponseCache(ttl_seconds=60)
    c.set("  tuprs  ", {"v": 1})
    assert c.get("TUPRS") == {"v": 1}
    assert c.get("tuprs") == {"v": 1}


def test_cache_expiry_lazy_evicts():
    c = ResponseCache(ttl_seconds=1)
    c.set("X", "val")
    time.sleep(1.2)
    assert c.get("X") is None
    # internal store cleaned
    assert c._store.get("X") is None


def test_invalidate_removes_entry():
    c = ResponseCache(ttl_seconds=60)
    c.set("A", 1)
    assert c.invalidate("A") is True
    assert c.get("A") is None
    assert c.invalidate("A") is False  # already gone


def test_clear_resets_store():
    c = ResponseCache(ttl_seconds=60)
    for s in ("A", "B", "C"):
        c.set(s, s)
    n = c.clear()
    assert n == 3
    assert c.get("A") is None


def test_ttl_zero_disables_cache():
    c = ResponseCache(ttl_seconds=0)
    c.set("X", 1)
    assert c.get("X") is None
