# -*- coding: utf-8 -*-
"""Bayat response cache: tazelik parmak izi (serving run id + piyasa CSV + gun)."""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.api.routers import analysis as analysis_router
from src.api.services.response_cache import (
    ResponseCache,
    compute_freshness_fingerprint,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_serving_db(path, run_id=5):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE global_model_runs (run_id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO global_model_runs VALUES (?)", (run_id,))
    conn.commit()
    conn.close()


def _insert_run(path, run_id):
    conn = sqlite3.connect(str(path))
    conn.execute("INSERT INTO global_model_runs VALUES (?)", (run_id,))
    conn.commit()
    conn.close()


def _write_csv(data_dir, symbol="TUPRS", rows=2):
    os.makedirs(str(data_dir), exist_ok=True)
    path = os.path.join(str(data_dir), f"{symbol}.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Date,Close\n" + "\n".join(f"2026-11-0{i + 1},{10 + i}" for i in range(rows)) + "\n")
    return path


class _Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


_DAY = date(2026, 11, 16)


# --------------------------------------------------------------------------- #
# fingerprint
# --------------------------------------------------------------------------- #
def test_fingerprint_missing_sources_is_none_components(tmp_path):
    fp = compute_freshness_fingerprint(
        "TUPRS", serving_db_path=str(tmp_path / "no.db"), data_dir=str(tmp_path / "nodir"), today=_DAY
    )
    assert fp == (None, None, "2026-11-16")


def test_fingerprint_reads_serving_run_and_csv(tmp_path):
    pool = tmp_path / "p.db"
    _make_serving_db(pool, run_id=5)
    _write_csv(tmp_path / "data")
    fp = compute_freshness_fingerprint(
        "tuprs", serving_db_path=str(pool), data_dir=str(tmp_path / "data"), today=_DAY
    )
    assert fp[0] == 5 and fp[1] is not None and fp[2] == "2026-11-16"


def test_fingerprint_unreadable_db_returns_none_component(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a sqlite file")
    fp = compute_freshness_fingerprint(
        "TUPRS", serving_db_path=str(bad), data_dir=str(tmp_path), today=_DAY
    )
    assert fp[0] is None


def test_fingerprint_changes_on_each_source(tmp_path):
    pool = tmp_path / "p.db"
    _make_serving_db(pool, run_id=5)
    data = tmp_path / "data"
    csv = _write_csv(data, rows=2)

    def fp(day=_DAY):
        return compute_freshness_fingerprint(
            "TUPRS", serving_db_path=str(pool), data_dir=str(data), today=day
        )

    base = fp()
    assert fp() == base
    _insert_run(pool, 6)
    after_run = fp()
    assert after_run != base
    _write_csv(data, rows=3)  # boyut degisir
    assert fp() != after_run
    assert fp(date(2026, 11, 17)) != fp(_DAY)
    assert os.path.exists(csv)


# --------------------------------------------------------------------------- #
# ResponseCache + fingerprint
# --------------------------------------------------------------------------- #
def test_default_ttl_is_one_hour(monkeypatch):
    monkeypatch.delenv("AI_CORE_RESPONSE_CACHE_TTL_SECONDS", raising=False)
    assert ResponseCache().ttl_seconds == 3600


def test_same_fingerprint_hits_different_misses():
    c = ResponseCache(ttl_seconds=60)
    c.set("X", "v", fingerprint=("a", 1))
    assert c.get("X", ("a", 1)) == "v"
    assert c.get("X", ("a", 2)) is None  # eski girdi atilir
    assert c.get("X", ("a", 1)) is None


def test_ttl_expiry_uses_injected_clock_no_sleep():
    clock = _Clock()
    c = ResponseCache(ttl_seconds=60, clock=clock)
    c.set("X", "v", fingerprint="f")
    clock.advance(seconds=30)
    assert c.get("X", "f") == "v"
    clock.advance(seconds=31)
    assert c.get("X", "f") is None


def test_invalidate_still_works_with_fingerprint():
    c = ResponseCache(ttl_seconds=60)
    c.set("X", 1, fingerprint="f")
    assert c.invalidate("X") is True
    assert c.get("X", "f") is None


# --------------------------------------------------------------------------- #
# router integration: nightly sonrasi bayat yanit servis edilmez
# --------------------------------------------------------------------------- #
@pytest.fixture()
def router_env(tmp_path, monkeypatch):
    pool = tmp_path / "p.db"
    data = tmp_path / "data"
    _make_serving_db(pool)
    _write_csv(data)
    calls = {"n": 0}

    class _FakeService:
        serving_db_path = str(pool)
        data_dir = str(data)

        def build(self, symbol):
            calls["n"] += 1
            return SimpleNamespace(
                symbol=str(symbol).upper(),
                analysis_status="ok",
                confidence=SimpleNamespace(label="medium"),
                serving_run=None,
                peer=None,
                build_no=calls["n"],
            )

    monkeypatch.setattr(analysis_router, "_service", _FakeService())
    monkeypatch.setattr(analysis_router, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(analysis_router, "_cache", ResponseCache(ttl_seconds=3600))
    return SimpleNamespace(pool=pool, data=data, calls=calls)


def test_router_same_fingerprint_is_cache_hit(router_env):
    first = analysis_router.get_analysis("tuprs")
    second = analysis_router.get_analysis("TUPRS")
    assert second is first
    assert router_env.calls["n"] == 1


def test_router_new_serving_run_invalidates_cache(router_env):
    first = analysis_router.get_analysis("TUPRS")
    _insert_run(router_env.pool, 6)
    second = analysis_router.get_analysis("TUPRS")
    assert second is not first and second.build_no == 2
    assert analysis_router.get_analysis("TUPRS") is second  # yeniden cache'lendi


def test_router_market_data_change_invalidates_cache(router_env):
    analysis_router.get_analysis("TUPRS")
    _write_csv(router_env.data, rows=3)
    analysis_router.get_analysis("TUPRS")
    assert router_env.calls["n"] == 2
