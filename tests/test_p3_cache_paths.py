# -*- coding: utf-8 -*-
"""P3 — F3 (bayat response cache: tazelik parmak izi) + F14 (python yorumlayici cozumu)."""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
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
def _make_stock_db(path, *, exp_id=1, updated_at="2026-11-01", run_id=10, run_at="2026-11-01T01:00"):
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE best_models (stock_symbol TEXT PRIMARY KEY, experiment_id INTEGER, updated_at TEXT);"
        "CREATE TABLE forecast_runs (id INTEGER PRIMARY KEY, stock_symbol TEXT, run_at TEXT);"
    )
    conn.execute("INSERT INTO best_models VALUES ('TUPRS', ?, ?)", (exp_id, updated_at))
    conn.execute("INSERT INTO forecast_runs VALUES (?, 'TUPRS', ?)", (run_id, run_at))
    conn.commit()
    conn.close()


def _make_serving_db(path, run_id=5):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE global_model_runs (run_id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO global_model_runs VALUES (?)", (run_id,))
    conn.commit()
    conn.close()


def _update(path, sql, params=()):
    conn = sqlite3.connect(str(path))
    conn.execute(sql, params)
    conn.commit()
    conn.close()


class _Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


# --------------------------------------------------------------------------- #
# fingerprint function
# --------------------------------------------------------------------------- #
def test_fingerprint_missing_dbs_is_all_none(tmp_path):
    fp = compute_freshness_fingerprint(
        "TUPRS", stock_db_path=str(tmp_path / "no.db"), serving_db_path=str(tmp_path / "no2.db")
    )
    assert fp == (None, None, None)


def test_fingerprint_reads_best_forecast_and_pool(tmp_path):
    s, p = tmp_path / "s.db", tmp_path / "p.db"
    _make_stock_db(s, exp_id=7, run_id=3)
    _make_serving_db(p, run_id=9)
    fp = compute_freshness_fingerprint("tuprs", stock_db_path=str(s), serving_db_path=str(p))
    assert fp[0][0] == 7 and fp[1][0] == 3 and fp[2] == 9


def test_fingerprint_unreadable_db_returns_none_components(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a sqlite file at all" * 10)
    fp = compute_freshness_fingerprint("TUPRS", stock_db_path=str(bad), serving_db_path=str(bad))
    assert fp == (None, None, None)


def test_fingerprint_changes_on_each_source(tmp_path):
    s, p = tmp_path / "s.db", tmp_path / "p.db"
    _make_stock_db(s)
    _make_serving_db(p)
    base = compute_freshness_fingerprint("TUPRS", str(s), str(p))

    _update(s, "UPDATE best_models SET experiment_id = 2")
    fp_best = compute_freshness_fingerprint("TUPRS", str(s), str(p))
    assert fp_best != base

    _update(s, "INSERT INTO forecast_runs VALUES (11, 'TUPRS', '2026-11-02T01:00')")
    fp_fc = compute_freshness_fingerprint("TUPRS", str(s), str(p))
    assert fp_fc != fp_best

    _update(p, "INSERT INTO global_model_runs VALUES (6)")
    assert compute_freshness_fingerprint("TUPRS", str(s), str(p)) != fp_fc


# --------------------------------------------------------------------------- #
# cache semantics
# --------------------------------------------------------------------------- #
def test_default_ttl_is_one_hour(monkeypatch):
    import importlib

    import src.api.services.response_cache as rc

    monkeypatch.delenv("AI_CORE_RESPONSE_CACHE_TTL_SECONDS", raising=False)
    try:
        mod = importlib.reload(rc)
        assert mod._DEFAULT_TTL_SECONDS == 3600
    finally:
        importlib.reload(rc)


def test_same_fingerprint_hits_different_misses():
    c = ResponseCache(ttl_seconds=60)
    c.set("X", "v", fingerprint=("a", 1))
    assert c.get("X", ("a", 1)) == "v"
    assert c.get("X", ("a", 2)) is None
    # mismatching read evicted the entry
    assert c.get("X", ("a", 1)) is None


def test_ttl_expiry_uses_injected_clock_no_sleep():
    clk = _Clock()
    c = ResponseCache(ttl_seconds=3600, clock=clk)
    c.set("X", "v", fingerprint="f")
    clk.advance(seconds=3599)
    assert c.get("X", "f") == "v"
    clk.advance(seconds=2)
    assert c.get("X", "f") is None


def test_invalidate_still_works_with_fingerprint():
    c = ResponseCache(ttl_seconds=60)
    c.set("X", 1, fingerprint="f")
    assert c.invalidate("X") is True
    assert c.get("X", "f") is None


# --------------------------------------------------------------------------- #
# router integration: stale response must not be served after nightly batch
# --------------------------------------------------------------------------- #
@pytest.fixture()
def router_env(tmp_path, monkeypatch):
    s, p = tmp_path / "s.db", tmp_path / "p.db"
    _make_stock_db(s)
    _make_serving_db(p)
    calls = {"n": 0}

    class _FakeService:
        _db_path = str(s)

        def build(self, symbol):
            calls["n"] += 1
            return SimpleNamespace(
                symbol=str(symbol).upper(),
                analysis_status="ok",
                refresh_status=None,
                refresh_reason=None,
                forecast_source=None,
                build_no=calls["n"],
            )

    monkeypatch.setattr(analysis_router, "_service", _FakeService())
    monkeypatch.setattr(analysis_router, "_peer_enricher", SimpleNamespace(enrich=lambda r: r))
    monkeypatch.setattr(analysis_router, "audit_append", lambda *a, **k: None)
    monkeypatch.setattr(analysis_router, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(analysis_router, "_cache", ResponseCache(ttl_seconds=3600))
    # point the serving-pool lookup at the tmp DB
    real = analysis_router.compute_freshness_fingerprint
    monkeypatch.setattr(
        analysis_router,
        "compute_freshness_fingerprint",
        lambda sym, stock_db_path=None: real(sym, stock_db_path=stock_db_path, serving_db_path=str(p)),
    )
    return SimpleNamespace(stock=s, pool=p, calls=calls)


def test_router_same_fingerprint_is_cache_hit(router_env):
    first = analysis_router.get_analysis("tuprs")
    second = analysis_router.get_analysis("TUPRS")
    assert second is first
    assert router_env.calls["n"] == 1


def test_router_best_model_change_invalidates_cache(router_env):
    first = analysis_router.get_analysis("TUPRS")
    _update(router_env.stock, "UPDATE best_models SET experiment_id = 99, updated_at = '2026-11-02'")
    second = analysis_router.get_analysis("TUPRS")
    assert second is not first
    assert second.build_no == 2
    assert router_env.calls["n"] == 2
    # and the new one is cached again
    assert analysis_router.get_analysis("TUPRS") is second


def test_router_new_forecast_run_invalidates_cache(router_env):
    analysis_router.get_analysis("TUPRS")
    _update(router_env.stock, "INSERT INTO forecast_runs VALUES (11, 'TUPRS', '2026-11-02T01:00')")
    analysis_router.get_analysis("TUPRS")
    assert router_env.calls["n"] == 2


def test_router_new_serving_run_invalidates_cache(router_env):
    analysis_router.get_analysis("TUPRS")
    _update(router_env.pool, "INSERT INTO global_model_runs VALUES (6)")
    analysis_router.get_analysis("TUPRS")
    assert router_env.calls["n"] == 2


# --------------------------------------------------------------------------- #
# F14: interpreter resolution order
# --------------------------------------------------------------------------- #
def _resolver():
    pytest.importorskip("PySide6")
    from src.gui.core.process_runner import resolve_python_interpreter

    return resolve_python_interpreter


def _venv_python(root):
    if os.name == "nt":
        return os.path.join(str(root), ".venv", "Scripts", "python.exe")
    return os.path.join(str(root), ".venv", "bin", "python")


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass
    return path


def test_python_env_var_wins_over_venv(tmp_path):
    resolve = _resolver()
    _touch(_venv_python(tmp_path))
    custom = _touch(str(tmp_path / "custom" / "py.exe"))
    assert resolve(str(tmp_path), {"AI_CORE_PYTHON": custom}) == custom


def test_python_env_var_missing_file_falls_to_venv(tmp_path):
    resolve = _resolver()
    venv = _touch(_venv_python(tmp_path))
    env = {"AI_CORE_PYTHON": str(tmp_path / "nope" / "python.exe")}
    assert resolve(str(tmp_path), env) == venv


def test_python_venv_used_when_no_env(tmp_path):
    resolve = _resolver()
    venv = _touch(_venv_python(tmp_path))
    assert resolve(str(tmp_path), {}) == venv


def test_python_falls_back_to_sys_executable(tmp_path):
    resolve = _resolver()
    assert resolve(str(tmp_path), {}) == sys.executable


def test_process_runner_has_no_hardcoded_user_path():
    pytest.importorskip("PySide6")
    import inspect

    import src.gui.core.process_runner as pr

    src = inspect.getsource(pr)
    assert "ysfygc" not in src and "anaconda3" not in src
