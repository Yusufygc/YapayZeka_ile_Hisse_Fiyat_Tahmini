# -*- coding: utf-8 -*-
"""K6/K7: serving havuzu tazelik ozeti + kapsama uyarilari + /health additive alan."""

from datetime import date

import pandas as pd

from src.api.services.serving_health import serving_pool_status
from src.serving.nightly_scoring import coverage_warnings
from src.serving.peer_store import GlobalRunMeta, PeerStore


def test_status_missing_db_is_unavailable_without_creating_file(tmp_path):
    path = tmp_path / "nope.db"
    out = serving_pool_status(str(path))
    assert out == {"available": False, "reason": "no_db"}
    assert not path.exists()                      # salt-okunur: dosya olusturmaz


def test_status_reports_age_and_staleness(tmp_path):
    db = str(tmp_path / "pool.db")
    PeerStore(db).insert_run(GlobalRunMeta(
        model_name="m", as_of_date="2026-05-26 00:00:00", n_symbols=585, n_rows=100))
    fresh = serving_pool_status(db, today=date(2026, 5, 28))
    assert fresh["available"] and fresh["as_of_date"] == "2026-05-26"
    assert fresh["age_days"] == 2 and fresh["stale"] is False and fresh["n_symbols"] == 585
    old = serving_pool_status(db, today=date(2026, 11, 9), stale_days=7)
    assert old["age_days"] == 167 and old["stale"] is True


def test_status_empty_or_corrupt_db_never_raises(tmp_path):
    empty = str(tmp_path / "empty.db")
    PeerStore(empty)                                # sema var, run yok
    assert serving_pool_status(empty)["reason"] == "no_runs"
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a sqlite file")
    out = serving_pool_status(str(bad))
    assert out["available"] is False and out["reason"] == "unreadable"


def test_coverage_warnings():
    today = pd.Timestamp("2026-11-09")
    w = coverage_warnings(585, 592, pd.Timestamp("2026-06-02"), today, stale_days=10)
    assert any("585/592" in x for x in w) and any("veri yasi" in x for x in w)
    ok = coverage_warnings(592, 592, pd.Timestamp("2026-11-08"), today, stale_days=10)
    assert ok == []
    assert coverage_warnings(5, 0, pd.NaT, today) == []     # evren bilinmiyor / tarih yok


def test_health_exposes_serving_pool_additively(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.api.main as api_main
    from src.database.stock_model_db import StockModelDB

    monkeypatch.setattr(api_main, "_get_db", lambda: StockModelDB(str(tmp_path / "s.db")))
    monkeypatch.setattr(api_main, "serving_pool_status",
                        lambda: {"available": True, "as_of_date": "2026-05-26", "stale": True})
    body = TestClient(api_main.app).get("/health").json()
    assert body["serving_pool"]["as_of_date"] == "2026-05-26"
    for key in ("status", "db_accessible", "schema", "registered_symbols", "timestamp"):
        assert key in body                          # mevcut alanlar degismedi
