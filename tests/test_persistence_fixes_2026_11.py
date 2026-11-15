# -*- coding: utf-8 -*-
"""2026-11 persistence audit fixes: cached DB, pragmas, atomicity, upsert, paths."""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from src.database import paths as db_paths
from src.database.repositories.best_model import BestModelRepository
from src.database.stock_model_db import StockModelDB, get_db
from src.gui.core.db_helper import DBHelper
from src.serving.peer_store import GlobalRunMeta, PeerStore

_METRICS = {"RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0, "Dir_Acc": 55.0, "Sharpe": 0.3, "Hit_Rate": 50.0}


def _path(tmp_path, name="stock_models.db") -> str:
    return os.path.join(str(tmp_path), name)


def _points(n=3, base="2026-03-"):
    return [
        {
            "target_date": f"{base}{10 + i:02d}",
            "horizon_index": i + 1,
            "bounded_predicted_close": 101.0 + i,
            "raw_predicted_close": 101.0 + i,
            "predicted_return": 0.01,
        }
        for i in range(n)
    ]


def _log_run(db, *, last_observed="2026-03-09", model="LSTM", points=None, **kw):
    return db.log_forecast_run(
        stock_symbol="TEST", model_name=model, source_experiment_id=None,
        last_observed_date=last_observed, last_close=100.0, horizon_days=3,
        trend_label="up", weekly_expected_return=0.02, trend_threshold=0.01,
        rules_version="v1", points=points or _points(), **kw,
    )


# ---------------------------------------------------------------- 1. get_db cache
def test_get_db_is_cached_and_schema_init_runs_once(tmp_path, monkeypatch):
    calls = {"n": 0}
    original = StockModelDB._init_db

    def counting(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(StockModelDB, "_init_db", counting)
    path = _path(tmp_path)

    first = get_db(path)
    second = get_db(os.path.abspath(path))
    third = StockModelDB(path)  # doğrudan yapıcı da şemayı tekrar kurmamalı

    assert first is second
    assert third is not first
    assert calls["n"] == 1
    third.ensure_schema(force=True)
    assert calls["n"] == 2


def test_schema_reinitialises_when_db_file_is_recreated(tmp_path):
    path = _path(tmp_path)
    StockModelDB(path)
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    db = StockModelDB(path)
    assert db.get_schema_status()["ok"] is True


# ----------------------------------------------------------- 2. WAL + busy_timeout
def test_main_db_connection_has_wal_and_busy_timeout(tmp_path):
    db = StockModelDB(_path(tmp_path))
    with db._connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_peer_store_connection_has_wal_and_busy_timeout(tmp_path):
    store = PeerStore(_path(tmp_path, "serving_pool.db"))
    with store._connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000


def test_peer_store_write_run_is_one_transaction(tmp_path):
    store = PeerStore(_path(tmp_path, "serving_pool.db"))
    meta = GlobalRunMeta(model_name="m", as_of_date="2026-06-03")
    good = pd.DataFrame({"symbol": ["AAA", "BBB"], "peer_score": [1.0, 2.0]})
    rid, n = store.write_run(meta, good)
    assert n == 2 and store.latest_run()["run_id"] == rid

    # Skor yazımı sırasında hata: NOT NULL symbol ihlali -> run başlığı da geri alınmalı.
    bad2 = pd.DataFrame({"symbol": [None], "peer_score": [1.0]})
    with pytest.raises(sqlite3.IntegrityError):
        store.write_run(meta, bad2)
    assert store.latest_run()["run_id"] == rid  # yetim run yok


# ---------------------------------------------------------- 3. log_experiment atomic
def test_log_experiment_rolls_back_on_best_model_failure(tmp_path, monkeypatch):
    db = StockModelDB(_path(tmp_path))

    def boom(*args, **kwargs):
        raise RuntimeError("best upsert failed")

    monkeypatch.setattr(BestModelRepository, "upsert_best_from_values", staticmethod(boom))
    with pytest.raises(RuntimeError):
        db.log_experiment(
            stock_symbol="TEST", model_name="Ridge Return", metrics=_METRICS,
            validation_mode="final_holdout", is_production_candidate=True, run_id="r1",
        )
    assert db.get_experiments(stock_symbol="TEST") == []
    assert db.get_best_model("TEST") is None


# ----------------------------------------------- 4. forecast_points keeps resolution
def test_resave_forecast_keeps_resolved_columns(tmp_path):
    db = StockModelDB(_path(tmp_path))
    run_id = _log_run(db)
    actuals = {"2026-03-10": 102.0, "2026-03-11": 103.0, "2026-03-12": 104.0}
    assert db.resolve_forecasts("TEST", actuals) == 3

    assert _log_run(db) == run_id  # aynı run_key -> yeniden kayıt

    pts = db.get_latest_forecast("TEST")["points"]
    assert [p["actual_close"] for p in pts] == [102.0, 103.0, 104.0]
    assert all(p["resolved_at"] is not None for p in pts)
    assert all(p["direction_correct"] is not None for p in pts)


# ------------------------------------------------------- 5. resolve_forecasts N+1
def test_resolve_skips_resolved_and_avoids_n_plus_one(tmp_path):
    db = StockModelDB(_path(tmp_path))
    for i in range(5):
        _log_run(db, last_observed=f"2026-03-{9 - i:02d}", model=f"M{i}",
                 points=_points(3, base="2026-03-"))
    actuals = {"2026-03-10": 102.0, "2026-03-11": 103.0, "2026-03-12": 104.0}

    statements: list[str] = []
    original = db._connect

    def traced():
        conn = original()
        conn.set_trace_callback(statements.append)
        return conn

    db._connect = traced  # type: ignore[assignment]
    assert db.resolve_forecasts("TEST", actuals) == 15
    point_selects = [s for s in statements if "FROM forecast_points" in s and s.lstrip().upper().startswith("SELECT")
                     and "actual_close IS NOT NULL" not in s]
    assert len(point_selects) == 1  # run başına değil, tek sorgu

    # İkinci çağrı: farklı gerçekleşme verilse de çözülmüş noktalar yeniden yazılmaz.
    with original() as conn:
        conn.execute("UPDATE forecast_points SET resolved_at = '2000-01-01T00:00:00'")
    changed = {k: v + 50.0 for k, v in actuals.items()}
    assert db.resolve_forecasts("TEST", changed) == 0
    with original() as conn:
        rows = conn.execute("SELECT actual_close, resolved_at FROM forecast_points").fetchall()
    assert {r["resolved_at"] for r in rows} == {"2000-01-01T00:00:00"}
    assert {r["actual_close"] for r in rows} == {102.0, 103.0, 104.0}


def test_resolve_continues_chain_after_partially_resolved_run(tmp_path):
    db = StockModelDB(_path(tmp_path))
    _log_run(db)
    assert db.resolve_forecasts("TEST", {"2026-03-10": 102.0}) == 1
    assert db.resolve_forecasts("TEST", {"2026-03-11": 103.0, "2026-03-12": 104.0}) == 2
    pts = db.get_latest_forecast("TEST")["points"]
    assert pts[1]["actual_return"] == pytest.approx(103.0 / 102.0 - 1.0)


# ------------------------------------------------------------------- 6. GUI helper
def test_gui_helper_returns_latest_pending_forecast_run(tmp_path):
    path = _path(tmp_path)
    db = StockModelDB(path)
    _log_run(db, last_observed="2026-03-02", run_at="2026-03-02T10:00:00+00:00")
    _log_run(db, last_observed="2026-03-09", run_at="2026-03-09T10:00:00+00:00")
    with db._connect() as conn:
        statuses = {r[0] for r in conn.execute("SELECT status FROM forecast_runs")}
    assert statuses == {"pending"}  # hiçbir yazıcı 'completed' yazmaz

    result = DBHelper(db_path=path).get_forecasts_by_symbol("test")

    assert result["run"]["last_observed_date"] == "2026-03-09"
    assert len(result["points"]) == 3


# --------------------------------------------------------------- 7. portable paths
def test_model_path_relative_round_trip(tmp_path):
    db = StockModelDB(_path(tmp_path))
    abs_path = os.path.join(db_paths.PROJECT_ROOT, "outputs", "TEST", "runs", "r1", "models", "m.pt")

    db.log_experiment(
        stock_symbol="TEST", model_name="Ridge Return", metrics=_METRICS, model_path=abs_path,
        validation_mode="final_holdout", is_production_candidate=True, run_id="r1",
    )

    with db._connect() as conn:
        raw_exp = conn.execute("SELECT model_path FROM experiments").fetchone()[0]
        raw_best = conn.execute("SELECT model_path FROM best_models").fetchone()[0]
    assert raw_exp == raw_best == "outputs/TEST/runs/r1/models/m.pt"
    expected = os.path.normpath(abs_path)
    assert os.path.normpath(db.get_best_model("TEST")["model_path"]) == expected
    assert os.path.normpath(db.get_experiments(stock_symbol="TEST")[0]["model_path"]) == expected


def test_legacy_absolute_path_is_remapped_to_local_outputs(tmp_path):
    model = tmp_path / "outputs" / "X" / "runs" / "r" / "models" / "m.pt"
    model.parent.mkdir(parents=True)
    model.write_text("w")
    legacy = r"D:\old_machine\ts_lab\outputs\X\runs\r\models\m.pt"

    resolved = db_paths.resolve_model_path(legacy, project_root=str(tmp_path))

    assert os.path.normpath(resolved) == os.path.normpath(str(model))
    # Yerelde karşılığı yoksa orijinal döner (sessizce yanlış dosya uydurulmaz).
    missing = r"D:\old_machine\outputs\Y\m.pt"
    assert db_paths.resolve_model_path(missing, project_root=str(tmp_path)) == missing
    # Göreceli yol proje köküne çözülür; outputs_root verilirse onun altına.
    rel = db_paths.resolve_model_path("outputs/X/runs/r/models/m.pt", project_root=str(tmp_path))
    assert os.path.normpath(rel) == os.path.normpath(str(model))
    other = tmp_path / "elsewhere"
    via_root = db_paths.resolve_model_path(
        "outputs/X/m.pt", project_root=str(tmp_path), outputs_root=str(other)
    )
    assert os.path.normpath(via_root) == os.path.normpath(str(other / "X" / "m.pt"))


def test_to_storable_path_is_idempotent_and_keeps_foreign_paths(tmp_path):
    inside = os.path.join(db_paths.PROJECT_ROOT, "outputs", "A", "m.pt")
    once = db_paths.to_storable_path(inside)
    assert once == "outputs/A/m.pt"
    assert db_paths.to_storable_path(once) == once
    assert db_paths.to_storable_path("") == ""
    outside = str(tmp_path / "other" / "m.pt")
    assert db_paths.to_storable_path(outside) == outside


def test_migration_rewrites_legacy_absolute_paths_idempotently(tmp_path):
    path = _path(tmp_path)
    db = StockModelDB(path)
    exp_id = db.log_experiment(
        stock_symbol="TEST", model_name="Ridge Return", metrics=_METRICS,
        validation_mode="final_holdout", is_production_candidate=True, run_id="r1",
    )
    legacy = r"D:\old_machine\ts_lab\outputs\TEST\runs\r1\models\m.pt"
    no_segment = r"D:\somewhere\else\m.pt"
    with db._connect() as conn:
        conn.execute("UPDATE experiments SET model_path = ? WHERE id = ?", (legacy, exp_id))
        conn.execute("UPDATE best_models SET model_path = ?", (legacy,))
        conn.execute(
            "INSERT INTO experiments (stock_symbol, model_name, model_path, trained_at) "
            "VALUES ('TEST', 'Other', ?, '2026-01-01 00:00:00')", (no_segment,),
        )

    db.ensure_schema(force=True)

    with db._connect() as conn:
        paths = [r[0] for r in conn.execute("SELECT model_path FROM experiments ORDER BY id")]
        best = conn.execute("SELECT model_path FROM best_models").fetchone()[0]
        again = db.schema_repository.migrate_model_paths(conn)
    assert paths == ["outputs/TEST/runs/r1/models/m.pt", no_segment]
    assert best == "outputs/TEST/runs/r1/models/m.pt"
    assert again == 0
