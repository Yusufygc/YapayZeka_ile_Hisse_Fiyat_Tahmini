# -*- coding: utf-8 -*-
"""DuckDB analiz katmanı (AnalyticsDB + adlandırılmış sorgular + CLI) testleri."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import analytics_sources as srcs
from src.analysis.analytics_db import AnalyticsDB, AnalyticsError
from src.analysis.analytics_queries import QUERIES
from src.analysis.run_leaderboard import build_multi_symbol_leaderboard
from src.database.stock_model_db import StockModelDB
from src.serving.peer_store import GlobalRunMeta, PeerStore

REPORT_COLS = ["Model", "Net_Return", "BuyHold_Return", "Sharpe", "Trade_Count", "Signal_Diagnosis"]
EXP_COLS = ["Timestamp", "Model", "Validation_Mode", "Run_ID", "Dir_Acc", "RMSE_vs_benchmark", "Sharpe",
            "Net_Return", "Composite_Score"]


def _report(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [";".join(REPORT_COLS)] + [";".join(str(c) for c in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _make_run(root: Path, sym: str, run_id: str, models: list[str], wf: list, final: list,
              exp: list | None = None, manifest: bool = True, with_final: bool = True) -> Path:
    run = root / "outputs" / sym / "runs" / run_id
    (run / "csv").mkdir(parents=True)
    (run / "experiments").mkdir()
    _report(run / "csv" / "backtest_report_wf.csv", wf)
    if with_final:
        _report(run / "csv" / "backtest_report_final_holdout.csv", final)
        (run / "csv" / "validation_protocol_report.csv").write_text("Split;Protocol\n1;wf\n", encoding="utf-8")
    if manifest:
        (run / "run_manifest.json").write_text(json.dumps({
            "run_id": run_id, "generated_at": "2026-01-01T00:00:00+00:00", "stock_symbol": sym,
            "model_list": models, "validation_protocol": "walk_forward", "random_seed": 42,
            "final_holdout_status": {"status": "success"}}), encoding="utf-8")
    if exp is not None:
        lines = [",".join(EXP_COLS)] + [",".join(str(c) for c in (*r, 0.01, 50.0)) for r in exp]
        (run / "experiments" / "experiment_log.csv").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return run


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """2 sembol; AAA: 2 tamam + 1 incomplete, BBB: 1 tamam + 1 failed + 1 running."""
    r = tmp_path
    # AAA run1: M2 lider (Naive model_list dışı -> aday olmamalı)
    _make_run(r, "AAA", "20260101_000000_AAA_x", ["M1", "M2"],
              wf=[("M1", 0.05, 0.02, 0.5, 12, "ok"), ("M2", 0.40, 0.02, 0.9, 3, "ok"), ("Naive", 0.1, 0.0, 0, 0, "n")],
              final=[("M1", 0.10, 0.02, 0.5, 12, "ok"), ("M2", 0.30, 0.02, 0.9, 3, "ok"), ("Naive", 0.9, 0.0, 9, 15, "n")],
              exp=[("t", "M1", "walk_forward", "r1", 50.0, "", 0.1), ("t", "M1", "final_holdout", "r1", 52.0, 0.9, 0.2)])
    # AAA run2 (en yeni tamam): M1 stable
    _make_run(r, "AAA", "20260102_000000_AAA_x", ["M1"],
              wf=[("M1", 0.05, 0.02, 0.5, 12, "ok")], final=[("M1", 0.10, 0.02, 0.5, 12, "ok")],
              exp=[("t", "M1", "walk_forward", "r2", 60.0, "", 0.3), ("t", "M1", "final_holdout", "r2", 56.0, 0.7, 0.4)])
    # AAA run3: manifest var, final rapor yok -> incomplete (en yeni run_id)
    _make_run(r, "AAA", "20260103_000000_AAA_x", ["M1"], wf=[("M1", 0.5, 0.0, 1, 5, "ok")], final=[], with_final=False)
    # BBB run1: tamam, defensive (trade band, net>=0, sharpe>0, excess<=0)
    _make_run(r, "BBB", "20260101_000000_BBB_x", ["M1"],
              wf=[("M1", 0.01, 0.0, 0.2, 10, "ok")], final=[("M1", 0.03, 0.05, 0.2, 10, "ok")])
    # BBB run2: manifest yok + eski -> failed; run3: manifest yok + taze -> running
    failed = _make_run(r, "BBB", "20260102_000000_BBB_x", ["M1"], wf=[("M1", 0, 0, 0, 0, "x")], final=[], manifest=False, with_final=False)
    old = time.time() - 48 * 3600
    os.utime(failed, (old, old))
    _make_run(r, "BBB", "20260103_000000_BBB_x", ["M1"], wf=[("M1", 0, 0, 0, 0, "x")], final=[], manifest=False, with_final=False)
    return r


@pytest.fixture()
def full_project(project: Path) -> Path:
    data = project / "data"
    data.mkdir()
    db = StockModelDB(str(data / "stock_models.db"))
    pts = [
        {"target_date": "2026-03-02", "horizon_index": 1, "bounded_predicted_close": 11.0, "predicted_return": 0.10,
         "p10_close": 10.0, "p90_close": 12.0},
        {"target_date": "2026-03-03", "horizon_index": 2, "bounded_predicted_close": 10.5, "predicted_return": -0.05,
         "p10_close": 9.0, "p90_close": 10.0},
        {"target_date": "2026-03-04", "horizon_index": 3, "bounded_predicted_close": 10.2, "predicted_return": 0.02},
    ]
    db.log_forecast_run(stock_symbol="AAA", model_name="M1", source_experiment_id=None, last_observed_date="2026-02-27",
                        last_close=10.0, horizon_days=3, trend_label="up", weekly_expected_return=0.02,
                        trend_threshold=0.01, rules_version="v1", points=pts)
    db.resolve_forecasts("AAA", {"2026-03-02": 11.5, "2026-03-03": 11.0, "2026-03-04": 10.8})
    db.log_experiment("AAA", "M1", {"RMSE": 1.0, "MAE": 0.8, "Dir_Acc": 55.0}, validation_mode="walk_forward", run_id="r1")
    store = PeerStore(str(data / "serving_pool.db"))
    scored = pd.DataFrame({"symbol": ["AAA", "BBB"], "peer_score": [0.3, -0.2]})
    for i, icir in enumerate((1.0, 1.5)):
        store.write_run(GlobalRunMeta(model_name="G", as_of_date="2026-01-01", n_symbols=2, n_rows=10, horizon=5,
                                      ic_mean=0.05 * (i + 1), icir=icir, pct_ic_positive=0.6, config={"boost": 400}),
                        scored, created_at=f"2026-01-0{i + 1}T00:00:00")
    clean = data / "clean"
    clean.mkdir()
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    pd.concat([
        pd.DataFrame({"symbol": "AAA", "Date": dates, "Close": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"symbol": "BBB", "Date": dates[:2], "Close": [1.0, 2.0]}),
    ]).to_parquet(clean / "panel.parquet")
    pd.DataFrame({
        "symbol": ["AAA", "BBB"], "n_rows": [3, 2], "first_date": pd.to_datetime(["2026-01-01"] * 2),
        "last_date": pd.to_datetime(["2026-01-05", "2026-01-02"]), "rejected_rows": [0, 4],
        "cleaned_at": ["x", "y"]}).to_parquet(clean / "manifest.parquet")
    return project


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_views_created_only_when_sources_exist(tmp_path: Path, project: Path, full_project: Path) -> None:
    with AnalyticsDB(project_root=tmp_path / "empty") as empty:
        assert empty.views() == []
        with pytest.raises(AnalyticsError, match="gereken kaynak yok"):
            empty.query("leaderboard")
    # Yalnız outputs (project fixture'ı data/ içermez; full_project henüz data yazar -> ayrı kök kullan)
    only_out = tmp_path / "only_out"
    (only_out / "outputs").mkdir(parents=True)
    _make_run(only_out, "AAA", "r1", ["M1"], wf=[("M1", 1, 1, 1, 1, "x")], final=[("M1", 1, 1, 1, 1, "x")])
    with AnalyticsDB(project_root=only_out) as part:
        assert part.views() == ["runs", "run_model_metrics"]  # experiment_log yok -> run_experiments yok
    with AnalyticsDB(project_root=full_project) as full:
        assert set(full.views()) == {
            "runs", "run_model_metrics", "run_experiments", "experiments", "best_models", "forecast_runs",
            "forecast_points", "peer_runs", "peer_scores", "panel", "panel_manifest"}
        assert full.sql("SELECT count(*) AS n FROM peer_scores").iloc[0, 0] == 4


def test_runs_view_status_and_columns(project: Path) -> None:
    with AnalyticsDB(project_root=project) as db:
        runs = db.sql("SELECT * FROM runs ORDER BY symbol, run_id")
        assert len(runs) == 6
        status = dict(zip(runs.run_id.str[:8] + runs.symbol, runs.status))
        assert status["20260101AAA"] == "completed" and status["20260103AAA"] == "incomplete"
        assert status["20260102BBB"] == "failed" and status["20260103BBB"] == "running"
        first = runs.iloc[0]
        assert first["mode"] == "walk_forward" and first.random_seed == 42 and list(first.model_list) == ["M1", "M2"]
        assert first.final_holdout_status == "success" and first.generated_at.startswith("2026-01-01")


def test_run_status_query(project: Path) -> None:
    with AnalyticsDB(project_root=project) as db:
        out = db.query("run_status").set_index("symbol")
        assert out.loc["AAA", ["n_runs", "completed", "incomplete", "failed", "running"]].tolist() == [3, 2, 1, 0, 0]
        assert out.loc["BBB", ["n_runs", "completed", "incomplete", "failed", "running"]].tolist() == [3, 1, 0, 1, 1]
        assert out.loc["AAA", "latest_completed_run_id"] == "20260102_000000_AAA_x"
        only = db.query("run_status", symbol="bbb")
        assert only.symbol.tolist() == ["BBB"]


def test_run_model_metrics_keeps_raw_columns(project: Path) -> None:
    with AnalyticsDB(project_root=project) as db:
        m = db.sql("SELECT * FROM run_model_metrics WHERE symbol='AAA' AND split='final_holdout' AND model='M2'")
        assert len(m) == 1 and m.iloc[0]["Net_Return"] == pytest.approx(0.30) and m.iloc[0]["Signal_Diagnosis"] == "ok"
        assert set(db.sql("SELECT DISTINCT split FROM run_model_metrics").split) == {"wf", "final_holdout"}


def test_leaderboard_rows(project: Path) -> None:
    with AnalyticsDB(project_root=project) as db:
        lb = db.query("leaderboard").set_index("symbol")
        assert lb.loc["AAA", "run_id"] == "20260102_000000_AAA_x" and lb.loc["AAA", "model"] == "M1"
        assert lb.loc["AAA", "leader_reliability_class"] == "stable"
        assert lb.loc["BBB", "run_id"] == "20260101_000000_BBB_x"
        assert lb.loc["BBB", "leader_reliability_class"] == "defensive"
        assert lb.loc["BBB", "final_excess_vs_buyhold"] == pytest.approx(-0.02)


def test_leaderboard_ignores_models_outside_manifest_list(project: Path) -> None:
    """run1'de Naive final'de en iyi görünür ama model_list'te yok -> lider M2."""
    with AnalyticsDB(project_root=project) as db:
        run1 = db.sql(
            "SELECT model FROM run_model_metrics m JOIN runs r USING (symbol, run_id) "
            "WHERE m.symbol='AAA' AND m.split='final_holdout' AND m.run_id LIKE '20260101%' "
            "AND list_contains(r.model_list, m.model) ORDER BY model").model.tolist()
        assert run1 == ["M1", "M2"]


def test_leaderboard_parity_with_run_leaderboard(project: Path) -> None:
    ref = build_multi_symbol_leaderboard(outputs_base=project / "outputs", symbols=["AAA", "BBB"])
    complete = ref[ref.missing_required_files.fillna("") == ""]
    with AnalyticsDB(project_root=project) as db:
        mine = db.query("leaderboard")
    assert sorted(mine.symbol) == sorted(complete.symbol.unique())
    for _, row in mine.iterrows():
        sub = complete[complete.symbol == row.symbol]
        sub = sub[sub.run_id == sub.run_id.max()]
        top = sub[sub.trading_leader_rank == sub.trading_leader_rank.min()]
        assert row.model in set(top.model)
        ref_row = top[top.model == row.model].iloc[0]
        for col in ("wf_net_return", "final_net_return", "final_excess_vs_buyhold", "final_trade_count",
                    "final_sharpe", "wf_final_net_gap"):
            assert row[col] == pytest.approx(ref_row[col], abs=1e-9), col
        assert row.leader_reliability_class == ref_row.leader_reliability_class
        assert bool(row.trade_sufficiency_flag) == bool(ref_row.trade_sufficiency_flag)


def test_model_stability(project: Path) -> None:
    with AnalyticsDB(project_root=project) as db:
        out = db.query("model_stability", symbol="AAA").set_index("validation_mode")
        wf = out.loc["walk_forward"]
        assert wf.n_runs == 2 and wf.dir_acc_mean == pytest.approx(55.0) and wf.dir_acc_std == pytest.approx(7.0710678)
        assert wf.n_rmse_obs == 0 and pd.isna(wf.rmse_vs_benchmark_mean)  # boş hücreler NULL
        fh = out.loc["final_holdout"]
        assert fh.rmse_vs_benchmark_mean == pytest.approx(0.8) and fh.dir_acc_mean == pytest.approx(54.0)


def test_forecast_accuracy_matches_sqlite(full_project: Path) -> None:
    with AnalyticsDB(project_root=full_project) as db:
        out = db.query("forecast_accuracy").set_index("horizon_step")
    conn = sqlite3.connect(str(full_project / "data" / "stock_models.db"))
    raw = pd.read_sql_query("SELECT * FROM forecast_points WHERE actual_close IS NOT NULL ORDER BY horizon_index", conn)
    conn.close()
    assert len(raw) == 3 and out.n.tolist() == [1, 1, 1]
    for _, p in raw.iterrows():
        row = out.loc[p.horizon_index]
        assert row.direction_accuracy == pytest.approx(p.direction_correct)
        assert row.mae_return == pytest.approx(abs(p.predicted_return - p.actual_return))
    assert out.loc[1, "n_interval"] == 1 and out.loc[1, "interval_coverage"] == 1.0  # 11.5 in [10,12]
    assert out.loc[2, "interval_coverage"] == 0.0 and out.loc[2, "mean_interval_width"] == pytest.approx(1.0)
    assert out.loc[3, "n_interval"] == 0 and pd.isna(out.loc[3, "interval_coverage"])


def test_ic_history(full_project: Path) -> None:
    with AnalyticsDB(project_root=full_project) as db:
        out = db.query("ic_history")
        assert out.run_id.tolist() == [1, 2] and out.icir.tolist() == [1.0, 1.5]
        assert pd.isna(out.icir_change.iloc[0]) and out.icir_change.iloc[1] == pytest.approx(0.5)
        assert out.boost.tolist() == [400, 400]


def test_data_coverage_manifest_and_panel_fallback(full_project: Path, tmp_path: Path) -> None:
    with AnalyticsDB(project_root=full_project) as db:
        cov = db.query("data_coverage").set_index("symbol")
        assert cov.loc["AAA", "n_rows"] == 3 and cov.loc["AAA", "lag_days"] == 0
        assert cov.loc["BBB", "lag_days"] == 3 and cov.loc["BBB", "rejected_rows"] == 4
    (full_project / "data" / "clean" / "manifest.parquet").unlink()
    with AnalyticsDB(project_root=full_project) as db:
        assert "panel_manifest" not in db.views()
        cov = db.query("data_coverage").set_index("symbol")
        assert cov.loc["BBB", "lag_days"] == 3 and cov.loc["AAA", "n_rows"] == 3


@pytest.mark.parametrize("stmt", [
    "DELETE FROM runs", "DROP VIEW runs", "CREATE TABLE x AS SELECT 1", "INSERT INTO x VALUES (1)",
    "COPY (SELECT 1) TO 'x.csv'", "SELECT 1; DROP VIEW runs", "WITH a AS (SELECT 1 i) INSERT INTO t SELECT * FROM a",
    "ATTACH 'x.db'", "", "not sql at all"])
def test_sql_rejects_non_select(project: Path, stmt: str) -> None:
    with AnalyticsDB(project_root=project) as db:
        with pytest.raises(AnalyticsError):
            db.sql(stmt)
        assert len(db.sql("SELECT * FROM runs")) == 6  # view sağlam


def test_sql_accepts_select_and_with(project: Path) -> None:
    with AnalyticsDB(project_root=project) as db:
        assert db.sql("select 1 as x;").x.tolist() == [1]
        assert db.sql("WITH a AS (SELECT 2 AS y) SELECT y FROM a").y.tolist() == [2]


def test_symbol_filter_is_validated(project: Path) -> None:
    with AnalyticsDB(project_root=project) as db:
        with pytest.raises(AnalyticsError, match="Geçersiz sembol"):
            db.query("run_status", symbol="AAA' OR '1'='1")
        with pytest.raises(AnalyticsError, match="Bilinmeyen"):
            db.query("nope")


def test_sqlite_opened_read_only_and_unchanged(full_project: Path) -> None:
    paths = [full_project / "data" / "stock_models.db", full_project / "data" / "serving_pool.db"]
    before = [_sha(p) for p in paths]
    conn = srcs.open_readonly(paths[0])
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM experiments")
    finally:
        conn.close()
    with AnalyticsDB(project_root=full_project) as db:
        for name in QUERIES:
            db.query(name)
    assert [_sha(p) for p in paths] == before


def test_main_db_and_serving_db_in_same_file(tmp_path: Path) -> None:
    db_path = tmp_path / "one.db"
    StockModelDB(str(db_path))
    PeerStore(str(db_path)).write_run(
        GlobalRunMeta(model_name="G", as_of_date="2026-01-01", icir=1.0), pd.DataFrame({"symbol": ["AAA"]}))
    with AnalyticsDB(project_root=tmp_path, main_db=db_path, serving_db=db_path) as db:
        assert {"experiments", "peer_runs", "peer_scores"} <= set(db.views())
        assert db.sql("SELECT count(*) AS n FROM peer_scores").n.iloc[0] == 1


def test_cli_query_and_sql(project: Path, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "analytics_cli", Path(__file__).resolve().parents[1] / "tools" / "analytics.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    root = ["--project-root", str(project)]
    assert cli.run(root + ["query", "run_status", "--symbol", "AAA", "--format", "csv"]) == 0
    assert capsys.readouterr().out.splitlines()[1].startswith("AAA,3,2,1")
    out = tmp_path / "o" / "lb.md"
    assert cli.run(root + ["query", "leaderboard", "--format", "md", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("| symbol |")
    assert cli.run(root + ["sql", "DELETE FROM runs"]) == 2
    assert "HATA" in capsys.readouterr().err
    assert cli.run(root + ["list"]) == 0 and "leaderboard" in capsys.readouterr().out
    assert cli.run(root + ["views"]) == 0 and "run_model_metrics" in capsys.readouterr().out
    assert cli.run(root + ["query", "data_coverage"]) == 2  # panel yok
