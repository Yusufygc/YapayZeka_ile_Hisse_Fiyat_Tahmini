# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def test_data_updater_returns_failed_result_for_missing_file():
    from src.data.data_updater import DataUpdater

    result = DataUpdater.check_and_update(
        str(Path(tempfile.mkdtemp()) / "MISSING.csv"),
        "ASELS",
        interactive=False,
    )

    assert result.status == "failed"
    assert result.rows_added == 0
    assert result.error


def test_data_refresh_service_stops_on_data_update_failure():
    from src.api.services.data_refresh_service import DataRefreshError, DataRefreshService
    from src.database.stock_model_db import StockModelDB

    project_root = Path(tempfile.mkdtemp())
    (project_root / "data").mkdir(parents=True)
    db = StockModelDB(str(project_root / "data" / "stock_models.db"))

    service = DataRefreshService(
        db=db,
        project_root=str(project_root),
        start_background=False,
    )

    with pytest.raises(DataRefreshError) as exc:
        service.refresh_symbol(symbol="ASELS", best_model=None)

    assert exc.value.reason == "data_update_failed"
    assert exc.value.payload["data_update"]["status"] == "failed"


def test_db_maintenance_backup_reset_creates_clean_schema(tmp_path):
    from src.cli.db_maintenance import backup_reset
    from src.database.stock_model_db import StockModelDB

    db_path = tmp_path / "stock_models.db"
    db = StockModelDB(str(db_path))
    db.log_experiment(
        stock_symbol="ASELS",
        model_name="XGBoost",
        metrics={"RMSE": 1.0, "MAE": 1.0, "Dir_Acc": 55.0, "Sharpe": 0.1, "Trade_Count": 10},
        validation_mode="final_holdout",
        is_production_candidate=True,
    )

    result = backup_reset(db_path)

    assert result["backup_path"]
    assert Path(result["backup_path"]).exists()
    assert result["schema_ok"] is True
    assert result["table_counts"]["experiments"] == 0
    assert result["table_counts"]["analysis_refresh_jobs"] == 0


def test_db_maintenance_backfills_trade_metrics_from_backtest_reports(tmp_path):
    from src.cli.db_maintenance import backfill_run_metrics
    from src.database.stock_model_db import StockModelDB

    db_path = tmp_path / "stock_models.db"
    db = StockModelDB(str(db_path))
    exp_id = db.log_experiment(
        stock_symbol="ASELS",
        model_name="LSTM",
        metrics={"RMSE": 1.0, "MAE": 1.0, "Dir_Acc": 62.0, "Sharpe": 0.8},
        validation_mode="final_holdout",
        is_production_candidate=True,
        run_id="run_1",
    )
    outputs = tmp_path / "outputs"
    report_dir = outputs / "ASELS" / "runs" / "run_1" / "csv"
    report_dir.mkdir(parents=True)
    (report_dir / "backtest_report_final_holdout.csv").write_text(
        "\n".join([
            "Model;Net_Return;BuyHold_Return;Max_Drawdown;Trade_Count;Signal_Diagnosis",
            "LSTM;0.12;0.05;-0.03;1;insufficient_trades",
        ]),
        encoding="utf-8",
    )

    first = backfill_run_metrics(db_path=db_path, outputs_base=outputs, symbol="ASELS")
    second = backfill_run_metrics(db_path=db_path, outputs_base=outputs, symbol="ASELS")

    refreshed = StockModelDB(str(db_path))
    experiment = refreshed.get_experiments(stock_symbol="ASELS", limit=1)[0]
    best = refreshed.get_best_model("ASELS")

    assert exp_id
    assert first["updated_rows"] == 1
    assert second["updated_rows"] == 1
    assert experiment["trade_count"] == 1
    assert experiment["signal_diagnosis"] == "insufficient_trades"
    assert best["trade_count"] == 1
    assert best["eligibility_status"] == "insufficient_trades"
    assert "Trade sayisi (1)" in best["eligibility_reason"]
