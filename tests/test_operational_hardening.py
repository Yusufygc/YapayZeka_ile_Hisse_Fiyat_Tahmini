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
