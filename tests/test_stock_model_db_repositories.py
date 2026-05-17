# -*- coding: utf-8 -*-
"""Repository-boundary tests for StockModelDB Phase 4 decomposition."""

import os

from src.database.repositories import (
    BestModelRepository,
    ExperimentRepository,
    ForecastRepository,
    ForecastResolutionRepository,
    SchemaRepository,
)
from src.database.stock_model_db import StockModelDB


def _db(tmp_path) -> StockModelDB:
    return StockModelDB(os.path.join(str(tmp_path), "stock_models.db"))


def test_stock_model_db_composes_internal_repositories(tmp_path):
    db = _db(tmp_path)

    assert isinstance(db.schema_repository, SchemaRepository)
    assert isinstance(db.experiment_repository, ExperimentRepository)
    assert isinstance(db.best_model_repository, BestModelRepository)
    assert isinstance(db.forecast_repository, ForecastRepository)
    assert isinstance(db.forecast_resolution_repository, ForecastResolutionRepository)


def test_schema_initialization_is_idempotent(tmp_path):
    db = _db(tmp_path)

    db._init_db()
    db._init_db()

    assert db.get_leaderboard() == []


def test_production_best_update_is_preserved(tmp_path):
    db = _db(tmp_path)
    metadata = {
        "target_mode": "log_return",
        "feature_mode": "stationary_features",
        "scaling_mode": "robust_x_standard_y_clip",
    }

    exp_id = db.log_experiment(
        stock_symbol="TEST",
        model_name="Ridge Return",
        metrics={"RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0, "Dir_Acc": 55.0, "Sharpe": 0.3, "Hit_Rate": 50.0},
        validation_mode="final_holdout",
        dataset_metadata=metadata,
        is_production_candidate=True,
        selection_source="test",
        run_id="run_1",
    )

    best = db.get_best_model("TEST")

    assert best["experiment_id"] == exp_id
    assert best["model_name"] == "Ridge Return"
    assert best["target_mode"] == "log_return"


def test_forecast_repository_keeps_run_idempotency_and_resolution(tmp_path):
    db = _db(tmp_path)
    points = [
        {
            "target_date": "2026-05-04",
            "horizon_index": 1,
            "raw_predicted_close": 101.0,
            "bounded_predicted_close": 101.0,
            "predicted_return": 0.01,
            "lower_band": 90.0,
            "upper_band": 110.0,
            "price_tick": 0.05,
        }
    ]
    kwargs = dict(
        stock_symbol="TEST",
        model_name="Ridge Return",
        source_experiment_id=None,
        last_observed_date="2026-04-30",
        last_close=100.0,
        horizon_days=1,
        trend_label="UP",
        weekly_expected_return=0.01,
        trend_threshold=0.005,
        rules_version="test",
        points=points,
    )

    first = db.log_forecast_run(**kwargs)
    second = db.log_forecast_run(**kwargs)
    resolved = db.resolve_forecasts("TEST", {"2026-05-04": 101.5})
    latest = db.get_latest_forecast("TEST")

    assert first == second
    assert resolved == 1
    assert len(latest["points"]) == 1
    assert latest["accuracy_summary"]["resolved_points"] == 1
