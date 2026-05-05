# -*- coding: utf-8 -*-

import os
import shutil

import numpy as np
import pandas as pd

from src.database.stock_model_db import StockModelDB
from src.forecasting.bist_rules import BistMarketRules
from src.forecasting.runner import ForecastRunner
from src.pipeline.data_manager import DataManager


def _workspace_tmp(name: str) -> str:
    path = os.path.abspath(os.path.join("outputs", "_test_forecasting", name))
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def test_bist_price_tick_rounding_and_band():
    rules = BistMarketRules(calendar_path=None)

    assert rules.price_tick(19.99) == 0.01
    assert rules.price_tick(20.00) == 0.02
    assert rules.round_to_tick(12.126, direction="nearest") == 12.13

    band = rules.price_band(101.13)
    assert band.lower_band >= band.base_price * 0.90
    assert band.upper_band <= band.base_price * 1.10
    bounded, _ = rules.bound_forecast_price(999.0, 101.13)
    assert bounded == band.upper_band


def test_bist_calendar_skips_weekends_and_closed_days():
    tmp = _workspace_tmp("calendar")
    calendar = os.path.join(tmp, "bist_calendar.csv")
    with open(calendar, "w", encoding="utf-8") as handle:
        handle.write(
        "\n".join([
            "Date,Is_Trading_Day,Session_Type,Note",
            "2026-05-01,False,closed,holiday",
            "2026-05-04,True,full,open",
            "2026-05-05,True,full,open",
        ])
    )
    rules = BistMarketRules(calendar)

    dates = rules.next_trading_days("2026-04-30", 2)

    assert [date.strftime("%Y-%m-%d") for date in dates] == ["2026-05-04", "2026-05-05"]


def test_sparse_bist_calendar_does_not_jump_to_first_seed_date():
    tmp = _workspace_tmp("sparse_calendar")
    calendar = os.path.join(tmp, "bist_calendar.csv")
    with open(calendar, "w", encoding="utf-8") as handle:
        handle.write(
        "\n".join([
            "Date,Is_Trading_Day,Session_Type,Note",
            "2026-04-23,False,closed,holiday",
            "2026-05-01,False,closed,holiday",
        ])
    )
    rules = BistMarketRules(calendar)

    dates = rules.next_trading_days("2025-10-24", 3)

    assert [date.strftime("%Y-%m-%d") for date in dates] == [
        "2025-10-27",
        "2025-10-28",
        "2025-10-29",
    ]


def test_trend_label_thresholds():
    assert BistMarketRules.trend_label(0.03, 0.02) == "UP"
    assert BistMarketRules.trend_label(-0.03, 0.02) == "DOWN"
    assert BistMarketRules.trend_label(0.01, 0.02) == "FLAT"


def test_forecast_db_idempotency_and_resolution():
    tmp = _workspace_tmp("db")
    db = StockModelDB(os.path.join(tmp, "stock_models.db"))
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
        },
        {
            "target_date": "2026-05-05",
            "horizon_index": 2,
            "raw_predicted_close": 102.0,
            "bounded_predicted_close": 102.0,
            "predicted_return": 0.0099,
            "lower_band": 90.9,
            "upper_band": 111.1,
            "price_tick": 0.05,
        },
    ]

    kwargs = dict(
        stock_symbol="TEST",
        model_name="Ridge Return",
        source_experiment_id=None,
        last_observed_date="2026-04-30",
        last_close=100.0,
        horizon_days=2,
        trend_label="UP",
        weekly_expected_return=0.02,
        trend_threshold=0.005,
        rules_version="test_rules",
        points=points,
    )
    first = db.log_forecast_run(**kwargs)
    second = db.log_forecast_run(**kwargs)

    assert first == second
    latest = db.get_latest_forecast("TEST")
    assert latest is not None
    assert len(latest["points"]) == 2

    resolved = db.resolve_forecasts("TEST", {"2026-05-04": 101.5, "2026-05-05": 102.5})
    latest = db.get_latest_forecast("TEST")

    assert resolved == 2
    assert latest["accuracy_summary"]["resolved_points"] == 2
    assert latest["points"][0]["actual_close"] == 101.5


def test_forecast_runner_replacement_excludes_baselines():
    project_root = _workspace_tmp("baseline_selection")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "stock_models.db")
    db = StockModelDB(db_path)
    metadata = {
        "target_mode": "log_return",
        "feature_mode": "stationary_features",
        "scaling_mode": "robust_x_standard_y_clip",
    }
    db.log_experiment(
        "TEST",
        "Naive Zero Return",
        {"RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0, "Dir_Acc": 60.0, "Sharpe": 1.0, "Hit_Rate": 60.0},
        dataset_metadata=metadata,
    )
    db.log_experiment(
        "TEST",
        "Ridge Return",
        {"RMSE": 2.0, "MAE": 2.0, "MAPE": 2.0, "Dir_Acc": 45.0, "Sharpe": 0.0, "Hit_Rate": 45.0},
        dataset_metadata=metadata,
    )
    runner = ForecastRunner(project_root=project_root, db_path=db_path)

    replacement = runner._best_trainable_experiment("TEST")

    assert replacement is not None
    assert replacement["model_name"] == "Ridge Return"


def test_forecast_runner_creates_five_bist_bounded_points(monkeypatch):
    project_root = _workspace_tmp("runner")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, "TEST.csv")
    pd.DataFrame({
        "Date": pd.date_range("2026-03-01", periods=45, freq="B"),
        "Close": np.linspace(100.0, 120.0, 45),
    }).to_csv(data_file, index=False)
    calendar_path = os.path.join(data_dir, "bist_calendar.csv")
    with open(calendar_path, "w", encoding="utf-8") as handle:
        handle.write(
        "\n".join([
            "Date,Is_Trading_Day,Session_Type,Note",
            "2026-05-01,False,closed,holiday",
            "2026-05-04,True,full,open",
            "2026-05-05,True,full,open",
            "2026-05-06,True,full,open",
            "2026-05-07,True,full,open",
            "2026-05-08,True,full,open",
        ])
    )
    db_path = os.path.join(data_dir, "stock_models.db")
    db = StockModelDB(db_path)
    db.log_experiment(
        "TEST",
        "Ridge Return",
        {"RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0, "Dir_Acc": 60.0, "Sharpe": 0.5, "Hit_Rate": 55.0},
        validation_mode="final_holdout",
        dataset_metadata={
            "target_mode": "log_return",
            "feature_mode": "stationary_features",
            "scaling_mode": "robust_x_standard_y_clip",
        },
        is_production_candidate=True,
        selection_source="test_final_holdout",
        run_id="test_run",
    )

    def fake_ingest(self):
        self.df = pd.DataFrame({
            "Date": pd.date_range("2026-03-02", periods=45, freq="B"),
            "Close": np.linspace(100.0, 120.0, 45),
            "Feature": np.linspace(0.0, 1.0, 45),
        })
        self.feature_names = ["Feature"]
        self.dataset_metadata = {
            "target_mode": self.data_cfg.target_mode,
            "feature_mode": self.data_cfg.feature_mode,
            "scaling_mode": self.data_cfg.scaling_mode,
        }

    monkeypatch.setattr(DataManager, "ingest_and_engineer", fake_ingest)
    runner = ForecastRunner(
        project_root=project_root,
        db_path=db_path,
        calendar_path=calendar_path,
    )

    result = runner.run_symbol(
        symbol="TEST",
        data_file=data_file,
        horizon_days=5,
        use_macro=False,
    )

    latest = runner.db.get_latest_forecast("TEST")
    assert result.run_id == latest["id"]
    assert len(result.points) == 5
    for point in result.points:
        assert point["lower_band"] <= point["bounded_predicted_close"] <= point["upper_band"]
