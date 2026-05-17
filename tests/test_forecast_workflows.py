# -*- coding: utf-8 -*-
"""Workflow-boundary tests for ForecastRunner Phase 4 decomposition."""

import os

import numpy as np

from src.forecasting.runner import ForecastRunner
from src.forecasting.workflows import (
    BestModelResolver,
    ForecastDataPreparationService,
    ForecastPointGenerator,
    ForecastSymbolWorkflow,
    LatestTargetPredictionWorkflow,
    ProductionTrainingWorkflow,
)


def _runner(tmp_path) -> ForecastRunner:
    return ForecastRunner(
        project_root=str(tmp_path),
        db_path=os.path.join(str(tmp_path), "data", "stock_models.db"),
        calendar_path=None,
    )


def test_forecast_runner_composes_internal_workflows(tmp_path):
    runner = _runner(tmp_path)

    assert isinstance(runner.model_resolver, BestModelResolver)
    assert isinstance(runner.data_preparation_service, ForecastDataPreparationService)
    assert isinstance(runner.production_training_workflow, ProductionTrainingWorkflow)
    assert isinstance(runner.latest_target_prediction_workflow, LatestTargetPredictionWorkflow)
    assert isinstance(runner.forecast_point_generator, ForecastPointGenerator)
    assert isinstance(runner.symbol_workflow, ForecastSymbolWorkflow)


def test_private_forecast_helpers_delegate_and_preserve_characterization(tmp_path):
    runner = _runner(tmp_path)

    target = runner._make_target(np.array([100.0, 102.0, 101.0]), "return")
    points = runner._roll_forward_points(
        predicted_target=0.01,
        horizon_days=2,
        last_close=100.0,
        last_observed_date=np.datetime64("2026-05-01"),
        target_mode="return",
    )

    np.testing.assert_allclose(target, np.array([0.02, -0.009803921568627416]))
    assert len(points) == 2
    assert points[0]["bounded_predicted_close"] == runner._target_to_price(0.01, 100.0, "return")


def test_model_resolver_forced_model_uses_best_metadata_when_available(tmp_path):
    runner = _runner(tmp_path)
    runner.db.log_experiment(
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
        selection_source="test",
    )

    selection = runner.model_resolver.resolve("TEST", force_model_name="XGBoost")

    assert selection["model_name"] == "XGBoost"
    assert selection["target_mode"] == "log_return"
    assert selection["source_experiment_id"] is not None
