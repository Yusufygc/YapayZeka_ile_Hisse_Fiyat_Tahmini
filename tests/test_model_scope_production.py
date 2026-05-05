# -*- coding: utf-8 -*-

import os
import shutil

from src.database.stock_model_db import StockModelDB
from src.experiments.experiment_tracker import ExperimentTracker
from src.pipeline.evaluation_services import MetricsReportingService
from src.pipeline.model_scope import BENCHMARK_MODELS, reportable_model_names
from src.pipeline.model_trainer import ModelTrainer


def _workspace_tmp(name: str) -> str:
    path = os.path.abspath(os.path.join("outputs", "_test_model_scope", name))
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def test_selected_tft_scope_keeps_only_tft_as_candidate_and_naive_as_benchmarks():
    tmp = _workspace_tmp("trainer_scope")
    trainer = ModelTrainer(
        stock_symbol="TEST",
        tracker=ExperimentTracker(tmp),
        feature_names=["f1"],
        selected_models=["TFT"],
        dataset_metadata={"target_mode": "log_return"},
    )

    assert trainer.candidate_models == {"TFT"}
    assert trainer.benchmark_models == set(BENCHMARK_MODELS)
    assert reportable_model_names(
        ["TFT", "ElasticNet Return", "Naive Last Value", "Naive Zero Return"],
        trainer.candidate_models,
    ) == {"TFT", "Naive Last Value", "Naive Zero Return"}


def test_select_best_model_ignores_benchmarks_and_non_candidates():
    metrics = {
        "Naive Last Value": {
            "Composite_Score": 99.0,
            "RMSE": 0.1,
            "Candidate_For_Selection": False,
        },
        "ElasticNet Return": {
            "Composite_Score": 98.0,
            "RMSE": 0.2,
            "Candidate_For_Selection": False,
        },
        "TFT": {
            "Composite_Score": 40.0,
            "RMSE": 1.0,
            "Candidate_For_Selection": True,
        },
    }

    assert MetricsReportingService._select_best_model(metrics) == "TFT"


def test_best_models_latest_final_holdout_production_candidate_wins():
    tmp = _workspace_tmp("db_policy")
    db = StockModelDB(os.path.join(tmp, "stock_models.db"))
    metadata = {
        "target_mode": "log_return",
        "feature_mode": "stationary_features",
        "scaling_mode": "robust_x_standard_y_clip",
    }

    db.log_experiment(
        "TEST",
        "LSTM",
        {"RMSE": 0.1, "MAE": 0.1, "MAPE": 0.1, "Dir_Acc": 90.0, "Sharpe": 5.0, "Hit_Rate": 90.0},
        validation_mode="walk_forward",
        dataset_metadata=metadata,
    )
    assert db.get_best_model("TEST") is None

    first_id = db.log_experiment(
        "TEST",
        "ElasticNet Return",
        {"RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0, "Dir_Acc": 55.0, "Sharpe": 0.4, "Hit_Rate": 55.0},
        validation_mode="final_holdout",
        dataset_metadata=metadata,
        is_production_candidate=True,
        selection_source="wf_selection",
        run_id="run_1",
    )
    first_best = db.get_best_model("TEST")
    assert first_best["experiment_id"] == first_id
    assert first_best["model_name"] == "ElasticNet Return"

    second_id = db.log_experiment(
        "TEST",
        "Ridge Return",
        {"RMSE": 2.0, "MAE": 2.0, "MAPE": 2.0, "Dir_Acc": 45.0, "Sharpe": -0.2, "Hit_Rate": 45.0},
        validation_mode="final_holdout",
        dataset_metadata=metadata,
        is_production_candidate=True,
        selection_source="wf_selection",
        run_id="run_2",
    )
    latest_best = db.get_best_model("TEST")

    assert latest_best["experiment_id"] == second_id
    assert latest_best["model_name"] == "Ridge Return"
    assert latest_best["validation_mode"] == "final_holdout"
    assert latest_best["run_id"] == "run_2"
