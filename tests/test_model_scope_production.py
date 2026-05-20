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


def test_selected_lstm_scope_keeps_only_lstm_as_candidate_and_naive_as_benchmarks():
    tmp = _workspace_tmp("trainer_scope")
    trainer = ModelTrainer(
        stock_symbol="TEST",
        tracker=ExperimentTracker(tmp),
        feature_names=["f1"],
        selected_models=["LSTM"],
        dataset_metadata={"target_mode": "log_return"},
    )

    assert trainer.candidate_models == {"LSTM"}
    assert trainer.benchmark_models == set(BENCHMARK_MODELS)
    assert reportable_model_names(
        ["LSTM", "ElasticNet Return", "Naive Last Value", "Naive Zero Return"],
        trainer.candidate_models,
    ) == {"LSTM", "Naive Last Value", "Naive Zero Return"}


def test_selected_lstm_lite_scope_keeps_only_lstm_lite_as_candidate():
    tmp = _workspace_tmp("trainer_lstm_lite_scope")
    trainer = ModelTrainer(
        stock_symbol="TEST",
        tracker=ExperimentTracker(tmp),
        feature_names=["f1"],
        selected_models=["LSTM Lite"],
        dataset_metadata={"target_mode": "log_return"},
    )

    assert trainer.candidate_models == {"LSTM Lite"}
    assert reportable_model_names(
        ["LSTM", "LSTM Lite", "Naive Last Value", "Naive Zero Return"],
        trainer.candidate_models,
    ) == {"LSTM Lite", "Naive Last Value", "Naive Zero Return"}


def test_selected_attention_lstm_v2_is_opt_in_candidate_only():
    tmp = _workspace_tmp("trainer_attention_v2_scope")
    default_trainer = ModelTrainer(
        stock_symbol="TEST",
        tracker=ExperimentTracker(tmp),
        feature_names=["f1"],
        selected_models=None,
        dataset_metadata={"target_mode": "log_return"},
    )
    selected_trainer = ModelTrainer(
        stock_symbol="TEST",
        tracker=ExperimentTracker(tmp),
        feature_names=["f1"],
        selected_models=["AttentionLSTM v2"],
        dataset_metadata={"target_mode": "log_return"},
    )

    assert "AttentionLSTM v2" not in default_trainer.candidate_models
    assert selected_trainer.candidate_models == {"AttentionLSTM v2"}


def test_reportable_models_include_only_production_ensembles():
    assert reportable_model_names(
        ["Ensemble Inverse RMSE", "Ensemble Meta-Stacker", "Ensemble Cash-Gated"],
        {"LSTM"},
    ) == {"Ensemble Inverse RMSE", "Ensemble Cash-Gated"}


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
        "LSTM": {
            "Composite_Score": 40.0,
            "RMSE": 1.0,
            "Candidate_For_Selection": True,
        },
    }

    assert MetricsReportingService._select_best_model(metrics) == "LSTM"


def test_best_models_highest_scored_final_holdout_candidate_wins():
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
        {
            "RMSE": 1.0,
            "MAE": 1.0,
            "MAPE": 1.0,
            "Dir_Acc": 55.0,
            "Sharpe": 0.4,
            "Hit_Rate": 55.0,
            "Trade_Count": 10,
        },
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
        {
            "RMSE": 2.0,
            "MAE": 2.0,
            "MAPE": 2.0,
            "Dir_Acc": 45.0,
            "Sharpe": -0.2,
            "Hit_Rate": 45.0,
            "Trade_Count": 10,
        },
        validation_mode="final_holdout",
        dataset_metadata=metadata,
        is_production_candidate=True,
        selection_source="wf_selection",
        run_id="run_2",
    )
    lower_score_best = db.get_best_model("TEST")

    assert second_id
    assert lower_score_best["experiment_id"] == first_id
    assert lower_score_best["model_name"] == "ElasticNet Return"

    third_id = db.log_experiment(
        "TEST",
        "LSTM",
        {
            "RMSE": 0.1,
            "MAE": 0.1,
            "MAPE": 0.1,
            "Dir_Acc": 90.0,
            "Sharpe": 5.0,
            "Hit_Rate": 90.0,
            "Trade_Count": 10,
        },
        validation_mode="final_holdout",
        dataset_metadata=metadata,
        is_production_candidate=True,
        selection_source="wf_selection",
        run_id="run_3",
    )
    latest_best = db.get_best_model("TEST")

    assert latest_best["experiment_id"] == third_id
    assert latest_best["model_name"] == "LSTM"
    assert latest_best["validation_mode"] == "final_holdout"
    assert latest_best["run_id"] == "run_3"


def test_production_ensemble_can_be_best_with_metadata():
    tmp = _workspace_tmp("db_ensemble_policy")
    db = StockModelDB(os.path.join(tmp, "stock_models.db"))
    metadata = {
        "target_mode": "log_return",
        "feature_mode": "stationary_features",
        "scaling_mode": "robust_x_standard_y_clip",
    }

    exp_id = db.log_experiment(
        "TEST",
        "Ensemble Inverse RMSE",
        {
            "RMSE": 0.8,
            "MAE": 0.8,
            "MAPE": 1.0,
            "Dir_Acc": 60.0,
            "Sharpe": 0.5,
            "Hit_Rate": 60.0,
            "RMSE_vs_benchmark": 0.95,
            "Trade_Count": 8,
            "Ensemble_Method": "Inverse RMSE",
            "Ensemble_Weights": '{"Ridge Return": 0.4, "LSTM": 0.6}',
        },
        validation_mode="walk_forward",
        dataset_metadata=metadata,
        is_production_candidate=True,
        selection_source="walk_forward_production_ensemble",
        run_id="run_ens",
    )

    best = db.get_best_model("TEST")
    assert best["experiment_id"] == exp_id
    assert best["model_name"] == "Ensemble Inverse RMSE"
    assert best["eligibility_status"] == "eligible"
    assert "Ridge Return" in best["ensemble_metadata_json"]
