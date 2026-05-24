# -*- coding: utf-8 -*-
"""Model-scoped run result export tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.pipeline.model_result_exporter import (
    export_single_split_result,
    export_walk_forward_results,
    model_result_slug,
)


def test_model_result_slug_is_stable():
    assert model_result_slug("Prophet-ML/DL Hybrid") == "prophet_ml_dl_hybrid"
    assert model_result_slug("LSTM Lite") == "lstm_lite"


def test_single_split_export_writes_model_folder(tmp_path):
    owner = SimpleNamespace(
        outputs_dir=str(tmp_path),
        dataset_metadata={"run_id": "run-1"},
        latest_tensors={
            "dates_test": ["2026-01-01", "2026-01-02"],
            "dates_prediction": ["2026-01-02", "2026-01-05"],
        },
        y_true_aligned=np.array([100.0, 101.0]),
        predictions={"XGBoost": np.array([99.5, 102.0])},
        y_true_target_aligned=np.array([0.01, 0.02]),
        prediction_targets={"XGBoost": np.array([0.005, 0.025])},
        prev_close_aligned=np.array([99.0, 100.0]),
    )

    export_single_split_result(
        owner,
        model_name="XGBoost",
        metrics={"RMSE": np.float64(1.25), "Trade_Count": np.int64(3)},
        model_path=str(tmp_path / "models" / "xgboost_model.pkl"),
    )

    result_dir = tmp_path / "model_results" / "xgboost"
    assert result_dir.exists()

    metrics = json.loads((result_dir / "metrics_single_split.json").read_text(encoding="utf-8"))
    assert metrics["RMSE"] == 1.25
    assert metrics["Trade_Count"] == 3

    preds = pd.read_csv(result_dir / "predictions_single_split.csv", sep=";")
    assert list(preds.columns) == [
        "date",
        "prediction_date",
        "y_true_price",
        "y_pred_price",
        "y_true_target",
        "y_pred_target",
        "prev_close",
    ]
    assert len(preds) == 2


def test_walk_forward_export_writes_fold_and_prediction_files(tmp_path):
    owner = SimpleNamespace(outputs_dir=str(tmp_path), dataset_metadata={"run_id": "run-2"})

    export_walk_forward_results(
        owner,
        metrics_by_model={"LSTM Lite": {"RMSE": 2.0}},
        fold_metrics_by_model={"LSTM Lite": [{"Model": "LSTM Lite", "Fold": 1, "RMSE": 2.2}]},
        backtest_inputs_by_model={
            "LSTM Lite": {
                "dates": ["2026-01-01"],
                "prediction_dates": ["2026-01-02"],
                "fold_ids": [1],
                "y_true_price": [100.0],
                "pred_price": [101.0],
                "y_true_target": [0.01],
                "pred_target": [0.02],
                "prev_close": [99.0],
            }
        },
    )

    result_dir = tmp_path / "model_results" / "lstm_lite"
    assert (result_dir / "metrics_walk_forward.json").exists()
    assert (result_dir / "fold_metrics_walk_forward.csv").exists()
    assert (result_dir / "predictions_walk_forward.csv").exists()
