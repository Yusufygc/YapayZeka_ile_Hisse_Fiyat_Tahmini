# -*- coding: utf-8 -*-
"""stability_score hesaplama ve cross-run leaderboard birim testleri."""
import math
import sqlite3
import tempfile

import numpy as np
import pytest

from src.pipeline.evaluation_workflows import WalkForwardEvaluationWorkflow


class TestComputeStabilityScore:
    def test_all_positive_sharpes(self):
        fold_rows = [{"Sharpe": 1.0}, {"Sharpe": 0.5}, {"Sharpe": 0.8}]
        score = WalkForwardEvaluationWorkflow._compute_stability_score(fold_rows)
        # ratio=1.0, std(1.0,0.5,0.8)=0.2055..., score = 1.0 - 0.5*0.2055 ≈ 0.897
        assert score > 0.8

    def test_mixed_sharpes(self):
        fold_rows = [{"Sharpe": 1.0}, {"Sharpe": -0.5}, {"Sharpe": 0.2}]
        score = WalkForwardEvaluationWorkflow._compute_stability_score(fold_rows)
        # ratio=2/3, std > 0 → score < 2/3
        assert score < 2 / 3 + 0.01

    def test_all_negative_sharpes(self):
        fold_rows = [{"Sharpe": -1.0}, {"Sharpe": -0.5}]
        score = WalkForwardEvaluationWorkflow._compute_stability_score(fold_rows)
        assert score < 0.0

    def test_empty_rows_returns_nan(self):
        score = WalkForwardEvaluationWorkflow._compute_stability_score([])
        assert math.isnan(score)

    def test_nan_sharpe_ignored(self):
        fold_rows = [{"Sharpe": float("nan")}, {"Sharpe": 0.5}]
        score = WalkForwardEvaluationWorkflow._compute_stability_score(fold_rows)
        assert math.isfinite(score)

    def test_single_fold_no_std(self):
        fold_rows = [{"Sharpe": 0.7}]
        score = WalkForwardEvaluationWorkflow._compute_stability_score(fold_rows)
        # std=0 → score = positive_ratio = 1.0
        assert abs(score - 1.0) < 1e-9


class TestCrossRunLeaderboard:
    def _make_db(self):
        from src.database.stock_model_db import StockModelDB

        tmp = tempfile.mktemp(suffix=".db")
        db = StockModelDB(tmp)
        return db

    def test_empty_returns_empty_list(self):
        db = self._make_db()
        result = db.get_cross_run_leaderboard("TUPRS", n_runs=5)
        assert result == []

    def test_returns_model_rows(self):
        db = self._make_db()
        db.log_experiment(
            stock_symbol="TUPRS",
            model_name="XGBoost",
            metrics={"MAE": 1.0, "RMSE": 2.0, "Dir_Acc": 55.0, "Sharpe": 0.5, "Stability_Score": 0.7},
            dataset_hash="hash1",
            validation_mode="walk_forward",
            run_id="run001",
        )
        db.log_experiment(
            stock_symbol="TUPRS",
            model_name="XGBoost",
            metrics={"MAE": 1.1, "RMSE": 2.1, "Dir_Acc": 53.0, "Sharpe": 0.3, "Stability_Score": 0.5},
            dataset_hash="hash2",
            validation_mode="walk_forward",
            run_id="run002",
        )
        result = db.get_cross_run_leaderboard("TUPRS", n_runs=5)
        assert len(result) == 1
        row = result[0]
        assert row["model_name"] == "XGBoost"
        assert row["run_count"] == 2
        assert abs(row["avg_stability_score"] - 0.6) < 1e-6
