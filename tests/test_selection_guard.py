# -*- coding: utf-8 -*-
"""selection_guard.py birim testleri."""
import pytest

from src.pipeline.selection_guard import compute_eligibility, evaluate_best_model_eligibility


class TestComputeEligibility:
    def test_eligible_prod_candidate(self):
        status, reason = compute_eligibility(
            model_name="XGBoost",
            is_production_candidate=True,
            is_baseline=False,
            total_trade_count=10,
            min_trades=6,
        )
        assert status == "eligible"
        assert reason == ""

    def test_no_candidate_not_prod(self):
        status, reason = compute_eligibility(
            model_name="Naive Last Value",
            is_production_candidate=False,
            is_baseline=True,
            total_trade_count=20,
            min_trades=6,
        )
        assert status == "no_candidate"

    def test_naive_low_trades(self):
        status, reason = compute_eligibility(
            model_name="Naive Drift",
            is_production_candidate=True,
            is_baseline=True,
            total_trade_count=1,
            min_trades=6,
        )
        assert status == "naive_low_trades"
        assert "1" in reason

    def test_insufficient_trades_non_naive(self):
        status, reason = compute_eligibility(
            model_name="XGBoost",
            is_production_candidate=True,
            is_baseline=False,
            total_trade_count=2,
            min_trades=6,
        )
        assert status == "insufficient_trades"

    def test_naive_with_sufficient_trades_eligible(self):
        # Naive, production candidate, ve yeterli trade varsa eligible
        status, reason = compute_eligibility(
            model_name="Naive Drift",
            is_production_candidate=True,
            is_baseline=True,
            total_trade_count=10,
            min_trades=6,
        )
        assert status == "eligible"


class TestEvaluateBestModelEligibility:
    def test_eligible_row(self):
        row = {
            "model_name": "LightGBM Return",
            "is_production_candidate": 1,
            "Trade_Count": 15,
        }
        status, reason = evaluate_best_model_eligibility(row, min_trades=6)
        assert status == "eligible"

    def test_benchmark_model_row_no_prod_flag(self):
        row = {
            "model_name": "Naive Last Value",
            "is_production_candidate": 0,
            "Trade_Count": 20,
        }
        status, _ = evaluate_best_model_eligibility(row, min_trades=6)
        assert status == "no_candidate"

    def test_benchmark_model_inferred_via_scope(self):
        row = {
            "model_name": "Naive Last Value",
            "is_production_candidate": 1,  # hatalı kayıt
            "Trade_Count": 2,
        }
        status, _ = evaluate_best_model_eligibility(row, min_trades=6)
        assert status == "naive_low_trades"

    def test_missing_trade_count_defaults_zero(self):
        row = {
            "model_name": "Ridge Return",
            "is_production_candidate": 1,
        }
        status, _ = evaluate_best_model_eligibility(row, min_trades=6)
        assert status == "insufficient_trades"

    def test_none_trade_count(self):
        row = {
            "model_name": "ElasticNet Return",
            "is_production_candidate": 1,
            "Trade_Count": None,
        }
        status, _ = evaluate_best_model_eligibility(row, min_trades=6)
        assert status == "insufficient_trades"
