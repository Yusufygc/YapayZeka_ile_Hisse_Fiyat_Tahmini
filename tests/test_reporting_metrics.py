# -*- coding: utf-8 -*-

import os
import shutil
import unittest

import numpy as np
import pandas as pd

from src.backtesting.metrics import summarize_backtest
from src.database.stock_model_db import compute_composite_score
from src.evaluation.financial_metrics import compute_financial_metrics, compute_quantile_metrics
from src.evaluator import enrich_with_benchmark_metrics, save_metrics_report
from src.pipeline.evaluation_manager import EvaluationManager


class ReportingMetricsTests(unittest.TestCase):
    def test_quantile_metrics_include_coverage_and_winkler(self):
        y_true = np.array([10.0, 11.0, 12.0])
        q_preds = np.array([
            [9.0, 10.0, 11.0],
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0],
        ])

        metrics = compute_quantile_metrics(y_true, q_preds)

        self.assertIn("Pinball_Loss", metrics)
        self.assertIn("P10_P90_Coverage", metrics)
        self.assertIn("Winkler_Score", metrics)
        self.assertGreaterEqual(metrics["P10_P90_Coverage"], 0.0)

    def test_benchmark_ineligible_model_is_not_report_leader(self):
        metrics = {
            "Naive Zero Return": {
                "RMSE": 1.0,
                "MAE": 1.0,
                "MAPE": 0.01,
                "Dir_Acc": 50.0,
                "Sharpe": 0.0,
                "Hit_Rate": 50.0,
                "Neutral_Rate": 0.0,
                "BuyHold_Sharpe": 0.0,
            },
            "Overfit Model": {
                "RMSE": 1.2,
                "MAE": 0.8,
                "MAPE": 0.01,
                "Dir_Acc": 80.0,
                "Sharpe": 3.0,
                "Hit_Rate": 80.0,
                "Neutral_Rate": 0.0,
                "BuyHold_Sharpe": 0.0,
                "Composite_Score": 99.0,
            },
        }
        enriched = enrich_with_benchmark_metrics(metrics)
        enriched["Naive Zero Return"]["Composite_Score"] = 50.0
        enriched["Overfit Model"]["Composite_Score"] = 99.0

        tmp = os.path.abspath(os.path.join("outputs", "_test_reporting_metrics"))
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        try:
            report_path = os.path.join(tmp, "metrics_report.csv")
            df = save_metrics_report(enriched, report_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertEqual(df.index[0], "Naive Zero Return")
        self.assertFalse(bool(enriched["Overfit Model"]["Eligible_For_Leader"]))

    def test_backtest_summary_includes_cagr_sortino_and_cost_drag(self):
        equity_curve = pd.DataFrame({
            "Equity": [1.01, 1.0, 1.03],
            "BuyHold_Equity": [1.01, 1.02, 1.03],
            "Net_Return": [0.01, -0.0099, 0.03],
            "Realized_Return": [0.01, 0.0099, 0.0098],
            "Position": [1.0, 1.0, 0.0],
            "Signal": [1.0, 1.0, 0.0],
            "Transaction_Cost": [0.001, 0.0, 0.001],
            "Commission_Cost": [0.0005, 0.0, 0.0005],
            "Slippage_Cost": [0.0005, 0.0, 0.0005],
            "Entry_Transaction_Cost": [0.001, 0.0, 0.0],
            "Exit_Transaction_Cost": [0.0, 0.0, 0.001],
            "Entry_Event": [1.0, 0.0, 0.0],
            "Exit_Event": [0.0, 0.0, 1.0],
            "Gross_Return": [0.011, -0.0099, 0.031],
        })
        trades = pd.DataFrame({
            "Net_Return": [0.02],
            "Holding_Period": [2],
        })
        summary = summarize_backtest({
            "model_name": "Model",
            "equity_curve": equity_curve,
            "trades": trades,
        })

        self.assertIn("CAGR", summary)
        self.assertIn("Sortino", summary)
        self.assertIn("Cost_Drag", summary)
        self.assertIn("Commission_Drag", summary)
        self.assertIn("Slippage_Drag", summary)
        self.assertEqual(summary["Avg_Holding_Period"], 2.0)
        self.assertEqual(summary["Turnover"], 2.0)
        self.assertIn("VaR_95", summary)
        self.assertIn("CVaR_95", summary)
        self.assertIn("BuyHold_VaR_95", summary)
        self.assertIn("Deflated_Sharpe", summary)

    def test_deflated_sharpe_penalizes_multiple_trials(self):
        equity_curve = pd.DataFrame({
            "Equity": np.cumprod(1.0 + np.array([0.01, -0.004, 0.006, 0.002, -0.003, 0.008])),
            "BuyHold_Equity": np.cumprod(1.0 + np.array([0.005, 0.004, -0.001, 0.002, 0.001, 0.003])),
            "Net_Return": [0.01, -0.004, 0.006, 0.002, -0.003, 0.008],
            "Realized_Return": [0.005, 0.004, -0.001, 0.002, 0.001, 0.003],
            "Position": [1.0] * 6,
            "Signal": [1.0] * 6,
        })
        result = {"model_name": "Model", "equity_curve": equity_curve, "trades": pd.DataFrame()}

        single = summarize_backtest(result, risk_free_annual=0.0, trial_count=1)
        many = summarize_backtest(result, risk_free_annual=0.0, trial_count=16)

        self.assertLess(many["Deflated_Sharpe"], single["Deflated_Sharpe"])
        self.assertLess(many["Sharpe_Probabilistic_Score"], single["Sharpe_Probabilistic_Score"])

    def test_risk_free_rate_reduces_forecast_sharpe(self):
        prev_close = np.full(5, 100.0)
        returns = np.array([0.010, 0.012, 0.009, 0.011, 0.013])
        y_true = prev_close * (1.0 + returns)
        y_pred = y_true.copy()

        zero_rf = compute_financial_metrics(
            y_true,
            y_pred,
            prev_close=prev_close,
            target_mode="price",
            risk_free_annual=0.0,
        )
        high_rf = compute_financial_metrics(
            y_true,
            y_pred,
            prev_close=prev_close,
            target_mode="price",
            risk_free_annual=0.40,
        )

        self.assertLess(high_rf["Sharpe"], zero_rf["Sharpe"])
        self.assertLess(high_rf["BuyHold_Sharpe"], zero_rf["BuyHold_Sharpe"])

    def test_zero_drawdown_calmar_is_infinite(self):
        equity_curve = pd.DataFrame({
            "Equity": [1.01, 1.02, 1.03],
            "BuyHold_Equity": [1.01, 1.02, 1.03],
            "Net_Return": [0.01, 0.0099, 0.0098],
            "Realized_Return": [0.01, 0.0099, 0.0098],
            "Position": [1.0, 1.0, 1.0],
            "Signal": [1.0, 1.0, 1.0],
        })
        summary = summarize_backtest({
            "model_name": "NoDD",
            "equity_curve": equity_curve,
            "trades": pd.DataFrame(),
        })

        self.assertTrue(np.isinf(summary["Calmar"]))

    def test_composite_score_ignores_mape(self):
        metrics = {
            "RMSE_vs_benchmark": 0.90,
            "DirAcc_vs_benchmark": 5.0,
            "Sharpe_excess_vs_buy_hold": 0.5,
            "Neutral_Rate": 0.0,
            "Dir_Acc": 55.0,
            "Eligible_For_Leader": True,
        }

        low_mape = dict(metrics, MAPE=0.01)
        high_mape = dict(metrics, MAPE=0.95)

        self.assertEqual(compute_composite_score(low_mape), compute_composite_score(high_mape))

    def test_single_split_ensemble_rows_are_derived_from_predictions(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.ensemble_enabled = True
        manager.ensemble_weights = {}
        manager.y_true_aligned = np.array([10.0, 11.0, 12.0])
        manager.prev_close_aligned = np.array([9.5, 10.5, 11.5])
        manager.predictions = {
            "Model A": np.array([10.0, 11.0, 12.0]),
            "Model B": np.array([9.0, 10.0, 11.0]),
        }
        manager.prediction_targets = {
            "Model A": np.array([0.01, 0.02, 0.03]),
            "Model B": np.array([0.00, 0.01, 0.02]),
        }
        manager.single_backtest_inputs = {
            "Model A": {
                "dates": pd.date_range("2024-01-01", periods=3),
                "prediction_dates": pd.date_range("2024-01-01", periods=3),
                "y_true_price": np.array([10.0, 11.0, 12.0]),
                "pred_price": np.array([10.0, 11.0, 12.0]),
                "prev_close": np.array([9.5, 10.5, 11.5]),
                "pred_target": np.array([0.01, 0.02, 0.03]),
                "y_true_target": np.array([0.01, 0.02, 0.03]),
            }
        }

        manager._add_single_split_ensembles()

        self.assertIn("Ensemble Equal Weight", manager.predictions)
        self.assertIn("Ensemble Inverse RMSE", manager.predictions)
        self.assertIn("Ensemble Equal Weight", manager.single_backtest_inputs)
        np.testing.assert_allclose(manager.predictions["Ensemble Equal Weight"], np.array([9.5, 10.5, 11.5]))


if __name__ == "__main__":
    unittest.main()
