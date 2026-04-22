# -*- coding: utf-8 -*-

import os
import shutil
import unittest

import numpy as np
import pandas as pd

from src.backtesting.metrics import summarize_backtest
from src.evaluation.financial_metrics import compute_quantile_metrics
from src.evaluator import enrich_with_benchmark_metrics, save_metrics_report


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


if __name__ == "__main__":
    unittest.main()
