# -*- coding: utf-8 -*-

import os
import shutil
import unittest

import numpy as np
import pandas as pd

from src.backtesting.engine import run_backtest
from src.backtesting.metrics import summarize_backtest
from src.backtesting.reporting import save_backtest_report, save_fold_backtest_report
from src.backtesting.signals import SignalConfig
from src.utils.reporting_utils import with_output_extension

try:
    from src.pipeline.evaluation_manager import EvaluationManager
except Exception as exc:  # pragma: no cover - optional runtime dependency guard
    EvaluationManager = None
    EVALUATION_IMPORT_ERROR = exc


class Phase6BacktestStandardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.abspath(os.path.join("outputs", "_test_phase6_backtest_standard"))
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sample_backtest(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        prev_close = np.full(6, 100.0)
        pred_price = np.array([101.0, 101.0, 99.0, 101.0, 99.0, 101.0])
        return run_backtest(
            dates=dates,
            prediction_dates=dates - pd.Timedelta(days=1),
            y_true_price=np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0]),
            pred_price=pred_price,
            prev_close=prev_close,
            fold_ids=np.array([0, 0, 0, 1, 1, 1]),
            model_name="Phase6 Model",
            validation_mode="wf",
            target_mode="price",
            signal_mode="legacy",
            commission_bps=10.0,
            slippage_bps=5.0,
        )

    def test_backtest_propagates_fold_ids_and_separate_cost_columns(self):
        result = self._sample_backtest()
        curve = result["equity_curve"]

        self.assertEqual(curve["Fold"].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertIn("Entry_Transaction_Cost", curve.columns)
        self.assertIn("Exit_Transaction_Cost", curve.columns)
        self.assertIn("Commission_Cost", curve.columns)
        self.assertIn("Slippage_Cost", curve.columns)

        trades = result["trades"]
        if not trades.empty:
            self.assertIn("Fold", trades.columns)

    def test_fold_backtest_report_writes_distribution_and_worst_fold(self):
        result = self._sample_backtest()
        fold_df, worst_df = save_fold_backtest_report(
            {"Phase6 Model": result},
            os.path.join(self.tmp, "fold_report.csv"),
            initial_capital=100000.0,
        )

        self.assertEqual(set(fold_df["Fold"].astype(int).tolist()), {0, 1})
        self.assertEqual(worst_df["Worst_Fold_Rule"].iloc[0], "min_net_return_then_min_sharpe")
        self.assertTrue(os.path.exists(with_output_extension(os.path.join(self.tmp, "fold_report.csv"), ".md")))

    def test_multiple_testing_warning_is_written_to_backtest_report(self):
        result = self._sample_backtest()
        summary = summarize_backtest(result)
        metrics = {f"Model {idx}": dict(summary, Model=f"Model {idx}") for idx in range(8)}

        save_backtest_report(metrics, os.path.join(self.tmp, "backtest_report.csv"))
        md_path = with_output_extension(os.path.join(self.tmp, "backtest_report.csv"), ".md")

        with open(md_path, "r", encoding="utf-8") as handle:
            report = handle.read()

        self.assertIn("Multiple testing risk", report)

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_signal_threshold_calibration_uses_walk_forward_folds_only(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.signal_config = SignalConfig(
            min_directional_accuracy=52.0,
            max_rmse_vs_benchmark=1.05,
            min_composite_score=50.0,
        )
        manager.default_signal_config = manager.signal_config
        manager.signal_threshold_source = "default_config"
        manager.signal_threshold_calibration_summary = {}
        manager.dataset_metadata = {}
        manager.commission_bps = 10.0
        manager.slippage_bps = 5.0

        manager._calibrate_signal_quality_thresholds({
            "Candidate": [
                {"Fold": 0, "Dir_Acc": 53.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0},
                {"Fold": 1, "Dir_Acc": 58.0, "RMSE_vs_benchmark": 0.90, "Composite_Score": 62.0},
                {"Fold": 2, "Dir_Acc": 61.0, "RMSE_vs_benchmark": 0.85, "Composite_Score": 70.0},
            ],
            "Naive Zero Return": [
                {"Fold": 0, "Dir_Acc": 10.0, "RMSE_vs_benchmark": 9.0, "Composite_Score": 1.0},
            ],
        })

        metadata = manager.dataset_metadata["signal_threshold_config"]
        self.assertEqual(metadata["source"], "walk_forward_calibration_folds")
        self.assertEqual(metadata["selection_scope"], "walk_forward_calibration_folds")
        self.assertFalse(metadata["final_holdout_optimized"])
        self.assertEqual(metadata["calibration_summary"]["calibration_set"], "walk_forward_folds_only")
        self.assertFalse(metadata["calibration_summary"]["final_holdout_used"])
        self.assertEqual(metadata["active_from_stage"], "walk_forward_backtest_signal_filtering")
        self.assertEqual(metadata["calibration_summary"]["active_from_stage"], "walk_forward_backtest_signal_filtering")
        self.assertEqual(metadata["calibration_summary"]["calibration_fold_count"], 3)
        self.assertGreaterEqual(metadata["quality_thresholds"]["min_directional_accuracy"], 52.0)


if __name__ == "__main__":
    unittest.main()
