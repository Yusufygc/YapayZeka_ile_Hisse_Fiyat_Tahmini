# -*- coding: utf-8 -*-

import os
import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
except Exception as exc:  # pragma: no cover - dependency guard
    np = None
    pd = None
    CORE_IMPORT_ERROR = exc

if pd is not None and np is not None:
    from src.backtesting.engine import run_backtest
    from src.backtesting.signals import SignalConfig, generate_professional_signals
else:  # pragma: no cover - dependency guard
    run_backtest = None
    SignalConfig = None
    generate_professional_signals = None

try:
    from src.pipeline.data_manager import DataManager
except Exception as exc:  # pragma: no cover - optional dependency guard
    DataManager = None
    DATA_MANAGER_IMPORT_ERROR = exc

try:
    from src.pipeline.evaluation_manager import EvaluationManager
except Exception as exc:  # pragma: no cover - optional dependency guard
    EvaluationManager = None
    EVALUATION_IMPORT_ERROR = exc

try:
    from src.pipeline.orchestrator import ForecastingPipeline
except Exception as exc:  # pragma: no cover - optional dependency guard
    ForecastingPipeline = None
    ORCHESTRATOR_IMPORT_ERROR = exc


@unittest.skipIf(pd is None or np is None, f"Core dependencies import failed: {globals().get('CORE_IMPORT_ERROR')}")
class Phase8AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.abspath(os.path.join("outputs", "_test_phase8_acceptance"))
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _data_manager_for_window_policy(self, years=5, min_history_days=504, new_listing_min_days=252):
        dm = DataManager.__new__(DataManager)
        dm.training_window_years = years
        dm.window_candidates = [3, 5, 7, 10, None]
        dm.min_history_days = min_history_days
        dm.new_listing_min_days = new_listing_min_days
        dm.training_window_report = {}
        return dm

    @unittest.skipIf(DataManager is None, f"DataManager import failed: {globals().get('DATA_MANAGER_IMPORT_ERROR')}")
    def test_training_window_policy_uses_recent_five_years_and_keeps_short_history(self):
        long_df = pd.DataFrame({
            "Date": pd.date_range("2010-01-01", periods=3200, freq="B"),
            "Close": np.linspace(10.0, 100.0, 3200),
            "Volume": np.full(3200, 1000),
        })
        dm = self._data_manager_for_window_policy()
        filtered = DataManager._apply_training_window(dm, long_df)

        cutoff = pd.to_datetime(long_df["Date"].iloc[-1]).normalize() - pd.DateOffset(years=5)
        self.assertLess(len(filtered), len(long_df))
        self.assertGreaterEqual(pd.to_datetime(filtered["Date"].iloc[0]).normalize(), cutoff)
        self.assertEqual(dm.training_window_report["effective_training_window_years"], 5)
        self.assertEqual(dm.training_window_report["effective_training_window_years_label"], "5y")
        self.assertFalse(dm.training_window_report["new_listing_mode"])

        short_df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=240, freq="B"),
            "Close": np.linspace(10.0, 20.0, 240),
            "Volume": np.full(240, 1000),
        })
        dm = self._data_manager_for_window_policy()
        filtered_short = DataManager._apply_training_window(dm, short_df)

        self.assertEqual(len(filtered_short), len(short_df))
        self.assertEqual(dm.training_window_report["effective_training_window_years_label"], "all")
        self.assertTrue(dm.training_window_report["new_listing_mode"])
        self.assertTrue(dm.training_window_report["insufficient_history_warning"])

    def test_quality_gate_soft_does_not_block_and_hard_preserves_old_block(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="B")
        prev_close = np.full(12, 100.0)
        y_true = prev_close * 1.01
        pred_price = prev_close * 1.04
        poor_metrics = {"Dir_Acc": 40.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}

        soft = run_backtest(
            dates=dates,
            y_true_price=y_true,
            pred_price=pred_price,
            prev_close=prev_close,
            model_name="Candidate",
            validation_mode="phase8",
            target_mode="price",
            signal_mode="professional",
            signal_config=SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=5),
            model_metrics=poor_metrics,
        )
        soft_curve = soft["equity_curve"]
        self.assertIn("BUY", set(soft_curve["Decision"]))
        self.assertNotIn("quality_dir_acc", set(soft_curve["Risk_State"]))
        self.assertGreaterEqual(float(soft_curve["Quality_Threshold_Multiplier"].max()), 1.75)

        hard = run_backtest(
            dates=dates,
            y_true_price=y_true,
            pred_price=pred_price,
            prev_close=prev_close,
            model_name="Candidate",
            validation_mode="phase8",
            target_mode="price",
            signal_mode="professional",
            signal_config=SignalConfig(quality_gate_mode="hard", min_holding_bars=1, max_holding_bars=5),
            model_metrics=poor_metrics,
        )
        hard_curve = hard["equity_curve"]
        self.assertNotIn("BUY", set(hard_curve["Decision"]))
        self.assertEqual(set(hard_curve["Risk_State"]), {"quality_dir_acc"})

    def test_entry_threshold_is_adjusted_by_market_regime_and_volatility(self):
        cfg = SignalConfig(
            min_entry_threshold=0.01,
            volatility_window=4,
            regime_bull_entry_multiplier=0.80,
            regime_neutral_entry_multiplier=1.00,
            regime_bear_entry_multiplier=1.30,
            volatility_low_entry_multiplier=0.75,
            volatility_normal_entry_multiplier=1.00,
            volatility_high_entry_multiplier=1.40,
        )
        frame = generate_professional_signals(
            pred_target=np.full(12, 0.005),
            pred_price=np.full(12, 101.0),
            prev_close=np.full(12, 100.0),
            target_mode="return",
            observed_returns=np.array([0.0, 0.0, 0.0, 0.002, -0.002, 0.008, -0.009, 0.001, 0.0, 0.006, -0.007, 0.002]),
            market_regime=np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0]),
            commission_bps=0.0,
            slippage_bps=0.0,
            config=cfg,
        )

        self.assertLess(frame.loc[0, "Entry_Threshold"], frame.loc[1, "Entry_Threshold"])
        self.assertGreater(frame.loc[2, "Entry_Threshold"], frame.loc[1, "Entry_Threshold"])
        np.testing.assert_allclose(
            frame["Entry_Threshold"].to_numpy(),
            frame["Base_Entry_Threshold"].to_numpy() * frame["Final_Threshold_Multiplier"].to_numpy(),
        )
        self.assertTrue(set(frame["Volatility_Regime"]).issubset({"normal_vol", "low_vol", "high_vol"}))

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_shadow_backtest_report_writes_three_modes_per_model(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.outputs_dir = self.tmp
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=5)
        manager.commission_bps = 10.0
        manager.slippage_bps = 5.0
        manager.initial_capital = 100000.0

        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        payload = {
            "dates": dates,
            "prediction_dates": dates - pd.Timedelta(days=1),
            "y_true_price": np.full(10, 101.0),
            "pred_price": np.full(10, 104.0),
            "prev_close": np.full(10, 100.0),
            "market_regime": np.zeros(10),
        }
        EvaluationManager._run_shadow_backtests(
            manager,
            backtest_inputs={"Candidate": payload},
            model_metrics_by_model={"Candidate": {"Dir_Acc": 40.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}},
            suffix="phase8",
            target_mode="price",
        )

        report = pd.read_csv(os.path.join(self.tmp, "shadow_backtest_comparison_v1_phase8.csv"))
        self.assertEqual(set(report["Shadow_Mode"]), {"professional_current", "professional_soft_gate", "legacy_directional"})
        self.assertEqual(len(report), 3)
        hard_row = report.loc[report["Shadow_Mode"] == "professional_current"].iloc[0]
        self.assertGreater(int(hard_row["Blocked_By_DirAcc"]), 0)
        soft_row = report.loc[report["Shadow_Mode"] == "professional_soft_gate"].iloc[0]
        self.assertEqual(int(soft_row["Blocked_By_DirAcc"]), 0)

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_walk_forward_signal_calibration_excludes_final_holdout_and_writes_reports(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.outputs_dir = self.tmp
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=6)
        manager.default_signal_config = manager.signal_config
        manager.signal_threshold_source = "default_config"
        manager.signal_threshold_calibration_summary = {"status": "applied", "final_holdout_used": False}
        manager.dataset_metadata = {"target_mode": "price"}
        manager.commission_bps = 10.0
        manager.slippage_bps = 5.0
        manager.initial_capital = 100000.0

        dates = pd.date_range("2024-01-01", periods=12, freq="B")
        payload = {
            "dates": dates,
            "prediction_dates": dates - pd.Timedelta(days=1),
            "y_true_price": np.full(12, 101.0),
            "pred_price": np.full(12, 104.0),
            "prev_close": np.full(12, 100.0),
            "market_regime": np.zeros(12),
        }
        tiny_grid = [
            {
                "min_directional_accuracy": 48.0,
                "volatility_multiplier": 0.10,
                "entry_cost_multiplier": 1.5,
                "min_entry_threshold": 0.0,
                "max_holding_bars": 6,
                "take_profit_vol_multiplier": 1.0,
                "stop_loss_vol_multiplier": 0.75,
            }
        ]

        with patch.object(EvaluationManager, "_signal_calibration_grid", return_value=tiny_grid):
            EvaluationManager._calibrate_walk_forward_signal_parameters(
                manager,
                wf_backtest_inputs={"Candidate": payload},
                model_metrics_by_model={"Candidate": {"Dir_Acc": 53.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}},
            )

        summary = manager.signal_threshold_calibration_summary
        self.assertEqual(summary["execution_calibration_set"], "walk_forward_backtest_inputs_only")
        self.assertFalse(summary["final_holdout_used"])
        self.assertEqual(manager.signal_threshold_source, "walk_forward_signal_calibration")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "signal_calibration_v1.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "signal_calibration_decision_v1.md")))

    @unittest.skipIf(ForecastingPipeline is None, f"ForecastingPipeline import failed: {globals().get('ORCHESTRATOR_IMPORT_ERROR')}")
    def test_window_selection_rows_and_decision_keep_final_holdout_out_of_selection(self):
        pipeline = ForecastingPipeline.__new__(ForecastingPipeline)
        pipeline.outputs_dir = self.tmp

        child = SimpleNamespace(
            outputs_dir=os.path.join(self.tmp, "window_selection", "5y"),
            evaluation_manager=SimpleNamespace(
                latest_model_metrics={
                    "wf": {
                        "Candidate": {
                            "Dir_Acc": 55.0,
                            "RMSE": 1.0,
                            "RMSE_vs_benchmark": 0.95,
                            "Sharpe_excess_vs_buy_hold": 0.2,
                            "Composite_Score": 60.0,
                        }
                    }
                },
                latest_backtest_metrics={
                    "wf": {
                        "Candidate": {
                            "Trade_Count": 4,
                            "Exposure": 0.4,
                            "Net_Return": 0.08,
                            "BuyHold_Return": 0.03,
                            "Max_Drawdown": -0.02,
                            "Sharpe": 1.1,
                        }
                    },
                    "final_holdout": {
                        "Candidate": {
                            "Net_Return": 0.01,
                            "BuyHold_Return": 0.02,
                        }
                    },
                },
            ),
        )

        rows = ForecastingPipeline._window_selection_rows(pipeline, "5y", 5, child)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["Final_Holdout_Used_For_Selection"])
        self.assertAlmostEqual(rows[0]["Final_Holdout_BuyHold_Gap"], -0.01)

        comparison = pd.DataFrame(rows)
        decision_path = ForecastingPipeline._write_window_selection_decision(
            pipeline,
            comparison,
            os.path.join(self.tmp, "training_window_comparison_v1.csv"),
        )
        with open(decision_path, "r", encoding="utf-8") as handle:
            decision = handle.read()
        self.assertIn("Final holdout used for selection: `False`", decision)
        self.assertIn("`Window_Label`: `5y`", decision)


if __name__ == "__main__":
    unittest.main()
