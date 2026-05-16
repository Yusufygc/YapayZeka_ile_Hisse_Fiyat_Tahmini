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
    def test_run_backtests_disables_gate_and_shadow_by_default(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.outputs_dir = self.tmp
        manager.backtest_enabled = True
        manager.dataset_metadata = {"target_mode": "price"}
        manager.signal_mode = "professional"
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=5)
        manager.commission_bps = 10.0
        manager.slippage_bps = 5.0
        manager.initial_capital = 100000.0
        manager.latest_backtest_results = {}
        manager.latest_backtest_metrics = {}

        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        payload = {
            "dates": dates,
            "prediction_dates": dates - pd.Timedelta(days=1),
            "y_true_price": np.full(8, 101.0),
            "pred_price": np.full(8, 104.0),
            "prev_close": np.full(8, 100.0),
            "market_regime": np.zeros(8),
        }

        with patch.object(EvaluationManager, "_get_signal_gate_diagnostics") as gate_probe, \
             patch.object(EvaluationManager, "_get_shadow_backtests") as shadow_probe:
            result = EvaluationManager._run_backtests(
                manager,
                {"Candidate": payload},
                suffix="default_diag",
                model_metrics_by_model={"Candidate": {"Dir_Acc": 53.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}},
            )

        gate_probe.assert_not_called()
        shadow_probe.assert_not_called()
        self.assertEqual(result["gate_diagnostics"]["status"], "disabled")
        self.assertEqual(result["shadow_results"]["status"], "disabled")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "signal_gate_diagnostics_v1_default_diag.csv")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "shadow_backtest_comparison_v1_default_diag.csv")))

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_run_backtests_writes_simple_order_report(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.outputs_dir = self.tmp
        manager.backtest_enabled = True
        manager.auto_signal_diagnostics = False
        manager.enable_gate_diagnostics = False
        manager.enable_shadow_backtests = False
        manager.dataset_metadata = {"target_mode": "price"}
        manager.signal_mode = "simple"
        manager.signal_config = SignalConfig()
        manager.commission_bps = 0.0
        manager.slippage_bps = 0.0
        manager.initial_capital = 100000.0
        manager.latest_backtest_results = {}
        manager.latest_backtest_metrics = {}

        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        payload = {
            "dates": dates,
            "prediction_dates": dates - pd.Timedelta(days=1),
            "y_true_price": np.array([101.0, 99.0, 102.0]),
            "pred_price": np.array([102.0, 98.0, 103.0]),
            "prev_close": np.full(3, 100.0),
            "market_regime": np.zeros(3),
        }
        result = EvaluationManager._run_backtests(
            manager,
            {"Candidate": payload},
            suffix="simple_orders",
            model_metrics_by_model={"Candidate": {"Dir_Acc": 50.0}},
        )

        orders_path = os.path.join(self.tmp, "csv", "backtest_orders_simple_orders.csv")
        self.assertTrue(os.path.exists(orders_path))
        orders = pd.read_csv(orders_path, sep=";")
        self.assertIn("Prediction_Date", orders.columns)
        self.assertIn("Execution_Date", orders.columns)
        self.assertEqual(orders["Executable_Order_TR"].tolist(), ["AL", "SAT", "AL"])
        self.assertEqual(float(result["metrics"]["Candidate"]["Cost_Drag"]), 0.0)

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_run_backtests_writes_gate_and_shadow_when_enabled(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.outputs_dir = self.tmp
        manager.backtest_enabled = True
        manager.enable_gate_diagnostics = True
        manager.enable_shadow_backtests = True
        manager.dataset_metadata = {"target_mode": "price"}
        manager.signal_mode = "professional"
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=5)
        manager.commission_bps = 10.0
        manager.slippage_bps = 5.0
        manager.initial_capital = 100000.0
        manager.latest_backtest_results = {}
        manager.latest_backtest_metrics = {}

        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        payload = {
            "dates": dates,
            "prediction_dates": dates - pd.Timedelta(days=1),
            "y_true_price": np.full(8, 101.0),
            "pred_price": np.full(8, 104.0),
            "prev_close": np.full(8, 100.0),
            "market_regime": np.zeros(8),
        }
        result = EvaluationManager._run_backtests(
            manager,
            {"Candidate": payload},
            suffix="enabled_diag",
            model_metrics_by_model={"Candidate": {"Dir_Acc": 53.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}},
        )

        self.assertIsInstance(result["gate_diagnostics"], pd.DataFrame)
        self.assertIn("comparison_df", result["shadow_results"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "signal_gate_diagnostics_v1_enabled_diag.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "shadow_backtest_comparison_v1_enabled_diag.csv")))

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_run_backtests_auto_writes_wf_and_final_diagnostics_with_underperform_label(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.outputs_dir = self.tmp
        manager.backtest_enabled = True
        manager.auto_signal_diagnostics = True
        manager.enable_gate_diagnostics = False
        manager.enable_shadow_backtests = False
        manager.dataset_metadata = {"target_mode": "price"}
        manager.signal_mode = "professional"
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=5)
        manager.commission_bps = 10.0
        manager.slippage_bps = 5.0
        manager.initial_capital = 100000.0
        manager.signal_calibration_min_trades = 6
        manager.latest_backtest_results = {}
        manager.latest_backtest_metrics = {}

        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        payload = {
            "dates": dates,
            "prediction_dates": dates - pd.Timedelta(days=1),
            "y_true_price": np.linspace(101.0, 108.0, 8),
            "pred_price": np.full(8, 100.0),
            "prev_close": np.full(8, 100.0),
            "market_regime": np.zeros(8),
        }

        result = EvaluationManager._run_backtests(
            manager,
            {"Candidate": payload},
            suffix="final_holdout",
            model_metrics_by_model={"Candidate": {"Dir_Acc": 53.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}},
        )

        self.assertIsInstance(result["gate_diagnostics"], pd.DataFrame)
        self.assertIn("comparison_df", result["shadow_results"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "signal_gate_diagnostics_v1_final_holdout.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "shadow_backtest_comparison_v1_final_holdout.csv")))
        report = pd.read_csv(os.path.join(self.tmp, "csv", "backtest_report_final_holdout.csv"), sep=";")
        self.assertIn("Signal_Diagnosis", report.columns)
        self.assertIn("underperform_buyhold", str(report.loc[0, "Signal_Diagnosis"]))

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
        self.assertEqual(summary["grid_size"], 1)
        self.assertEqual(summary["executed_trials"], 1)
        self.assertEqual(summary["trial_cap"], 64)
        self.assertEqual(summary["calibration_profile"], "production")
        self.assertEqual(manager.signal_threshold_source, "walk_forward_signal_calibration")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "csv", "signal_calibration_v1.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "md", "signal_calibration_decision_v1.md")))
        report = pd.read_csv(os.path.join(self.tmp, "csv", "signal_calibration_v1.csv"), sep=";")
        for column in [
            "Mean_BuyHold_Return",
            "Mean_Excess_Return",
            "Risk_Adjusted_Score",
            "Beats_BuyHold_Count",
            "Mean_Calmar",
            "Eval_Net_Return",
            "Eval_BuyHold_Return",
            "Eval_Excess_Return",
            "Eval_Sharpe",
            "Eval_Max_Drawdown",
            "Eval_Trade_Count",
            "OOS_Constraint_Passed",
            "Reject_Reason",
            "Active_For_Execution",
            "Selection_Rank",
            "Sampler",
            "Seed",
            "Grid_Size",
            "Executed_Trials",
            "Adaptive_Expanded",
            "Coverage_Status",
        ]:
            self.assertIn(column, report.columns)
        with open(os.path.join(self.tmp, "md", "signal_calibration_decision_v1.md"), "r", encoding="utf-8") as handle:
            decision_md = handle.read()
        self.assertIn("Risk_Adjusted_Score", decision_md)
        self.assertIn("Mean_Excess_Return", decision_md)
        self.assertIn("Executed_Trials", decision_md)
        self.assertIn("Adaptive_Expanded", decision_md)
        self.assertIn("OOS_Constraint_Passed", decision_md)
        self.assertIn("Active_For_Execution", decision_md)

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_signal_calibration_production_sampler_is_stratified_and_deterministic(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=6)
        manager.signal_calibration_profile = "production"
        manager.signal_calibration_sampler = "adaptive_stratified"
        manager.signal_calibration_seed = 42
        manager.signal_calibration_max_trials = 64

        grid = EvaluationManager._signal_calibration_grid(manager.signal_config)
        selected_a, metadata_a = EvaluationManager._apply_signal_calibration_trial_policy(manager, grid)
        selected_b, metadata_b = EvaluationManager._apply_signal_calibration_trial_policy(manager, grid)

        self.assertEqual(selected_a, selected_b)
        self.assertEqual(metadata_a["sampler"], "adaptive_stratified")
        self.assertEqual(metadata_a["coverage_status"], "complete")
        self.assertEqual(metadata_b["coverage_status"], "complete")
        self.assertEqual(len(selected_a), 64)
        self.assertNotEqual(selected_a, grid[:64])
        selected_df = pd.DataFrame(selected_a)
        for column in [
            "min_directional_accuracy",
            "volatility_multiplier",
            "entry_cost_multiplier",
            "min_entry_threshold",
            "max_holding_bars",
            "take_profit_vol_multiplier",
            "stop_loss_vol_multiplier",
        ]:
            self.assertGreater(selected_df[column].nunique(), 1)

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_signal_calibration_sampler_respects_exclusions_and_keeps_coverage(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=6)
        grid = EvaluationManager._signal_calibration_grid(manager.signal_config)
        from src.pipeline.evaluation_services import SignalCalibrationService

        first_batch = SignalCalibrationService._sample_signal_calibration_grid(grid, cap=32, seed=42)
        excluded = {SignalCalibrationService._grid_param_key(params) for params in first_batch}

        second_batch = SignalCalibrationService._sample_signal_calibration_grid(
            grid,
            cap=64,
            seed=43,
            exclude_keys=excluded,
        )

        self.assertEqual(len(second_batch), 64)
        self.assertTrue(all(SignalCalibrationService._grid_param_key(params) not in excluded for params in second_batch))
        self.assertEqual(
            SignalCalibrationService._signal_calibration_coverage_status(grid, first_batch + second_batch),
            "complete",
        )

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_signal_calibration_adaptive_expands_when_selected_sharpe_is_weak(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.outputs_dir = self.tmp
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=6)
        manager.default_signal_config = manager.signal_config
        manager.signal_threshold_source = "default_config"
        manager.signal_threshold_calibration_summary = {}
        manager.dataset_metadata = {"target_mode": "price"}
        manager.commission_bps = 10.0
        manager.slippage_bps = 5.0
        manager.initial_capital = 100000.0
        manager.signal_calibration_profile = "production"
        manager.signal_calibration_sampler = "adaptive_stratified"
        manager.signal_calibration_seed = 42
        manager.signal_calibration_max_trials = 64
        manager.signal_calibration_min_trades = 6
        manager.signal_calibration_objective = "risk_adjusted"

        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        payload = {
            "dates": dates,
            "prediction_dates": dates - pd.Timedelta(days=1),
            "y_true_price": np.full(8, 101.0),
            "pred_price": np.full(8, 104.0),
            "prev_close": np.full(8, 100.0),
            "market_regime": np.zeros(8),
        }
        weak_summary = {
            "Model": "Candidate",
            "Net_Return": 0.05,
            "BuyHold_Return": 0.01,
            "Max_Drawdown": -0.02,
            "Trade_Count": 6,
            "Sharpe": -0.2,
            "Calmar": 2.5,
            "Beats_BuyHold_NetReturn": True,
        }

        with patch("src.pipeline.signal_calibrator.run_backtest", return_value={}), \
             patch("src.pipeline.signal_calibrator.summarize_backtest", return_value=weak_summary):
            result = EvaluationManager._calibrate_walk_forward_signal_parameters(
                manager,
                wf_backtest_inputs={"Candidate": payload},
                model_metrics_by_model={"Candidate": {"Dir_Acc": 53.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}},
            )

        summary = manager.signal_threshold_calibration_summary
        self.assertTrue(summary["adaptive_expanded"])
        self.assertEqual(summary["executed_trials"], 128)
        self.assertEqual(len(result["calibration_df"]), 128)
        self.assertEqual(summary["coverage_status"], "complete")

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_signal_calibration_research_profile_runs_full_grid(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=6)
        manager.signal_calibration_profile = "research"
        manager.signal_calibration_sampler = "adaptive_stratified"
        manager.signal_calibration_seed = 42
        manager.signal_calibration_max_trials = 2
        grid = EvaluationManager._signal_calibration_grid(manager.signal_config)

        selected, metadata = EvaluationManager._apply_signal_calibration_trial_policy(manager, grid)

        self.assertEqual(len(selected), len(grid))
        self.assertIsNone(metadata["trial_cap"])
        self.assertEqual(metadata["sampler"], "full_grid")

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_oos_confirmation_rejects_high_calibration_return_with_negative_eval_sharpe(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.outputs_dir = self.tmp
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=6)
        manager.default_signal_config = manager.signal_config
        manager.signal_threshold_source = "default_config"
        manager.signal_threshold_calibration_summary = {}
        manager.dataset_metadata = {"target_mode": "price"}
        manager.commission_bps = 10.0
        manager.slippage_bps = 5.0
        manager.initial_capital = 100000.0
        manager.signal_calibration_profile = "production"
        manager.signal_calibration_sampler = "prefix"
        manager.signal_calibration_max_trials = 2
        manager.signal_calibration_min_trades = 6
        manager.signal_calibration_require_oos_confirmation = True
        manager.signal_calibration_min_eval_excess_return = 0.0
        manager.signal_calibration_min_eval_sharpe = 0.0
        manager.signal_calibration_reject_behavior = "no_trade"

        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        payload = {
            "dates": dates,
            "prediction_dates": dates - pd.Timedelta(days=1),
            "y_true_price": np.full(8, 101.0),
            "pred_price": np.full(8, 104.0),
            "prev_close": np.full(8, 100.0),
            "market_regime": np.zeros(8),
        }
        grid = [
            {
                "min_directional_accuracy": 48.0,
                "volatility_multiplier": 0.10,
                "entry_cost_multiplier": 1.5,
                "min_entry_threshold": 0.0,
                "max_holding_bars": 6,
                "take_profit_vol_multiplier": 1.0,
                "stop_loss_vol_multiplier": 0.75,
            },
            {
                "min_directional_accuracy": 50.0,
                "volatility_multiplier": 0.20,
                "entry_cost_multiplier": 2.0,
                "min_entry_threshold": 0.001,
                "max_holding_bars": 10,
                "take_profit_vol_multiplier": 1.5,
                "stop_loss_vol_multiplier": 1.0,
            },
        ]
        summaries = [
            {"Model": "Candidate", "Net_Return": 0.20, "BuyHold_Return": 0.0, "Max_Drawdown": -0.02, "Trade_Count": 6, "Sharpe": 1.2, "Calmar": 10.0, "Beats_BuyHold_NetReturn": True},
            {"Model": "Candidate", "Net_Return": 0.05, "BuyHold_Return": 0.0, "Max_Drawdown": -0.01, "Trade_Count": 6, "Sharpe": 0.8, "Calmar": 5.0, "Beats_BuyHold_NetReturn": True},
            {"Model": "Candidate", "Net_Return": 0.08, "BuyHold_Return": 0.0, "Max_Drawdown": -0.04, "Trade_Count": 6, "Sharpe": -0.1, "Calmar": 2.0, "Beats_BuyHold_NetReturn": True},
            {"Model": "Candidate", "Net_Return": 0.03, "BuyHold_Return": 0.0, "Max_Drawdown": -0.01, "Trade_Count": 6, "Sharpe": 0.4, "Calmar": 3.0, "Beats_BuyHold_NetReturn": True},
        ]

        with patch.object(EvaluationManager, "_signal_calibration_grid", return_value=grid), \
             patch("src.pipeline.signal_calibrator.run_backtest", return_value={}), \
             patch("src.pipeline.signal_calibrator.summarize_backtest", side_effect=summaries):
            result = EvaluationManager._calibrate_walk_forward_signal_parameters(
                manager,
                wf_backtest_inputs={"Candidate": payload},
                wf_evaluation_backtest_inputs={"Candidate": payload},
                model_metrics_by_model={"Candidate": {"Dir_Acc": 53.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}},
            )

        self.assertEqual(result["best_row"]["Trial"], 2)
        self.assertTrue(result["best_row"]["OOS_Constraint_Passed"])
        self.assertEqual(manager.signal_threshold_source, "walk_forward_signal_calibration")

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_oos_confirmation_rejects_all_trials_and_final_backtest_no_trades(self):
        manager = EvaluationManager.__new__(EvaluationManager)
        manager.outputs_dir = self.tmp
        manager.signal_config = SignalConfig(quality_gate_mode="soft", min_holding_bars=1, max_holding_bars=6)
        manager.default_signal_config = manager.signal_config
        manager.signal_threshold_source = "default_config"
        manager.signal_threshold_calibration_summary = {}
        manager.dataset_metadata = {"target_mode": "price"}
        manager.commission_bps = 10.0
        manager.slippage_bps = 5.0
        manager.initial_capital = 100000.0
        manager.signal_mode = "professional"
        manager.backtest_enabled = True
        manager.auto_signal_diagnostics = True
        manager.enable_gate_diagnostics = False
        manager.enable_shadow_backtests = False
        manager.latest_backtest_results = {}
        manager.latest_backtest_metrics = {}
        manager.signal_calibration_profile = "production"
        manager.signal_calibration_sampler = "prefix"
        manager.signal_calibration_max_trials = 1
        manager.signal_calibration_min_trades = 6
        manager.signal_calibration_require_oos_confirmation = True
        manager.signal_calibration_min_eval_excess_return = 0.0
        manager.signal_calibration_min_eval_sharpe = 0.0
        manager.signal_calibration_reject_behavior = "no_trade"

        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        payload = {
            "dates": dates,
            "prediction_dates": dates - pd.Timedelta(days=1),
            "y_true_price": np.linspace(101.0, 108.0, 8),
            "pred_price": np.full(8, 104.0),
            "prev_close": np.full(8, 100.0),
            "market_regime": np.zeros(8),
        }
        grid = [{
            "min_directional_accuracy": 48.0,
            "volatility_multiplier": 0.10,
            "entry_cost_multiplier": 1.5,
            "min_entry_threshold": 0.0,
            "max_holding_bars": 6,
            "take_profit_vol_multiplier": 1.0,
            "stop_loss_vol_multiplier": 0.75,
        }]
        summaries = [
            {"Model": "Candidate", "Net_Return": 0.20, "BuyHold_Return": 0.0, "Max_Drawdown": -0.02, "Trade_Count": 6, "Sharpe": 1.0, "Calmar": 10.0, "Beats_BuyHold_NetReturn": True},
            {"Model": "Candidate", "Net_Return": 0.08, "BuyHold_Return": 0.0, "Max_Drawdown": -0.04, "Trade_Count": 6, "Sharpe": -0.1, "Calmar": 2.0, "Beats_BuyHold_NetReturn": True},
        ]

        with patch.object(EvaluationManager, "_signal_calibration_grid", return_value=grid), \
             patch("src.pipeline.signal_calibrator.run_backtest", return_value={}), \
             patch("src.pipeline.signal_calibrator.summarize_backtest", side_effect=summaries):
            result = EvaluationManager._calibrate_walk_forward_signal_parameters(
                manager,
                wf_backtest_inputs={"Candidate": payload},
                wf_evaluation_backtest_inputs={"Candidate": payload},
                model_metrics_by_model={"Candidate": {"Dir_Acc": 53.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}},
            )

        self.assertTrue(result["rejected"])
        self.assertEqual(manager.signal_threshold_source, "walk_forward_signal_rejected")
        self.assertEqual(manager.signal_threshold_calibration_summary["execution_calibration_status"], "rejected_no_valid_oos_trial")
        self.assertFalse(result["best_row"]["Active_For_Execution"])

        bt = EvaluationManager._run_backtests(
            manager,
            {"Candidate": payload},
            suffix="final_holdout",
            model_metrics_by_model={"Candidate": {"Dir_Acc": 53.0, "RMSE_vs_benchmark": 0.95, "Composite_Score": 55.0}},
        )
        metrics = bt["metrics"]["Candidate"]
        self.assertEqual(int(metrics["Trade_Count"]), 0)
        self.assertIn("rejected_no_trade", metrics["Signal_Diagnosis"])
        report = pd.read_csv(os.path.join(self.tmp, "csv", "backtest_report_final_holdout.csv"), sep=";")
        self.assertIn("rejected_no_trade", str(report.loc[0, "Signal_Diagnosis"]))

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_signal_calibration_prefers_traded_trial_over_zero_trade_when_min_not_met(self):
        rows = [
            {
                "Trial": 1,
                "Mean_Net_Return": 0.0,
                "Median_Net_Return": 0.0,
                "Mean_Max_Drawdown": 0.0,
                "Total_Trade_Count": 0,
                "Min_Trade_Count": 3,
                "Meets_Min_Trade_Count": False,
                "Mean_Sharpe": 0.0,
                "Positive_Net_Return": False,
                "Status": "ok",
            },
            {
                "Trial": 2,
                "Mean_Net_Return": -0.01,
                "Median_Net_Return": -0.01,
                "Mean_Max_Drawdown": -0.02,
                "Total_Trade_Count": 2,
                "Min_Trade_Count": 3,
                "Meets_Min_Trade_Count": False,
                "Mean_Sharpe": -0.5,
                "Positive_Net_Return": False,
                "Status": "ok",
            },
        ]

        selected = EvaluationManager._select_signal_calibration_row(rows)
        self.assertEqual(selected["Trial"], 2)

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_signal_calibration_prefers_risk_adjusted_trial_over_raw_return(self):
        aggressive = EvaluationManager._summarize_signal_calibration_trial(
            trial_idx=1,
            params={},
            summaries=[{
                "Net_Return": 0.20,
                "BuyHold_Return": 0.24,
                "Max_Drawdown": -0.45,
                "Sharpe": -2.0,
                "Calmar": 0.44,
                "Trade_Count": 6,
                "Beats_BuyHold_NetReturn": False,
            }],
            min_trade_count=6,
        )
        balanced = EvaluationManager._summarize_signal_calibration_trial(
            trial_idx=2,
            params={},
            summaries=[{
                "Net_Return": 0.10,
                "BuyHold_Return": 0.02,
                "Max_Drawdown": -0.05,
                "Sharpe": 1.5,
                "Calmar": 2.0,
                "Trade_Count": 6,
                "Beats_BuyHold_NetReturn": True,
            }],
            min_trade_count=6,
        )

        selected = EvaluationManager._select_signal_calibration_row([aggressive, balanced])

        self.assertGreater(aggressive["Mean_Net_Return"], balanced["Mean_Net_Return"])
        self.assertGreater(balanced["Risk_Adjusted_Score"], aggressive["Risk_Adjusted_Score"])
        self.assertEqual(selected["Trial"], 2)

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_signal_calibration_prefers_excess_return_when_net_return_ties(self):
        underperformer = EvaluationManager._summarize_signal_calibration_trial(
            trial_idx=1,
            params={},
            summaries=[{
                "Net_Return": 0.05,
                "BuyHold_Return": 0.08,
                "Max_Drawdown": -0.10,
                "Sharpe": 0.6,
                "Calmar": 0.5,
                "Trade_Count": 6,
                "Beats_BuyHold_NetReturn": False,
            }],
            min_trade_count=6,
        )
        outperformer = EvaluationManager._summarize_signal_calibration_trial(
            trial_idx=2,
            params={},
            summaries=[{
                "Net_Return": 0.05,
                "BuyHold_Return": 0.01,
                "Max_Drawdown": -0.10,
                "Sharpe": 0.6,
                "Calmar": 0.5,
                "Trade_Count": 6,
                "Beats_BuyHold_NetReturn": True,
            }],
            min_trade_count=6,
        )

        selected = EvaluationManager._select_signal_calibration_row([underperformer, outperformer])

        self.assertEqual(underperformer["Mean_Net_Return"], outperformer["Mean_Net_Return"])
        self.assertGreater(outperformer["Mean_Excess_Return"], underperformer["Mean_Excess_Return"])
        self.assertEqual(selected["Trial"], 2)

    @unittest.skipIf(EvaluationManager is None, f"EvaluationManager import failed: {globals().get('EVALUATION_IMPORT_ERROR')}")
    def test_signal_calibration_rejects_under_min_trade_trial_when_valid_exists(self):
        under_min = EvaluationManager._summarize_signal_calibration_trial(
            trial_idx=1,
            params={},
            summaries=[{
                "Net_Return": 0.30,
                "BuyHold_Return": 0.00,
                "Max_Drawdown": -0.05,
                "Sharpe": 2.0,
                "Calmar": 6.0,
                "Trade_Count": 2,
                "Beats_BuyHold_NetReturn": True,
            }],
            min_trade_count=6,
        )
        valid = EvaluationManager._summarize_signal_calibration_trial(
            trial_idx=2,
            params={},
            summaries=[{
                "Net_Return": 0.02,
                "BuyHold_Return": 0.00,
                "Max_Drawdown": -0.05,
                "Sharpe": 0.5,
                "Calmar": 0.4,
                "Trade_Count": 6,
                "Beats_BuyHold_NetReturn": True,
            }],
            min_trade_count=6,
        )

        selected = EvaluationManager._select_signal_calibration_row([under_min, valid])

        self.assertFalse(under_min["Meets_Min_Trade_Count"])
        self.assertTrue(valid["Meets_Min_Trade_Count"])
        self.assertEqual(selected["Trial"], 2)

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
