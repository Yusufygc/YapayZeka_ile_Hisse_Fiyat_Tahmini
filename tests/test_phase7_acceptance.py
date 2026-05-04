# -*- coding: utf-8 -*-

import json
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtesting.engine import run_backtest
from src.backtesting.metrics import summarize_backtest
from src.backtesting.reporting import save_backtest_report
from src.evaluation.financial_metrics import compute_financial_metrics
from src.evaluation.evaluator import save_metrics_report
from src.model_registry.model_registry import ModelRegistry
from src.utils.data_splitter import TimeSeriesSplitter


class Phase7AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.abspath(os.path.join("outputs", "_test_phase7_acceptance"))
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_data_manager_prepare_tensors_aligns_x_t_to_y_t_plus_1(self):
        try:
            from src.pipeline.data_manager import DataManager
        except ImportError as exc:
            self.skipTest(f"DataManager dependency missing: {exc}")

        dm = DataManager.__new__(DataManager)
        dm.target_mode = "log_return"
        dm.scaling_mode = "standard"
        dm.models_dir = self.tmp
        dm.time_steps = 2
        dm.scaling_reports = []
        dm._prepare_tensors_call_idx = 0
        dm.clip_shift_warning_threshold_pct = 1.0

        train = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "Close": [100.0, 110.0, 121.0, 133.1],
            "Feature": [1.0, 2.0, 3.0, 4.0],
        })
        test = pd.DataFrame({
            "Date": pd.date_range("2024-01-05", periods=3, freq="D"),
            "Close": [133.1, 146.41, 161.051],
            "Feature": [5.0, 6.0, 7.0],
        })

        class IdentityScaler:
            clip_report_ = {}

        def fake_scale(X_train, X_test, y_train, y_test, save_dir, scaling_mode):
            return X_train, X_test, y_train, y_test, IdentityScaler(), IdentityScaler()

        with patch("src.pipeline.data_manager.scale_data", side_effect=fake_scale):
            tensors = dm.prepare_tensors(train, test)

        np.testing.assert_allclose(tensors["X_train"].ravel(), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(tensors["y_train"].ravel(), np.log(np.array([1.1, 1.1, 1.1])))
        self.assertEqual(list(tensors["dates_prediction"].dt.strftime("%Y-%m-%d")), ["2024-01-05", "2024-01-06"])
        self.assertEqual(list(tensors["dates_test"].dt.strftime("%Y-%m-%d")), ["2024-01-06", "2024-01-07"])

    def test_small_synthetic_single_split_data_pipeline_completes(self):
        try:
            from src.pipeline.data_manager import DataManager
        except ImportError as exc:
            self.skipTest(f"DataManager dependency missing: {exc}")

        dm = DataManager.__new__(DataManager)
        dm.df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "Close": np.linspace(100.0, 120.0, 20),
            "Feature": np.arange(20, dtype=float),
        })
        dm.test_ratio = 0.25

        with patch.object(DataManager, "prepare_tensors", return_value={"ok": True}) as mocked_prepare:
            DataManager.split_data(dm, "single_split")

        self.assertEqual(dm.tensors, {"ok": True})
        self.assertEqual(len(dm.selection_df), 15)
        self.assertEqual(len(dm.final_holdout_df), 5)
        mocked_prepare.assert_called_once()

    def test_scaler_fit_uses_train_only_statistics(self):
        try:
            from src.data.preprocessor import scale_data
        except ImportError as exc:
            self.skipTest(f"preprocessor dependency missing: {exc}")

        X_train = np.array([[0.0], [1.0], [2.0]])
        X_test = np.array([[100.0]])
        y_train = np.array([[0.0], [1.0], [2.0]])
        y_test = np.array([[100.0]])

        _, X_test_s, _, _, scaler_X, scaler_y = scale_data(
            X_train,
            X_test,
            y_train,
            y_test,
            save_dir=self.tmp,
            scaling_mode="standard",
        )

        self.assertAlmostEqual(float(scaler_X.mean_[0]), 1.0)
        self.assertAlmostEqual(float(scaler_y.mean_[0]), 1.0)
        self.assertGreater(float(X_test_s[0, 0]), 90.0)

    def test_macro_release_lag_prevents_future_cpi_and_rate_visibility(self):
        from src.features.macro_pipeline import MacroPipeline

        mp = MacroPipeline(rate_release_lag_days=1, cpi_release_lag_days=15)
        raw_rate = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "INTEREST_RATE": [40.0, 45.0],
        })
        raw_cpi = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=14, freq="MS"),
            "CPI": np.arange(100.0, 114.0),
        })

        rate_feats = mp._engineer_monthly_rate(raw_rate)
        rate_feats["Date"] = rate_feats["Date"] + pd.to_timedelta(mp.rate_release_lag_days, unit="D")
        cpi_feats = mp._engineer_monthly_cpi(raw_cpi)
        cpi_feats["Date"] = cpi_feats["Date"] + pd.to_timedelta(mp.cpi_release_lag_days, unit="D")

        self.assertEqual(rate_feats["Date"].iloc[0], pd.Timestamp("2024-01-02"))
        self.assertEqual(cpi_feats["Date"].iloc[-1], pd.Timestamp("2024-02-16"))

        daily = pd.DataFrame({"Date": pd.date_range("2024-02-01", "2024-02-20", freq="D")})
        merged = pd.merge(daily, cpi_feats, on="Date", how="left")
        merged[["CPI_MoM", "CPI_YoY"]] = merged[["CPI_MoM", "CPI_YoY"]].ffill()
        before_release = merged.loc[merged["Date"] < pd.Timestamp("2024-02-16"), "CPI_YoY"]
        self.assertTrue(before_release.isna().all())
        self.assertTrue(np.isfinite(merged.loc[merged["Date"] == pd.Timestamp("2024-02-16"), "CPI_YoY"].iloc[0]))

    def test_backtest_report_trade_equity_and_metric_columns_are_consistent(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        result = run_backtest(
            dates=dates,
            prediction_dates=dates - pd.Timedelta(days=1),
            y_true_price=np.array([101.0, 102.0, 101.0, 103.0]),
            pred_price=np.array([102.0, 103.0, 99.0, 104.0]),
            prev_close=np.full(4, 100.0),
            model_name="Model",
            validation_mode="test",
            target_mode="price",
            signal_mode="legacy",
            commission_bps=10.0,
            slippage_bps=5.0,
        )
        summary = summarize_backtest(result)
        summary.update({
            "Target_Semantics": "X[t] -> y[t+1]",
            "Execution_Lag": "next_bar",
            "Macro_Release_Lag": "{}",
            "Transaction_Costs": "commission_bps=10; slippage_bps=5",
            "Threshold_Config": "{}",
            "Validation_Protocol": "{'mode': 'test'}",
        })
        df = save_backtest_report({"Model": summary}, os.path.join(self.tmp, "backtest_report.csv"))

        self.assertEqual(len(result["equity_curve"]), len(dates))
        self.assertIn("Entry_Transaction_Cost", result["equity_curve"].columns)
        self.assertIn("Exit_Transaction_Cost", result["equity_curve"].columns)
        self.assertIn("Trade_Count", df.columns)
        self.assertIn("Cost_Drag", df.columns)
        self.assertIn("Exposure", df.columns)

        md_path = os.path.join(self.tmp, "md", "backtest_report.md")
        with open(md_path, "r", encoding="utf-8") as handle:
            md = handle.read()
        self.assertIn("Leakage Guard", md)
        self.assertIn("Transaction_Costs", md)

    def test_registry_preserves_dataset_hash_and_metadata_per_protocol(self):
        registry = ModelRegistry(os.path.join(self.tmp, "registry"))
        registry.register(
            "Model",
            "v1",
            ["f1"],
            {"RMSE": 1.0},
            "model_a.pkl",
            dataset_hash="hash_a",
            dataset_metadata={"validation_config": {"mode": "single"}, "target_mode": "log_return"},
        )
        registry.register(
            "Model",
            "v1",
            ["f1"],
            {"RMSE": 1.0},
            "model_b.pkl",
            dataset_hash="hash_b",
            dataset_metadata={"validation_config": {"mode": "walk_forward"}, "target_mode": "log_return"},
        )

        with open(os.path.join(self.tmp, "registry", "registry.json"), "r", encoding="utf-8") as handle:
            entries = json.load(handle)
        self.assertEqual({entry["dataset_hash"] for entry in entries}, {"hash_a", "hash_b"})
        self.assertNotEqual(entries[0]["dataset_metadata"]["validation_config"], entries[1]["dataset_metadata"]["validation_config"])

    def test_regression_zero_return_reconstructs_prev_close_and_metrics_are_stable(self):
        try:
            from src.data.preprocessor import reconstruct_prices_from_logret
        except ImportError as exc:
            self.skipTest(f"preprocessor dependency missing: {exc}")

        prev_close = np.array([100.0, 101.0, 102.0])
        pred_price = reconstruct_prices_from_logret(np.zeros(3), prev_close)
        np.testing.assert_allclose(pred_price, prev_close)

        y_true_target = np.array([0.01, -0.01, 0.02])
        metrics = compute_financial_metrics(
            prev_close * np.exp(y_true_target),
            pred_price,
            y_true_target=y_true_target,
            y_pred_target=np.zeros(3),
            prev_close=prev_close,
            target_mode="log_return",
        )
        self.assertAlmostEqual(metrics["Dir_Acc"], 0.0)
        self.assertAlmostEqual(metrics["Hit_Rate"], 0.0)
        self.assertLess(metrics["Return_RMSE"], 0.02)

    def test_final_holdout_metadata_is_generated_as_not_used_for_selection(self):
        try:
            from src.pipeline.data_manager import DataManager
        except ImportError as exc:
            self.skipTest(f"DataManager dependency missing: {exc}")

        dm = DataManager.__new__(DataManager)
        dm.dataset_metadata = {"target_mode": "log_return"}
        dm.selection_df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=10)})
        dm.final_holdout_df = pd.DataFrame({"Date": pd.date_range("2024-01-11", periods=3)})
        dm.scaling_reports = []

        metadata, _ = DataManager.build_run_metadata(dm, "walk_forward", model_config={"model": "x"})
        self.assertFalse(metadata["nested_model_selection"]["final_holdout_used_for_selection"])
        self.assertEqual(metadata["evaluation_set"]["rows"], 3)

    def test_metrics_report_marks_research_score_and_blocks_ineligible_leader(self):
        metrics = {
            "Naive Zero Return": {
                "RMSE": 1.0,
                "MAE": 1.0,
                "MAPE": 0.0,
                "Return_RMSE": 0.01,
                "Return_MAE": 0.01,
                "Dir_Acc": 50.0,
                "Hit_Rate": 50.0,
                "Neutral_Rate": 0.0,
                "Sharpe": 0.0,
                "BuyHold_Sharpe": 0.0,
                "Benchmark_Model": "Naive Zero Return",
                "Benchmark_Source": "best_baseline_by_rmse",
                "RMSE_vs_benchmark": 1.0,
                "Mandatory_Zero_Return_RMSE": 1.0,
                "RMSE_vs_zero_return": 1.0,
                "DirAcc_vs_benchmark": 0.0,
                "Sharpe_excess_vs_buy_hold": 0.0,
                "Beats_Benchmark_RMSE": True,
                "Beats_Zero_Return_RMSE": True,
                "Beats_BuyHold_Sharpe": True,
                "Eligible_For_Leader": True,
                "Composite_Score": 40.0,
                "Target_Semantics": "X[t] -> y[t+1]",
                "Execution_Lag": "next_bar",
                "Macro_Release_Lag": "{}",
                "Transaction_Costs": "commission_bps=10; slippage_bps=5",
                "Validation_Protocol": "{'wf_n_splits': 12}",
            },
            "Overfit": {
                "RMSE": 1.2,
                "MAE": 0.5,
                "MAPE": 0.0,
                "Return_RMSE": 0.01,
                "Return_MAE": 0.01,
                "Dir_Acc": 90.0,
                "Hit_Rate": 90.0,
                "Neutral_Rate": 0.0,
                "Sharpe": 3.0,
                "BuyHold_Sharpe": 0.0,
                "Benchmark_Model": "Naive Zero Return",
                "Benchmark_Source": "best_baseline_by_rmse",
                "RMSE_vs_benchmark": 1.2,
                "Mandatory_Zero_Return_RMSE": 1.0,
                "RMSE_vs_zero_return": 1.2,
                "DirAcc_vs_benchmark": 40.0,
                "Sharpe_excess_vs_buy_hold": 3.0,
                "Beats_Benchmark_RMSE": False,
                "Beats_Zero_Return_RMSE": False,
                "Beats_BuyHold_Sharpe": True,
                "Eligible_For_Leader": False,
                "Composite_Score": 99.0,
                "Target_Semantics": "X[t] -> y[t+1]",
                "Execution_Lag": "next_bar",
                "Macro_Release_Lag": "{}",
                "Transaction_Costs": "commission_bps=10; slippage_bps=5",
                "Validation_Protocol": "{'wf_n_splits': 12}",
            },
        }
        df = save_metrics_report(metrics, os.path.join(self.tmp, "metrics_report_latest.csv"))
        self.assertEqual(df.index[0], "Naive Zero Return")
        self.assertEqual(df.loc["Naive Zero Return", "Score_Type"], "research_score")

        md_path = os.path.join(self.tmp, "md", "metrics_report_latest.md")
        with open(md_path, "r", encoding="utf-8") as handle:
            md = handle.read()
        self.assertIn("Validation_Protocol", md)
        self.assertIn("Target_Semantics", md)
        self.assertIn("Transaction_Costs", md)

    def test_walk_forward_acceptance_creates_at_least_twelve_folds(self):
        rows = 900
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "Close": np.linspace(100.0, 200.0, rows),
            "Feature": np.arange(rows, dtype=float),
        })
        splits = TimeSeriesSplitter.walk_forward_splits(
            df,
            n_splits=12,
            min_train_size=504,
            test_size=21,
            max_train_size=756,
        )
        self.assertGreaterEqual(len(splits), 12)


if __name__ == "__main__":
    unittest.main()
