# -*- coding: utf-8 -*-

import os
import shutil
import unittest

import numpy as np
import pandas as pd

from src.xai.feature_dictionary import feature_group


class Phase5DataQualityTests(unittest.TestCase):
    def test_feature_group_taxonomy_matches_phase5_groups(self):
        self.assertEqual(feature_group("SMA_14_rel"), "technical")
        self.assertEqual(feature_group("USDTRY_Return"), "macro")
        self.assertEqual(feature_group("BIST100_Return"), "market_relative")
        self.assertEqual(feature_group("RollStd_14_norm"), "volatility")
        self.assertEqual(feature_group("LogRet_Lag_3"), "lag")
        self.assertEqual(feature_group("OBV_Norm_20"), "volume")
        self.assertEqual(feature_group("VWAP_20_rel"), "volume")
        self.assertEqual(feature_group("Market_Regime_SMA200"), "regime")

    def test_feature_pipeline_adds_volume_and_lag_features(self):
        try:
            from src.features.feature_pipeline import FeaturePipeline
        except ImportError as exc:
            self.skipTest(f"FeaturePipeline dependency missing: {exc}")

        rows = 80
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Open": np.linspace(99.0, 119.0, rows),
            "High": np.linspace(101.0, 121.0, rows),
            "Low": np.linspace(98.0, 118.0, rows),
            "Close": np.linspace(100.0, 120.0, rows),
            "Volume": np.linspace(1000.0, 2000.0, rows),
        })

        fp = FeaturePipeline(lag_feature_count=5)
        engineered = fp.engineer_features(df)

        self.assertIn("OBV_Norm_20", engineered.columns)
        self.assertIn("VWAP_20_rel", engineered.columns)
        self.assertIn("LogRet_Lag_5", engineered.columns)
        self.assertFalse(engineered[["OBV_Norm_20", "VWAP_20_rel", "LogRet_Lag_5"]].isna().any().any())
        self.assertEqual(fp.feature_groups["OBV_Norm_20"], "volume")
        self.assertEqual(fp.feature_groups["LogRet_Lag_5"], "lag")

    def test_correlation_pruning_keeps_lowest_average_correlation_feature(self):
        try:
            from src.features.feature_pipeline import FeaturePipeline
        except ImportError as exc:
            self.skipTest(f"FeaturePipeline dependency missing: {exc}")

        base = np.linspace(0.0, 1.0, 60)
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=60, freq="B"),
            "Close": np.linspace(100.0, 160.0, 60),
            "KeepMe": base,
            "DropMe": base * 1.001,
            "AlsoDrop": base * 1.002,
        })
        fp = FeaturePipeline(prune_correlated_features=True, correlation_threshold=0.99)
        pruned, features = fp._prune_correlated(df, ["KeepMe", "DropMe", "AlsoDrop"])

        self.assertIn("KeepMe", features)
        self.assertNotIn("DropMe", features)
        self.assertNotIn("AlsoDrop", features)
        self.assertIn("kept_feature", fp.pruning_report["dropped_features"][0])
        self.assertEqual(fp.pruning_report["dropped_features"][0]["kept_feature"], "KeepMe")

    def test_load_and_clean_records_adj_close_report_when_available(self):
        try:
            from src.data.data_loader import load_and_clean
        except ImportError as exc:
            self.skipTest(f"data_loader dependency missing: {exc}")

        tmp = os.path.abspath(os.path.join("outputs", "_test_phase5_data_quality"))
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        path = os.path.join(tmp, "adj.csv")
        try:
            pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=3),
                "Open": [10.0, 11.0, 12.0],
                "High": [11.0, 12.0, 13.0],
                "Low": [9.0, 10.0, 11.0],
                "Close": [10.0, 20.0, 30.0],
                "Adj_Close": [10.0, 10.0, 15.0],
                "Volume": [100, 100, 100],
            }).to_csv(path, index=False)
            df = load_and_clean(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        report = df.attrs["corporate_action_report"]
        self.assertTrue(report["adj_close_available"])
        self.assertEqual(report["price_source"], "Adj_Close")
        self.assertTrue(report["adjusted_price_trusted"])
        self.assertFalse(report["corporate_action_anomaly"])
        np.testing.assert_allclose(df["Close"].to_numpy(), np.array([10.0, 10.0, 15.0]))

    def test_load_and_clean_rejects_anomalous_adj_close_and_keeps_raw_close(self):
        try:
            from src.data.data_loader import load_and_clean
        except ImportError as exc:
            self.skipTest(f"data_loader dependency missing: {exc}")

        tmp = os.path.abspath(os.path.join("outputs", "_test_phase5_adj_anomaly"))
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        path = os.path.join(tmp, "adj_anomaly.csv")
        try:
            pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=3),
                "Open": [10.0, 11.0, 12.0],
                "High": [11.0, 12.0, 13.0],
                "Low": [9.0, 10.0, 11.0],
                "Close": [10.0, 20.0, 30.0],
                "Adj_Close": [10.0, 500000.0, 15.0],
                "Volume": [100, 100, 100],
            }).to_csv(path, index=False)
            df = load_and_clean(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        report = df.attrs["corporate_action_report"]
        self.assertTrue(report["adj_close_available"])
        self.assertEqual(report["price_source"], "Close")
        self.assertFalse(report["adjusted_price_trusted"])
        self.assertTrue(report["corporate_action_anomaly"])
        self.assertGreater(report["max_abs_adj_close_diff_pct"], report["anomaly_threshold_pct"])
        np.testing.assert_allclose(df["Close"].to_numpy(), np.array([10.0, 20.0, 30.0]))

    def test_scaling_report_warns_on_high_test_clip(self):
        try:
            from src.pipeline.data_manager import DataManager
        except ImportError as exc:
            self.skipTest(f"DataManager dependency missing: {exc}")

        dm = DataManager(
            data_file="data/DUMMY.csv",
            test_ratio=0.2,
            time_steps=2,
            models_dir="outputs/_test_phase5_data_quality",
            use_macro=False,
            clip_shift_warning_threshold_pct=1.0,
        )

        class FakeScaler:
            clip_report_ = {
                "train_clip_rate_pct": 0.0,
                "test_clip_rate_pct": 2.5,
                "clip_low": -5.0,
                "clip_high": 5.0,
            }

        train_df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=3)})
        test_df = pd.DataFrame({"Date": pd.date_range("2024-01-04", periods=2)})
        dm._record_scaling_report(train_df, test_df, FakeScaler())

        self.assertEqual(dm.scaling_reports[0]["scaler_fit_scope"], "train_only")
        self.assertIn("distribution_shift_warning", dm.scaling_reports[0]["warning"])

    def test_market_regime_sma200_uses_first_199_days_as_neutral(self):
        try:
            from src.features.feature_pipeline import FeaturePipeline
        except ImportError as exc:
            self.skipTest(f"FeaturePipeline dependency missing: {exc}")

        close = np.concatenate([np.linspace(100.0, 140.0, 210), np.linspace(120.0, 80.0, 20)])
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=len(close), freq="B"),
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(len(close), 1000.0),
        })
        fp = FeaturePipeline()
        with_regime = fp._add_market_regime(df.copy())

        self.assertTrue((with_regime["Market_Regime_SMA200"].iloc[:199] == 0).all())
        self.assertEqual(with_regime["Market_Regime_SMA200"].iloc[199], 1)
        self.assertIn(-1, set(with_regime["Market_Regime_SMA200"].iloc[210:]))

    def test_survivorship_bias_missing_universe_warns_and_existing_file_covers_symbol(self):
        try:
            from src.pipeline.data_manager import DataManager
        except ImportError as exc:
            self.skipTest(f"DataManager dependency missing: {exc}")

        dm = DataManager(
            data_file="data/TEST.csv",
            test_ratio=0.2,
            time_steps=2,
            models_dir="outputs/_test_phase5_data_quality",
            use_macro=False,
            universe_file="outputs/_test_phase5_data_quality/missing_universe.csv",
        )
        dm.df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=5)})
        missing_report = dm._check_survivorship_bias()
        self.assertTrue(missing_report["survivorship_bias_warning"])
        self.assertFalse(missing_report["universe_file_exists"])

        tmp = os.path.abspath(os.path.join("outputs", "_test_phase5_data_quality"))
        os.makedirs(tmp, exist_ok=True)
        universe_path = os.path.join(tmp, "bist_universe.csv")
        try:
            pd.DataFrame({
                "Symbol": ["TEST"],
                "Listed_Date": ["2020-01-01"],
                "Delisted_Date": [""],
                "Status": ["Active"],
            }).to_csv(universe_path, index=False)
            dm.universe_file = universe_path
            covered_report = dm._check_survivorship_bias()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertTrue(covered_report["coverage_ok"])
        self.assertFalse(covered_report["survivorship_bias_warning"])


if __name__ == "__main__":
    unittest.main()
