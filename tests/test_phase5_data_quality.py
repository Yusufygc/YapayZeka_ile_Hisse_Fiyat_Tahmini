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

    def test_load_and_clean_records_adj_close_report_when_available(self):
        try:
            from src.data_loader import load_and_clean
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
        np.testing.assert_allclose(df["Close"].to_numpy(), np.array([10.0, 10.0, 15.0]))

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


if __name__ == "__main__":
    unittest.main()
