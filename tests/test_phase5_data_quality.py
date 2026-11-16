# -*- coding: utf-8 -*-

import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.xai.feature_dictionary import feature_group, feature_group_label, feature_group_reason


class Phase5DataQualityTests(unittest.TestCase):
    def test_feature_group_taxonomy_matches_phase5_groups(self):
        self.assertEqual(feature_group("SMA_21_rel"), "technical")
        self.assertEqual(feature_group("USDTRY_Return"), "macro")
        self.assertEqual(feature_group("BIST100_Return"), "market_relative")
        self.assertEqual(feature_group("RollStd_14_norm"), "volatility")
        self.assertEqual(feature_group("LogRet_Lag_1"), "lag")
        self.assertEqual(feature_group("Market_Regime_SMA200"), "regime")
        self.assertEqual(feature_group("momentum_60d"), "technical")
        self.assertEqual(feature_group("market_momentum_60d"), "market_relative")
        self.assertEqual(feature_group("relative_to_market_60d"), "market_relative")
        self.assertEqual(feature_group("day_of_week"), "calendar")

    def test_feature_group_product_labels_and_reasons(self):
        self.assertEqual(feature_group_label("macro"), "Makro ekonomik sinyaller")
        reason = feature_group_reason("macro", "asagi", context="forecast")
        self.assertIn("model tahminini asagi", reason)
        peer_reason = feature_group_reason("technical", "yukari", context="peer")
        self.assertIn("akran siralamasini yukari", peer_reason)

    def test_feature_pipeline_adds_lag_features_and_drops_volume_features(self):
        try:
            from src.features.feature_pipeline import FeaturePipeline
            import ta as _ta_module
        except ImportError as exc:
            self.skipTest(f"FeaturePipeline dependency missing: {exc}")
        if isinstance(_ta_module, MagicMock):
            self.skipTest("ta conftest stub (MagicMock) - production dep gerekli")

        rows = 80
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
                "Open": np.linspace(99.0, 119.0, rows),
                "High": np.linspace(101.0, 121.0, rows),
                "Low": np.linspace(98.0, 118.0, rows),
                "Close": np.linspace(100.0, 120.0, rows),
                "Volume": np.linspace(1000.0, 2000.0, rows),
            }
        )

        fp = FeaturePipeline(lag_feature_count=5)
        engineered = fp.engineer_features(df)

        # OBV/VWAP IC denetimiyle kaldirildi.
        self.assertNotIn("OBV_Norm_20", engineered.columns)
        self.assertNotIn("VWAP_20_rel", engineered.columns)
        self.assertIn("LogRet_Lag_5", engineered.columns)
        self.assertFalse(engineered[["LogRet_Lag_5"]].isna().any().any())
        self.assertEqual(fp.feature_groups["LogRet_Lag_5"], "lag")

        # Varsayilan lag sayisi 1'dir.
        default_fp = FeaturePipeline()
        self.assertEqual(default_fp.lag_feature_count, 1)
        default_out = default_fp.engineer_features(df)
        self.assertIn("LogRet_Lag_1", default_out.columns)
        self.assertNotIn("LogRet_Lag_2", default_out.columns)

    def test_correlation_pruning_keeps_lowest_average_correlation_feature(self):
        try:
            from src.features.feature_pipeline import FeaturePipeline
        except ImportError as exc:
            self.skipTest(f"FeaturePipeline dependency missing: {exc}")

        base = np.linspace(0.0, 1.0, 60)
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=60, freq="B"),
                "Close": np.linspace(100.0, 160.0, 60),
                "KeepMe": base,
                "DropMe": base * 1.001,
                "AlsoDrop": base * 1.002,
            }
        )
        fp = FeaturePipeline(prune_correlated_features=True, correlation_threshold=0.99)
        pruned, features = fp._prune_correlated(df, ["KeepMe", "DropMe", "AlsoDrop"])

        self.assertIn("KeepMe", features)
        self.assertNotIn("DropMe", features)
        self.assertNotIn("AlsoDrop", features)
        self.assertIn("kept_feature", fp.pruning_report["dropped_features"][0])
        self.assertEqual(fp.pruning_report["dropped_features"][0]["kept_feature"], "KeepMe")

    def test_market_regime_sma200_uses_first_199_days_as_neutral(self):
        try:
            from src.features.feature_pipeline import FeaturePipeline
        except ImportError as exc:
            self.skipTest(f"FeaturePipeline dependency missing: {exc}")

        close = np.concatenate([np.linspace(100.0, 140.0, 210), np.linspace(120.0, 80.0, 20)])
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=len(close), freq="B"),
                "Open": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": np.full(len(close), 1000.0),
            }
        )
        fp = FeaturePipeline()
        with_regime = fp._add_market_regime(df.copy())

        self.assertTrue((with_regime["Market_Regime_SMA200"].iloc[:199] == 0).all())
        self.assertEqual(with_regime["Market_Regime_SMA200"].iloc[199], 1)
        self.assertIn(-1, set(with_regime["Market_Regime_SMA200"].iloc[210:]))


if __name__ == "__main__":
    unittest.main()
