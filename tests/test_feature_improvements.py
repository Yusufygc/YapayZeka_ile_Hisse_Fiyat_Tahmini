# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np
from src.features.feature_pipeline import FeaturePipeline, _SYMBOL_SECTORS

def _make_synthetic_ohlcv(n=100):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, size=n))
    open_ = close + rng.normal(0, 0.5, size=n)
    high = np.maximum(close, open_) + rng.uniform(0.1, 1.0, size=n)
    low = np.minimum(close, open_) - rng.uniform(0.1, 1.0, size=n)
    volume = rng.uniform(1000, 5000, size=n)
    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })

def _make_synthetic_macro(n=100):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Date": dates,
        "BIST100_Return": rng.normal(0.0005, 0.01, size=n),
        "BIST100_MA7": rng.normal(0, 0.02, size=n),
        "USDTRY_Return": rng.normal(0.001, 0.005, size=n),
        "USDTRY_Volatility7": rng.uniform(0.001, 0.01, size=n),
        "XBANK_Return": rng.normal(0.0006, 0.012, size=n),
        "XUSIN_Return": rng.normal(0.0004, 0.008, size=n),
        "Rate_Level": np.full(n, 8.5),
        "Rate_Change": np.zeros(n),
        "CPI_YoY": np.full(n, 50.0),
        "CPI_MoM": np.full(n, 2.5),
        "Real_Rate": np.full(n, -41.5)
    })

class TestFeatureImprovements(unittest.TestCase):
    def test_default_correlation_threshold(self):
        pipeline = FeaturePipeline()
        self.assertEqual(pipeline.correlation_threshold, 0.88)

    def test_new_indicators_present_and_no_nan(self):
        df_ohlcv = _make_synthetic_ohlcv(200)
        pipeline = FeaturePipeline(prune_correlated_features=False)
        df_features = pipeline.engineer_features(df_ohlcv)
        
        # Check new indicators
        self.assertIn("NATR_14", df_features.columns)
        self.assertIn("MFI_14", df_features.columns)
        self.assertIn("ADX_14", df_features.columns)
        self.assertIn("CMF_20", df_features.columns)
        
        # Check no NaN values
        self.assertFalse(df_features["NATR_14"].isna().any())
        self.assertFalse(df_features["MFI_14"].isna().any())
        self.assertFalse(df_features["ADX_14"].isna().any())
        self.assertFalse(df_features["CMF_20"].isna().any())

    def test_non_stationary_features_absent(self):
        df_ohlcv = _make_synthetic_ohlcv(100)
        macro_df = _make_synthetic_macro(100)
        
        # Add non-stationary variables to macro manually just in case
        macro_df["BIST100_Norm"] = np.linspace(100, 150, 100)
        macro_df["USDTRY_MA7"] = np.linspace(18.5, 20.0, 100)
        
        pipeline = FeaturePipeline()
        df_features = pipeline.engineer_features(df_ohlcv, macro_df=macro_df, symbol="AKBNK")
        
        self.assertNotIn("BIST100_Norm", df_features.columns)
        self.assertNotIn("USDTRY_MA7", df_features.columns)

    def test_sector_relative_strength_calculation(self):
        df_ohlcv = _make_synthetic_ohlcv(150)
        macro_df = _make_synthetic_macro(150)
        
        pipeline = FeaturePipeline()
        
        # AKBNK -> XBANK
        df_akbnk = pipeline.engineer_features(df_ohlcv, macro_df=macro_df, symbol="AKBNK")
        self.assertIn("Sector_Relative_Strength", df_akbnk.columns)
        
        # Verify manual calculation: Sector_Relative_Strength = Return - XBANK_Return
        row = df_akbnk.iloc[10]
        expected_diff = row["Return"] - row["XBANK_Return"]
        self.assertAlmostEqual(row["Sector_Relative_Strength"], expected_diff)

        # Default/other -> XUSIN
        df_default = pipeline.engineer_features(df_ohlcv, macro_df=macro_df, symbol="SASA")
        self.assertIn("Sector_Relative_Strength", df_default.columns)
        row_def = df_default.iloc[10]
        expected_diff_def = row_def["Return"] - row_def["XUSIN_Return"]
        self.assertAlmostEqual(row_def["Sector_Relative_Strength"], expected_diff_def)

    def test_sector_relative_strength_fallback(self):
        df_ohlcv = _make_synthetic_ohlcv(150)
        macro_df = _make_synthetic_macro(150)
        
        # Remove XBANK_Return to trigger fallback
        if "XBANK_Return" in macro_df.columns:
            macro_df = macro_df.drop(columns=["XBANK_Return"])
            
        pipeline = FeaturePipeline()
        
        # AKBNK -> XBANK (missing) -> Fallback to BIST100_Return
        df_akbnk = pipeline.engineer_features(df_ohlcv, macro_df=macro_df, symbol="AKBNK")
        self.assertIn("Sector_Relative_Strength", df_akbnk.columns)
        
        row = df_akbnk.iloc[10]
        expected_diff = row["Return"] - row["BIST100_Return"]
        self.assertAlmostEqual(row["Sector_Relative_Strength"], expected_diff)

if __name__ == "__main__":
    unittest.main()
