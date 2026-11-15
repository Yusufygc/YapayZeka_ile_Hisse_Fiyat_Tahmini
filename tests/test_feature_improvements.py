# -*- coding: utf-8 -*-
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
from src.features.feature_cache import FeatureCache
from src.features.feature_pipeline import FeaturePipeline
from src.features.sector_mapping import resolve_sector_mapping


def _make_synthetic_ohlcv(n=100):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, size=n))
    open_ = close + rng.normal(0, 0.5, size=n)
    high = np.maximum(close, open_) + rng.uniform(0.1, 1.0, size=n)
    low = np.minimum(close, open_) - rng.uniform(0.1, 1.0, size=n)
    volume = rng.uniform(1000, 5000, size=n)
    return pd.DataFrame(
        {"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def _make_synthetic_macro(n=100):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Date": dates,
            "BIST100_Return": rng.normal(0.0005, 0.01, size=n),
            "BIST100_MA7": rng.normal(0, 0.02, size=n),
            "USDTRY_Return": rng.normal(0.001, 0.005, size=n),
            "USDTRY_Volatility7": rng.uniform(0.001, 0.01, size=n),
        }
    )


def _write_universe(path, rows):
    lines = ["Symbol,Listed_Date,Delisted_Date,Status,Source,Sector,Sector_Index,Note"]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines), encoding="utf-8")


class TestFeatureImprovements(unittest.TestCase):
    def test_default_correlation_threshold(self):
        pipeline = FeaturePipeline()
        self.assertEqual(pipeline.correlation_threshold, 0.88)

    def test_new_indicators_present_and_no_nan(self):
        df_ohlcv = _make_synthetic_ohlcv(200)
        pipeline = FeaturePipeline(prune_correlated_features=False)
        df_features = pipeline.engineer_features(df_ohlcv)

        # Kurate sette kalan gostergeler
        for col in ("NATR_14", "RSI_14", "MACD_Diff_norm", "SMA_7_rel", "SMA_21_rel", "SMA_50_rel"):
            self.assertIn(col, df_features.columns)
            self.assertFalse(df_features[col].isna().any())

        # IC denetimi: tam/yakin kopya veya gurultu ozellikler uretilmez
        for col in (
            "MFI_14", "ADX_14", "CMF_20", "OBV_Norm_20", "VWAP_20_rel",
            "BB_Width_14", "BB_Width_21", "MACD_norm", "MACD_Signal_norm",
            "EMA_7_rel", "EMA_14_rel", "EMA_21_rel", "EMA_50_rel", "SMA_14_rel",
            "Log_Return",
        ):
            self.assertNotIn(col, df_features.columns)

        # Market_Regime_SMA200: sinyal kapisi girdisi olarak frame'de kalir ama OZELLIK DEGIL
        self.assertIn("Market_Regime_SMA200", df_features.columns)
        self.assertNotIn("Market_Regime_SMA200", pipeline.feature_names)

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

    def test_sector_features_are_removed_and_report_is_disabled(self):
        df_ohlcv = _make_synthetic_ohlcv(150)
        macro_df = _make_synthetic_macro(150)
        # Sektor endeksi serisi makro frame'de olsa bile sektor feature'i uretilmez.
        macro_df["XBANK_Return"] = 0.001

        pipeline = FeaturePipeline()
        df_features = pipeline.engineer_features(df_ohlcv, macro_df=macro_df, symbol="AKBNK")

        for col in ("Sector_Relative_Strength", "sector_momentum_60d", "relative_momentum_60d"):
            self.assertNotIn(col, df_features.columns)
            self.assertNotIn(col, pipeline.feature_names)
        report = pipeline.sector_mapping_report
        self.assertEqual(report["status"], "disabled")
        self.assertFalse(report["feature_created"])

    def test_sector_label_resolver_still_works(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            universe_path = Path(tmp) / "universe.csv"
            _write_universe(
                universe_path,
                [["AKBNK", "2015-01-01", "", "Active", "test", "Bank", "XBANK", ""]],
            )
            mapping = resolve_sector_mapping("AKBNK", universe_path)
        self.assertEqual(mapping.sector, "Bank")
        self.assertEqual(mapping.status, "matched")

    def test_relative_strength_still_created_with_macro(self):
        df_ohlcv = _make_synthetic_ohlcv(150)
        macro_df = _make_synthetic_macro(150)
        pipeline = FeaturePipeline()
        df_features = pipeline.engineer_features(df_ohlcv, macro_df=macro_df, symbol="SASA")
        row = df_features.iloc[10]
        self.assertAlmostEqual(row["Relative_Strength"], row["Return"] - row["BIST100_Return"])

    def test_feature_cache_key_changes_when_universe_changes(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            universe_path = Path(tmp) / "universe.csv"
            universe_path.write_text("Symbol,Sector_Index\nAKBNK,XBANK\n", encoding="utf-8")
            cfg = SimpleNamespace(
                feature_mode="stationary_features",
                use_macro=True,
                macro_rate_lag_days=1,
                prune_correlated_features=False,
                correlation_threshold=0.88,
                lag_feature_count=1,
                training_window_years=5,
                universe_file=str(universe_path),
            )
            cache = FeatureCache(cache_dir=str(Path(tmp) / "cache"))
            data_path = Path(tmp) / "AKBNK.csv"
            data_path.write_text("Date,Close\n2024-01-01,1\n", encoding="utf-8")

            first_key = cache.make_key(str(data_path), cfg)
            universe_path.write_text("Symbol,Sector_Index\nAKBNK,XUSIN\n", encoding="utf-8")
            second_key = cache.make_key(str(data_path), cfg)

        self.assertNotEqual(first_key, second_key)


if __name__ == "__main__":
    unittest.main()
