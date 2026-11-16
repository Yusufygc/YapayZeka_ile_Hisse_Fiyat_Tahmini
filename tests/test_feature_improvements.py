# -*- coding: utf-8 -*-
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
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

    def test_sector_features_are_removed_and_report_is_disabled(self):
        df_ohlcv = _make_synthetic_ohlcv(150)

        pipeline = FeaturePipeline()
        df_features = pipeline.engineer_features(df_ohlcv, symbol="AKBNK")

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


if __name__ == "__main__":
    unittest.main()
