# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features.macro_pipeline import MacroPipeline


class MacroCacheSchemaTests(unittest.TestCase):
    def test_daily_cache_aliases_collapse_to_canonical_key(self):
        mp = MacroPipeline(cache_dir="data/macro")
        df = pd.DataFrame({
            "Date": pd.date_range("2026-01-01", periods=3, freq="B"),
            "XU100.IS": [100.0, 101.0, np.nan],
            "XU100": [np.nan, np.nan, 102.0],
        })

        normalized = mp._normalize_daily_cache_schema("BIST100", df)

        self.assertEqual(normalized.columns.tolist(), ["Date", "BIST100"])
        self.assertEqual(normalized["BIST100"].tolist(), [100.0, 101.0, 102.0])

    def test_global_ticker_cache_aliases_use_macro_key(self):
        mp = MacroPipeline(cache_dir="data/macro")
        df = pd.DataFrame({
            "Date": pd.date_range("2026-01-01", periods=2, freq="B"),
            "^VIX": [18.5, 19.0],
        })

        normalized = mp._normalize_daily_cache_schema("VIX", df)

        self.assertEqual(normalized.columns.tolist(), ["Date", "VIX"])
        self.assertEqual(normalized["VIX"].tolist(), [18.5, 19.0])

    def test_get_macro_features_uses_fresh_cache_without_network_and_finalizes_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates = pd.date_range("2026-01-01", "2026-05-15", freq="B")
            pd.DataFrame({"Date": dates, "USDTRY": np.linspace(29.0, 33.0, len(dates))}).to_csv(
                os.path.join(tmp, "USDTRY.csv"),
                index=False,
            )
            pd.DataFrame({"Date": dates, "BIST100": np.linspace(8000.0, 9500.0, len(dates))}).to_csv(
                os.path.join(tmp, "BIST100.csv"),
                index=False,
            )
            pd.DataFrame({
                "Date": pd.date_range("2025-01-01", "2026-05-01", freq="MS"),
                "INTEREST_RATE": np.linspace(42.5, 50.0, 17),
            }).to_csv(os.path.join(tmp, "INTEREST_RATE.csv"), index=False)
            pd.DataFrame({
                "Date": pd.date_range("2025-01-01", "2026-05-01", freq="MS"),
                "CPI": np.linspace(100.0, 130.0, 17),
            }).to_csv(os.path.join(tmp, "CPI.csv"), index=False)

            mp = MacroPipeline(cache_dir=tmp, rate_release_lag_days=1, cpi_release_lag_days=15)
            with patch.object(mp, "_is_stale", return_value=False), \
                 patch.object(mp, "_update_daily_cache") as daily_update, \
                 patch.object(mp, "_update_monthly_cache") as monthly_update:
                features = mp.get_macro_features("2026-04-01", "2026-05-15")

        daily_update.assert_not_called()
        monthly_update.assert_not_called()
        self.assertFalse(features.empty)
        self.assertTrue(features["Date"].is_monotonic_increasing)
        self.assertFalse(features["Date"].duplicated().any())
        self.assertGreaterEqual(features["Date"].min(), pd.Timestamp("2026-04-01"))
        for column in ["USDTRY_Return", "BIST100_Norm", "Rate_Level", "CPI_YoY", "Real_Rate"]:
            self.assertIn(column, features.columns)
        self.assertFalse(any(column.endswith("_Raw_Date") for column in features.columns))

    def test_monthly_cache_keeps_existing_manual_csv_when_fetch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manual_path = os.path.join(tmp, "INTEREST_RATE.csv")
            pd.DataFrame({
                "Date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                "Rate": [42.5, 45.0],
            }).to_csv(manual_path, index=False)

            mp = MacroPipeline(cache_dir=tmp)
            with patch.object(mp, "_fetch_evds_series", return_value=None):
                mp._update_monthly_cache("INTEREST_RATE", "2026-01-01")

            loaded = pd.read_csv(manual_path)

        self.assertEqual(list(loaded.columns), ["Date", "Rate"])
        self.assertEqual(loaded["Rate"].tolist(), [42.5, 45.0])


if __name__ == "__main__":
    unittest.main()
