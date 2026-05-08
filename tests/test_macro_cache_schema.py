# -*- coding: utf-8 -*-

import unittest

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


if __name__ == "__main__":
    unittest.main()
