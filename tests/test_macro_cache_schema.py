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

    def test_removed_global_tickers_are_not_macro_keys(self):
        from src.features import macro_pipeline

        self.assertEqual(set(macro_pipeline._YFINANCE_TICKERS), {"USDTRY", "BIST100"})
        self.assertEqual(
            set(macro_pipeline._CACHE_FILES), {"USDTRY", "BIST100", "INTEREST_RATE"}
        )
        self.assertEqual(
            MacroPipeline.macro_feature_names(),
            ["USDTRY_Return", "USDTRY_Volatility7", "BIST100_Return", "BIST100_MA7"],
        )

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

            mp = MacroPipeline(cache_dir=tmp, rate_release_lag_days=1)
            with patch.object(mp, "_is_stale", return_value=False), \
                 patch.object(mp, "_update_daily_cache") as daily_update, \
                 patch.object(mp, "_update_interest_rate_cache") as rate_update:
                features = mp.get_macro_features("2026-04-01", "2026-05-15")

        daily_update.assert_not_called()
        rate_update.assert_not_called()
        self.assertFalse(features.empty)
        self.assertTrue(features["Date"].is_monotonic_increasing)
        self.assertFalse(features["Date"].duplicated().any())
        self.assertGreaterEqual(features["Date"].min(), pd.Timestamp("2026-04-01"))
        self.assertEqual(
            [c for c in features.columns if c != "Date"],
            ["USDTRY_Return", "USDTRY_Volatility7", "BIST100_Return", "BIST100_MA7"],
        )
        self.assertFalse(features.isna().any().any())

    def test_stale_interest_rate_cache_is_still_refreshed(self):
        """Faiz feature degil ama risk-free kaynagi: get_macro_features cache'i tazelemeli."""
        with tempfile.TemporaryDirectory() as tmp:
            dates = pd.date_range("2026-01-01", "2026-05-15", freq="B")
            pd.DataFrame({"Date": dates, "USDTRY": np.linspace(29.0, 33.0, len(dates))}).to_csv(
                os.path.join(tmp, "USDTRY.csv"), index=False)
            pd.DataFrame({"Date": dates, "BIST100": np.linspace(8000.0, 9500.0, len(dates))}).to_csv(
                os.path.join(tmp, "BIST100.csv"), index=False)

            mp = MacroPipeline(cache_dir=tmp)
            with patch.object(mp, "_is_stale", side_effect=lambda key, days: key == "INTEREST_RATE"), \
                 patch.object(mp, "_update_daily_cache") as daily_update, \
                 patch.object(mp, "_update_interest_rate_cache") as rate_update:
                features = mp.get_macro_features("2026-04-01", "2026-05-15")

        daily_update.assert_not_called()
        rate_update.assert_called_once()
        self.assertFalse(features.empty)

    def test_interest_rate_refresh_failure_does_not_break_macro_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates = pd.date_range("2026-01-01", "2026-05-15", freq="B")
            pd.DataFrame({"Date": dates, "USDTRY": np.linspace(29.0, 33.0, len(dates))}).to_csv(
                os.path.join(tmp, "USDTRY.csv"), index=False)
            pd.DataFrame({"Date": dates, "BIST100": np.linspace(8000.0, 9500.0, len(dates))}).to_csv(
                os.path.join(tmp, "BIST100.csv"), index=False)

            mp = MacroPipeline(cache_dir=tmp)
            with patch.object(mp, "_is_stale", side_effect=lambda key, days: key == "INTEREST_RATE"), \
                 patch.object(mp, "_update_interest_rate_cache", side_effect=RuntimeError("evds down")), \
                 self.assertLogs("src.features.macro_pipeline", level="WARNING") as logs:
                features = mp.get_macro_features("2026-04-01", "2026-05-15")

        self.assertFalse(features.empty)
        self.assertTrue(any("INTEREST_RATE" in line for line in logs.output))

    def test_macro_matrix_starts_where_usdtry_bist100_start(self):
        """Eski aylik CPI YoY + dropna hisse gecmisini 2021-07'ye kesiyordu; artik kesmemeli."""
        from src.features.feature_pipeline import FeaturePipeline

        with tempfile.TemporaryDirectory() as tmp:
            daily = pd.bdate_range("2014-10-03", "2026-05-15")
            rng = np.random.default_rng(7)
            pd.DataFrame({
                "Date": daily,
                "USDTRY": 2.3 * np.exp(np.cumsum(rng.normal(0.0004, 0.004, len(daily)))),
            }).to_csv(os.path.join(tmp, "USDTRY.csv"), index=False)
            pd.DataFrame({
                "Date": daily,
                "BIST100": 740.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(daily)))),
            }).to_csv(os.path.join(tmp, "BIST100.csv"), index=False)
            # Eski truncation sebebi: CPI cache 2020-06'da basliyordu (+12 ay YoY -> 2021-07).
            pd.DataFrame({
                "Date": pd.date_range("2020-06-01", "2025-04-01", freq="MS"),
                "CPI": np.linspace(178.0, 2500.0, 59),
            }).to_csv(os.path.join(tmp, "CPI.csv"), index=False)
            pd.DataFrame({
                "Date": pd.date_range("2020-01-01", "2026-05-01", freq="MS"),
                "INTEREST_RATE": np.linspace(12.0, 45.0, 77),
            }).to_csv(os.path.join(tmp, "INTEREST_RATE.csv"), index=False)

            mp = MacroPipeline(cache_dir=tmp)
            with patch.object(mp, "_is_stale", return_value=False):
                macro = mp.get_macro_features("2015-01-02", "2026-05-15")

            self.assertFalse(macro.empty)
            # 2015-01-02 baslangic; 7-gunluk rolling + tampon sonrasi ilk satir 2015-01-02
            self.assertLessEqual(macro["Date"].min(), pd.Timestamp("2015-01-05"))
            self.assertLess(macro["Date"].min(), pd.Timestamp("2021-07-01"))

            stock_dates = pd.bdate_range("2015-01-02", "2026-05-15")
            close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(stock_dates))))
            raw = pd.DataFrame({
                "Date": stock_dates, "Open": close, "High": close * 1.01,
                "Low": close * 0.99, "Close": close, "Volume": 1e6,
            })
            frame = FeaturePipeline().engineer_features(raw, macro_df=macro, symbol="TEST")

        # Hisse gecmisi 2021-07'de kesilmez; yalniz pencere isinmasi (~60 satir) kadar kaybolur.
        self.assertLess(frame["Date"].min(), pd.Timestamp("2015-06-01"))
        self.assertGreater(len(frame), 0.95 * len(raw))

    def test_interest_rate_cache_keeps_existing_manual_csv_when_fetch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manual_path = os.path.join(tmp, "INTEREST_RATE.csv")
            pd.DataFrame({
                "Date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                "Rate": [42.5, 45.0],
            }).to_csv(manual_path, index=False)

            mp = MacroPipeline(cache_dir=tmp)
            with patch.object(mp, "_fetch_evds_series", return_value=None):
                mp._update_interest_rate_cache("2026-01-01")

            loaded = pd.read_csv(manual_path)

        self.assertEqual(list(loaded.columns), ["Date", "Rate"])
        self.assertEqual(loaded["Rate"].tolist(), [42.5, 45.0])

    def test_interest_rate_csv_still_feeds_risk_free_rate(self):
        from src.utils.risk_free_rate import get_current_risk_free_rate

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({
                "Date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                "INTEREST_RATE": [42.5, 45.0],
            }).to_csv(os.path.join(tmp, "INTEREST_RATE.csv"), index=False)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("RISK_FREE_RATE_ANNUAL", None)
                rf = get_current_risk_free_rate(macro_cache_dir=tmp)

        self.assertEqual(rf, 0.45)


if __name__ == "__main__":
    unittest.main()
