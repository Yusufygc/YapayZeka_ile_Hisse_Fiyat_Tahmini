# -*- coding: utf-8 -*-

import os
import unittest
from unittest.mock import patch

import pandas as pd

from src.features.macro_pipeline import MacroPipeline


class LeakageGuardTests(unittest.TestCase):
    def test_macro_release_lag_shifts_monthly_features(self):
        # Faiz artik feature degil; yayim kurali risk-free serisinde korunur:
        # ay damgasi -> ay sonu + lag_days.
        from src.features import macro_transforms

        feats = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Rate_Level": [10.0, 12.0],
        })
        lagged = macro_transforms.lag_monthly_features(
            feats, lag_days=3, raw_date_column="Rate_Raw_Date"
        )

        self.assertEqual(lagged["Date"].iloc[0], pd.Timestamp("2024-02-03"))
        self.assertEqual(lagged["Date"].iloc[1], pd.Timestamp("2024-03-03"))
        self.assertEqual(lagged["Rate_Raw_Date"].iloc[0], pd.Timestamp("2024-01-01"))

    def test_evds_interest_rate_fetch_normalizes_json_response(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "items": [
                        {"Tarih": "01-01-2024", "TP_PPK_H01": "42,50", "UNIXTIME": 1},
                        {"Tarih": "01-02-2024", "TP_PPK_H01": "45.0", "UNIXTIME": 2},
                    ]
                }

        mp = MacroPipeline()
        with patch.dict(os.environ, {"TCMB_EVDS_API_KEY": "test-key"}):
            with patch("src.features.macro_pipeline.requests.get", return_value=FakeResponse()) as get_mock:
                df = mp._fetch_evds_series("TP.PPK.H01", "2024-01-01", "2024-02-01", "INTEREST_RATE")

        self.assertEqual(list(df.columns), ["Date", "INTEREST_RATE"])
        self.assertEqual(list(df["INTEREST_RATE"]), [42.5, 45.0])
        self.assertEqual(get_mock.call_args.kwargs["params"]["startDate"], "01-01-2024")
        self.assertEqual(get_mock.call_args.kwargs["headers"]["key"], "test-key")

    def test_evds_interest_rate_fetch_skips_without_api_key(self):
        mp = MacroPipeline()
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.features.macro_pipeline.requests.get") as get_mock:
                df = mp._fetch_evds_series("TP.PPK.H01", "2024-01-01", "2024-02-01", "INTEREST_RATE")

        self.assertIsNone(df)
        get_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
