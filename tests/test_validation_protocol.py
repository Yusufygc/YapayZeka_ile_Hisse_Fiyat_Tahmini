# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd

from src.utils.data_splitter import TimeSeriesSplitter


class ValidationProtocolTests(unittest.TestCase):
    def _frame(self, rows: int = 900) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=rows, freq="B")
        return pd.DataFrame({
            "Date": dates,
            "Close": np.linspace(100.0, 200.0, rows),
            "Feature": np.arange(rows, dtype=float),
        })

    def test_default_walk_forward_protocol_creates_twelve_sliding_folds(self):
        splits = TimeSeriesSplitter.walk_forward_splits(
            self._frame(),
            n_splits=12,
            min_train_size=504,
            test_size=21,
            max_train_size=756,
        )

        self.assertEqual(len(splits), 12)
        self.assertTrue(all(len(split["test"]) == 21 for split in splits))
        self.assertTrue(all(len(split["train"]) <= 756 for split in splits))
        self.assertLess(splits[0]["train"]["Date"].max(), splits[0]["test"]["Date"].min())

    def test_walk_forward_embargo_creates_boundary_gap(self):
        splits = TimeSeriesSplitter.walk_forward_splits(
            self._frame(),
            n_splits=3,
            min_train_size=504,
            test_size=21,
            max_train_size=756,
            embargo_size=30,
        )

        split = splits[0]
        self.assertEqual(len(split["embargo_context"]), 30)
        self.assertLess(split["train"]["Date"].max(), split["embargo_context"]["Date"].min())
        self.assertLess(split["embargo_context"]["Date"].max(), split["test"]["Date"].min())
        self.assertEqual(split["effective_train_end"], split["embargo_start"])
        self.assertEqual(split["embargo_end"], split["test_start"])

    def test_expanding_walk_forward_keeps_train_start_at_zero(self):
        splits = TimeSeriesSplitter.walk_forward_splits(
            self._frame(),
            n_splits=12,
            min_train_size=504,
            test_size=21,
            max_train_size=None,
        )

        self.assertEqual(len(splits), 12)
        self.assertTrue(all(split["train_start"] == 0 for split in splits))
        self.assertGreater(len(splits[-1]["train"]), len(splits[0]["train"]))

    def test_final_holdout_metadata_is_selection_safe_shape(self):
        selection = self._frame(840)
        holdout = self._frame(60)
        metadata = {
            "selection_set": {
                "rows": len(selection),
                "date_start": selection["Date"].iloc[0].strftime("%Y-%m-%d"),
                "date_end": selection["Date"].iloc[-1].strftime("%Y-%m-%d"),
            },
            "evaluation_set": {
                "rows": len(holdout),
                "date_start": holdout["Date"].iloc[0].strftime("%Y-%m-%d"),
                "date_end": holdout["Date"].iloc[-1].strftime("%Y-%m-%d"),
            },
            "nested_model_selection": {
                "final_holdout_used_for_selection": False,
            },
        }

        self.assertFalse(metadata["nested_model_selection"]["final_holdout_used_for_selection"])
        self.assertEqual(metadata["evaluation_set"]["rows"], 60)


if __name__ == "__main__":
    unittest.main()
