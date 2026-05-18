# -*- coding: utf-8 -*-
"""src/data/quality.py testleri (Adim 1.9)."""
import numpy as np
import pandas as pd
import pytest

from src.data.quality import compute_quality_flags, compute_psi


def _make_df(n=500, start="2020-01-01", freq="B") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=n, freq=freq)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"Date": dates, "Close": close, "Volume": 1e6})


class TestComputePsi:
    def test_identical_distributions_near_zero(self):
        df = _make_df(200)
        df["feat"] = np.random.default_rng(1).normal(0, 1, 200)
        train = df.iloc[:160]
        holdout = df.iloc[160:]
        scores = compute_psi(train, holdout, exclude_cols=["Close", "Volume"])
        assert scores["feat"] >= 0.0

    def test_shifted_distribution_high_psi(self):
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=300, freq="B"),
                               "feat": rng.normal(0, 1, 300)})
        holdout = pd.DataFrame({"Date": pd.date_range("2021-01-01", periods=100, freq="B"),
                                  "feat": rng.normal(10, 1, 100)})
        scores = compute_psi(train, holdout)
        assert scores["feat"] > 0.25, "Tamamen farklı dağılımda PSI > 0.25 olmalı"

    def test_excludes_date_column(self):
        df = _make_df(200)
        df["feat"] = 1.0
        train = df.iloc[:160]
        holdout = df.iloc[160:]
        scores = compute_psi(train, holdout)
        assert "Date" not in scores

    def test_insufficient_data_returns_zero(self):
        train = pd.DataFrame({"feat": [1.0, 2.0]})
        holdout = pd.DataFrame({"feat": [3.0, 4.0]})
        scores = compute_psi(train, holdout)
        assert scores["feat"] == 0.0


class TestComputeQualityFlags:
    def test_clean_data_no_flags(self):
        df = _make_df(600)
        flags = compute_quality_flags(df, "TEST")
        assert flags["corporate_action_anomaly"] is False
        assert flags["survivorship_warning"] is False
        assert flags["psi_high"] is False
        assert isinstance(flags["clip_rate"], float)

    def test_survivorship_short_history(self):
        df = _make_df(100, start="2024-01-01", freq="B")
        flags = compute_quality_flags(df, "SHORTLISTED")
        assert flags["survivorship_warning"] is True

    def test_survivorship_large_gap(self):
        dates_a = pd.date_range("2020-01-01", periods=100, freq="B")
        dates_b = pd.date_range("2022-01-01", periods=200, freq="B")
        all_dates = dates_a.append(dates_b)
        df = pd.DataFrame({"Date": all_dates, "Close": 100.0, "Volume": 1e6})
        flags = compute_quality_flags(df, "GAPPY")
        assert flags["survivorship_warning"] is True

    def test_psi_high_when_distributions_shift(self):
        rng = np.random.default_rng(7)
        train = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=300, freq="B"),
            "feat": rng.normal(0, 1, 300),
        })
        holdout = pd.DataFrame({
            "Date": pd.date_range("2021-01-01", periods=100, freq="B"),
            "feat": rng.normal(10, 1, 100),
        })
        flags = compute_quality_flags(_make_df(300), "TEST", train_df=train, holdout_df=holdout)
        assert flags["psi_high"] is True
        assert flags["psi_max"] > 0.25

    def test_corporate_action_from_attrs(self):
        df = _make_df(300)
        df.attrs["corporate_action_report"] = {"corporate_action_anomaly": True}
        flags = compute_quality_flags(df, "CATEST")
        assert flags["corporate_action_anomaly"] is True

    def test_flags_dict_has_all_keys(self):
        df = _make_df(300)
        flags = compute_quality_flags(df, "X")
        for key in ["corporate_action_anomaly", "survivorship_warning", "psi_high",
                    "psi_max", "psi_scores", "clip_rate"]:
            assert key in flags
