# -*- coding: utf-8 -*-
"""
Sprint 7 (2026-05-25) A7.1 — Calendar features testleri.

FeaturePipeline._add_calendar_features:
  - day_of_week, day_of_month, days_to_month_end, days_to_quarter_end,
    is_quarter_end_week, days_to_next_fomc sutunlarini uretir.
  - FOMC CSV yoksa days_to_next_fomc placeholder (>=365) doner.
  - Quarter sonu haftasi binary flag dogru.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

try:
    from src.features.feature_pipeline import FeaturePipeline
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest.skip(f"feature_pipeline import basarisiz: {exc}", allow_module_level=True)

# `ta` modulu conftest stub'i ile MagicMock olabilir; calendar bagimsiz.
import ta as _ta_module
from unittest.mock import MagicMock as _MagicMock


def _make_pipeline(fomc_path=None, enable_cal=True) -> FeaturePipeline:
    return FeaturePipeline(
        feature_mode="stationary_features",
        lag_feature_count=0,
        enable_calendar_features=enable_cal,
        fomc_calendar_path=fomc_path,
    )


def test_calendar_columns_added_and_dtypes():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=10, freq="D")})
    fp = _make_pipeline()
    out = fp._add_calendar_features(df.copy())
    for col in (
        "day_of_week",
        "day_of_month",
        "days_to_month_end",
        "days_to_quarter_end",
        "is_quarter_end_week",
        "days_to_next_fomc",
    ):
        assert col in out.columns


def test_day_of_week_matches_pandas():
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-05"])})
    fp = _make_pipeline()
    out = fp._add_calendar_features(df.copy())
    # 2024-01-01 Pazartesi=0; 2024-01-05 Cuma=4
    assert int(out["day_of_week"].iloc[0]) == 0
    assert int(out["day_of_week"].iloc[1]) == 4


def test_days_to_month_end_correct():
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-03-15", "2024-03-31"])})
    fp = _make_pipeline()
    out = fp._add_calendar_features(df.copy())
    assert int(out["days_to_month_end"].iloc[0]) == 16
    assert int(out["days_to_month_end"].iloc[1]) == 0


def test_is_quarter_end_week_flag():
    # 2024 Q1 sonu 2024-03-31. 2024-03-28 -> 3 gun, flag 1.
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-02-15", "2024-03-28"])})
    fp = _make_pipeline()
    out = fp._add_calendar_features(df.copy())
    assert int(out["is_quarter_end_week"].iloc[0]) == 0
    assert int(out["is_quarter_end_week"].iloc[1]) == 1


def test_fomc_missing_csv_sets_placeholder():
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-15"])})
    fp = _make_pipeline(fomc_path="nonexistent_fomc.csv")
    out = fp._add_calendar_features(df.copy())
    assert float(out["days_to_next_fomc"].iloc[0]) >= 365.0


def test_fomc_csv_used_for_distance():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fomc.csv")
        pd.DataFrame({"Date": ["2024-01-31"], "Note": ["test"]}).to_csv(
            path, index=False
        )
        fp = _make_pipeline(fomc_path=path)
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-15", "2024-01-31"])})
        out = fp._add_calendar_features(df.copy())
        assert int(out["days_to_next_fomc"].iloc[0]) == 16
        assert int(out["days_to_next_fomc"].iloc[1]) == 0


def test_calendar_features_no_date_column_passthrough():
    df = pd.DataFrame({"Close": [1.0, 2.0]})
    fp = _make_pipeline()
    out = fp._add_calendar_features(df.copy())
    # Date yok -> tum hesap atlanir, ek sutun yok
    assert "day_of_week" not in out.columns
