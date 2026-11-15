# -*- coding: utf-8 -*-
"""
Takvim feature'lari testleri (2026-11 IC denetimi sonrasi kurate set).

FeaturePipeline._add_calendar_features yalniz ``day_of_week`` uretir.
Kaldirilanlar: day_of_month, days_to_month_end, days_to_quarter_end,
is_quarter_end_week, days_to_next_fomc (+ FOMC takvim yukleyicisi).
"""
from __future__ import annotations

import pandas as pd
import pytest

try:
    from src.features.feature_pipeline import FeaturePipeline
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest.skip(f"feature_pipeline import basarisiz: {exc}", allow_module_level=True)


def _make_pipeline(enable_cal=True) -> FeaturePipeline:
    return FeaturePipeline(
        feature_mode="stationary_features",
        lag_feature_count=0,
        enable_calendar_features=enable_cal,
    )


def test_only_day_of_week_is_added():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=10, freq="D")})
    out = _make_pipeline()._add_calendar_features(df.copy())
    assert "day_of_week" in out.columns
    for removed in (
        "day_of_month",
        "days_to_month_end",
        "days_to_quarter_end",
        "is_quarter_end_week",
        "days_to_next_fomc",
    ):
        assert removed not in out.columns
    assert out["day_of_week"].dtype == float


def test_day_of_week_matches_pandas():
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-05"])})
    out = _make_pipeline()._add_calendar_features(df.copy())
    # 2024-01-01 Pazartesi=0; 2024-01-05 Cuma=4
    assert int(out["day_of_week"].iloc[0]) == 0
    assert int(out["day_of_week"].iloc[1]) == 4


def test_fomc_loader_removed():
    fp = _make_pipeline()
    assert not hasattr(fp, "_load_fomc_dates")
    assert not hasattr(fp, "fomc_calendar_path")


def test_calendar_features_no_date_column_passthrough():
    df = pd.DataFrame({"Close": [1.0, 2.0]})
    out = _make_pipeline()._add_calendar_features(df.copy())
    # Date yok -> tum hesap atlanir, ek sutun yok
    assert "day_of_week" not in out.columns


def test_calendar_disabled_adds_no_column():
    n = 80
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(range(100, 100 + n), dtype=float)
    raw = pd.DataFrame({
        "Date": dates, "Open": close, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": 1000.0,
    })
    out = _make_pipeline(enable_cal=False).engineer_features(raw)
    assert "day_of_week" not in out.columns
