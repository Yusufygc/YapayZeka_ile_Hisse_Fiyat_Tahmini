# -*- coding: utf-8 -*-
"""P0-3/P0-4/P0-5/P1-15 — makro look-ahead regresyon testleri.

2026-11: faiz/CPI/ABD-seansi FEATURE'lari kaldirildi. Faiz yayim kurali artik
yalniz risk-free serisinde (``risk_free_rate.get_risk_free_rate_series`` ->
``macro_transforms.lag_monthly_features`` + merge_asof) gecerlidir; testler onu korur.

Network yok: tum testler gecici cache dizini / DataFrame kullanir.
"""
from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.features import macro_feature_engineering, macro_transforms
from src.features.feature_pipeline import FeaturePipeline
from src.utils.risk_free_rate import get_risk_free_rate_series


def _rf_series(rate: pd.DataFrame, dates, lag_days: int = 1) -> pd.Series:
    """INTEREST_RATE.csv (gecici dizin) -> risk-free serisi (YILLIK, ondalik)."""
    with tempfile.TemporaryDirectory() as tmp:
        rate.to_csv(os.path.join(tmp, "INTEREST_RATE.csv"), index=False)
        old = os.environ.pop("RISK_FREE_RATE_ANNUAL", None)
        try:
            return get_risk_free_rate_series(
                dates, macro_cache_dir=tmp, lag_days=lag_days, annual=True
            )
        finally:
            if old is not None:
                os.environ["RISK_FREE_RATE_ANNUAL"] = old


# ── P0-3a: politika faizi ay sonu damgasi (risk-free serisi) ────────────────
def test_rate_jump_not_visible_before_month_end_plus_lag():
    months = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
    values = [9.0] * len(months)
    for i, m in enumerate(months):
        if m >= pd.Timestamp("2023-06-01"):
            values[i] = 15.0  # 2023-06-01 damgali deger = ay sonu seviyesi
    rate = pd.DataFrame({"Date": months, "INTEREST_RATE": values})
    dates = pd.bdate_range("2023-01-02", "2023-12-29")

    out = _rf_series(rate, dates)

    early = out[out.index < pd.Timestamp("2023-07-01")]
    assert not (early == 0.15).any(), (
        "Haziran ay-sonu faizi 2023-07-01 oncesinde gorunuyor (look-ahead)"
    )
    assert (out[out.index >= pd.Timestamp("2023-07-03")] == 0.15).all()


def test_lag_monthly_features_anchors_to_month_end():
    feats = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-03-01", "2024-04-01"]), "Rate_Level": [50.0, 50.0]}
    )
    lagged = macro_transforms.lag_monthly_features(
        feats, lag_days=1, raw_date_column="Rate_Raw_Date"
    )
    assert list(lagged["Date"]) == [pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01")]
    # ham tarih (ay damgasi) korunur
    assert list(lagged["Rate_Raw_Date"]) == [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-04-01")]


# ── P0-4: aylik veri kaybi ──────────────────────────────────────────────────
def test_every_month_reflected_even_when_release_date_is_weekend():
    months = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
    rate = pd.DataFrame({"Date": months, "INTEREST_RATE": np.arange(len(months), dtype=float) + 10.0})
    out = _rf_series(rate, pd.bdate_range("2021-12-01", "2024-03-29"))

    seen = {round(v, 6) for v in out.dropna().tolist()}
    expected = {round((v) / 100.0, 6) for v in rate["INTEREST_RATE"]}
    missing = sorted(expected - seen)
    assert not missing, f"Aylik degerler gunluk cikti da yok: {missing}"


# ── P0-5: ABD-seansi gecikme mantigi kaldirildi; USDTRY/BIST100 gecikmesiz ──
def test_us_session_lag_logic_removed():
    assert not hasattr(macro_feature_engineering, "US_SESSION_COLUMNS")
    assert not hasattr(macro_feature_engineering, "US_SESSION_LAG_ROWS")


def test_usdtry_bist100_features_are_same_day_unlagged():
    n = 30
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=n),
            "USDTRY": 30 + rng.random(n),
            "BIST100": 9000 + rng.random(n) * 100,
        }
    )
    raw = df.set_index("Date")
    out = macro_feature_engineering.engineer_daily(df).set_index("Date")

    assert list(out.columns) == [
        "USDTRY_Return", "USDTRY_Volatility7", "BIST100_Return", "BIST100_MA7",
    ]
    d = out.index[5]
    prev = raw.index[raw.index.get_loc(d) - 1]
    assert out.loc[d, "USDTRY_Return"] == pytest.approx(raw.loc[d, "USDTRY"] / raw.loc[prev, "USDTRY"] - 1)
    assert out.loc[d, "BIST100_Return"] == pytest.approx(
        raw.loc[d, "BIST100"] / raw.loc[prev, "BIST100"] - 1
    )


# ── P1-15: kapsam disi getiri 0 ─────────────────────────────────────────────
def _stock_and_macro(macro_days: int = 10, stock_days: int = 40):
    stock = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=stock_days),
            "Close": np.linspace(10, 20, stock_days),
            "Return": 0.01,
        }
    )
    macro = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=macro_days),
            "USDTRY_Return": 0.05,
            "BIST100_Return": 0.02,
            "USDTRY_Volatility7": 0.3,
            "BIST100_MA7": 0.01,
        }
    )
    return stock, macro


def test_return_columns_beyond_macro_coverage_are_zero_not_repeated():
    stock, macro = _stock_and_macro()
    merged = FeaturePipeline()._merge_macro(stock, macro)
    beyond = merged[merged["Date"] > macro["Date"].max()]
    assert len(beyond) > 0
    for col in ["USDTRY_Return", "BIST100_Return"]:
        assert (beyond[col] == 0.0).all(), f"{col} kapsam disinda tekrarlandi"
    # seviye / volatilite kolonlari ffill kalir
    assert (beyond["BIST100_MA7"] == 0.01).all()
    assert (beyond["USDTRY_Volatility7"] == 0.3).all()


def test_stale_macro_coverage_emits_warning(caplog):
    stock, macro = _stock_and_macro()
    with caplog.at_level(logging.WARNING, logger="src.features.feature_pipeline"):
        FeaturePipeline()._merge_macro(stock, macro)
    assert any("macro" in r.getMessage().lower() for r in caplog.records)


def test_no_warning_when_macro_covers_stock_dates(caplog):
    stock, macro = _stock_and_macro(macro_days=40, stock_days=40)
    with caplog.at_level(logging.WARNING, logger="src.features.feature_pipeline"):
        FeaturePipeline()._merge_macro(stock, macro)
    assert not [r for r in caplog.records if r.name == "src.features.feature_pipeline"]
