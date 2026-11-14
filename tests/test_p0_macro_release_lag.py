# -*- coding: utf-8 -*-
"""P0-3/P0-4/P0-5/P1-15 — makro look-ahead regresyon testleri.

Network yok: tum testler dogrudan DataFrame ile saf helper'lari cagirir.
"""
from __future__ import annotations

import logging
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.features import macro_feature_engineering, macro_transforms
from src.features.feature_pipeline import FeaturePipeline
from src.features.macro_pipeline import MacroPipeline


def _daily_macro(start: str, end: str) -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.bdate_range(start, end)})


def _pipeline() -> MacroPipeline:
    # Varsayilan lag degerleri test edilir; cache dizini gecici.
    tmp = tempfile.mkdtemp(prefix="macro_lag_")
    return MacroPipeline(cache_dir=tmp)


def _merge(pipe: MacroPipeline, daily, rate=None, cpi=None) -> pd.DataFrame:
    rate_feats = pipe._lag_monthly_rate_features(rate) if rate is not None else None
    cpi_feats = pipe._lag_monthly_cpi_features(cpi) if cpi is not None else None
    return macro_transforms.merge_monthly_feature_frames(
        daily, monthly_rate_feats=rate_feats, monthly_cpi_feats=cpi_feats
    )


# ── P0-3a: politika faizi ay sonu damgasi ───────────────────────────────────
def test_rate_jump_not_visible_before_month_end_plus_lag():
    months = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
    values = [9.0] * len(months)
    for i, m in enumerate(months):
        if m >= pd.Timestamp("2023-06-01"):
            values[i] = 15.0  # 2023-06-01 damgali deger = ay sonu seviyesi
    rate = pd.DataFrame({"Date": months, "INTEREST_RATE": values})

    out = _merge(_pipeline(), _daily_macro("2023-01-02", "2023-12-29"), rate=rate)

    early = out[out["Date"] < pd.Timestamp("2023-07-01")]
    assert not (early["Rate_Level"] == 15.0).any(), (
        "Haziran ay-sonu faizi 2023-07-01 oncesinde gorunuyor (look-ahead)"
    )
    assert (out.loc[out["Date"] >= pd.Timestamp("2023-07-03"), "Rate_Level"] == 15.0).all()


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


# ── P0-3b: CPI ──────────────────────────────────────────────────────────────
def test_may_cpi_not_visible_before_june_5():
    months = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
    cpi = [100.0]
    for m in months[1:]:
        cpi.append(cpi[-1] * (1.10 if m == pd.Timestamp("2023-05-01") else 1.01))
    raw = pd.DataFrame({"Date": months, "CPI": cpi})
    may_mom = (cpi[list(months).index(pd.Timestamp("2023-05-01"))] / cpi[
        list(months).index(pd.Timestamp("2023-04-01"))] - 1) * 100

    out = _merge(_pipeline(), _daily_macro("2023-01-02", "2023-12-29"), cpi=raw)

    visible = out[np.isclose(out["CPI_MoM"], may_mom)]
    assert not visible.empty
    assert visible["Date"].min() >= pd.Timestamp("2023-06-05"), (
        f"Mayis CPI {visible['Date'].min().date()} tarihinde gorunuyor (>= 2023-06-05 olmali)"
    )


# ── P0-4: aylik veri kaybi ──────────────────────────────────────────────────
def test_every_month_reflected_even_when_release_date_is_weekend():
    months = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
    rate = pd.DataFrame({"Date": months, "INTEREST_RATE": np.arange(len(months), dtype=float) + 10.0})
    out = _merge(_pipeline(), _daily_macro("2021-12-01", "2024-03-29"), rate=rate)

    seen = set(out["Rate_Level"].dropna().tolist())
    missing = [v for v in rate["INTEREST_RATE"] if v not in seen]
    assert not missing, f"Aylik degerler gunluk cikti da yok: {missing}"


def test_merge_asof_carries_value_over_non_business_release_date():
    daily = _daily_macro("2024-01-01", "2024-03-29")
    feats = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-02-03"]),  # Cumartesi
            "Rate_Level": [42.0],
            "Rate_Change": [1.0],
            "Rate_Raw_Date": pd.to_datetime(["2024-01-01"]),
        }
    )
    out = macro_transforms.merge_ffill_monthly_features(
        daily, feats, columns=["Rate_Level", "Rate_Change"]
    )
    mon = out.loc[out["Date"] == pd.Timestamp("2024-02-05"), "Rate_Level"].iloc[0]
    assert mon == 42.0
    assert out.loc[out["Date"] == pd.Timestamp("2024-02-02"), "Rate_Level"].isna().all()


# ── P0-5: ABD seansi serileri 1 satir gecikmeli ─────────────────────────────
def test_us_session_columns_constant_is_explicit():
    assert set(macro_feature_engineering.US_SESSION_COLUMNS) == {
        "VIX", "DXY", "US10Y", "GOLD_USD", "OIL_USD",
    }


def test_us_session_series_are_lagged_one_row():
    n = 30
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=n),
            "USDTRY": 30 + rng.random(n),
            "BIST100": 9000 + rng.random(n) * 100,
            "VIX": 15 + rng.random(n) * 5,
            "DXY": 100 + rng.random(n),
            "US10Y": 4 + rng.random(n),
            "GOLD_USD": 2000 + rng.random(n) * 10,
            "OIL_USD": 80 + rng.random(n),
        }
    )
    raw = df.set_index("Date")
    out = macro_feature_engineering.engineer_daily(df).set_index("Date")

    d = out.index[5]
    prev = raw.index[raw.index.get_loc(d) - 1]
    prev2 = raw.index[raw.index.get_loc(d) - 2]
    assert out.loc[d, "VIX_Level"] == pytest.approx(raw.loc[prev, "VIX"])
    assert out.loc[d, "US10Y_Level"] == pytest.approx(raw.loc[prev, "US10Y"])
    assert out.loc[d, "VIX_Change"] == pytest.approx(raw.loc[prev, "VIX"] - raw.loc[prev2, "VIX"])
    assert out.loc[d, "DXY_Return"] == pytest.approx(raw.loc[prev, "DXY"] / raw.loc[prev2, "DXY"] - 1)
    # BIST/USDTRY ayni gun kalir (BIST seansi icinde gozlenebilir)
    prev_usd = raw.loc[prev, "USDTRY"]
    assert out.loc[d, "USDTRY_Return"] == pytest.approx(raw.loc[d, "USDTRY"] / prev_usd - 1)
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
            "VIX_Change": 1.5,
            "USDTRY_Volatility7": 0.3,
            "Rate_Level": 45.0,
        }
    )
    return stock, macro


def test_return_columns_beyond_macro_coverage_are_zero_not_repeated():
    stock, macro = _stock_and_macro()
    merged = FeaturePipeline()._merge_macro(stock, macro)
    beyond = merged[merged["Date"] > macro["Date"].max()]
    assert len(beyond) > 0
    for col in ["USDTRY_Return", "BIST100_Return", "VIX_Change"]:
        assert (beyond[col] == 0.0).all(), f"{col} kapsam disinda tekrarlandi"
    # seviye / volatilite kolonlari ffill kalir
    assert (beyond["Rate_Level"] == 45.0).all()
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
