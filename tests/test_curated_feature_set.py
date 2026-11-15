# -*- coding: utf-8 -*-
"""Kurate ozellik seti guard testleri (2026-11 IC denetimi).

63 ozellikten yalniz bagimsiz bilgi tasiyanlar kaldi. Bu testler setin sessizce
buyumesini (kopya/gurultu ozelligin geri gelmesini) engeller.

Not: |Spearman| guard'i GARCH-benzeri (volatilite kumelenmeli, kalin kuyruklu)
sentetik veri kullanir. Homoskedastik saf random-walk'ta SMA_21_rel ile RSI_14
yapisal olarak ~0.95-0.96 rank-korelasyona cikar; gercek BIST verisinde degil.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.data.panel_columns import PIPELINE_RAW_COLS
from src.features.feature_pipeline import FeaturePipeline


def model_feature_names(columns) -> list[str]:
    """Kolon listesinden model girdisi olacak ozellikler (ham OHLCV/yardimci haric)."""
    return [c for c in columns if c not in PIPELINE_RAW_COLS and c != "Adj_Close"]

# stationary_features + makro var + takvim acik -> model ozellikleri
CURATED_STATIONARY_FEATURES = [
    "Return",
    "SMA_7_rel", "SMA_21_rel", "SMA_50_rel",
    "RollStd_14_norm", "RollStd_21_norm", "NATR_14",
    "RSI_14", "MACD_Diff_norm",
    "LogRet_Lag_1",
    "day_of_week",
    "USDTRY_Return", "USDTRY_Volatility7", "BIST100_Return", "BIST100_MA7",
    "Relative_Strength",
    "momentum_60d", "market_momentum_60d", "relative_to_market_60d",
]

N_ROWS = 1500
MAX_ABS_SPEARMAN = 0.95


def _synthetic_ohlcv_and_macro(seed: int = 0, n: int = N_ROWS):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    eps = rng.standard_t(5, n) / np.sqrt(5 / 3)
    sig = np.empty(n)
    ret = np.empty(n)
    sig[0] = 0.02
    for t in range(n):
        if t > 0:
            sig[t] = np.sqrt(4e-6 + 0.08 * ret[t - 1] ** 2 + 0.9 * sig[t - 1] ** 2)
        ret[t] = 0.0003 + sig[t] * eps[t]
    close = 50.0 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    raw = pd.DataFrame({
        "Date": dates, "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(100_000, 1_000_000, n).astype(float),
    })
    bist_ret = rng.normal(0.0004, 0.012, n)
    bist_lvl = 1000.0 * np.exp(np.cumsum(bist_ret))
    usd_ret = rng.normal(0.0005, 0.006, n)
    macro = pd.DataFrame({
        "Date": dates,
        "USDTRY_Return": usd_ret,
        "USDTRY_Volatility7": pd.Series(usd_ret).rolling(7).std().to_numpy(),
        "BIST100_Return": bist_ret,
        "BIST100_MA7": bist_lvl / pd.Series(bist_lvl).rolling(7).mean().to_numpy() - 1.0,
    }).dropna().reset_index(drop=True)
    return raw, macro


@pytest.fixture(scope="module")
def engineered():
    raw, macro = _synthetic_ohlcv_and_macro()
    fp = FeaturePipeline()
    frame = fp.engineer_features(raw, macro_df=macro, symbol="SYN")
    return fp, frame


def test_stationary_feature_set_equals_curated_list(engineered):
    fp, frame = engineered
    features = model_feature_names(fp.feature_names)
    assert sorted(features) == sorted(CURATED_STATIONARY_FEATURES)
    assert len(features) == len(set(features)) == 19
    # frame kolonlarindan turetilen model girdisi (tensor hazirlama yolu) ayni sette
    assert sorted(model_feature_names(frame.columns)) == sorted(CURATED_STATIONARY_FEATURES)


def test_no_feature_pair_exceeds_spearman_threshold(engineered):
    fp, frame = engineered
    features = model_feature_names(fp.feature_names)
    corr = frame[features].corr(method="spearman").abs()
    offenders = [
        (a, b, round(float(corr.loc[a, b]), 3))
        for a, b in itertools.combinations(features, 2)
        if corr.loc[a, b] > MAX_ABS_SPEARMAN
    ]
    assert not offenders, f"|Spearman| > {MAX_ABS_SPEARMAN}: {offenders}"


def test_frame_has_no_nan_and_history_is_not_truncated(engineered):
    _, frame = engineered
    assert not frame.isna().any().any()
    # yalniz pencere isinmasi (momentum_60d -> ~60 satir) kaybolur
    assert len(frame) >= N_ROWS - 80


def test_market_regime_is_computed_but_not_a_model_feature(engineered):
    fp, frame = engineered
    assert "Market_Regime_SMA200" in frame.columns  # sinyal kapisi girdisi
    assert set(frame["Market_Regime_SMA200"].unique()) <= {-1, 0, 1}
    assert "Market_Regime_SMA200" not in fp.feature_names
    assert "Market_Regime_SMA200" not in model_feature_names(frame.columns)
    assert "Market_Regime_SMA200" in PIPELINE_RAW_COLS
    assert "Log_Return" not in frame.columns


def test_without_macro_only_stock_level_features_remain():
    raw, _ = _synthetic_ohlcv_and_macro(seed=3, n=400)
    fp = FeaturePipeline()
    fp.engineer_features(raw, macro_df=None)
    features = model_feature_names(fp.feature_names)
    assert sorted(features) == sorted(
        [
            "Return", "SMA_7_rel", "SMA_21_rel", "SMA_50_rel",
            "RollStd_14_norm", "RollStd_21_norm", "NATR_14",
            "RSI_14", "MACD_Diff_norm", "LogRet_Lag_1", "day_of_week",
        ]
    )
    assert fp.sector_mapping_report["status"] == "disabled"
    assert fp.sector_mapping_report["feature_created"] is False


def test_default_lag_feature_count_is_one_everywhere(tmp_path):
    from types import SimpleNamespace

    from src.features.feature_cache import FeatureCache

    assert FeaturePipeline().lag_feature_count == 1

    # lag_feature_count verilmeyen config, cache anahtarinda 1 olarak sayilir.
    cache = FeatureCache(cache_dir=str(tmp_path / "cache"))
    data_file = tmp_path / "TEST.csv"
    data_file.write_text("Date,Close\n2024-01-01,1\n", encoding="utf-8")
    base = {"feature_mode": "stationary_features", "use_macro": True}
    implicit = SimpleNamespace(**base)
    explicit_one = SimpleNamespace(**base, lag_feature_count=1)
    explicit_five = SimpleNamespace(**base, lag_feature_count=5)
    assert cache.make_key(str(data_file), implicit) == cache.make_key(str(data_file), explicit_one)
    assert cache.make_key(str(data_file), implicit) != cache.make_key(str(data_file), explicit_five)
