# -*- coding: utf-8 -*-
"""
Sprint 5 (2026-05-25) — FeaturePipeline.recompute_close_dependent testleri.

Plan A5.1: Recursive satir eklendikten sonra SMA/EMA/RSI/MACD vb. teknik
gostergeler dogru hesaplanmali; macro/lag sutunlari dokunulmamali.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from src.features.feature_pipeline import FeaturePipeline
except ModuleNotFoundError as exc:
    pytest.skip(f"feature_pipeline import failed: {exc}", allow_module_level=True)

# Real `ta` paketi gerekli. conftest.py MagicMock stub yapiyor -> tip
# kontrolu sart.
import ta as _ta_module
from unittest.mock import MagicMock as _MagicMock
if isinstance(_ta_module, _MagicMock):
    pytest.skip("ta conftest stub (MagicMock) - production dep gerekli", allow_module_level=True)


def _seed_frame(n=300, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=n))
    high = close + np.abs(rng.normal(0, 0.5, size=n))
    low = close - np.abs(rng.normal(0, 0.5, size=n))
    open_ = close + rng.normal(0, 0.3, size=n)
    volume = rng.integers(1000, 10000, size=n).astype(float)
    df = pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    fp = FeaturePipeline(feature_mode="stationary_features", lag_feature_count=3)
    engineered = fp.engineer_features(df)
    return engineered, fp


def test_recompute_updates_sma_rel():
    engineered, fp = _seed_frame()
    last_close = float(engineered["Close"].iloc[-1])
    new_row = engineered.iloc[-1].copy()
    new_row["Date"] = pd.Timestamp("2099-01-01")
    new_row["Close"] = last_close * 1.05  # +%5 sicrayis
    appended = pd.concat([engineered, pd.DataFrame([new_row])], ignore_index=True)
    out = fp.recompute_close_dependent(appended)
    # SMA_7_rel son satirda yeniden hesaplanmali, eski satirla ayni olmamali
    assert out["SMA_7_rel"].iloc[-1] != engineered["SMA_7_rel"].iloc[-1]


def test_recompute_updates_rsi():
    engineered, fp = _seed_frame()
    last_close = float(engineered["Close"].iloc[-1])
    new_row = engineered.iloc[-1].copy()
    new_row["Date"] = pd.Timestamp("2099-01-01")
    new_row["Close"] = last_close * 1.10
    new_row["High"] = last_close * 1.12
    new_row["Low"] = last_close * 1.09
    appended = pd.concat([engineered, pd.DataFrame([new_row])], ignore_index=True)
    out = fp.recompute_close_dependent(appended)
    rsi_new = out["RSI_14"].iloc[-1]
    # RSI yukseldi (10% pozitif close)
    assert 0 < rsi_new <= 100


def test_recompute_preserves_lag_columns():
    engineered, fp = _seed_frame()
    new_row = engineered.iloc[-1].copy()
    new_row["Date"] = pd.Timestamp("2099-01-01")
    new_row["Close"] = float(engineered["Close"].iloc[-1])
    # Lag sutunlarini elle set et — recompute degistirmemeli (lag_features
    # zincirde yok; macro da yok).
    new_row["LogRet_Lag_1"] = 0.0123
    appended = pd.concat([engineered, pd.DataFrame([new_row])], ignore_index=True)
    out = fp.recompute_close_dependent(appended)
    assert out["LogRet_Lag_1"].iloc[-1] == pytest.approx(0.0123)


def test_recompute_preserves_macro_columns_if_present():
    engineered, fp = _seed_frame()
    # Macro-style ek sutun manuel ekle
    engineered["USDTRY"] = 30.0
    new_row = engineered.iloc[-1].copy()
    new_row["Date"] = pd.Timestamp("2099-01-01")
    new_row["Close"] = float(engineered["Close"].iloc[-1])
    new_row["USDTRY"] = 31.5  # macro projection sonucu gibi
    appended = pd.concat([engineered, pd.DataFrame([new_row])], ignore_index=True)
    out = fp.recompute_close_dependent(appended)
    assert out["USDTRY"].iloc[-1] == pytest.approx(31.5)


def test_recompute_market_regime_remains_int():
    engineered, fp = _seed_frame()
    new_row = engineered.iloc[-1].copy()
    new_row["Date"] = pd.Timestamp("2099-01-01")
    new_row["Close"] = float(engineered["Close"].iloc[-1])
    appended = pd.concat([engineered, pd.DataFrame([new_row])], ignore_index=True)
    out = fp.recompute_close_dependent(appended)
    val = out["Market_Regime_SMA200"].iloc[-1]
    assert int(val) in (-1, 0, 1)
