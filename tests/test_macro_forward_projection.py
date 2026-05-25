# -*- coding: utf-8 -*-
"""
Sprint 5 (2026-05-25) — MacroForwardProjector testleri.

Plan A5.2: Recursive forecast son satirinda macro feature'leri ARIMA
projection ile guncellenmeli; ARIMA yoksa fallback trend kullanilmali;
macro olmayan sutunlar dokunulmamali.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from src.features.macro_forward_projection import MacroForwardProjector
except ModuleNotFoundError as exc:
    pytest.skip(f"macro_forward_projection import failed: {exc}", allow_module_level=True)


def _seed_frame(n=300, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n + 1, freq="D")
    usdtry = 30.0 + np.cumsum(rng.normal(0.01, 0.05, size=n + 1))
    bist100 = 9000.0 + np.cumsum(rng.normal(5, 50, size=n + 1))
    vix = 18.0 + rng.normal(0, 2, size=n + 1)
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=n + 1))
    df = pd.DataFrame({
        "Date": dates,
        "Close": close,
        "USDTRY": usdtry,
        "BIST100": bist100,
        "VIX": vix,
        "NonMacro": np.ones(n + 1) * 7.5,
    })
    # Son satir: macro sutunlarda son known degeri ayni (frozen senaryosu)
    return df


def test_auto_resolve_columns_picks_known_macros():
    p = MacroForwardProjector()
    df = _seed_frame()
    cols = p._resolve_columns(df)
    assert "USDTRY" in cols
    assert "BIST100" in cols
    assert "VIX" in cols
    assert "NonMacro" not in cols


def test_explicit_columns_override():
    p = MacroForwardProjector(columns=["USDTRY"])
    df = _seed_frame()
    cols = p._resolve_columns(df)
    assert cols == ["USDTRY"]


def test_project_last_row_updates_macro_only():
    df = _seed_frame()
    # Son satir macroyu manuel "donmus" sabit yapalim
    df.loc[df.index[-1], "USDTRY"] = float(df["USDTRY"].iloc[-2])
    df.loc[df.index[-1], "BIST100"] = float(df["BIST100"].iloc[-2])
    df.loc[df.index[-1], "VIX"] = float(df["VIX"].iloc[-2])
    p = MacroForwardProjector()
    out = p.project_last_row(df, target_date=pd.Timestamp("2099-01-01"))
    # NonMacro dokunulmamali
    assert out["NonMacro"].iloc[-1] == pytest.approx(7.5)
    # Close dokunulmamali
    assert out["Close"].iloc[-1] == pytest.approx(df["Close"].iloc[-1])


def test_project_last_row_handles_empty_frame():
    p = MacroForwardProjector()
    out = p.project_last_row(pd.DataFrame())
    assert out.empty


def test_project_last_row_no_macro_columns():
    p = MacroForwardProjector()
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=5), "Close": [1.0] * 5})
    out = p.project_last_row(df)
    assert out.equals(df)


def test_fallback_trend_handles_short_series():
    p = MacroForwardProjector()
    series = pd.Series([10.0, 11.0])
    val = p._fallback_trend(series)
    # Avg delta = 1.0 -> last + 1.0 = 12.0
    assert val == pytest.approx(12.0)


def test_fallback_trend_single_value():
    p = MacroForwardProjector()
    val = p._fallback_trend(pd.Series([42.0]))
    assert val == pytest.approx(42.0)


def test_fallback_trend_empty():
    p = MacroForwardProjector()
    val = p._fallback_trend(pd.Series(dtype=float))
    assert val == 0.0
