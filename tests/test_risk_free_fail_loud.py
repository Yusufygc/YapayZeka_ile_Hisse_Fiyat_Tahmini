# -*- coding: utf-8 -*-
"""
Sprint 1 (2026-05-25) Plan A1.1 — Risk-free fail-loud testleri.

Macro INTEREST_RATE.csv + env yoksa:
  - get_current_risk_free_rate() None doner
  - compute_financial_metrics() Sharpe/BuyHold_Sharpe NaN doner
  - sozlukte Risk_Free_Unavailable=True ve Sharpe_Warning='risk_free_unavailable'
  - summarize_backtest() ayni bayraklari doldurur

Bu uyari ileride confidence.warnings'a baglanir (Sprint 8).
"""

from __future__ import annotations

import math
import os
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.utils.risk_free_rate import get_current_risk_free_rate


def _isolated_dir():
    """tempfile.TemporaryDirectory wrapper; pytest tmp_path fixture'una bagimli degil."""
    return tempfile.TemporaryDirectory()


def test_get_rf_returns_none_when_cache_and_env_missing(monkeypatch):
    """Cache yok + env yok -> None (fail-loud)."""
    monkeypatch.delenv("RISK_FREE_RATE_ANNUAL", raising=False)
    with _isolated_dir() as tmp:
        rf = get_current_risk_free_rate(
            macro_cache_dir=str(tmp),
            project_root=str(tmp),
        )
    assert rf is None


def test_get_rf_uses_env_when_provided(monkeypatch):
    """Env varsa cache'siz dahi okunmali."""
    monkeypatch.setenv("RISK_FREE_RATE_ANNUAL", "0.35")
    with _isolated_dir() as tmp:
        rf = get_current_risk_free_rate(
            macro_cache_dir=str(tmp),
            project_root=str(tmp),
        )
    assert rf == pytest.approx(0.35)


def test_get_rf_no_legacy_40_fallback(monkeypatch):
    """Sprint 1: 0.40 sabit fallback artik yok."""
    monkeypatch.delenv("RISK_FREE_RATE_ANNUAL", raising=False)
    with _isolated_dir() as tmp:
        rf = get_current_risk_free_rate(
            macro_cache_dir=str(tmp),
            project_root=str(tmp),
        )
    assert rf != 0.40
    assert rf is None  # explicit fail-loud


def test_get_rf_explicit_fallback_honored(monkeypatch):
    """Kullanici explicit fallback verirse o deger kullanilir."""
    monkeypatch.delenv("RISK_FREE_RATE_ANNUAL", raising=False)
    with _isolated_dir() as tmp:
        rf = get_current_risk_free_rate(
            macro_cache_dir=str(tmp),
            project_root=str(tmp),
            fallback=0.10,
        )
    assert rf == pytest.approx(0.10)


def test_compute_financial_metrics_marks_unavailable(monkeypatch):
    """Macro yok + env yok -> Sharpe NaN + Risk_Free_Unavailable=True."""
    monkeypatch.delenv("RISK_FREE_RATE_ANNUAL", raising=False)
    # Patch _get_rf icindeki cache yolunu calismayan tmp_path'e cevir
    with mock.patch(
        "src.evaluation.financial_metrics._get_rf",
        side_effect=lambda **kwargs: None,
    ):
        from src.evaluation.financial_metrics import compute_financial_metrics
        prev_close = np.full(5, 100.0)
        returns = np.array([0.010, 0.012, 0.009, 0.011, 0.013])
        y_true = prev_close * (1.0 + returns)
        y_pred = y_true.copy()

        result = compute_financial_metrics(
            y_true,
            y_pred,
            prev_close=prev_close,
            target_mode="price",
            risk_free_annual=None,
        )

        assert result["Risk_Free_Unavailable"] is True
        assert result["Sharpe_Warning"] == "risk_free_unavailable"
        assert math.isnan(result["Sharpe"])
        assert math.isnan(result["BuyHold_Sharpe"])


def test_summarize_backtest_marks_unavailable():
    """summarize_backtest rf yoksa ayni bayragi doldurur."""
    from src.backtesting.metrics import summarize_backtest

    equity_curve = pd.DataFrame({
        "Equity": [1.01, 1.02, 1.03],
        "BuyHold_Equity": [1.01, 1.02, 1.03],
        "Net_Return": [0.01, 0.0099, 0.0098],
        "Realized_Return": [0.01, 0.0099, 0.0098],
        "Position": [1.0, 1.0, 1.0],
        "Signal": [1.0, 1.0, 1.0],
    })
    with mock.patch(
        "src.backtesting.metrics._get_rf",
        side_effect=lambda **kwargs: None,
    ):
        summary = summarize_backtest({
            "model_name": "TestModel",
            "equity_curve": equity_curve,
            "trades": pd.DataFrame(),
        })

    assert summary["Risk_Free_Unavailable"] is True
    assert summary["Sharpe_Warning"] == "risk_free_unavailable"
    # Sharpe NaN olarak isaretlenir (rounded NaN = NaN)
    sharpe_val = summary["Sharpe"]
    assert sharpe_val is None or math.isnan(float(sharpe_val))


def test_summarize_backtest_clean_when_rf_provided():
    """Explicit risk_free_annual verilirse bayrak False kalir."""
    from src.backtesting.metrics import summarize_backtest

    equity_curve = pd.DataFrame({
        "Equity": [1.01, 1.02, 1.03],
        "BuyHold_Equity": [1.01, 1.02, 1.03],
        "Net_Return": [0.01, 0.0099, 0.0098],
        "Realized_Return": [0.01, 0.0099, 0.0098],
        "Position": [1.0, 1.0, 1.0],
        "Signal": [1.0, 1.0, 1.0],
    })
    summary = summarize_backtest({
        "model_name": "TestModel",
        "equity_curve": equity_curve,
        "trades": pd.DataFrame(),
    }, risk_free_annual=0.0)

    assert summary["Risk_Free_Unavailable"] is False
    assert summary["Sharpe_Warning"] == ""
    assert summary["Risk_Free_Annual_Used"] == 0.0
