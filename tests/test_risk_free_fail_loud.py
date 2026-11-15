# -*- coding: utf-8 -*-
"""
Sprint 1 (2026-05-25) Plan A1.1 — Risk-free fail-loud testleri.

Macro INTEREST_RATE.csv + env yoksa:
  - get_current_risk_free_rate() None doner

(Backtest/financial_metrics tuketicileri Kol-A ile emekli oldu; yalniz kaynak
fonksiyon test edilir.)
"""

from __future__ import annotations

import tempfile

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
