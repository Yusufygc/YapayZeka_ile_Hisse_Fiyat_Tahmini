# -*- coding: utf-8 -*-
"""
Sprint 0 (2026-05-25) — WF embargo auto-default testleri.

Plan A0.2: wf_embargo_size None veya 0 ise auto max(200, time_steps).
Sebep: Market_Regime_SMA200 ve diger rolling-200 feature'lar train/test
arasinda sizinti yaratir; tampon en az 200 olmalidir.
"""

from __future__ import annotations

import pytest

from src.utils.data_splitter import _MIN_AUTO_EMBARGO_SIZE, _resolve_wf_embargo_size


def test_resolve_embargo_none_uses_min_floor():
    """None verildiginde en az 200 dondurulur (time_steps daha kucukse bile)."""
    assert _resolve_wf_embargo_size(None, time_steps=30) == 200
    assert _resolve_wf_embargo_size(None, time_steps=10) == 200


def test_resolve_embargo_none_returns_time_steps_when_larger():
    """time_steps 200'den buyukse o deger kullanilir."""
    assert _resolve_wf_embargo_size(None, time_steps=250) == 250


def test_resolve_embargo_zero_treated_as_auto():
    """0 explicit deger bile auto'yu tetikler — leakage'i engellemek icin."""
    assert _resolve_wf_embargo_size(0, time_steps=30) == 200


def test_resolve_embargo_positive_int_passes_through():
    """Pozitif explicit int (>0) aynen dondurulur."""
    assert _resolve_wf_embargo_size(50, time_steps=30) == 50
    assert _resolve_wf_embargo_size(500, time_steps=30) == 500


def test_resolve_embargo_negative_treated_as_auto():
    """Negatif deger guvenli sekilde auto'ya cevrilir."""
    assert _resolve_wf_embargo_size(-5, time_steps=30) == 200


def test_min_floor_constant():
    """Sabit 200 olmali — SMA_200 lookback'i bunu zorunlar."""
    assert _MIN_AUTO_EMBARGO_SIZE == 200
