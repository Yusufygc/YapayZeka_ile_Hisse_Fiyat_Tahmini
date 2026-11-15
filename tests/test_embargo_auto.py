# -*- coding: utf-8 -*-
"""
WF embargo auto-default testleri (P1 derin denetim 2026-11 ile guncellendi).

Eski kural `max(200, sequence uzunlugu)` "SMA200 sizintisi" gerekcesiyle konmustu; ozellikler
nedensel oldugundan yanlisti. Yeni kural: auto = target_horizon + 5 (pooled CV E=h+5);
explicit embargo < target_horizon -> ValueError.
"""

from __future__ import annotations

import pytest

from src.utils.data_splitter import _resolve_wf_embargo_size


def test_resolve_embargo_none_is_horizon_plus_five():
    assert _resolve_wf_embargo_size(None) == 6
    assert _resolve_wf_embargo_size(None, target_horizon=5) == 10


def test_resolve_embargo_zero_treated_as_auto():
    assert _resolve_wf_embargo_size(0) == 6


def test_resolve_embargo_positive_int_passes_through():
    assert _resolve_wf_embargo_size(50) == 50
    assert _resolve_wf_embargo_size(500) == 500


def test_resolve_embargo_negative_treated_as_auto():
    assert _resolve_wf_embargo_size(-5) == 6


def test_resolve_embargo_below_horizon_raises():
    with pytest.raises(ValueError):
        _resolve_wf_embargo_size(2, target_horizon=5)
