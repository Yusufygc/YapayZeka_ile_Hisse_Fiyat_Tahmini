# -*- coding: utf-8 -*-
"""Interactive CLI model menu coverage."""

from __future__ import annotations

REMOVED = {"LSTM", "LSTM Lite", "AttentionLSTM v2", "DLinear", "NLinear", "Prophet", "ARIMA"}


def test_interactive_manual_menu_has_only_supported_candidates():
    from src.cli.interactive import AVAILABLE_MODELS

    assert {"XGBoost", "Ridge Return", "LightGBM Return"} <= set(AVAILABLE_MODELS)
    assert not REMOVED & set(AVAILABLE_MODELS)


def test_interactive_presets_never_offer_removed_models():
    from src.cli.interactive import AVAILABLE_MODELS, PRESETS

    for label, models in PRESETS.values():
        if models is None:
            continue
        assert not REMOVED & set(models), label
        assert set(models) <= set(AVAILABLE_MODELS), label
