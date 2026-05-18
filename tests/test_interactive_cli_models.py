# -*- coding: utf-8 -*-
"""Interactive CLI model menu coverage."""

from __future__ import annotations


def test_interactive_manual_menu_contains_lstm_lite():
    from src.cli.interactive import AVAILABLE_MODELS

    assert "LSTM Lite" in AVAILABLE_MODELS
    assert AVAILABLE_MODELS.index("LSTM Lite") == AVAILABLE_MODELS.index("LSTM") + 1


def test_interactive_deep_learning_preset_contains_lstm_lite():
    from src.cli.interactive import PRESETS

    label, models = PRESETS["3"]
    assert "Derin" in label
    assert models == ["LSTM", "LSTM Lite"]
