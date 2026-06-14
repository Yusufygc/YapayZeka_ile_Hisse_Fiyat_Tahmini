# -*- coding: utf-8 -*-
"""Kol-A horizon support policy guards."""

from __future__ import annotations

from typing import Any


def assert_kola_production_horizon_supported(data_cfg: Any) -> None:
    """Fail loud for Kol-A production paths that cannot yet honor h > 1."""
    horizon = max(1, int(getattr(data_cfg, "target_horizon", 1) or 1))
    if horizon <= 1:
        return
    raise ValueError(
        "Kol-A production backtest/forecast semantics currently support only "
        "target_horizon=1. target_horizon>1 requires the separate horizon-aware "
        "evaluation/backtest refactor before production use."
    )
