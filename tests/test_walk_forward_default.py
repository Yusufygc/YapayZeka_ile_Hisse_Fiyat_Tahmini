# -*- coding: utf-8 -*-
"""
Sprint 0 (2026-05-25) — Walk-forward default validation modu testi.

Plan: docs/wiki/log.md + plans/sistematik-ad-m-ad-m-yap-lacak-expressive-eich.md
"""

from __future__ import annotations

import pytest

from src.pipeline.config import (
    DataConfig,
    PipelineConfig,
    ValidationConfig,
)


def test_validation_config_default_is_walk_forward():
    """Sprint 0 (A0.1): ValidationConfig default validation_mode = 'walk_forward'."""
    cfg = ValidationConfig()
    assert cfg.validation_mode == "walk_forward", (
        f"ValidationConfig default validation_mode 'walk_forward' olmali, "
        f"bulundu: {cfg.validation_mode!r}"
    )


def test_pipeline_config_default_is_walk_forward():
    """PipelineConfig() default'u walk_forward modunu uretir."""
    cfg = PipelineConfig(data=DataConfig(data_file="dummy.csv"))
    assert cfg.validation.validation_mode == "walk_forward"


def test_validation_config_single_split_value_still_allowed_as_string():
    """single_split string degeri hala dataclass'a verilebilir (research-only
    akis icin gereklidir). Sadece default deger walk_forward olmali."""
    cfg = ValidationConfig(validation_mode="single_split")
    assert cfg.validation_mode == "single_split"
