# -*- coding: utf-8 -*-
"""Kol-A target_horizon production guard tests."""

from __future__ import annotations

import pytest


def test_kola_horizon_policy_allows_default_horizon():
    from src.pipeline.config import DataConfig
    from src.pipeline.horizon_policy import assert_kola_production_horizon_supported

    assert_kola_production_horizon_supported(DataConfig(data_file="TEST.csv", target_horizon=1))


def test_forecasting_pipeline_rejects_target_horizon_gt_one_before_data_manager(tmp_path, monkeypatch):
    from src.pipeline.config import DataConfig, ModelConfig, PipelineConfig, ValidationConfig
    from src.pipeline.orchestrator import ForecastingPipeline

    csv_path = tmp_path / "TEST.csv"
    csv_path.write_text("Date,Open,High,Low,Close,Volume\n", encoding="utf-8")

    data_cfg = DataConfig(
        data_file=str(csv_path),
        target_horizon=5,
        use_macro=False,
        universe_auto_sync=False,
    )
    cfg = PipelineConfig(
        data=data_cfg,
        validation=ValidationConfig(validation_mode="walk_forward"),
        models=ModelConfig(selected_models=["Ridge Return"], require_available=False),
    )

    calls = {"data_manager": 0}

    def _boom(*args, **kwargs):
        calls["data_manager"] += 1
        raise AssertionError("DataManager must not be constructed before horizon guard")

    monkeypatch.setattr("src.pipeline.orchestrator.DataManager", _boom)

    with pytest.raises(ValueError, match="target_horizon=1"):
        ForecastingPipeline(cfg)
    assert calls["data_manager"] == 0
