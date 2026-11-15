# -*- coding: utf-8 -*-
"""Evaluation servislerini manager/__new__ hileleri olmadan kurmak icin test yardimcilari."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.evaluation_manager import EvaluationManager
from src.pipeline.evaluation_services import (
    BacktestService,
    EvaluationContext,
    EvaluationState,
    MetricsReportingService,
    PredictionService,
    SignalCalibrationService,
)


def make_eval_services(ctx: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> SimpleNamespace:
    """Yalnizca verilen alanlarla EvaluationContext/State + 4 servis kurar.

    ``ctx`` EvaluationContext alanlarini (config/identity), ``state`` EvaluationState
    alanlarini (mutable runtime) override eder; digerleri dataclass default'larindadir.
    """
    context = EvaluationContext(**(ctx or {}))
    run_state = EvaluationState(**(state or {}))
    return SimpleNamespace(
        ctx=context,
        state=run_state,
        prediction=PredictionService(context, run_state),
        backtest=BacktestService(context, run_state),
        signal_calibration=SignalCalibrationService(context, run_state),
        metrics=MetricsReportingService(context, run_state),
    )


def make_eval_manager(tmpdir: str, *, ensemble_enabled: bool = True, target_mode: str = "log_return", exe_cfg: ExecutionConfig | None = None) -> EvaluationManager:
    """Gercek EvaluationManager (tam __init__) kurar."""
    models_dir = os.path.join(tmpdir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return EvaluationManager(
        stock_symbol="TEST",
        outputs_dir=tmpdir,
        models_dir=models_dir,
        feature_names=["f1", "f2", "f3"],
        dataset_hash="abc123",
        dataset_metadata={"target_mode": target_mode, "signal_threshold_config": {}},
        exe_cfg=exe_cfg or ExecutionConfig(),
        model_cfg=ModelConfig(ensemble_enabled=ensemble_enabled),
    )
