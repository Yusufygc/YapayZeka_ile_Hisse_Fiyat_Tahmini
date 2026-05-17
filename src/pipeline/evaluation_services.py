# -*- coding: utf-8 -*-
"""
evaluation_services.py - EvaluationManager servis kompozisyon katmani.

Bu modul, eski mixin is mantigini manager mirasindan ayirip servis
nesnelerine tasir. Servisler mevcut davranisi korumak icin owner-backed
calisir; EvaluationManager public orkestrasyon yuzeyi olarak kalir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from src.database.stock_model_db import StockModelDB
from src.experiments.experiment_tracker import ExperimentTracker
from src.pipeline.backtest_runner import _BacktestRunnerMixin
from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.metrics_reporter import _MetricsReporterMixin
from src.pipeline.prediction_engine import _PredictionEngineMixin
from src.pipeline.signal_calibrator import _SignalCalibratorMixin


@dataclass
class EvaluationContext:
    """Immutable-ish dependency bag for evaluation services."""

    stock_symbol: str
    outputs_dir: str
    models_dir: str
    tracker: ExperimentTracker
    feature_names: list
    dataset_hash: str
    dataset_metadata: Dict[str, Any]
    exe_cfg: ExecutionConfig
    model_cfg: ModelConfig
    stock_db: Optional[StockModelDB] = None


@dataclass
class EvaluationState:
    """Runtime outputs preserved for backward-compatible manager attributes."""

    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    prediction_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    quantile_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    single_backtest_inputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    latest_tensors: Dict[str, Any] = field(default_factory=dict)
    latest_backtest_results: Dict[str, Any] = field(default_factory=dict)
    latest_backtest_metrics: Dict[str, Any] = field(default_factory=dict)
    latest_model_metrics: Dict[str, Any] = field(default_factory=dict)
    ensemble_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)


class _OwnerBackedService:
    """
    Compatibility adapter while the old mixin logic is moved behind services.

    The mixin methods read/write attributes such as predictions, signal_config
    and dataset_metadata. Forwarding those operations to the owner keeps public
    behavior stable while EvaluationManager no longer inherits the mixins.
    """

    def __init__(self, owner: Any) -> None:
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_owner":
            object.__setattr__(self, name, value)
            return
        setattr(self._owner, name, value)


class PredictionService(_OwnerBackedService, _PredictionEngineMixin):
    """Prediction, inverse-target conversion and ensemble coordination."""


class BacktestService(_OwnerBackedService, _BacktestRunnerMixin):
    """Backtest execution, gate diagnostics and shadow scenarios."""


class SignalCalibrationService(_OwnerBackedService, _SignalCalibratorMixin):
    """Signal quality threshold and execution-parameter calibration."""


class MetricsReportingService(_OwnerBackedService, _MetricsReporterMixin):
    """Metric enrichment, model selection and XAI report routing."""
