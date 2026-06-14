# -*- coding: utf-8 -*-
"""
evaluation_services.py - EvaluationManager servis kompozisyon katmani.

Bu modul, eski mixin is mantigini manager mirasindan ayirip servis
nesnelerine tasir. EvaluationManager public orkestrasyon yuzeyi olarak kalir.
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
    """Salt-okunur bağımlılık/kimlik torbası (evaluation servisleri için).

    Faz 2 (E1 owner-forward epiği): servislerin owner'dan OKUDUĞU tüm config/
    identity attribute'ları burada toplanır. EvaluationManager bu alanları
    property forward ile (`manager.X` <-> `manager.context.X`) açar.

    Tüm alanlar default'ludur: `__init__`'i atlayan (`__new__`) mekanizma testleri
    için boş `EvaluationContext()` kurulabilir olmalı (bkz. `manager.context` lazy
    property). Gerçek `__init__` zaten tüm alanları explicit kurar.
    """

    stock_symbol: str = ""
    outputs_dir: str = ""
    models_dir: str = ""
    tracker: Optional[ExperimentTracker] = None
    feature_names: list = field(default_factory=list)
    dataset_hash: str = ""
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)
    exe_cfg: Optional[ExecutionConfig] = None
    model_cfg: Optional[ModelConfig] = None
    stock_db: Optional[StockModelDB] = None
    # Faz 2: exe_cfg/model_cfg/outputs_dir'den türetilen config/identity (READ-ONLY).
    ensemble_enabled: bool = False
    selected_models: Optional[set] = None
    backtest_enabled: bool = False
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    initial_capital: float = 0.0
    signal_mode: str = "legacy"
    default_signal_config: Any = None
    xai_dir: str = ""
    # Faz 3.2: BacktestService'in okuduğu exe_cfg flag'leri (READ-ONLY).
    write_trade_logs: bool = False
    signal_calibration_train_ratio: float = 0.70
    min_signal_evaluation_folds: int = 3
    signal_calibration_min_trades: int = 6
    signal_calibration_reject_behavior: str = "no_trade"
    auto_signal_diagnostics: bool = True
    enable_gate_diagnostics: bool = False
    enable_shadow_backtests: bool = False
    # Faz 3.3: SignalCalibrationService'in okuduğu exe_cfg flag'leri (READ-ONLY).
    # Default'lar eski getattr(self, "X", default) fallback'leriyle birebir eşleşir
    # (mixin + signal_calibration/grid.apply_trial_policy).
    calibration_scope: str = "wf_train"
    signal_calibration_require_oos_confirmation: bool = True
    signal_calibration_min_eval_excess_return: float = 0.0
    signal_calibration_min_eval_sharpe: float = 0.0
    signal_calibration_objective: str = "risk_adjusted"
    signal_calibration_profile: str = "production"
    signal_calibration_sampler: str = "adaptive_stratified"
    signal_calibration_seed: int = 42
    signal_calibration_max_trials: Optional[int] = 64
    # Faz 3.4: MetricsReportingService'in (XAI yazımı) okuduğu exe_cfg flag'leri
    # (READ-ONLY). Default'lar eski getattr(self, "X", default) fallback'leriyle
    # birebir eşleşir (metrics_reporter._write_xai_reports).
    write_xai_tables: bool = False
    write_markdown_reports: bool = True


@dataclass
class EvaluationState:
    """Runtime outputs preserved for backward-compatible manager attributes.

    Faz 1 (E1 owner-forward epiği): EvaluationManager'in tüm mutable evaluation
    state'i artık burada tutulur. Manager bu alanları property forward ile
    (`manager.X` <-> `manager.state.X`) açar; servisler ve workflow'lar explicit
    DI ile aynı state'e yazar/okur.
    """

    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    prediction_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    quantile_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    single_backtest_inputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    latest_tensors: Dict[str, Any] = field(default_factory=dict)
    latest_backtest_results: Dict[str, Any] = field(default_factory=dict)
    latest_backtest_metrics: Dict[str, Any] = field(default_factory=dict)
    latest_model_metrics: Dict[str, Any] = field(default_factory=dict)
    ensemble_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ensemble_weight_scope: Dict[str, str] = field(default_factory=dict)
    # Hizalanmis tahmin/gerçek diziler (None = henüz üretilmedi).
    y_true_aligned: Optional[np.ndarray] = None
    y_true_target_aligned: Optional[np.ndarray] = None
    prev_close_aligned: Optional[np.ndarray] = None
    # Sinyal eşik kalibrasyon durumu (WF kalibrasyonunda mutasyona uğrar).
    signal_config: Any = None
    signal_threshold_source: str = "default_config"
    signal_threshold_calibration_summary: Dict[str, Any] = field(default_factory=dict)


class PredictionService(_PredictionEngineMixin):
    """Prediction, inverse-target conversion and ensemble coordination.

    Faz 3 (E1 owner-forward epiği): owner-forward kaldırıldı. Bağımlılıklar
    açıkça enjekte edilir — READ-ONLY config/identity ``ctx`` (EvaluationContext),
    mutable runtime çıktı ``state`` (EvaluationState). Mixin gövdesi artık
    ``self.ctx.X`` / ``self.state.X`` kullanır (``self._owner`` forward yok).
    """

    def __init__(self, ctx: EvaluationContext, state: EvaluationState) -> None:
        self.ctx = ctx
        self.state = state


class BacktestService(_BacktestRunnerMixin):
    """Backtest execution, gate diagnostics and shadow scenarios.

    Faz 3.2 (E1 owner-forward epiği): owner-forward kaldırıldı; ctor `(ctx, state)`
    enjekte alır. Mixin gövdesi `self.ctx.X` (READ-ONLY config) / `self.state.X`
    (mutable runtime) kullanır.
    """

    def __init__(self, ctx: EvaluationContext, state: EvaluationState) -> None:
        self.ctx = ctx
        self.state = state


class SignalCalibrationService(_SignalCalibratorMixin):
    """Signal quality threshold and execution-parameter calibration.

    Faz 3.3 (E1 owner-forward epiği): owner-forward kaldırıldı; ctor `(ctx, state)`
    enjekte alır. Mixin gövdesi `self.ctx.X` (READ-ONLY config: calibration_scope,
    commission/slippage/initial_capital, outputs_dir, default_signal_config,
    dataset_metadata, signal_calibration_* flag'leri) / `self.state.X` (mutable
    runtime: signal_config, signal_threshold_source/calibration_summary) kullanır.
    """

    def __init__(self, ctx: EvaluationContext, state: EvaluationState) -> None:
        self.ctx = ctx
        self.state = state


class MetricsReportingService(_MetricsReporterMixin):
    """Metric enrichment, model selection and XAI report routing.

    Faz 3.4 (E1 owner-forward epiği): owner-forward kaldırıldı; ctor `(ctx, state)`
    enjekte alır. Mixin gövdesi `self.ctx.X` (READ-ONLY config: dataset_metadata,
    commission/slippage_bps, stock_symbol, feature_names, xai_dir,
    write_xai_tables, write_markdown_reports) / `self.state.X` (mutable runtime:
    predictions, prediction_targets, quantile_predictions, y_true_aligned,
    ensemble_weights, latest_backtest_results) kullanır.
    """

    def __init__(self, ctx: EvaluationContext, state: EvaluationState) -> None:
        self.ctx = ctx
        self.state = state
