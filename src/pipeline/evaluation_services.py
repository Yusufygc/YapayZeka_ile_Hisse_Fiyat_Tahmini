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
    """Salt-okunur bağımlılık/kimlik torbası (evaluation servisleri için).

    Faz 2 (E1 owner-forward epiği): servislerin owner'dan OKUDUĞU tüm config/
    identity attribute'ları burada toplanır. EvaluationManager bu alanları
    property forward ile (`manager.X` <-> `manager.context.X`) açar; owner-forward
    servisler/workflow'lar getattr üzerinden aynı context'ten okur. Faz 3'te
    servisler doğrudan `self.ctx.X` kullanacak.

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
    signal_calibration_min_trades: int = 6
    signal_calibration_reject_behavior: str = "no_trade"
    auto_signal_diagnostics: bool = True
    enable_gate_diagnostics: bool = False
    enable_shadow_backtests: bool = False


@dataclass
class EvaluationState:
    """Runtime outputs preserved for backward-compatible manager attributes.

    Faz 1 (E1 owner-forward epiği): EvaluationManager'in tüm mutable evaluation
    state'i artık burada tutulur. Manager bu alanları property forward ile
    (`manager.X` <-> `manager.state.X`) açar; servisler/workflow'lar owner-forward
    üzerinden aynı state'e yazar/okur. Faz 3'te servisler `self.state.X`'e geçer.
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


class _OwnerBackedService:
    """
    Compatibility adapter while the old mixin logic is moved behind services.

    The mixin methods read/write attributes such as predictions, signal_config
    and dataset_metadata. Forwarding those operations to the owner keeps public
    behavior stable while EvaluationManager no longer inherits the mixins.

    Writes are fail-loud: a forwarded assignment must either target an attribute
    that already exists on the owner (every owner pre-initializes its own state
    in ``__init__``) or be one of the few attributes that are legitimately
    lazy-created on first use (``_LAZY_FORWARDED_WRITES``). This closes the
    silent-typo encapsulation hole of the old blanket ``__setattr__`` (a mistyped
    attribute used to create a new owner attribute silently) while keeping the
    shared base usable across every service family (evaluation, training/eval
    workflows, data-manager services), each of which forwards to a different
    owner with a different state surface.
    """

    # Attributes that may be created on the owner on first write because they are
    # intentionally lazy-initialized rather than set in the owner's __init__.
    _LAZY_FORWARDED_WRITES = frozenset({"ensemble_weight_scope"})

    # Opt-in fail-loud guard. Enabled for the evaluation services and workflows
    # (the E1/B1 owner-forward god-object). DataManager service families keep the
    # permissive forwarding until their owner state surface is hardened too.
    _FAIL_LOUD = True

    def __init__(self, owner: Any) -> None:
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_owner":
            object.__setattr__(self, name, value)
            return
        owner = object.__getattribute__(self, "_owner")
        if (
            self._FAIL_LOUD
            and not hasattr(owner, name)
            and name not in self._LAZY_FORWARDED_WRITES
        ):
            raise AttributeError(
                f"{type(self).__name__} tried to set unknown owner attribute "
                f"'{name}'. Owner-forwarded writes must target an attribute the "
                f"owner initialized (or a declared lazy attribute); this guards "
                f"against silent typos. Initialize it in the owner's __init__ or "
                f"add it to _OwnerBackedService._LAZY_FORWARDED_WRITES if the "
                f"write is intentional."
            )
        setattr(owner, name, value)


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


class SignalCalibrationService(_OwnerBackedService, _SignalCalibratorMixin):
    """Signal quality threshold and execution-parameter calibration."""


class MetricsReportingService(_OwnerBackedService, _MetricsReporterMixin):
    """Metric enrichment, model selection and XAI report routing."""
