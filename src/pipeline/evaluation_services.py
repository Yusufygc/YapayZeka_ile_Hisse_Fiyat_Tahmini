# -*- coding: utf-8 -*-
"""
evaluation_services.py - EvaluationManager servis kompozisyon katmani.

Bu modul, eski mixin is mantigini manager mirasindan ayirip servis
nesnelerine tasir. EvaluationManager public orkestrasyon yuzeyi olarak kalir.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import os
from typing import Any, Dict, Optional

import numpy as np

from src.database.stock_model_db import StockModelDB
from src.pipeline.backtest_runner import _BacktestRunnerMixin
from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.metrics_reporter import _MetricsReporterMixin
from src.pipeline.prediction_engine import _PredictionEngineMixin
from src.pipeline.signal_calibrator import _SignalCalibratorMixin


@dataclass
class EvaluationContext:
    """Salt-okunur bağımlılık/kimlik torbası (evaluation servisleri için).

    Servislerin OKUDUĞU tüm config/identity attribute'ları burada toplanır.
    Gerçek kurulum `EvaluationContext.from_config(...)` ile yapılır: ExecutionConfig
    ile aynı isimli alanlar otomatik kopyalanır (tek tek atama yok). Tüm alanlar
    default'ludur; servisleri izole test etmek için boş `EvaluationContext()` veya
    yalnızca ilgili alanlarla kurulabilir.
    """

    stock_symbol: str = ""
    outputs_dir: str = ""
    models_dir: str = ""
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
    commission_bps: float = 10.5  # ExecutionConfig ile ayni (10 bps + %5 BSMV)
    slippage_bps: float = 5.0
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

    @classmethod
    def from_config(
        cls,
        *,
        stock_symbol: str,
        outputs_dir: str,
        models_dir: str,
        feature_names: list,
        dataset_hash: str,
        dataset_metadata: Dict[str, Any],
        exe_cfg: ExecutionConfig,
        model_cfg: ModelConfig,
        stock_db: Optional[StockModelDB] = None,
    ) -> "EvaluationContext":
        """Identity + ExecutionConfig/ModelConfig'ten context kurar.

        ExecutionConfig'te AYNI isimle bulunan context alanları (backtest_enabled,
        commission_bps, signal_calibration_* ...) olduğu gibi kopyalanır; yalnızca
        isim değiştiren/türetilen alanlar (signal_config -> default_signal_config,
        model_cfg.selected_models/ensemble_enabled, xai_dir) elle eşlenir.
        """
        exe_values = {
            f.name: getattr(exe_cfg, f.name)
            for f in fields(cls)
            if hasattr(exe_cfg, f.name)
        }
        return cls(
            stock_symbol=stock_symbol,
            outputs_dir=outputs_dir,
            models_dir=models_dir,
            feature_names=feature_names,
            dataset_hash=dataset_hash,
            dataset_metadata=dataset_metadata,
            exe_cfg=exe_cfg,
            model_cfg=model_cfg,
            stock_db=stock_db,
            **{
                **exe_values,
                "selected_models": set(model_cfg.selected_models) if model_cfg.selected_models else None,
                "ensemble_enabled": model_cfg.ensemble_enabled,
                "default_signal_config": exe_cfg.signal_config,
                "xai_dir": os.path.join(outputs_dir, "xai"),
            },
        )


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
    wf_xai_records: Dict[str, Any] = field(default_factory=dict)
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


class _EvaluationService:
    """Ortak ctor: READ-ONLY ``ctx`` (EvaluationContext) + mutable ``state`` (EvaluationState).

    Mixin gövdeleri ``self.ctx.X`` (config/identity) ve ``self.state.X`` (runtime
    çıktı/durum) kullanır; owner-forward yoktur.
    """

    def __init__(self, ctx: EvaluationContext, state: EvaluationState) -> None:
        self.ctx = ctx
        self.state = state


class PredictionService(_EvaluationService, _PredictionEngineMixin):
    """Prediction, inverse-target conversion and ensemble coordination."""


class BacktestService(_EvaluationService, _BacktestRunnerMixin):
    """Backtest execution, gate diagnostics and shadow scenarios."""


class SignalCalibrationService(_EvaluationService, _SignalCalibratorMixin):
    """Signal quality threshold and execution-parameter calibration."""


class MetricsReportingService(_EvaluationService, _MetricsReporterMixin):
    """Metric enrichment, model selection and XAI report routing."""
