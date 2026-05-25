# -*- coding: utf-8 -*-
"""
config.py - Pipeline Configuration Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Grup parametrelerini mantiksal nesnelerde toplayarak
parameter explosion ve over-engineering hantalligi giderir.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.backtesting.signals import SignalConfig


@dataclass
class DataConfig:
    """Veri yukleme, ozellik muhendisligi ve kalite kontrolleri ayarlari."""

    data_file: str
    test_ratio: float = 0.20
    time_steps: int = 30
    target_mode: str = "log_return"
    feature_mode: str = "stationary_features"
    scaling_mode: str = "robust_x_standard_y_clip"
    use_macro: bool = True
    macro_rate_lag_days: int = 1
    macro_cpi_lag_days: int = 15
    prune_correlated_features: bool = False
    correlation_threshold: float = 0.98
    lag_feature_count: int = 5
    universe_file: str = "data/bist_universe.csv"
    clip_shift_warning_threshold_pct: float = 1.0
    training_window_years: Optional[int] = 5
    window_candidates: List[Optional[int]] = field(default_factory=lambda: [3, 5, 7, 10, None])
    min_history_days: int = 504
    new_listing_min_days: int = 252
    auto_update_data: bool = False
    auto_update_interactive: bool = False
    universe_auto_sync: bool = True


@dataclass
class ValidationConfig:
    """Validasyon protokolu (Single Split, Walk-Forward) ayarlari."""

    # Sprint 0 (2026-05-25): Default `walk_forward`. `single_split` modu yalniz
    # `--debug-quick` bayragiyla erisilebilir; uretime gitmez.
    validation_mode: str = "walk_forward"  # "walk_forward" (default) | "single_split" (debug only)
    wf_n_splits: int = 12
    wf_min_train_size: int = 504
    wf_test_size: int = 21
    wf_max_train_size: Optional[int] = 756
    wf_window_type: str = "sliding"
    wf_embargo_size: Optional[int] = None
    final_holdout_size: int = 60


@dataclass
class ModelConfig:
    """Model secimi, hiperparametreler ve ensemble ayarlari."""

    selected_models: Optional[List[str]] = None
    # Faz 4: registry'de var ama bu koşuda eğitilmesin / raporlanmasin.
    disabled_models: List[str] = field(default_factory=list)
    # Eksik optional dep durumunda davranış: True → fail, False → sessizce atla.
    require_available: bool = False
    # Spec.ensemble_eligible alanını runtime'da override etme imkanı.
    ensemble_eligibility_overrides: Dict[str, bool] = field(default_factory=dict)
    registry_version: str = "v5"
    ensemble_enabled: bool = True
    model_settings: Dict[str, Any] = field(
        default_factory=lambda: {
            "arima": {"auto_order": False, "order": (1, 0, 0)},
            "deep_learning": {
                "min_sequence_samples": 64,
                "validation_ratio": 0.1,
                "min_validation_samples": 32,
                "lstm": {
                    "epochs_single": 80,
                    "epochs_wf": 50,
                    "epochs_final": 50,
                    "patience": 15,
                    "lr_patience": 5,
                    "dropout": 0.2,
                    "batch_size": 32,
                },
                "lstm_lite_min_sequence_samples": 252,
                "lstm_lite": {
                    "units": 32,
                    "dense_units": 16,
                    "dropout": 0.25,
                    "learning_rate": 0.0003,
                    "epochs_single": 80,
                    "epochs_wf": 50,
                    "epochs_final": 50,
                    "patience": 12,
                    "lr_patience": 4,
                    "batch_size": 32,
                    "tune_on_fit": False,
                    "tune_n_trials": 12,
                },
                "attention_lstm_v2_min_sequence_samples": 252,
                "attention_lstm_v2": {
                    "units_1": 64,
                    "units_2": 32,
                    "dense_units": 32,
                    "dropout": 0.30,
                    "learning_rate": 0.0005,
                    "loss": "huber",
                    "epochs_single": 80,
                    "epochs_wf": 50,
                    "epochs_final": 50,
                    "patience": 12,
                    "lr_patience": 4,
                    "batch_size": 32,
                    "tune_on_fit": False,
                    "tune_n_trials": 12,
                },
            },
            "experimental_sequence_baselines": {},
            "gradient_boosting": {"lightgbm_optional": True},
            "prophet": {"use_regressors": True},
        }
    )


@dataclass
class ExecutionConfig:
    """Backtest, maliyetler ve sinyal uretim ayarlari."""

    backtest_enabled: bool = True
    initial_capital: float = 100000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    signal_mode: str = "simple"  # "simple", "legacy" veya "professional"
    signal_config: SignalConfig = field(default_factory=SignalConfig)
    # Leakage korumasi (Faz 2.5):
    # "wf_train" → kalibrasyon yalnizca WF fold verisi kullanir; final holdout ASLA.
    # Baska bir deger atanirsa _assert_wf_train_scope() RuntimeError firlatir.
    calibration_scope: str = "wf_train"
    signal_calibration_train_ratio: float = 0.70
    min_signal_evaluation_folds: int = 3
    enable_signal_execution_calibration: bool = False
    enable_gate_diagnostics: bool = False
    enable_shadow_backtests: bool = False
    signal_calibration_max_trials: int = 64
    signal_calibration_profile: str = "production"  # "production" veya "research"
    signal_calibration_sampler: str = "adaptive_stratified"
    signal_calibration_seed: int = 42
    signal_calibration_objective: str = "risk_adjusted"
    signal_calibration_min_trades: int = 6
    signal_calibration_require_oos_confirmation: bool = True
    signal_calibration_min_eval_excess_return: float = 0.0
    signal_calibration_min_eval_sharpe: float = 0.0
    signal_calibration_reject_behavior: str = "no_trade"
    auto_signal_diagnostics: bool = True
    report_detail_level: str = "summary"  # "summary" veya "research"
    write_text_reports: bool = False
    write_markdown_reports: bool = True
    write_xai_tables: bool = True
    write_trade_logs: bool = False
    auto_generate_forecast_after_training: bool = True
    research_policy: Optional[str] = None
    research_phase: Optional[str] = None
    research_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Tum pipeline'i kapsayan kok konfigurasyon nesnesi."""

    data: DataConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
