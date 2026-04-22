# -*- coding: utf-8 -*-
"""
orchestrator.py - Ana ForecastingPipeline sinifi
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

from src.database.stock_model_db import StockModelDB
from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry
from src.pipeline.data_manager import DataManager
from src.pipeline.evaluation_manager import EvaluationManager
from src.pipeline.model_trainer import ModelTrainer
from src.utils.reproducibility import set_global_seed


class ForecastingPipeline:
    def __init__(
        self,
        data_file: str,
        test_ratio: float = 0.20,
        time_steps: int = 30,
        validation_mode: str = "single_split",
        selected_models: list = None,
        target_mode: str = "log_return",
        feature_mode: str = "stationary_features",
        scaling_mode: str = "robust_x_standard_y_clip",
        backtest_enabled: bool = True,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        initial_capital: float = 100000.0,
        signal_mode: str = "legacy",
        signal_entry_cost_multiplier: float = 2.0,
        signal_volatility_multiplier: float = 0.25,
        min_holding_bars: int = 3,
        max_holding_bars: int = 20,
        take_profit_vol_multiplier: float = 1.5,
        stop_loss_vol_multiplier: float = 1.0,
        min_directional_accuracy: float = 52.0,
        max_rmse_vs_benchmark: float = 1.05,
        min_composite_score: float = 50.0,
        emergency_stop_overrides_min_hold: bool = True,
        macro_rate_lag_days: int = 1,
        macro_cpi_lag_days: int = 15,
        wf_n_splits: int = 12,
        wf_min_train_size: int = 504,
        wf_test_size: int = 21,
        wf_max_train_size: int | None = 756,
        wf_window_type: str = "sliding",
        final_holdout_size: int = 60,
        model_config: dict | None = None,
        prune_correlated_features: bool = False,
        correlation_threshold: float = 0.98,
        clip_shift_warning_threshold_pct: float = 1.0,
    ):
        self.data_file = data_file
        self.validation_mode = validation_mode
        self.selected_models = selected_models
        self.target_mode = target_mode
        self.feature_mode = feature_mode
        self.scaling_mode = scaling_mode
        self.backtest_enabled = backtest_enabled
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.initial_capital = initial_capital
        self.signal_mode = signal_mode
        self.signal_entry_cost_multiplier = signal_entry_cost_multiplier
        self.signal_volatility_multiplier = signal_volatility_multiplier
        self.min_holding_bars = min_holding_bars
        self.max_holding_bars = max_holding_bars
        self.take_profit_vol_multiplier = take_profit_vol_multiplier
        self.stop_loss_vol_multiplier = stop_loss_vol_multiplier
        self.min_directional_accuracy = min_directional_accuracy
        self.max_rmse_vs_benchmark = max_rmse_vs_benchmark
        self.min_composite_score = min_composite_score
        self.emergency_stop_overrides_min_hold = emergency_stop_overrides_min_hold
        self.macro_rate_lag_days = macro_rate_lag_days
        self.macro_cpi_lag_days = macro_cpi_lag_days
        self.wf_n_splits = wf_n_splits
        self.wf_min_train_size = wf_min_train_size
        self.wf_test_size = wf_test_size
        self.wf_max_train_size = wf_max_train_size
        self.wf_window_type = wf_window_type
        self.final_holdout_size = final_holdout_size
        self.prune_correlated_features = prune_correlated_features
        self.correlation_threshold = correlation_threshold
        self.clip_shift_warning_threshold_pct = clip_shift_warning_threshold_pct
        self.model_config = model_config or {
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
                "tft": {
                    "model_label": "TFT-like Quantile Sequence Model",
                    "epochs_single": 80,
                    "epochs_wf": 50,
                    "epochs_final": 50,
                    "patience_single": 15,
                    "patience_wf": 12,
                    "patience_final": 12,
                    "lr_patience": 5,
                    "dropout": 0.3,
                    "batch_size": 32,
                },
            },
            "experimental_sequence_baselines": {
                "enabled_models": ["DLinear", "NLinear", "PatchTST Experimental"],
                "patchtst_status": "evaluation_path_prepared_not_production",
                "patchtst_config": {"lookback": 128, "patch_length": 16, "stride": 8, "alpha": 1.0},
            },
            "gradient_boosting": {"lightgbm_optional": True},
        }

        set_global_seed(42)

        self.stock_symbol = os.path.splitext(os.path.basename(self.data_file))[0]
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.outputs_dir = os.path.join(self.project_root, "outputs", self.stock_symbol)
        self.models_dir = os.path.join(self.outputs_dir, "models")
        self.experiment_dir = os.path.join(self.outputs_dir, "experiments")
        self.registry_dir = self.models_dir

        self.tracker = ExperimentTracker(self.experiment_dir)
        self.registry = ModelRegistry(self.registry_dir)

        db_path = os.path.join(self.project_root, "stock_models.db")
        self.stock_db = StockModelDB(db_path)

        self.data_manager = DataManager(
            data_file,
            test_ratio,
            time_steps,
            self.models_dir,
            use_macro=True,
            macro_cache_dir=os.path.join(self.project_root, "data", "macro"),
            target_mode=self.target_mode,
            feature_mode=self.feature_mode,
            scaling_mode=self.scaling_mode,
            macro_rate_lag_days=self.macro_rate_lag_days,
            macro_cpi_lag_days=self.macro_cpi_lag_days,
            wf_n_splits=self.wf_n_splits,
            wf_min_train_size=self.wf_min_train_size,
            wf_test_size=self.wf_test_size,
            wf_max_train_size=self.wf_max_train_size,
            wf_window_type=self.wf_window_type,
            final_holdout_size=self.final_holdout_size,
            prune_correlated_features=self.prune_correlated_features,
            correlation_threshold=self.correlation_threshold,
            clip_shift_warning_threshold_pct=self.clip_shift_warning_threshold_pct,
        )

        self.model_trainer = None
        self.evaluation_manager = None
        self.registry_version = "v5"
        self.run_dataset_metadata = {}
        self.run_dataset_hash = "N/A"

    def setup_environment(self) -> None:
        for directory in [self.models_dir, self.outputs_dir, self.experiment_dir, self.registry_dir]:
            os.makedirs(directory, exist_ok=True)

        print(f"\n  [INFO] Pipeline Modu: {self.validation_mode}")
        print(f"  [INFO] Hisse Sembolu: {self.stock_symbol}")
        print(f"  [INFO] Target Mode : {self.target_mode}")
        print(f"  [INFO] Feature Mode: {self.feature_mode}")
        print(f"  [INFO] Scaling Mode: {self.scaling_mode}")
        print(f"  [INFO] Macro Lag   : rate={self.macro_rate_lag_days}d, CPI={self.macro_cpi_lag_days}d")
        print(
            "  [INFO] WF Config   : "
            f"splits={self.wf_n_splits}, test={self.wf_test_size}, "
            f"min_train={self.wf_min_train_size}, max_train={self.wf_max_train_size}, "
            f"type={self.wf_window_type}, holdout={self.final_holdout_size}"
        )
        print(f"  [INFO] Backtest    : {'acik' if self.backtest_enabled else 'kapali'}")
        print(f"  [INFO] Signal Mode : {self.signal_mode}")
        print(
            "  [INFO] Feature QC  : "
            f"corr_prune={self.prune_correlated_features}, "
            f"corr_threshold={self.correlation_threshold}, "
            f"clip_warn={self.clip_shift_warning_threshold_pct}%"
        )
        deep_cfg = self.model_config.get("deep_learning", {})
        print(
            "  [INFO] Deep Config : "
            f"min_seq={deep_cfg.get('min_sequence_samples')}, "
            f"val_ratio={deep_cfg.get('validation_ratio')}, "
            f"min_val={deep_cfg.get('min_validation_samples')}"
        )

    def run_all(self) -> None:
        self.setup_environment()

        self.data_manager.ingest_and_engineer()
        self.data_manager.split_data(self.validation_mode)
        self.data_manager.save_validation_protocol_report(self.outputs_dir)
        self.data_manager.save_data_quality_reports(self.outputs_dir)
        self.run_dataset_metadata, self.run_dataset_hash = self.data_manager.build_run_metadata(
            self.validation_mode,
            model_config={
                "selected_models": self.selected_models or "all",
                "registry_version": self.registry_version,
                "xgboost_hpo": {"scope": "train_only_temporal_cv", "n_trials": 5, "n_splits": 3},
                "random_forest_hpo": {"scope": "train_only_temporal_cv", "n_trials": 5, "n_splits": 3},
                **self.model_config,
            },
        )
        self.run_dataset_metadata["signal_threshold_config"] = {
            "phase": "phase6_backtest_standard",
            "source": "default_config",
            "selection_scope": "configured_defaults_or_calibration_set",
            "final_holdout_optimized": False,
            "quality_thresholds": {
                "min_directional_accuracy": self.min_directional_accuracy,
                "max_rmse_vs_benchmark": self.max_rmse_vs_benchmark,
                "min_composite_score": self.min_composite_score,
            },
            "execution_policy": "decision_applies_to_next_bar_return",
            "cost_policy": {
                "commission_bps": self.commission_bps,
                "slippage_bps": self.slippage_bps,
                "entry_exit_accounted_separately": True,
            },
        }

        self.model_trainer = ModelTrainer(
            self.stock_symbol,
            self.tracker,
            self.registry,
            self.data_manager.feature_names,
            self.selected_models,
            dataset_hash=self.run_dataset_hash,
            dataset_metadata=self.run_dataset_metadata,
            registry_version=self.registry_version,
            model_config=self.model_config,
        )
        self.evaluation_manager = EvaluationManager(
            self.stock_symbol,
            self.outputs_dir,
            self.models_dir,
            self.tracker,
            self.registry,
            self.data_manager.feature_names,
            self.run_dataset_hash,
            self.run_dataset_metadata,
            selected_models=self.selected_models,
            registry_version=self.registry_version,
            stock_db=self.stock_db,
            backtest_enabled=self.backtest_enabled,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            initial_capital=self.initial_capital,
            signal_mode=self.signal_mode,
            signal_entry_cost_multiplier=self.signal_entry_cost_multiplier,
            signal_volatility_multiplier=self.signal_volatility_multiplier,
            min_holding_bars=self.min_holding_bars,
            max_holding_bars=self.max_holding_bars,
            take_profit_vol_multiplier=self.take_profit_vol_multiplier,
            stop_loss_vol_multiplier=self.stop_loss_vol_multiplier,
            min_directional_accuracy=self.min_directional_accuracy,
            max_rmse_vs_benchmark=self.max_rmse_vs_benchmark,
            min_composite_score=self.min_composite_score,
            emergency_stop_overrides_min_hold=self.emergency_stop_overrides_min_hold,
        )

        print("\n" + "=" * 60)
        print("  ADIM 4 | Model Egitimi ve Tracking (Orchestrator)")
        print("=" * 60)

        if self.validation_mode == "single_split":
            self.model_trainer.train_single_split(self.data_manager.tensors)
            self.evaluation_manager.generate_predictions(self.model_trainer.trained_models, self.data_manager.tensors)
            self.evaluation_manager.evaluate_single_split(self.model_trainer.trained_models)
        elif self.validation_mode == "walk_forward":
            self.model_trainer.train_walk_forward(self.data_manager.wf_splits, self.data_manager)
            best_model_name = self.evaluation_manager.evaluate_walk_forward(
                self.model_trainer.wf_results,
                self.model_trainer.wf_predictions,
                self.model_trainer.wf_y_true,
                self.model_trainer.wf_backtest_inputs,
                self.model_trainer.wf_fold_metrics,
            )
            if best_model_name and self.data_manager.final_holdout_df is not None and not self.data_manager.final_holdout_df.empty:
                try:
                    final_model, final_tensors = self.model_trainer.train_final_holdout_model(
                        best_model_name,
                        self.data_manager,
                    )
                    self.evaluation_manager.evaluate_final_holdout(
                        best_model_name,
                        final_model,
                        final_tensors,
                    )
                except Exception as exc:
                    print(f"  [WARN] Final holdout degerlendirmesi basarisiz, atlaniyor: {exc}")

        self.data_manager.save_data_quality_reports(self.outputs_dir)

        print("\n  [OK] Pipeline completed successfully!")
