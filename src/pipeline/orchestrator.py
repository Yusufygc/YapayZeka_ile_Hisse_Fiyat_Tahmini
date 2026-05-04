# -*- coding: utf-8 -*-
"""
orchestrator.py - Ana ForecastingPipeline sinifi
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

from src.database.stock_model_db import StockModelDB
from src.experiments.experiment_tracker import ExperimentTracker
from src.pipeline.config import PipelineConfig, DataConfig, ValidationConfig, ModelConfig, ExecutionConfig
from src.pipeline.data_manager import DataManager
from src.pipeline.evaluation_manager import EvaluationManager
from src.pipeline.model_trainer import ModelTrainer
from src.utils.reproducibility import set_global_seed


class ForecastingPipeline:
    def __init__(self, cfg: PipelineConfig):
        # ── config nesnelerini çöz ─────────────────────────────────────────
        d  = cfg.data
        v  = cfg.validation
        m  = cfg.models
        e  = cfg.execution
        sc = e.signal_config

        self.data_file       = d.data_file
        self.validation_mode = v.validation_mode
        self.selected_models = m.selected_models
        self.target_mode     = d.target_mode
        self.feature_mode    = d.feature_mode
        self.scaling_mode    = d.scaling_mode
        self.backtest_enabled = e.backtest_enabled
        self.commission_bps  = e.commission_bps
        self.slippage_bps    = e.slippage_bps
        self.initial_capital = e.initial_capital
        self.signal_mode     = e.signal_mode
        self.quality_gate_mode             = getattr(sc, "quality_gate_mode", "soft")
        self.signal_entry_cost_multiplier  = sc.entry_cost_multiplier
        self.signal_volatility_multiplier  = sc.volatility_multiplier
        self.min_holding_bars              = sc.min_holding_bars
        self.max_holding_bars              = sc.max_holding_bars
        self.take_profit_vol_multiplier    = sc.take_profit_vol_multiplier
        self.stop_loss_vol_multiplier      = sc.stop_loss_vol_multiplier
        self.min_directional_accuracy      = sc.min_directional_accuracy
        self.max_rmse_vs_benchmark         = sc.max_rmse_vs_benchmark
        self.min_composite_score           = sc.min_composite_score
        self.emergency_stop_overrides_min_hold = sc.emergency_stop_overrides_min_hold
        self.macro_rate_lag_days  = d.macro_rate_lag_days
        self.macro_cpi_lag_days   = d.macro_cpi_lag_days
        self.wf_n_splits          = v.wf_n_splits
        self.wf_min_train_size    = v.wf_min_train_size
        self.wf_test_size         = v.wf_test_size
        self.wf_max_train_size    = v.wf_max_train_size
        self.wf_window_type       = v.wf_window_type
        self.wf_embargo_size      = v.wf_embargo_size
        self.final_holdout_size   = v.final_holdout_size
        self.prune_correlated_features     = d.prune_correlated_features
        self.correlation_threshold         = d.correlation_threshold
        self.lag_feature_count             = d.lag_feature_count
        self.use_prophet_macro_regressors  = m.model_settings.get("prophet", {}).get("use_regressors", True)
        self.ensemble_enabled              = m.ensemble_enabled
        self.universe_file                 = d.universe_file
        self.clip_shift_warning_threshold_pct = d.clip_shift_warning_threshold_pct
        self.training_window_years         = d.training_window_years
        self.window_candidates             = d.window_candidates or [3, 5, 7, 10, None]
        self.min_history_days              = d.min_history_days
        self.new_listing_min_days          = d.new_listing_min_days

        if self.wf_window_type == "sliding" and self.training_window_years is not None:
            self.wf_max_train_size = max(
                int(self.training_window_years) * 252,
                int(self.wf_max_train_size or 0),
            )

        self.model_config = m.model_settings
        self.model_config.setdefault("prophet", {})
        self.model_config["prophet"].setdefault("use_regressors", self.use_prophet_macro_regressors)

        # cfg'yi saklıyoruz — EvaluationManager'a geçirilecek
        self._cfg = cfg

        set_global_seed(42)

        self.stock_symbol = os.path.splitext(os.path.basename(self.data_file))[0]
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.outputs_dir = os.path.join(self.project_root, "outputs", self.stock_symbol)
        self.models_dir = os.path.join(self.outputs_dir, "models")
        self.experiment_dir = os.path.join(self.outputs_dir, "experiments")
        self.registry_dir = self.models_dir

        self.tracker = ExperimentTracker(self.experiment_dir)

        db_path = os.path.join(self.project_root, "data", "stock_models.db")
        self.stock_db = StockModelDB(db_path)

        self.data_manager = DataManager(
            data_cfg=self._cfg.data,
            val_cfg=self._cfg.validation,
            models_dir=self.models_dir,
            macro_cache_dir=os.path.join(self.project_root, "data", "macro"),
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
        print(
            "  [INFO] Data Window : "
            f"training_window_years={self.training_window_years}, "
            f"min_history={self.min_history_days}, new_listing_min={self.new_listing_min_days}"
        )
        print(f"  [INFO] Backtest    : {'acik' if self.backtest_enabled else 'kapali'}")
        print(f"  [INFO] Signal Mode : {self.signal_mode}")
        print(f"  [INFO] Quality Gate: {self.quality_gate_mode}")
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
        # --- Save validation protocol report ---
        try:
            import pandas as _pd
            vp_df = self.data_manager.get_validation_protocol_data()
            if vp_df is not None and not vp_df.empty:
                vp_path = os.path.join(self.outputs_dir, "validation_protocol_report.csv")
                vp_df.to_csv(vp_path, index=False)
                print(f"  [OK] Validation protocol raporu kaydedildi -> {vp_path}")
        except Exception as _e:
            print(f"  [WARN] Validation protocol raporu kaydedilemedi: {_e}")

        # --- Save data quality reports ---
        try:
            import pandas as _pd
            dq_reports = self.data_manager.get_data_quality_reports()
            for report_name, report_data in dq_reports.items():
                if report_data:
                    try:
                        dq_df = _pd.DataFrame(report_data)
                        dq_path = os.path.join(self.outputs_dir, f"data_quality_{report_name}.csv")
                        dq_df.to_csv(dq_path, index=False)
                    except Exception:
                        pass
        except Exception as _e:
            print(f"  [WARN] Data quality raporlari kaydedilemedi: {_e}")
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
                "quality_gate_mode": self.quality_gate_mode,
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
            self.data_manager.feature_names,
            self.selected_models,
            dataset_hash=self.run_dataset_hash,
            dataset_metadata=self.run_dataset_metadata,
            model_config=self.model_config,
        )
        self.evaluation_manager = EvaluationManager(
            self.stock_symbol,
            self.outputs_dir,
            self.models_dir,
            self.tracker,
            self.data_manager.feature_names,
            self.run_dataset_hash,
            self.run_dataset_metadata,
            exe_cfg=self._cfg.execution,
            model_cfg=self._cfg.models,
            stock_db=self.stock_db,
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
            wf_result = self.evaluation_manager.evaluate_walk_forward(
                self.model_trainer.wf_results,
                self.model_trainer.wf_predictions,
                self.model_trainer.wf_y_true,
                self.model_trainer.wf_backtest_inputs,
                self.model_trainer.wf_fold_metrics,
            )
            best_model_name = wf_result["best_model_name"]
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

        # --- Save data quality reports (end of pipeline) ---
        try:
            import pandas as _pd
            dq_reports = self.data_manager.get_data_quality_reports()
            for report_name, report_data in dq_reports.items():
                if report_data:
                    try:
                        dq_df = _pd.DataFrame(report_data)
                        dq_path = os.path.join(self.outputs_dir, f"data_quality_{report_name}.csv")
                        dq_df.to_csv(dq_path, index=False)
                    except Exception:
                        pass
        except Exception as _e:
            print(f"  [WARN] Data quality raporlari kaydedilemedi: {_e}")

        print("\n  [OK] Pipeline completed successfully!")
