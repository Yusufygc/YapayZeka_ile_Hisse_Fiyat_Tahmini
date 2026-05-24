# -*- coding: utf-8 -*-
"""
orchestrator.py - Ana ForecastingPipeline sinifi
"""

import json
import os
import shutil
import hashlib
import re
import subprocess
import sys
import time
import warnings
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

from src.database.stock_model_db import StockModelDB
from src.experiments.experiment_tracker import ExperimentTracker
from src.pipeline.config import (
    PipelineConfig,
    DataConfig,
    ValidationConfig,
    ModelConfig,
    ExecutionConfig,
)
from src.pipeline.data_manager import DataManager
from src.pipeline.evaluation_manager import EvaluationManager
from src.pipeline.model_scope import (
    BENCHMARK_MODELS,
    normalize_candidate_models,
    resolve_candidates,
)
from src.pipeline.model_trainer import ModelTrainer
from src.utils.reproducibility import set_global_seed
from src.pipeline.artifacts import (
    write_window_selection_decision,
    write_validation_and_quality_reports,
    write_run_manifest,
)


class ForecastingPipeline:
    def __init__(self, cfg: PipelineConfig):
        # ── config nesnelerini çöz ─────────────────────────────────────────
        d = cfg.data
        v = cfg.validation
        m = cfg.models
        e = cfg.execution
        sc = e.signal_config

        self.data_file = d.data_file
        self.validation_mode = v.validation_mode
        self.selected_models = m.selected_models
        self.target_mode = d.target_mode
        self.feature_mode = d.feature_mode
        self.scaling_mode = d.scaling_mode
        self.backtest_enabled = e.backtest_enabled
        self.commission_bps = e.commission_bps
        self.slippage_bps = e.slippage_bps
        self.initial_capital = e.initial_capital
        self.signal_mode = e.signal_mode
        self.quality_gate_mode = getattr(sc, "quality_gate_mode", "soft")
        self.signal_entry_cost_multiplier = sc.entry_cost_multiplier
        self.signal_volatility_multiplier = sc.volatility_multiplier
        self.min_holding_bars = sc.min_holding_bars
        self.max_holding_bars = sc.max_holding_bars
        self.take_profit_vol_multiplier = sc.take_profit_vol_multiplier
        self.stop_loss_vol_multiplier = sc.stop_loss_vol_multiplier
        self.min_directional_accuracy = sc.min_directional_accuracy
        self.max_rmse_vs_benchmark = sc.max_rmse_vs_benchmark
        self.min_composite_score = sc.min_composite_score
        self.emergency_stop_overrides_min_hold = sc.emergency_stop_overrides_min_hold
        self.macro_rate_lag_days = d.macro_rate_lag_days
        self.macro_cpi_lag_days = d.macro_cpi_lag_days
        self.wf_n_splits = v.wf_n_splits
        self.wf_min_train_size = v.wf_min_train_size
        self.wf_test_size = v.wf_test_size
        self.wf_max_train_size = v.wf_max_train_size
        self.wf_window_type = v.wf_window_type
        self.wf_embargo_size = v.wf_embargo_size
        self.final_holdout_size = v.final_holdout_size
        self.prune_correlated_features = d.prune_correlated_features
        self.correlation_threshold = d.correlation_threshold
        self.lag_feature_count = d.lag_feature_count
        self.use_prophet_macro_regressors = m.model_settings.get("prophet", {}).get(
            "use_regressors", True
        )
        self.ensemble_enabled = m.ensemble_enabled
        self.universe_file = d.universe_file
        self.clip_shift_warning_threshold_pct = d.clip_shift_warning_threshold_pct
        self.training_window_years = d.training_window_years
        self.window_candidates = d.window_candidates or [3, 5, 7, 10, None]
        self.min_history_days = d.min_history_days
        self.new_listing_min_days = d.new_listing_min_days
        self.research_policy = getattr(e, "research_policy", None)
        self.research_phase = getattr(e, "research_phase", None)
        self.research_metadata = dict(getattr(e, "research_metadata", {}) or {})

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
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        # Faz 4: selected − disabled − unavailable
        self.disabled_models = list(getattr(m, "disabled_models", []) or [])
        self.require_available = bool(getattr(m, "require_available", False))
        self.ensemble_eligibility_overrides = dict(
            getattr(m, "ensemble_eligibility_overrides", {}) or {}
        )
        self.candidate_models = resolve_candidates(
            selected=self.selected_models,
            disabled=self.disabled_models,
            require_available=self.require_available,
        )
        self.benchmark_models = set(BENCHMARK_MODELS)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = self._model_slug_for_run_id(self.selected_models)
        if self.research_policy:
            policy_slug = self._slugify_model_name(self.research_policy)
            model_slug = f"{model_slug}_policy-{policy_slug}"
        self.run_id = f"{timestamp}_{self.stock_symbol}_{self.validation_mode}_{model_slug}"

        self.output_root = os.path.join(self.project_root, "outputs", self.stock_symbol)
        self.outputs_dir = os.path.join(self.output_root, "runs", self.run_id)
        self.latest_dir = os.path.join(self.output_root, "latest")
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
        self.final_holdout_status = {"status": "not_run"}
        self.report_detail_level = e.report_detail_level
        self.auto_generate_forecast_after_training = bool(
            getattr(e, "auto_generate_forecast_after_training", True)
        )

    @staticmethod
    def _slugify_model_name(model_name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "", str(model_name))
        return slug or "Model"

    @classmethod
    def _model_slug_for_run_id(cls, selected_models: Optional[List[str]]) -> str:
        models = [cls._slugify_model_name(model) for model in (selected_models or [])]
        models = [model for model in models if model]
        if not models:
            return "models-All"
        if len(models) == 1:
            return f"model-{models[0]}"
        if len(models) <= 3:
            return "models-" + "-".join(models)

        digest = hashlib.sha1("|".join(models).encode("utf-8")).hexdigest()[:8]
        visible = "-".join(models[:3])
        return f"models-{visible}-plus{len(models) - 3}-{digest}"

    def setup_environment(self) -> None:
        for directory in [
            self.output_root,
            self.outputs_dir,
            self.models_dir,
            self.experiment_dir,
            self.registry_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

        print(f"\n  [INFO] Pipeline Modu: {self.validation_mode}")
        print(f"  [INFO] Hisse Sembolu: {self.stock_symbol}")
        print(f"  [INFO] Target Mode : {self.target_mode}")
        print(f"  [INFO] Feature Mode: {self.feature_mode}")
        print(f"  [INFO] Scaling Mode: {self.scaling_mode}")
        print(
            f"  [INFO] Macro Lag   : rate={self.macro_rate_lag_days}d, CPI={self.macro_cpi_lag_days}d"
        )
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
        print(f"  [INFO] Run ID      : {self.run_id}")
        print(f"  [INFO] Output Dir  : {self.outputs_dir}")
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

    def _window_selection_rows(
        self,
        window_label: str,
        window_years: Optional[int],
        child: Any,
    ) -> List[Dict[str, Any]]:
        evaluator = getattr(child, "evaluation_manager", None)
        if evaluator is None:
            return []
        wf_metrics = getattr(evaluator, "latest_model_metrics", {}).get("wf", {})
        wf_backtests = getattr(evaluator, "latest_backtest_metrics", {}).get("wf", {})
        final_backtests = getattr(evaluator, "latest_backtest_metrics", {}).get("final_holdout", {})
        rows: List[Dict[str, Any]] = []
        for model_name, metrics in wf_metrics.items():
            bt = wf_backtests.get(model_name, {})
            final_bt = final_backtests.get(model_name, {})
            final_net = final_bt.get("Net_Return")
            final_bh = final_bt.get("BuyHold_Return")
            final_gap = (
                float(final_net) - float(final_bh)
                if final_net is not None and final_bh is not None
                else None
            )
            rows.append(
                {
                    "Window_Label": window_label,
                    "Window_Years": window_years,
                    "Model": model_name,
                    "Dir_Acc": metrics.get("Dir_Acc"),
                    "RMSE": metrics.get("RMSE"),
                    "RMSE_vs_benchmark": metrics.get("RMSE_vs_benchmark"),
                    "Sharpe_excess_vs_buy_hold": metrics.get("Sharpe_excess_vs_buy_hold"),
                    "Composite_Score": metrics.get("Composite_Score"),
                    "Trade_Count": bt.get("Trade_Count"),
                    "Exposure": bt.get("Exposure"),
                    "Net_Return": bt.get("Net_Return"),
                    "BuyHold_Return": bt.get("BuyHold_Return"),
                    "Max_Drawdown": bt.get("Max_Drawdown"),
                    "Sharpe": bt.get("Sharpe"),
                    "Final_Holdout_Net_Return": final_net,
                    "Final_Holdout_BuyHold_Return": final_bh,
                    "Final_Holdout_BuyHold_Gap": final_gap,
                    "Final_Holdout_Used_For_Selection": False,
                }
            )
        return rows

    def _write_window_selection_decision(self, comparison_df: Any, save_path: str) -> str:
        return write_window_selection_decision(comparison_df, save_path)

    def _write_validation_and_quality_reports(self) -> None:
        write_validation_and_quality_reports(self)

    def _write_run_manifest(self) -> None:
        write_run_manifest(self)

    def _sync_latest_output(self) -> None:
        root = os.path.abspath(self.output_root)
        run_dir = os.path.abspath(self.outputs_dir)
        latest_dir = os.path.abspath(self.latest_dir)
        tmp_dir = os.path.abspath(os.path.join(self.output_root, f"latest.__tmp__{self.run_id}"))
        lock_path = os.path.abspath(os.path.join(self.output_root, ".latest_sync.lock"))
        root_prefix = root + os.sep
        if not (
            run_dir.startswith(root_prefix)
            and latest_dir.startswith(root_prefix)
            and tmp_dir.startswith(root_prefix)
            and lock_path.startswith(root_prefix)
        ):
            raise RuntimeError("Output latest senkronizasyon hedefi output root disinda.")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        shutil.copytree(run_dir, tmp_dir)
        try:
            lock_fd = self._acquire_latest_sync_lock(lock_path)
        except Exception:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            raise
        try:
            if os.path.exists(latest_dir):
                shutil.rmtree(latest_dir)
            os.replace(tmp_dir, latest_dir)
        finally:
            try:
                os.close(lock_fd)
            finally:
                if os.path.exists(lock_path):
                    os.unlink(lock_path)
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
        print(f"  [OK] Latest output senkronlandi -> {latest_dir}")

    @staticmethod
    def _acquire_latest_sync_lock(
        lock_path: str,
        *,
        timeout_seconds: float = 60.0,
        stale_seconds: float = 900.0,
    ) -> int:
        start = time.monotonic()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
                return fd
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(lock_path) > stale_seconds:
                        os.unlink(lock_path)
                        continue
                except OSError:
                    continue
                if time.monotonic() - start >= timeout_seconds:
                    raise TimeoutError(f"Latest output lock alinamadi: {lock_path}")
                time.sleep(0.25)

    def _auto_generate_forecast_after_training(self) -> None:
        if not self.auto_generate_forecast_after_training:
            return
        try:
            from src.forecasting.bist_calendar import default_calendar_path, ensure_bist_calendar
            from src.forecasting.runner import ForecastRunner

            calendar_path = default_calendar_path(self.project_root)
            ensure_bist_calendar(calendar_path, years_back=5, years_forward=1)
            runner = ForecastRunner(
                project_root=self.project_root,
                db_path=self.stock_db.db_path,
                calendar_path=calendar_path,
                model_config=self._cfg.models,
            )
            result = runner.run_symbol(
                symbol=self.stock_symbol,
                data_file=self.data_file,
                horizon_days=5,
                use_macro=bool(self._cfg.data.use_macro),
                auto_update_data=False,
                auto_update_interactive=False,
            )
            print(
                "  [OK] Training sonrasi forecast uretildi -> "
                f"run_id={result.run_id} model={result.model_name}"
            )
        except Exception as exc:
            print(f"  [WARN] Training sonrasi forecast uretilemedi: {exc}")

    def run_all(self) -> None:
        self.setup_environment()

        self.data_manager.ingest_and_engineer()
        self.data_manager.split_data(self.validation_mode)
        self._write_validation_and_quality_reports()
        self.run_dataset_metadata, self.run_dataset_hash = self.data_manager.build_run_metadata(
            self.validation_mode,
            model_config={
                "selected_models": self.selected_models or "all",
                "candidate_models": sorted(self.candidate_models),
                "benchmark_models": list(BENCHMARK_MODELS),
                "registry_version": self.registry_version,
                "xgboost_hpo": {"scope": "train_only_temporal_cv", "n_trials": 40, "n_splits": 3},
                "random_forest_hpo": {
                    "scope": "train_only_temporal_cv",
                    "n_trials": 40,
                    "n_splits": 3,
                },
                **self.model_config,
            },
        )
        self.run_dataset_metadata.update(
            {
                "run_id": self.run_id,
                "output_run_dir": self.outputs_dir,
                "latest_dir": self.latest_dir,
                "selected_models": list(self.selected_models or []),
                "candidate_models": sorted(self.candidate_models),
                "benchmark_models": list(BENCHMARK_MODELS),
            }
        )
        if self.research_policy or self.research_metadata:
            self.run_dataset_metadata.update(
                {
                    "research_policy": self.research_policy,
                    "research_phase": self.research_phase,
                    "research_metadata": self.research_metadata,
                    "uses_final_holdout_for_selection": False,
                }
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
            "execution_policy": "decision_applies_to_aligned_next_bar_return",
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
            self.evaluation_manager.generate_predictions(
                self.model_trainer.trained_models, self.data_manager.tensors
            )
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
            if (
                best_model_name
                and self.data_manager.final_holdout_df is not None
                and not self.data_manager.final_holdout_df.empty
            ):
                try:
                    self.final_holdout_status = {
                        "status": "running",
                        "model_name": best_model_name,
                    }
                    final_model, final_tensors = self.model_trainer.train_final_holdout_model(
                        best_model_name,
                        self.data_manager,
                    )
                    self.evaluation_manager.evaluate_final_holdout(
                        best_model_name,
                        final_model,
                        final_tensors,
                    )
                    self.final_holdout_status = {
                        "status": "success",
                        "model_name": best_model_name,
                    }
                except Exception as exc:
                    self.final_holdout_status = {
                        "status": "failed",
                        "model_name": best_model_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    print(f"  [WARN] Final holdout degerlendirmesi basarisiz, atlaniyor: {exc}")
            else:
                self.final_holdout_status = {
                    "status": "skipped",
                    "reason": "missing_best_model_or_empty_holdout",
                    "model_name": best_model_name,
                }

        self._auto_generate_forecast_after_training()
        self._write_run_manifest()
        self._sync_latest_output()
        print("\n  [OK] Pipeline completed successfully!")
