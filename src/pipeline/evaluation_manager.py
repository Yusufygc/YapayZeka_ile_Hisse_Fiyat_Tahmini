# -*- coding: utf-8 -*-
"""
evaluation_manager.py - Evaluation orkestrasyon katmani.

Bu sinif artik yalnizca:
  - __init__(): durum ve konfigurasyon baslangici
  - evaluate_single_split(): tek bolunme degerlendirmesi
  - evaluate_walk_forward(): walk-forward degerlendirmesi
  - evaluate_final_holdout(): son holdout degerlendirmesi

Gercek is mantigi servis kompozisyonu tarafindan saglanir:
  - PredictionService
  - BacktestService
  - SignalCalibrationService
  - MetricsReportingService
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtesting.signals import SignalConfig
from src.database.stock_model_db import StockModelDB
from src.evaluation.financial_metrics import compute_quantile_metrics
from src.evaluation.evaluator import compute_metrics, plot_comparison, plot_prediction_interval
from src.experiments.experiment_tracker import ExperimentTracker
from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.evaluation_workflows import (
    FinalHoldoutEvaluationWorkflow,
    SingleSplitEvaluationWorkflow,
    WalkForwardEvaluationWorkflow,
)
from src.pipeline.evaluation_services import (
    BacktestService,
    EvaluationContext,
    EvaluationState,
    MetricsReportingService,
    PredictionService,
    SignalCalibrationService,
)
from src.pipeline.results import FinalHoldoutResult, SingleSplitResult, WalkForwardResult


class EvaluationManager:
    """
    Degerlendirme, kayit ve raporlama icin ince orkestrasyon sinifi.

    Is mantigi servisler uzerinden compose edilir; bu sinif durum yonetimi,
    public API ve geriye uyumlu helper delegasyonlarini icerir.
    """

    def __init__(
        self,
        stock_symbol: str,
        outputs_dir: str,
        models_dir: str,
        tracker: ExperimentTracker,
        feature_names: list,
        dataset_hash: str,
        dataset_metadata: Dict[str, Any],
        exe_cfg: ExecutionConfig,
        model_cfg: ModelConfig,
        stock_db: Optional[StockModelDB] = None,
    ):
        self.stock_symbol = stock_symbol
        self.outputs_dir = outputs_dir
        self.models_dir = models_dir
        self.tracker = tracker
        self.feature_names = feature_names
        self.dataset_hash = dataset_hash
        self.dataset_metadata = dataset_metadata

        self.exe_cfg = exe_cfg
        self.model_cfg = model_cfg
        self.stock_db = stock_db

        self._init_model_attrs()
        self._init_execution_attrs()
        # signal threshold metadata triggers an early service init (uses
        # signal_calibration_service); call order preserved intentionally.
        self._init_signal_calibration_state()
        self._init_mutable_state()
        self._init_context_and_state(
            stock_symbol=stock_symbol,
            outputs_dir=outputs_dir,
            models_dir=models_dir,
            tracker=tracker,
            feature_names=feature_names,
            dataset_hash=dataset_hash,
            dataset_metadata=dataset_metadata,
            exe_cfg=exe_cfg,
            model_cfg=model_cfg,
            stock_db=stock_db,
        )
        self._init_services()

    def _init_model_attrs(self) -> None:
        self.selected_models = set(self.model_cfg.selected_models) if self.model_cfg.selected_models else None
        self.ensemble_enabled = self.model_cfg.ensemble_enabled

    def _init_execution_attrs(self) -> None:
        e = self.exe_cfg
        self.backtest_enabled = e.backtest_enabled
        self.commission_bps = e.commission_bps
        self.slippage_bps = e.slippage_bps
        self.initial_capital = e.initial_capital
        self.signal_mode = e.signal_mode
        self.signal_config = e.signal_config
        self.calibration_scope: str = e.calibration_scope
        self.signal_calibration_train_ratio = e.signal_calibration_train_ratio
        self.min_signal_evaluation_folds = e.min_signal_evaluation_folds
        self.enable_signal_execution_calibration = e.enable_signal_execution_calibration
        self.enable_gate_diagnostics = e.enable_gate_diagnostics
        self.enable_shadow_backtests = e.enable_shadow_backtests
        self.signal_calibration_max_trials = e.signal_calibration_max_trials
        self.signal_calibration_profile = e.signal_calibration_profile
        self.signal_calibration_sampler = e.signal_calibration_sampler
        self.signal_calibration_seed = e.signal_calibration_seed
        self.signal_calibration_objective = e.signal_calibration_objective
        self.signal_calibration_min_trades = e.signal_calibration_min_trades
        self.signal_calibration_require_oos_confirmation = e.signal_calibration_require_oos_confirmation
        self.signal_calibration_min_eval_excess_return = e.signal_calibration_min_eval_excess_return
        self.signal_calibration_min_eval_sharpe = e.signal_calibration_min_eval_sharpe
        self.signal_calibration_reject_behavior = e.signal_calibration_reject_behavior
        self.auto_signal_diagnostics = e.auto_signal_diagnostics
        self.report_detail_level = e.report_detail_level
        self.write_text_reports = e.write_text_reports
        self.write_markdown_reports = e.write_markdown_reports
        self.write_xai_tables = e.write_xai_tables
        self.write_trade_logs = e.write_trade_logs

    def _init_signal_calibration_state(self) -> None:
        self.default_signal_config = self.signal_config
        self.signal_threshold_source = "default_config"
        self.signal_threshold_calibration_summary: Dict[str, Any] = {}
        self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()

    def _init_mutable_state(self) -> None:
        self.predictions: Dict[str, np.ndarray] = {}
        self.prediction_targets: Dict[str, np.ndarray] = {}
        self.quantile_predictions: Dict[str, np.ndarray] = {}
        self.single_backtest_inputs: Dict[str, Dict[str, Any]] = {}
        self.y_true_aligned: Optional[np.ndarray] = None
        self.y_true_target_aligned: Optional[np.ndarray] = None
        self.prev_close_aligned: Optional[np.ndarray] = None
        self.xai_dir = os.path.join(self.outputs_dir, "xai")
        self.latest_tensors: Dict[str, Any] = {}
        self.latest_backtest_results: Dict[str, Any] = {}
        self.latest_backtest_metrics: Dict[str, Any] = {}
        self.latest_model_metrics: Dict[str, Any] = {}
        self.ensemble_weights: Dict[str, Dict[str, float]] = {}

    def _init_context_and_state(self, **kwargs: Any) -> None:
        self.context = EvaluationContext(**kwargs)
        self.state = EvaluationState(
            predictions=self.predictions,
            prediction_targets=self.prediction_targets,
            quantile_predictions=self.quantile_predictions,
            single_backtest_inputs=self.single_backtest_inputs,
            latest_tensors=self.latest_tensors,
            latest_backtest_results=self.latest_backtest_results,
            latest_backtest_metrics=self.latest_backtest_metrics,
            latest_model_metrics=self.latest_model_metrics,
            ensemble_weights=self.ensemble_weights,
        )

    def _init_services(self) -> None:
        self.prediction_service = PredictionService(self)
        self.backtest_service = BacktestService(self)
        self.signal_calibration_service = SignalCalibrationService(self)
        self.metrics_reporting_service = MetricsReportingService(self)
        self.single_split_workflow = SingleSplitEvaluationWorkflow(self)
        self.walk_forward_workflow = WalkForwardEvaluationWorkflow(self)
        self.final_holdout_workflow = FinalHoldoutEvaluationWorkflow(self)

    def _ensure_services(self) -> None:
        if not all(
            hasattr(self, attr)
            for attr in (
                "prediction_service",
                "backtest_service",
                "signal_calibration_service",
                "metrics_reporting_service",
                "single_split_workflow",
                "walk_forward_workflow",
                "final_holdout_workflow",
            )
        ):
            self._init_services()

    # ------------------------------------------------------------------ #
    #  Backward-compatible service delegation                             #
    # ------------------------------------------------------------------ #

    def _target_to_price(self, preds_target: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        self._ensure_services()
        return self.prediction_service._target_to_price(preds_target, prev_close)

    @staticmethod
    def _weighted_average(predictions: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        return PredictionService._weighted_average(predictions, weights)

    @staticmethod
    def _base_predictions_for_ensemble(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return PredictionService._base_predictions_for_ensemble(predictions)

    def _add_single_split_ensembles(self) -> None:
        self._ensure_services()
        return self.prediction_service._add_single_split_ensembles()

    def _add_walk_forward_ensembles(
        self,
        wf_results: Dict[str, Dict[str, Any]],
        wf_predictions: Dict[str, np.ndarray],
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._ensure_services()
        return self.prediction_service._add_walk_forward_ensembles(
            wf_results,
            wf_predictions,
            wf_y_true,
            wf_backtest_inputs,
        )

    def _save_selected_models_plot(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        save_path: str,
        title: str,
    ) -> None:
        self._ensure_services()
        return self.prediction_service._save_selected_models_plot(y_true, predictions, save_path, title)

    def _predict_single_model(self, model_name: str, model: Any, tensors: dict):
        self._ensure_services()
        return self.prediction_service._predict_single_model(model_name, model, tensors)

    def generate_predictions(self, trained_models: dict, tensors: dict):
        self._ensure_services()
        return self.prediction_service.generate_predictions(trained_models, tensors)

    @staticmethod
    def _diagnostic_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
        return BacktestService._diagnostic_numeric(frame, column)

    @staticmethod
    def _diagnostic_float(value: Any) -> float:
        return BacktestService._diagnostic_float(value)

    @staticmethod
    def _count_decision(frame: pd.DataFrame, decision: str) -> int:
        return BacktestService._count_decision(frame, decision)

    @staticmethod
    def _payload_expected_return(payload: Dict[str, Any], target_mode: str) -> np.ndarray:
        return BacktestService._payload_expected_return(payload, target_mode)

    def _get_shadow_backtests(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str,
        target_mode: str,
    ) -> Dict[str, Any]:
        self._ensure_services()
        return self.backtest_service._get_shadow_backtests(
            backtest_inputs=backtest_inputs,
            model_metrics_by_model=model_metrics_by_model,
            suffix=suffix,
            target_mode=target_mode,
        )

    def _run_shadow_backtests(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str,
        target_mode: str,
    ) -> Dict[str, Any]:
        self._ensure_services()
        return self.backtest_service._run_shadow_backtests(
            backtest_inputs=backtest_inputs,
            model_metrics_by_model=model_metrics_by_model,
            suffix=suffix,
            target_mode=target_mode,
        )

    def _get_signal_gate_diagnostics(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        backtest_results: Dict[str, Dict[str, Any]],
        backtest_metrics: Dict[str, Dict[str, Any]],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str,
        target_mode: str,
    ) -> pd.DataFrame:
        self._ensure_services()
        return self.backtest_service._get_signal_gate_diagnostics(
            backtest_inputs=backtest_inputs,
            backtest_results=backtest_results,
            backtest_metrics=backtest_metrics,
            model_metrics_by_model=model_metrics_by_model,
            suffix=suffix,
            target_mode=target_mode,
        )

    def _write_signal_gate_diagnostics(self, diagnostics: pd.DataFrame, suffix: str) -> None:
        self._ensure_services()
        return self.backtest_service._write_signal_gate_diagnostics(diagnostics, suffix)

    def _write_shadow_backtest_reports(self, shadow_results: Dict[str, Any], suffix: str) -> None:
        self._ensure_services()
        return self.backtest_service._write_shadow_backtest_reports(shadow_results, suffix)

    def _run_backtests(
        self,
        backtest_inputs: Dict[str, Dict[str, Any]],
        suffix: str,
        model_metrics_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self._ensure_services()
        return self.backtest_service._run_backtests(backtest_inputs, suffix, model_metrics_by_model)

    def _assert_wf_train_scope(self) -> None:
        self._ensure_services()
        return self.signal_calibration_service._assert_wf_train_scope()

    def _signal_threshold_metadata(self) -> Dict[str, Any]:
        self._ensure_services()
        return self.signal_calibration_service._signal_threshold_metadata()

    def _calibrate_signal_quality_thresholds(
        self, wf_fold_metrics: Dict[str, list[Dict[str, Any]]]
    ) -> None:
        self._ensure_services()
        return self.signal_calibration_service._calibrate_signal_quality_thresholds(wf_fold_metrics)

    def _calibrate_walk_forward_signal_parameters(
        self,
        *,
        wf_backtest_inputs: Dict[str, Dict[str, Any]],
        wf_evaluation_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str = "",
    ) -> Dict[str, Any]:
        self._ensure_services()
        object.__setattr__(
            self.signal_calibration_service,
            "_signal_calibration_grid",
            self._signal_calibration_grid,
        )
        return self.signal_calibration_service._calibrate_walk_forward_signal_parameters(
            wf_backtest_inputs=wf_backtest_inputs,
            wf_evaluation_backtest_inputs=wf_evaluation_backtest_inputs,
            model_metrics_by_model=model_metrics_by_model,
            suffix=suffix,
        )

    @staticmethod
    def _signal_calibration_grid(base_cfg: SignalConfig) -> list[Dict[str, float | int]]:
        return SignalCalibrationService._signal_calibration_grid(base_cfg)

    def _apply_signal_calibration_trial_policy(
        self,
        grid: list[Dict[str, float | int]],
    ) -> tuple[list[Dict[str, float | int]], Dict[str, Any]]:
        self._ensure_services()
        return self.signal_calibration_service._apply_signal_calibration_trial_policy(grid)

    @staticmethod
    def _summarize_signal_calibration_trial(
        *,
        trial_idx: int,
        params: Dict[str, float | int],
        summaries: list[Dict[str, Any]],
        min_trade_count: int,
    ) -> Dict[str, Any]:
        return SignalCalibrationService._summarize_signal_calibration_trial(
            trial_idx=trial_idx,
            params=params,
            summaries=summaries,
            min_trade_count=min_trade_count,
        )

    @staticmethod
    def _signal_calibration_sort_key(row: Dict[str, Any]) -> tuple:
        return SignalCalibrationService._signal_calibration_sort_key(row)

    @classmethod
    def _select_signal_calibration_row(cls, rows: list[Dict[str, Any]]) -> Dict[str, Any] | None:
        return SignalCalibrationService._select_signal_calibration_row(rows)

    def _write_signal_calibration_reports(
        self,
        calibration_df: pd.DataFrame,
        decision_md: str,
        *,
        suffix: str = "",
    ) -> None:
        self._ensure_services()
        return self.signal_calibration_service._write_signal_calibration_reports(
            calibration_df,
            decision_md,
            suffix=suffix,
        )

    def _get_signal_calibration_decision_md(self, best_row: Dict[str, Any] | None) -> str:
        self._ensure_services()
        return self.signal_calibration_service._get_signal_calibration_decision_md(best_row)

    @staticmethod
    def _attach_composite_scores(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        return MetricsReportingService._attach_composite_scores(metrics_dict)

    def _attach_leakage_guard_metadata(
        self, metrics_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        self._ensure_services()
        return self.metrics_reporting_service._attach_leakage_guard_metadata(metrics_dict)

    def _attach_model_family_metadata(
        self, metrics_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        self._ensure_services()
        return self.metrics_reporting_service._attach_model_family_metadata(metrics_dict)

    def _attach_model_scope_metadata(
        self, metrics_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        self._ensure_services()
        return self.metrics_reporting_service._attach_model_scope_metadata(metrics_dict)

    def _filter_reportable_models(
        self,
        data: Dict[str, Any],
        metrics_dict: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        self._ensure_services()
        return self.metrics_reporting_service._filter_reportable_models(data, metrics_dict)

    def _enrich_wf_fold_metrics(
        self, wf_fold_metrics: Dict[str, list[Dict[str, Any]]]
    ) -> Dict[str, list[Dict[str, Any]]]:
        self._ensure_services()
        return self.metrics_reporting_service._enrich_wf_fold_metrics(wf_fold_metrics)

    def _get_wf_fold_metric_report(
        self, wf_fold_metrics: Dict[str, list[Dict[str, Any]]]
    ) -> Dict[str, pd.DataFrame]:
        self._ensure_services()
        return self.metrics_reporting_service._get_wf_fold_metric_report(wf_fold_metrics)

    @staticmethod
    def _select_best_model(metrics_dict: Dict[str, Dict[str, Any]]) -> Optional[str]:
        return MetricsReportingService._select_best_model(metrics_dict)

    def _get_xai_single_split(self, trained_models: dict, tensors: dict) -> Optional[Dict[str, Any]]:
        self._ensure_services()
        return self.metrics_reporting_service._get_xai_single_split(trained_models, tensors)

    def _get_xai_walk_forward(
        self,
        wf_predictions: dict,
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        self._ensure_services()
        return self.metrics_reporting_service._get_xai_walk_forward(
            wf_predictions,
            wf_y_true,
            wf_backtest_inputs,
        )

    def _write_xai_reports(self, payload, suffix: str) -> None:
        self._ensure_services()
        return self.metrics_reporting_service._write_xai_reports(payload, suffix)

    def _split_walk_forward_signal_sets(
        self,
        wf_fold_metrics: Dict[str, list[Dict[str, Any]]],
        wf_backtest_inputs: Dict[str, Dict[str, Any]],
    ) -> tuple[Dict[str, list[Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        fold_values = set()
        for model_rows in wf_fold_metrics.values():
            for row in model_rows:
                if row.get("Fold") is not None:
                    fold_values.add(row.get("Fold"))
        for payload in wf_backtest_inputs.values():
            fold_ids = payload.get("fold_ids")
            if fold_ids is not None:
                fold_values.update(np.asarray(fold_ids).ravel().tolist())

        folds = sorted(fold_values)
        min_eval = int(getattr(self, "min_signal_evaluation_folds", 3))
        train_ratio = float(getattr(self, "signal_calibration_train_ratio", 0.70))
        if len(folds) <= min_eval:
            metadata = {
                "status": "skipped_insufficient_folds",
                "fold_count": int(len(folds)),
                "min_signal_evaluation_folds": min_eval,
                "calibration_folds": [],
                "evaluation_folds": folds,
            }
            return {}, {}, wf_backtest_inputs, metadata

        split_idx = int(np.floor(len(folds) * train_ratio))
        split_idx = max(1, min(split_idx, len(folds) - min_eval))
        calibration_folds = set(folds[:split_idx])
        evaluation_folds = set(folds[split_idx:])

        calibration_metrics = {
            model_name: [row for row in rows if row.get("Fold") in calibration_folds]
            for model_name, rows in wf_fold_metrics.items()
        }
        calibration_inputs = self._filter_backtest_inputs_by_folds(wf_backtest_inputs, calibration_folds)
        evaluation_inputs = self._filter_backtest_inputs_by_folds(wf_backtest_inputs, evaluation_folds)
        metadata = {
            "status": "applied",
            "fold_count": int(len(folds)),
            "calibration_train_ratio": train_ratio,
            "min_signal_evaluation_folds": min_eval,
            "calibration_folds": list(folds[:split_idx]),
            "evaluation_folds": list(folds[split_idx:]),
        }
        return calibration_metrics, calibration_inputs, evaluation_inputs, metadata

    @staticmethod
    def _filter_backtest_inputs_by_folds(
        backtest_inputs: Dict[str, Dict[str, Any]],
        selected_folds: set,
    ) -> Dict[str, Dict[str, Any]]:
        filtered: Dict[str, Dict[str, Any]] = {}
        for model_name, payload in backtest_inputs.items():
            fold_ids = payload.get("fold_ids")
            if fold_ids is None:
                filtered[model_name] = payload
                continue
            fold_arr = np.asarray(fold_ids)
            mask = np.isin(fold_arr, list(selected_folds))
            if not np.any(mask):
                continue
            new_payload: Dict[str, Any] = {}
            for key, value in payload.items():
                arr = np.asarray(value)
                if arr.ndim > 0 and len(arr) == len(mask):
                    new_payload[key] = arr[mask]
                else:
                    new_payload[key] = value
            filtered[model_name] = new_payload
        return filtered

    # ------------------------------------------------------------------ #
    #  Public evaluation methods                                          #
    # ------------------------------------------------------------------ #

    def evaluate_single_split(self, trained_models: dict) -> SingleSplitResult:
        self._ensure_services()
        return self.single_split_workflow.run(trained_models)

    def evaluate_walk_forward(
        self,
        wf_results: dict,
        wf_predictions: dict,
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        wf_fold_metrics: Optional[Dict[str, list[Dict[str, Any]]]] = None,
    ) -> WalkForwardResult:
        self._ensure_services()
        return self.walk_forward_workflow.run(
            wf_results,
            wf_predictions,
            wf_y_true,
            wf_backtest_inputs=wf_backtest_inputs,
            wf_fold_metrics=wf_fold_metrics,
        )

    def evaluate_final_holdout(
        self, model_name: str, model: Any, tensors: dict
    ) -> FinalHoldoutResult:
        self._ensure_services()
        return self.final_holdout_workflow.run(model_name, model, tensors)
