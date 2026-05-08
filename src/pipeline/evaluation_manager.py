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

        # --- model attributes ------------------------------------------
        self.selected_models = set(self.model_cfg.selected_models) if self.model_cfg.selected_models else None
        self.ensemble_enabled = self.model_cfg.ensemble_enabled

        # --- execution attributes --------------------------------------
        self.backtest_enabled = self.exe_cfg.backtest_enabled
        self.commission_bps = self.exe_cfg.commission_bps
        self.slippage_bps = self.exe_cfg.slippage_bps
        self.initial_capital = self.exe_cfg.initial_capital
        self.signal_mode = self.exe_cfg.signal_mode
        self.signal_config = self.exe_cfg.signal_config
        self.calibration_scope: str = self.exe_cfg.calibration_scope
        self.signal_calibration_train_ratio = self.exe_cfg.signal_calibration_train_ratio
        self.min_signal_evaluation_folds = self.exe_cfg.min_signal_evaluation_folds
        self.enable_signal_execution_calibration = self.exe_cfg.enable_signal_execution_calibration
        self.enable_gate_diagnostics = self.exe_cfg.enable_gate_diagnostics
        self.enable_shadow_backtests = self.exe_cfg.enable_shadow_backtests
        self.signal_calibration_max_trials = self.exe_cfg.signal_calibration_max_trials
        self.signal_calibration_profile = self.exe_cfg.signal_calibration_profile
        self.signal_calibration_sampler = self.exe_cfg.signal_calibration_sampler
        self.signal_calibration_seed = self.exe_cfg.signal_calibration_seed
        self.signal_calibration_objective = self.exe_cfg.signal_calibration_objective
        self.signal_calibration_min_trades = self.exe_cfg.signal_calibration_min_trades
        self.signal_calibration_require_oos_confirmation = self.exe_cfg.signal_calibration_require_oos_confirmation
        self.signal_calibration_min_eval_excess_return = self.exe_cfg.signal_calibration_min_eval_excess_return
        self.signal_calibration_min_eval_sharpe = self.exe_cfg.signal_calibration_min_eval_sharpe
        self.signal_calibration_reject_behavior = self.exe_cfg.signal_calibration_reject_behavior
        self.auto_signal_diagnostics = self.exe_cfg.auto_signal_diagnostics
        self.report_detail_level = self.exe_cfg.report_detail_level
        self.write_text_reports = self.exe_cfg.write_text_reports
        self.write_markdown_reports = self.exe_cfg.write_markdown_reports
        self.write_xai_tables = self.exe_cfg.write_xai_tables
        self.write_trade_logs = self.exe_cfg.write_trade_logs

        self.stock_db = stock_db

        # --- signal calibration state ----------------------------------
        self.default_signal_config = self.signal_config
        self.signal_threshold_source = "default_config"
        self.signal_threshold_calibration_summary: Dict[str, Any] = {}
        self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()

        # --- mutable prediction state ----------------------------------
        self.predictions: Dict[str, np.ndarray] = {}
        self.prediction_targets: Dict[str, np.ndarray] = {}
        self.quantile_predictions: Dict[str, np.ndarray] = {}
        self.multihorizon_predictions: Dict[str, Dict[str, np.ndarray]] = {}
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
        self.context = EvaluationContext(
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
        self.state = EvaluationState(
            predictions=self.predictions,
            prediction_targets=self.prediction_targets,
            quantile_predictions=self.quantile_predictions,
            multihorizon_predictions=self.multihorizon_predictions,
            single_backtest_inputs=self.single_backtest_inputs,
            latest_tensors=self.latest_tensors,
            latest_backtest_results=self.latest_backtest_results,
            latest_backtest_metrics=self.latest_backtest_metrics,
            latest_model_metrics=self.latest_model_metrics,
            ensemble_weights=self.ensemble_weights,
        )
        self._init_services()

    def _init_services(self) -> None:
        self.prediction_service = PredictionService(self)
        self.backtest_service = BacktestService(self)
        self.signal_calibration_service = SignalCalibrationService(self)
        self.metrics_reporting_service = MetricsReportingService(self)

    def _ensure_services(self) -> None:
        if not all(
            hasattr(self, attr)
            for attr in (
                "prediction_service",
                "backtest_service",
                "signal_calibration_service",
                "metrics_reporting_service",
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

    def _save_multihorizon_report(self, suffix: str = "latest") -> None:
        self._ensure_services()
        return self.metrics_reporting_service._save_multihorizon_report(suffix)

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
        print("\n" + "=" * 60)
        print("  ADIM 7 | Degerlendirme ve Registry (EvaluationManager)")
        print("=" * 60)

        metrics = {
            name: compute_metrics(
                self.y_true_aligned,
                preds,
                y_true_target=self.y_true_target_aligned,
                y_pred_target=self.prediction_targets.get(name),
                prev_close=self.prev_close_aligned,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            for name, preds in self.predictions.items()
        }
        for name, q_preds in self.quantile_predictions.items():
            if name in metrics:
                q_metrics = compute_quantile_metrics(self.y_true_aligned, q_preds)
                metrics[name].update({key: round(value, 6) for key, value in q_metrics.items()})
        metrics = self._attach_composite_scores(metrics)
        metrics = self._attach_model_scope_metadata(metrics)
        metrics = self._filter_reportable_models(metrics, metrics)
        self.predictions = self._filter_reportable_models(self.predictions, metrics)
        self.prediction_targets = self._filter_reportable_models(self.prediction_targets, metrics)
        self.quantile_predictions = self._filter_reportable_models(self.quantile_predictions, metrics)
        self.single_backtest_inputs = self._filter_reportable_models(self.single_backtest_inputs, metrics)
        metrics = self._attach_leakage_guard_metadata(metrics)
        metrics = self._attach_model_family_metadata(metrics)
        self.latest_model_metrics["latest"] = metrics

        for name, model_metrics in metrics.items():
            self.tracker.log_run(
                name,
                {"validation": "single"},
                model_metrics,
                self.feature_names,
                self.dataset_hash,
                self.dataset_metadata,
            )

            model_ext = ".pt" if name == "TFT" else ".keras" if name == "LSTM" else ".pkl"
            model_filename = f"{name.replace(' ', '_').lower()}_model{model_ext}"
            model_path = os.path.join(self.models_dir, model_filename)

            original_model = trained_models.get(name)
            if original_model is None:
                if not name.startswith("Ensemble "):
                    print(f"  [WARN] {name} icin kayitli model bulunamadi, dosya kaydi atlaniyor.")
                model_path = ""
            else:
                original_model.save(model_path)

            if self.stock_db is not None:
                self.stock_db.log_experiment(
                    stock_symbol=self.stock_symbol,
                    model_name=name,
                    metrics=model_metrics,
                    model_path=model_path,
                    features=self.feature_names,
                    dataset_hash=self.dataset_hash,
                    validation_mode="single_split",
                    dataset_metadata=self.dataset_metadata,
                )

        backtest_results = self._run_backtests(
            self.single_backtest_inputs,
            suffix="latest",
            model_metrics_by_model=metrics,
        )
        self._save_multihorizon_report(suffix="latest")
        xai_payload = self._get_xai_single_split(trained_models, tensors=self.latest_tensors)

        tft_quantiles_df = None
        if "TFT" in self.quantile_predictions:
            tft_quantiles = self.quantile_predictions["TFT"]
            quantile_labels = [f"Q{idx + 1}" for idx in range(tft_quantiles.shape[1])]
            if tft_quantiles.shape[1] == 3:
                quantile_labels = ["P10", "P50", "P90"]
            tft_quantiles_df = pd.DataFrame(tft_quantiles, columns=quantile_labels)
            tft_quantiles_df.insert(0, "Actual", self.y_true_aligned[-len(tft_quantiles_df):])

        # ── Tahmin karşılaştırma grafikleri ─────────────────────────
        try:
            _plot_path = os.path.join(self.outputs_dir, 'predictions_latest.png')
            plot_comparison(
                self.y_true_aligned,
                self.predictions,
                save_path=_plot_path,
                title=f'{self.stock_symbol} — Gerçek vs Tahmin (latest)',
            )
        except Exception as _pe:
            print(f'  [WARN] Tahmin grafiği kaydedilemedi: {_pe}')
        try:
            if 'TFT' in self.quantile_predictions and self.quantile_predictions['TFT'].shape[1] >= 3:
                _q = self.quantile_predictions['TFT']
                plot_prediction_interval(
                    self.y_true_aligned,
                    median_pred=_q[:, 1],
                    lower_pred=_q[:, 0],
                    upper_pred=_q[:, 2],
                    save_path=os.path.join(self.outputs_dir, 'predictions_tft_interval_latest.png'),
                    title=f'{self.stock_symbol} TFT P10-P50-P90 (latest)',
                )
        except Exception as _pe:
            print(f'  [WARN] TFT interval grafiği kaydedilemedi: {_pe}')

        return {
            "metrics": metrics,
            "y_true": self.y_true_aligned,
            "predictions": self.predictions,
            "backtest": backtest_results,
            "xai_payload": xai_payload,
            "tft_quantiles_df": tft_quantiles_df,
            "quantile_predictions": self.quantile_predictions,
        }

    def evaluate_walk_forward(
        self,
        wf_results: dict,
        wf_predictions: dict,
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        wf_fold_metrics: Optional[Dict[str, list[Dict[str, Any]]]] = None,
    ) -> WalkForwardResult:
        print("\n" + "=" * 60)
        print("  ADIM 7 | Degerlendirme Gosterimi (Walk-Forward)")
        print("=" * 60)

        self._add_walk_forward_ensembles(wf_results, wf_predictions, wf_y_true, wf_backtest_inputs)
        wf_results = self._attach_composite_scores(wf_results)
        wf_results = self._attach_model_scope_metadata(wf_results)
        wf_results = self._filter_reportable_models(wf_results, wf_results)
        wf_predictions = self._filter_reportable_models(wf_predictions, wf_results)
        wf_backtest_inputs = self._filter_reportable_models(wf_backtest_inputs or {}, wf_results)
        wf_fold_metrics = self._filter_reportable_models(wf_fold_metrics or {}, wf_results)
        enriched_fold_metrics = self._enrich_wf_fold_metrics(wf_fold_metrics)
        (
            signal_calibration_fold_metrics,
            signal_calibration_backtest_inputs,
            signal_evaluation_backtest_inputs,
            signal_split_metadata,
        ) = self._split_walk_forward_signal_sets(enriched_fold_metrics, wf_backtest_inputs or {})
        self.dataset_metadata["signal_calibration_split"] = signal_split_metadata

        wf_fold_reports = self._get_wf_fold_metric_report(enriched_fold_metrics)
        if signal_calibration_fold_metrics:
            self._calibrate_signal_quality_thresholds(signal_calibration_fold_metrics)
        else:
            self.signal_threshold_calibration_summary.update({
                "status": "skipped_insufficient_signal_calibration_folds",
                "signal_calibration_split": signal_split_metadata,
                "final_holdout_used": False,
            })
            self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()

        if getattr(self, "enable_signal_execution_calibration", True) and signal_calibration_backtest_inputs:
            calibration_results = self._calibrate_walk_forward_signal_parameters(
                wf_backtest_inputs=signal_calibration_backtest_inputs,
                wf_evaluation_backtest_inputs=signal_evaluation_backtest_inputs,
                model_metrics_by_model=wf_results,
                suffix="wf_calibration",
            )
        else:
            self.signal_threshold_calibration_summary.update({
                "execution_calibration_status": "skipped_insufficient_signal_calibration_folds",
                "signal_calibration_split": signal_split_metadata,
                "final_holdout_used": False,
            })
            self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()
            calibration_results = {}
        wf_results = self._attach_leakage_guard_metadata(wf_results)
        wf_results = self._attach_model_family_metadata(wf_results)
        self.latest_model_metrics["wf"] = wf_results
        best_model_name = self._select_best_model(wf_results)
        if best_model_name:
            print(f"\n  [INFO] Walk-forward secim modeli: {best_model_name}")

        df_wf = pd.DataFrame(wf_results).T
        if "Composite_Score" in df_wf.columns:
            df_wf = df_wf.sort_values(by=["Composite_Score", "RMSE"], ascending=[False, True])
        print("\nWalk-Forward Ortalama Metrikleri:")
        print(df_wf)

        if self.stock_db is not None:
            for model_name, avg_metrics in wf_results.items():
                self.stock_db.log_experiment(
                    stock_symbol=self.stock_symbol,
                    model_name=model_name,
                    metrics=avg_metrics,
                    model_path="",
                    features=self.feature_names,
                    dataset_hash=self.dataset_hash,
                    validation_mode="walk_forward",
                    dataset_metadata=self.dataset_metadata,
                )

        backtest_results = self._run_backtests(
            signal_evaluation_backtest_inputs or {},
            suffix="wf",
            model_metrics_by_model=wf_results,
        )
        xai_payload = self._get_xai_walk_forward(wf_predictions, wf_y_true, wf_backtest_inputs or {})

        # ── Walk-forward tahmin karşılaştırma grafiği ───────────────
        try:
            _wf_true = np.asarray(wf_y_true).ravel() if wf_y_true is not None else np.array([])
            _wf_preds = {
                name: np.asarray(preds).ravel()
                for name, preds in wf_predictions.items()
                if np.asarray(preds).ndim <= 2
            }
            if _wf_true.size and _wf_preds:
                _k = min(len(_wf_true), min(len(v) for v in _wf_preds.values()))
                plot_comparison(
                    _wf_true[-_k:],
                    {n: v[-_k:] for n, v in _wf_preds.items()},
                    save_path=os.path.join(self.outputs_dir, 'predictions_wf.png'),
                    title=f'{self.stock_symbol} — Gerçek vs Tahmin (walk-forward)',
                )
        except Exception as _pe:
            print(f'  [WARN] WF tahmin grafiği kaydedilemedi: {_pe}')

        return {
            "metrics": wf_results,
            "best_model_name": best_model_name,
            "y_true": wf_y_true,
            "predictions": wf_predictions,
            "backtest": backtest_results,
            "xai_payload": xai_payload,
            "wf_fold_reports": wf_fold_reports,
            "calibration_results": calibration_results,
        }

    def evaluate_final_holdout(
        self, model_name: str, model: Any, tensors: dict
    ) -> FinalHoldoutResult:
        print("\n" + "=" * 60)
        print("  ADIM 8 | Final Untouched Holdout Degerlendirmesi")
        print("=" * 60)

        (
            pred_price,
            pred_target,
            y_true_price,
            y_true_target,
            prev_close,
            dates,
            prediction_dates,
            market_regime,
            quantile_price,
        ) = self._predict_single_model(model_name, model, tensors)

        metrics = {
            model_name: compute_metrics(
                y_true_price,
                pred_price,
                y_true_target=y_true_target,
                y_pred_target=pred_target,
                prev_close=prev_close,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
        }
        if quantile_price is not None:
            q_metrics = compute_quantile_metrics(y_true_price, quantile_price)
            metrics[model_name].update({key: round(value, 6) for key, value in q_metrics.items()})
        metrics = self._attach_composite_scores(metrics)
        metrics = self._attach_model_scope_metadata(metrics)
        metrics = self._attach_leakage_guard_metadata(metrics)
        metrics = self._attach_model_family_metadata(metrics)
        metrics[model_name]["Selection_Source"] = "walk_forward_composite_score"
        metrics[model_name]["Evaluation_Set_Name"] = "untouched_final_holdout"
        self.latest_model_metrics["final_holdout"] = metrics

        final_metadata = dict(self.dataset_metadata)
        final_metadata["validation_mode"] = "final_holdout"
        final_metadata["protocol_stage"] = "final_holdout_evaluation"
        final_metadata["selected_by"] = "walk_forward_composite_score"

        self.tracker.log_run(
            model_name,
            {"validation": "final_holdout", "selected_by": "walk_forward"},
            metrics[model_name],
            self.feature_names,
            self.dataset_hash,
            final_metadata,
        )

        model_ext = ".pt" if model_name == "TFT" else ".keras" if model_name == "LSTM" else ".pkl"
        model_filename = f"{model_name.replace(' ', '_').lower()}_final_holdout_model{model_ext}"
        model_path = os.path.join(self.models_dir, model_filename)
        model.save(model_path)

        if self.stock_db is not None:
            self.stock_db.log_experiment(
                stock_symbol=self.stock_symbol,
                model_name=model_name,
                metrics=metrics[model_name],
                model_path=model_path,
                features=self.feature_names,
                dataset_hash=self.dataset_hash,
                validation_mode="final_holdout",
                dataset_metadata=final_metadata,
                is_production_candidate=bool(metrics[model_name].get("Candidate_For_Selection", False)),
                selection_source="walk_forward_composite_score",
                run_id=self.dataset_metadata.get("run_id"),
            )

        quantiles_df = None
        if quantile_price is not None:
            quantile_labels = [f"Q{idx + 1}" for idx in range(quantile_price.shape[1])]
            if quantile_price.shape[1] == 3:
                quantile_labels = ["P10", "P50", "P90"]
            quantiles_df = pd.DataFrame(quantile_price, columns=quantile_labels)
            quantiles_df.insert(0, "Actual", y_true_price[-len(quantiles_df):])

        backtest_results = self._run_backtests(
            {
                model_name: {
                    "dates": dates,
                    "prediction_dates": prediction_dates,
                    "market_regime": market_regime,
                    "y_true_price": y_true_price,
                    "pred_price": pred_price,
                    "prev_close": prev_close,
                    "pred_target": pred_target,
                    "y_true_target": y_true_target,
                }
            },
            suffix="final_holdout",
            model_metrics_by_model=metrics,
        )

        # ── Final holdout grafikleri ─────────────────────────────────
        try:
            plot_comparison(
                y_true_price,
                {model_name: pred_price},
                save_path=os.path.join(self.outputs_dir, f'predictions_final_holdout_{model_name}.png'),
                title=f'{self.stock_symbol} {model_name} — Final Holdout',
            )
        except Exception as _pe:
            print(f'  [WARN] Final holdout grafiği kaydedilemedi: {_pe}')
        try:
            if quantile_price is not None and quantile_price.shape[1] >= 3:
                plot_prediction_interval(
                    y_true_price,
                    median_pred=quantile_price[:, 1],
                    lower_pred=quantile_price[:, 0],
                    upper_pred=quantile_price[:, 2],
                    save_path=os.path.join(self.outputs_dir, f'predictions_tft_interval_final_holdout.png'),
                    title=f'{self.stock_symbol} TFT P10-P50-P90 (final holdout)',
                )
        except Exception as _pe:
            print(f'  [WARN] TFT interval grafiği (final holdout) kaydedilemedi: {_pe}')

        return {
            "metrics": metrics,
            "y_true": y_true_price,
            "predictions": {model_name: pred_price},
            "quantiles_df": quantiles_df,
            "quantile_price": quantile_price,
            "backtest": backtest_results,
            "model_name": model_name,
        }
