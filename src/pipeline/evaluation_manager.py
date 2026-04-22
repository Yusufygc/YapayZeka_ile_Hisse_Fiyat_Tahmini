# -*- coding: utf-8 -*-
"""
evaluation_manager.py - Evaluation and reporting orchestration.
"""

import os
from dataclasses import asdict, replace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtesting import plot_equity_curves, run_backtest, save_backtest_report, save_fold_backtest_report, save_trade_logs, summarize_backtest
from src.backtesting.signals import SignalConfig
from src.database.stock_model_db import StockModelDB, compute_composite_score
from src.ensemble import EnsembleModel
from src.evaluation.financial_metrics import compute_quantile_metrics
from src.evaluator import compute_metrics, enrich_with_benchmark_metrics, plot_comparison, plot_prediction_interval, save_metrics_report
from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry
try:
    from src.preprocessor import reconstruct_prices_from_logret, reconstruct_prices_from_return
except ImportError:  # pragma: no cover - keeps reporting/calibration importable in minimal runtimes
    def reconstruct_prices_from_logret(log_returns: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        return np.asarray(prev_close, dtype=float).ravel() * np.exp(np.asarray(log_returns, dtype=float).ravel())

    def reconstruct_prices_from_return(returns: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        return np.asarray(prev_close, dtype=float).ravel() * (1.0 + np.asarray(returns, dtype=float).ravel())
from src.reporting_utils import route_output_path, write_csv_and_aligned_view
try:
    from src.xai import XAIExplainer, XAIReportWriter
except ImportError as _xai_import_error:  # pragma: no cover - XAI is optional in minimal runtimes
    XAIExplainer = None
    XAIReportWriter = None
    XAI_IMPORT_ERROR = _xai_import_error


class EvaluationManager:
    def __init__(
        self,
        stock_symbol: str,
        outputs_dir: str,
        models_dir: str,
        tracker: ExperimentTracker,
        registry: ModelRegistry,
        feature_names: list,
        dataset_hash: str,
        dataset_metadata: Dict[str, Any],
        selected_models: Optional[list] = None,
        registry_version: str = "v5",
        stock_db: Optional[StockModelDB] = None,
        backtest_enabled: bool = True,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        initial_capital: float = 100000.0,
        signal_mode: str = "legacy",
        quality_gate_mode: str = "soft",
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
        ensemble_enabled: bool = True,
    ):
        self.stock_symbol = stock_symbol
        self.outputs_dir = outputs_dir
        self.models_dir = models_dir
        self.tracker = tracker
        self.registry = registry
        self.feature_names = feature_names
        self.dataset_hash = dataset_hash
        self.dataset_metadata = dataset_metadata
        self.selected_models = set(selected_models) if selected_models else None
        self.registry_version = registry_version
        self.stock_db = stock_db
        self.backtest_enabled = backtest_enabled
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.initial_capital = initial_capital
        self.signal_mode = signal_mode
        self.signal_config = SignalConfig(
            quality_gate_mode=quality_gate_mode,
            entry_cost_multiplier=signal_entry_cost_multiplier,
            volatility_multiplier=signal_volatility_multiplier,
            min_holding_bars=min_holding_bars,
            max_holding_bars=max_holding_bars,
            take_profit_vol_multiplier=take_profit_vol_multiplier,
            stop_loss_vol_multiplier=stop_loss_vol_multiplier,
            min_directional_accuracy=min_directional_accuracy,
            max_rmse_vs_benchmark=max_rmse_vs_benchmark,
            min_composite_score=min_composite_score,
            emergency_stop_overrides_min_hold=emergency_stop_overrides_min_hold,
        )
        self.default_signal_config = self.signal_config
        self.signal_threshold_source = "default_config"
        self.signal_threshold_calibration_summary = {}
        self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()

        self.predictions = {}
        self.prediction_targets = {}
        self.quantile_predictions = {}
        self.single_backtest_inputs = {}
        self.y_true_aligned = None
        self.y_true_target_aligned = None
        self.prev_close_aligned = None
        self.xai_dir = os.path.join(self.outputs_dir, "xai")
        self.latest_tensors = {}
        self.latest_backtest_results = {}
        self.ensemble_enabled = ensemble_enabled
        self.ensemble_weights: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _attach_composite_scores(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        enriched = enrich_with_benchmark_metrics(metrics_dict)
        for _, model_metrics in enriched.items():
            model_metrics["Composite_Score"] = compute_composite_score(model_metrics)
        return enriched

    def _attach_leakage_guard_metadata(self, metrics_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        for model_metrics in metrics_dict.values():
            model_metrics["Target_Semantics"] = self.dataset_metadata.get("target_semantics", "")
            model_metrics["Execution_Lag"] = self.dataset_metadata.get("execution_lag", "")
            model_metrics["Macro_Release_Lag"] = str(self.dataset_metadata.get("macro_release_lag", {}))
            model_metrics["Transaction_Costs"] = f"commission_bps={self.commission_bps}; slippage_bps={self.slippage_bps}"
            model_metrics["Validation_Protocol"] = str(self.dataset_metadata.get("validation_config", {}))
            model_metrics["Selection_Set"] = str(self.dataset_metadata.get("selection_set", {}))
            model_metrics["Evaluation_Set"] = str(self.dataset_metadata.get("evaluation_set", {}))
            model_metrics["Final_Holdout_Used_For_Selection"] = False
            model_metrics["Corporate_Action_Adjustment"] = str(self.dataset_metadata.get("corporate_action", {}))
            model_metrics["Feature_Pruning"] = str(self.dataset_metadata.get("feature_pruning", {}))
            model_metrics["Feature_Groups"] = str(self.dataset_metadata.get("feature_groups", {}))
            model_metrics["Scaling_Clip_Report"] = str(self.dataset_metadata.get("scaling_reports", []))
            model_metrics["Threshold_Config"] = str(self.dataset_metadata.get("signal_threshold_config", {}))
            model_metrics["Market_Regime_Source"] = "Market_Regime_SMA200"
            model_metrics["Prophet_Regressors_Used"] = str(self.dataset_metadata.get("prophet_regressors_used", []))
            model_metrics["Survivorship_Bias_Check"] = str(self.dataset_metadata.get("survivorship_bias", {}))
        return metrics_dict

    def _signal_threshold_metadata(self) -> Dict[str, Any]:
        cfg = asdict(self.signal_config)
        return {
            "phase": "phase6_backtest_standard",
            "source": self.signal_threshold_source,
            "selection_scope": "walk_forward_calibration_folds" if self.signal_threshold_source != "default_config" else "configured_defaults",
            "active_from_stage": (
                "walk_forward_backtest_signal_filtering"
                if self.signal_threshold_source != "default_config"
                else "initial_signal_filtering"
            ),
            "final_holdout_optimized": False,
            "quality_thresholds": {
                "quality_gate_mode": self.signal_config.quality_gate_mode,
                "min_directional_accuracy": self.signal_config.min_directional_accuracy,
                "max_rmse_vs_benchmark": self.signal_config.max_rmse_vs_benchmark,
                "min_composite_score": self.signal_config.min_composite_score,
            },
            "default_quality_thresholds": {
                "quality_gate_mode": self.default_signal_config.quality_gate_mode,
                "min_directional_accuracy": self.default_signal_config.min_directional_accuracy,
                "max_rmse_vs_benchmark": self.default_signal_config.max_rmse_vs_benchmark,
                "min_composite_score": self.default_signal_config.min_composite_score,
            },
            "full_signal_config": cfg,
            "execution_policy": "decision_applies_to_next_bar_return",
            "cost_policy": {
                "commission_bps": self.commission_bps,
                "slippage_bps": self.slippage_bps,
                "entry_exit_accounted_separately": True,
            },
            "calibration_summary": self.signal_threshold_calibration_summary,
        }

    def _enrich_wf_fold_metrics(self, wf_fold_metrics: Dict[str, list[Dict[str, Any]]]) -> Dict[str, list[Dict[str, Any]]]:
        if not wf_fold_metrics:
            return {}

        by_fold: Dict[Any, Dict[str, Dict[str, Any]]] = {}
        for model_name, rows in wf_fold_metrics.items():
            for row in rows:
                fold_id = row.get("Fold")
                metrics = {key: value for key, value in row.items() if key not in {"Model", "Fold"}}
                by_fold.setdefault(fold_id, {})[model_name] = metrics

        enriched_by_model: Dict[str, list[Dict[str, Any]]] = {name: [] for name in wf_fold_metrics}
        for fold_id, fold_metrics in by_fold.items():
            enriched = self._attach_composite_scores(fold_metrics)
            for model_name, metrics in enriched.items():
                enriched_by_model.setdefault(model_name, []).append({
                    "Model": model_name,
                    "Fold": fold_id,
                    **metrics,
                })
        return enriched_by_model

    def _save_wf_fold_metric_report(self, wf_fold_metrics: Dict[str, list[Dict[str, Any]]]) -> None:
        rows = [row for rows in wf_fold_metrics.values() for row in rows]
        if not rows:
            return

        fold_df = pd.DataFrame(rows)
        fold_df.sort_values(by=["Model", "Fold"], inplace=True)
        fold_path = os.path.join(self.outputs_dir, "wf_fold_metrics_v6.csv")
        write_csv_and_aligned_view(fold_df, fold_path)

        worst_rows = []
        for model_name, model_df in fold_df.groupby("Model", sort=False):
            worst = model_df.sort_values(
                by=["Composite_Score", "RMSE", "Dir_Acc"],
                ascending=[True, False, True],
            ).iloc[0].copy()
            worst["Worst_Fold_Rule"] = "min_composite_then_max_rmse"
            worst_rows.append(worst)
        worst_df = pd.DataFrame(worst_rows)
        worst_path = os.path.join(self.outputs_dir, "wf_worst_fold_v6.csv")
        write_csv_and_aligned_view(worst_df, worst_path)

        print(f"[OK] Walk-forward fold metrik dagilimi kaydedildi -> {route_output_path(fold_path)}")
        print(f"[OK] Walk-forward worst-fold raporu kaydedildi -> {route_output_path(worst_path)}")

    def _calibrate_signal_quality_thresholds(self, wf_fold_metrics: Dict[str, list[Dict[str, Any]]]) -> None:
        rows = []
        for model_name, model_rows in wf_fold_metrics.items():
            if model_name in self.signal_config.benchmark_only_models:
                continue
            rows.extend(model_rows)

        if len(rows) < 3:
            self.signal_threshold_source = "default_config"
            self.signal_threshold_calibration_summary = {
                "status": "skipped_insufficient_calibration_folds",
                "fold_metric_rows": len(rows),
                "calibration_fold_count": len({row.get("Fold") for row in rows}),
                "active_from_stage": "initial_signal_filtering",
            }
            self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()
            return

        calibration_df = pd.DataFrame(rows)
        dir_values = pd.to_numeric(calibration_df.get("Dir_Acc"), errors="coerce").dropna()
        rmse_values = pd.to_numeric(calibration_df.get("RMSE_vs_benchmark"), errors="coerce").dropna()
        composite_values = pd.to_numeric(calibration_df.get("Composite_Score"), errors="coerce").dropna()

        min_directional_accuracy = self.default_signal_config.min_directional_accuracy
        max_rmse_vs_benchmark = self.default_signal_config.max_rmse_vs_benchmark
        min_composite_score = self.default_signal_config.min_composite_score

        if not dir_values.empty:
            min_directional_accuracy = max(min_directional_accuracy, float(dir_values.quantile(0.25)))
        if not rmse_values.empty:
            max_rmse_vs_benchmark = min(max_rmse_vs_benchmark, float(rmse_values.quantile(0.75)))
        if not composite_values.empty:
            min_composite_score = max(min_composite_score, float(composite_values.quantile(0.25)))

        self.signal_config = replace(
            self.signal_config,
            min_directional_accuracy=round(min_directional_accuracy, 2),
            max_rmse_vs_benchmark=round(max_rmse_vs_benchmark, 4),
            min_composite_score=round(min_composite_score, 4),
        )
        self.signal_threshold_source = "walk_forward_calibration_folds"
        self.signal_threshold_calibration_summary = {
            "status": "applied",
            "fold_metric_rows": int(len(rows)),
            "calibration_fold_count": int(calibration_df["Fold"].nunique()) if "Fold" in calibration_df.columns else None,
            "dir_acc_q25": round(float(dir_values.quantile(0.25)), 4) if not dir_values.empty else None,
            "rmse_vs_benchmark_q75": round(float(rmse_values.quantile(0.75)), 4) if not rmse_values.empty else None,
            "composite_score_q25": round(float(composite_values.quantile(0.25)), 4) if not composite_values.empty else None,
            "calibration_set": "walk_forward_folds_only",
            "active_from_stage": "walk_forward_backtest_signal_filtering",
            "final_holdout_used": False,
        }
        self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()

    def _attach_model_family_metadata(self, metrics_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        tft_label = (
            self.dataset_metadata
            .get("model_config", {})
            .get("deep_learning", {})
            .get("tft", {})
            .get("model_label", "TFT-like Quantile Sequence Model")
        )
        for model_name, model_metrics in metrics_dict.items():
            if model_name.startswith("Ensemble "):
                model_metrics["Model_Family"] = "ensemble"
                model_metrics["Ensemble_Method"] = model_name.replace("Ensemble ", "")
                model_metrics["Ensemble_Weights"] = str(self.ensemble_weights.get(model_name, {}))
            elif model_name == "TFT":
                model_metrics["Model_Family"] = tft_label
            elif model_name in {"DLinear", "NLinear", "PatchTST Experimental"}:
                model_metrics["Model_Family"] = "low_parameter_sequence_baseline"
            elif model_name == "LightGBM Return":
                model_metrics["Model_Family"] = "gradient_boosting_return_baseline"
            else:
                model_metrics.setdefault("Model_Family", model_name)
        return metrics_dict

    @staticmethod
    def _weighted_average(predictions: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        names = list(predictions)
        arrays = [np.asarray(predictions[name], dtype=float).ravel() for name in names]
        min_len = min(len(arr) for arr in arrays)
        stacked = np.stack([arr[-min_len:] for arr in arrays], axis=0)
        weight_array = np.asarray([weights.get(name, 0.0) for name in names], dtype=float)
        if weight_array.sum() <= 0:
            weight_array = np.ones(len(names), dtype=float) / len(names)
        else:
            weight_array = weight_array / weight_array.sum()
        return np.average(stacked, axis=0, weights=weight_array)

    def _base_predictions_for_ensemble(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            name: np.asarray(preds, dtype=float).ravel()
            for name, preds in predictions.items()
            if not name.startswith("Ensemble ") and len(np.asarray(preds).ravel()) > 0
        }

    def _add_single_split_ensembles(self) -> None:
        if not self.ensemble_enabled:
            return
        base_preds = self._base_predictions_for_ensemble(self.predictions)
        if len(base_preds) < 2 or self.y_true_aligned is None:
            return

        equal_name = "Ensemble Equal Weight"
        inv_name = "Ensemble Inverse RMSE"
        equal_preds = EnsembleModel().combine(base_preds)
        inverse_weights = EnsembleModel.optimize_inverse_rmse(np.asarray(self.y_true_aligned), base_preds)
        inverse_preds = EnsembleModel(inverse_weights).combine(base_preds)
        self.ensemble_weights[equal_name] = {name: round(1.0 / len(base_preds), 6) for name in base_preds}
        self.ensemble_weights[inv_name] = inverse_weights

        base_targets = {name: self.prediction_targets[name] for name in base_preds if name in self.prediction_targets}
        equal_target = EnsembleModel().combine(base_targets) if len(base_targets) >= 2 else None
        inverse_target = self._weighted_average(base_targets, inverse_weights) if len(base_targets) >= 2 else None

        for name, pred_price, pred_target in [
            (equal_name, equal_preds, equal_target),
            (inv_name, inverse_preds, inverse_target),
        ]:
            k = min(len(pred_price), len(self.y_true_aligned), len(self.prev_close_aligned))
            self.predictions[name] = np.asarray(pred_price)[-k:]
            if pred_target is not None:
                self.prediction_targets[name] = np.asarray(pred_target)[-k:]
            template = next(iter(self.single_backtest_inputs.values()), None)
            if template:
                payload = {}
                for key, value in template.items():
                    arr = np.asarray(value)
                    payload[key] = arr[-k:] if arr.ndim > 0 and len(arr) >= k else value
                payload["pred_price"] = self.predictions[name]
                payload["pred_target"] = self.prediction_targets.get(name)
                self.single_backtest_inputs[name] = payload
        print("  [OK] Ensemble tahminleri eklendi: Equal Weight, Inverse RMSE")

    def _add_walk_forward_ensembles(
        self,
        wf_results: Dict[str, Dict[str, Any]],
        wf_predictions: Dict[str, np.ndarray],
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not self.ensemble_enabled or wf_y_true is None:
            return
        base_preds = self._base_predictions_for_ensemble(wf_predictions)
        if len(base_preds) < 2:
            return

        equal_name = "Ensemble Equal Weight"
        inv_name = "Ensemble Inverse RMSE"
        equal_preds = EnsembleModel().combine(base_preds)
        inverse_weights = EnsembleModel.optimize_inverse_rmse(np.asarray(wf_y_true), base_preds)
        inverse_preds = EnsembleModel(inverse_weights).combine(base_preds)
        self.ensemble_weights[equal_name] = {name: round(1.0 / len(base_preds), 6) for name in base_preds}
        self.ensemble_weights[inv_name] = inverse_weights

        bt_inputs = wf_backtest_inputs or {}
        template = next(iter(bt_inputs.values()), None)
        base_targets = {name: bt_inputs[name]["pred_target"] for name in base_preds if name in bt_inputs and "pred_target" in bt_inputs[name]}
        equal_target = EnsembleModel().combine(base_targets) if len(base_targets) >= 2 else None
        inverse_target = self._weighted_average(base_targets, inverse_weights) if len(base_targets) >= 2 else None

        for name, pred_price, pred_target in [
            (equal_name, equal_preds, equal_target),
            (inv_name, inverse_preds, inverse_target),
        ]:
            k = min(len(pred_price), len(np.asarray(wf_y_true).ravel()))
            wf_predictions[name] = np.asarray(pred_price)[-k:]
            y_true_price = np.asarray(wf_y_true).ravel()[-k:]
            if template:
                y_true_target = np.asarray(template.get("y_true_target", []), dtype=float).ravel()[-k:]
                prev_close = np.asarray(template.get("prev_close", []), dtype=float).ravel()[-k:]
            else:
                y_true_target = None
                prev_close = None
            wf_results[name] = compute_metrics(
                y_true_price,
                wf_predictions[name],
                y_true_target=y_true_target,
                y_pred_target=np.asarray(pred_target)[-k:] if pred_target is not None else None,
                prev_close=prev_close,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            if template:
                payload = {}
                for key, value in template.items():
                    arr = np.asarray(value)
                    payload[key] = arr[-k:] if arr.ndim > 0 and len(arr) >= k else value
                payload["y_true_price"] = y_true_price
                payload["pred_price"] = wf_predictions[name]
                payload["pred_target"] = np.asarray(pred_target)[-k:] if pred_target is not None else None
                bt_inputs[name] = payload
        print("  [OK] Walk-forward ensemble tahminleri eklendi.")

    @staticmethod
    def _select_best_model(metrics_dict: Dict[str, Dict[str, Any]]) -> Optional[str]:
        if not metrics_dict:
            return None
        return max(
            metrics_dict,
            key=lambda name: (
                float(metrics_dict[name].get("Composite_Score", float("-inf"))),
                -float(metrics_dict[name].get("RMSE", float("inf"))),
            ),
        )

    def _target_to_price(self, preds_target: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        target_mode = self.dataset_metadata.get("target_mode", "log_return")
        preds_target = np.asarray(preds_target).ravel()
        prev_close = np.asarray(prev_close).ravel()

        if target_mode == "log_return":
            return reconstruct_prices_from_logret(preds_target, prev_close)
        if target_mode == "return":
            return reconstruct_prices_from_return(preds_target, prev_close)
        if target_mode == "price":
            return preds_target
        raise ValueError(f"Desteklenmeyen target_mode: {target_mode}")

    def _save_selected_models_plot(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray], save_path: str, title: str) -> None:
        if not self.selected_models:
            return
        selected_predictions = {name: preds for name, preds in predictions.items() if name in self.selected_models}
        if not selected_predictions:
            return
        plot_comparison(y_true, selected_predictions, save_path=save_path, title=title)
        print(f"[OK] Secilen modeller grafigi kaydedildi -> {route_output_path(save_path)}")

    def _run_backtests(
        self,
        backtest_inputs: Dict[str, Dict[str, Any]],
        suffix: str,
        model_metrics_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not self.backtest_enabled or not backtest_inputs:
            return

        results = {}
        metrics_by_model = {}
        trades_by_model = {}
        equity_curves = {}
        target_mode = self.dataset_metadata.get("target_mode", "log_return")

        for model_name, payload in backtest_inputs.items():
            try:
                result = run_backtest(
                    dates=payload.get("dates"),
                    prediction_dates=payload.get("prediction_dates"),
                    y_true_price=payload["y_true_price"],
                    pred_price=payload["pred_price"],
                    prev_close=payload["prev_close"],
                    fold_ids=payload.get("fold_ids"),
                    pred_target=payload.get("pred_target"),
                    model_name=model_name,
                    validation_mode=suffix,
                    target_mode=target_mode,
                    commission_bps=self.commission_bps,
                    slippage_bps=self.slippage_bps,
                    signal_mode=self.signal_mode,
                    signal_config=self.signal_config,
                    model_metrics=(model_metrics_by_model or {}).get(model_name, {}),
                )
                results[model_name] = result
                metrics_by_model[model_name] = summarize_backtest(
                    result,
                    initial_capital=self.initial_capital,
                    trial_count=max(1, len(backtest_inputs)),
                )
                metrics_by_model[model_name].update({
                    "Target_Semantics": self.dataset_metadata.get("target_semantics", ""),
                    "Execution_Lag": self.dataset_metadata.get("execution_lag", ""),
                    "Macro_Release_Lag": str(self.dataset_metadata.get("macro_release_lag", {})),
                    "Transaction_Costs": f"commission_bps={self.commission_bps}; slippage_bps={self.slippage_bps}",
                    "Validation_Protocol": str(self.dataset_metadata.get("validation_config", {})),
                    "Threshold_Config": str(self.dataset_metadata.get("signal_threshold_config", {})),
                })
                trades_by_model[model_name] = result["trades"]
                equity_curves[model_name] = result["equity_curve"]
            except Exception as exc:
                print(f"  [WARN] {model_name} backtest basarisiz, atlaniyor: {exc}")

        if not metrics_by_model:
            return

        self.latest_backtest_results[suffix] = results

        report_path = os.path.join(self.outputs_dir, f"backtest_report_v1_{suffix}.csv")
        trades_path = os.path.join(self.outputs_dir, f"backtest_trades_v1_{suffix}.csv")
        equity_path = os.path.join(self.outputs_dir, f"backtest_equity_curve_v1_{suffix}.png")

        save_backtest_report(metrics_by_model, report_path)
        save_trade_logs(trades_by_model, trades_path)
        self._save_signal_gate_diagnostics(
            backtest_inputs=backtest_inputs,
            backtest_results=results,
            backtest_metrics=metrics_by_model,
            model_metrics_by_model=model_metrics_by_model or {},
            suffix=suffix,
            target_mode=target_mode,
        )
        if suffix == "wf":
            fold_report_path = os.path.join(self.outputs_dir, "backtest_fold_report_v6_wf.csv")
            save_fold_backtest_report(
                results,
                fold_report_path,
                initial_capital=self.initial_capital,
                trial_count=max(1, len(backtest_inputs)),
            )
        plot_equity_curves(
            equity_curves,
            save_path=equity_path,
            title=f"{self.stock_symbol} - Backtest Equity Curve [{suffix}]",
        )

        if self.selected_models:
            selected_equity_path = os.path.join(self.outputs_dir, f"backtest_equity_curve_v1_{suffix}_selected.png")
            plot_equity_curves(
                equity_curves,
                save_path=selected_equity_path,
                title=f"{self.stock_symbol} - Selected Backtest Equity Curve [{suffix}]",
                selected_models=self.selected_models,
            )

    def _save_signal_gate_diagnostics(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        backtest_results: Dict[str, Dict[str, Any]],
        backtest_metrics: Dict[str, Dict[str, Any]],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str,
        target_mode: str,
    ) -> None:
        rows = []
        for model_name, payload in backtest_inputs.items():
            current_result = backtest_results.get(model_name, {})
            current_curve = current_result.get("equity_curve", pd.DataFrame())
            current_states = (
                current_curve["Risk_State"].astype(str)
                if isinstance(current_curve, pd.DataFrame) and "Risk_State" in current_curve.columns
                else pd.Series(dtype=str)
            )
            model_metrics = model_metrics_by_model.get(model_name, {})
            bt_metrics = backtest_metrics.get(model_name, {})
            n_bars = int(len(current_curve)) if isinstance(current_curve, pd.DataFrame) else 0
            dir_acc = self._diagnostic_float(model_metrics.get("Dir_Acc"))
            rmse_vs_benchmark = self._diagnostic_float(model_metrics.get("RMSE_vs_benchmark"))
            composite_score = self._diagnostic_float(model_metrics.get("Composite_Score"))

            probe_status = "skipped_benchmark_only"
            probe_curve = pd.DataFrame()
            if model_name not in self.signal_config.benchmark_only_models:
                try:
                    probe_result = run_backtest(
                        dates=payload.get("dates"),
                        prediction_dates=payload.get("prediction_dates"),
                        y_true_price=payload["y_true_price"],
                        pred_price=payload["pred_price"],
                        prev_close=payload["prev_close"],
                        fold_ids=payload.get("fold_ids"),
                        pred_target=payload.get("pred_target"),
                        model_name=model_name,
                        validation_mode=f"{suffix}_gate_probe",
                        target_mode=target_mode,
                        commission_bps=self.commission_bps,
                        slippage_bps=self.slippage_bps,
                        signal_mode="professional",
                        signal_config=self.signal_config,
                        model_metrics={},
                    )
                    probe_curve = probe_result.get("equity_curve", pd.DataFrame())
                    probe_status = "ok"
                except Exception as exc:
                    probe_status = f"failed: {exc}"

            expected_return = self._diagnostic_numeric(probe_curve, "Expected_Return")
            entry_threshold = self._diagnostic_numeric(probe_curve, "Entry_Threshold")
            if expected_return.size == 0:
                expected_return = self._payload_expected_return(payload, target_mode)

            above_entry = np.array([], dtype=bool)
            if expected_return.size and entry_threshold.size:
                k = min(expected_return.size, entry_threshold.size)
                above_entry = expected_return[-k:] > entry_threshold[-k:]

            rows.append({
                "Model": model_name,
                "Validation_Suffix": suffix,
                "Gate_Mode": f"{self.signal_mode}_current",
                "Probe_Status": probe_status,
                "Dir_Acc": dir_acc,
                "RMSE_vs_benchmark": rmse_vs_benchmark,
                "Composite_Score": composite_score,
                "Would_Buy_Count": self._count_decision(probe_curve, "BUY"),
                "Blocked_By_DirAcc": n_bars if np.isfinite(dir_acc) and dir_acc < self.signal_config.min_directional_accuracy else 0,
                "Blocked_By_RMSE": n_bars if np.isfinite(rmse_vs_benchmark) and rmse_vs_benchmark > self.signal_config.max_rmse_vs_benchmark else 0,
                "Blocked_By_Composite": n_bars if np.isfinite(composite_score) and composite_score < self.signal_config.min_composite_score else 0,
                "Primary_Blocked_By_DirAcc": int((current_states == "quality_dir_acc").sum()),
                "Primary_Blocked_By_RMSE": int((current_states == "quality_rmse").sum()),
                "Primary_Blocked_By_Composite": int((current_states == "quality_composite").sum()),
                "Blocked_By_BenchmarkOnly": int((current_states == "benchmark_only").sum()),
                "Below_Entry_Threshold": int((probe_curve.get("Risk_State", pd.Series(dtype=str)).astype(str) == "below_threshold").sum()) if isinstance(probe_curve, pd.DataFrame) else 0,
                "Trade_Count": self._diagnostic_float(bt_metrics.get("Trade_Count")),
                "Exposure": self._diagnostic_float(bt_metrics.get("Exposure")),
                "Net_Return": self._diagnostic_float(bt_metrics.get("Net_Return")),
                "BuyHold_Return": self._diagnostic_float(bt_metrics.get("BuyHold_Return")),
                "Mean_Abs_Predicted_Return": float(np.nanmean(np.abs(expected_return))) if expected_return.size else np.nan,
                "Median_Entry_Threshold": float(np.nanmedian(entry_threshold)) if entry_threshold.size else np.nan,
                "Pct_Pred_Above_Threshold": float(np.nanmean(above_entry) * 100.0) if above_entry.size else np.nan,
                "Min_Directional_Accuracy_Config": self.signal_config.min_directional_accuracy,
                "Max_RMSE_vs_Benchmark_Config": self.signal_config.max_rmse_vs_benchmark,
                "Min_Composite_Score_Config": self.signal_config.min_composite_score,
                "Entry_Cost_Multiplier": self.signal_config.entry_cost_multiplier,
                "Volatility_Multiplier": self.signal_config.volatility_multiplier,
            })

        if not rows:
            return

        diagnostics_df = pd.DataFrame(rows)
        diagnostics_path = os.path.join(self.outputs_dir, f"signal_gate_diagnostics_v1_{suffix}.csv")
        output_paths = write_csv_and_aligned_view(diagnostics_df, diagnostics_path)
        print(f"[OK] Signal gate diagnostik raporu kaydedildi -> {output_paths['csv']}")

    @staticmethod
    def _diagnostic_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)

    @staticmethod
    def _diagnostic_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _count_decision(frame: pd.DataFrame, decision: str) -> int:
        if not isinstance(frame, pd.DataFrame) or "Decision" not in frame.columns:
            return 0
        return int((frame["Decision"].astype(str) == decision).sum())

    @staticmethod
    def _payload_expected_return(payload: Dict[str, Any], target_mode: str) -> np.ndarray:
        pred_target = payload.get("pred_target")
        if pred_target is not None and target_mode in {"log_return", "return"}:
            return np.asarray(pred_target, dtype=float).ravel()
        pred_price = np.asarray(payload.get("pred_price", []), dtype=float).ravel()
        prev_close = np.asarray(payload.get("prev_close", []), dtype=float).ravel()
        k = min(len(pred_price), len(prev_close))
        if k == 0:
            return np.array([], dtype=float)
        return (pred_price[-k:] / np.maximum(prev_close[-k:], 1e-12)) - 1.0

    def _run_xai_single_split(self, trained_models: dict, tensors: dict) -> None:
        if not self.predictions:
            return
        try:
            if XAIExplainer is None or XAIReportWriter is None:
                raise ImportError(f"XAI dependency unavailable: {XAI_IMPORT_ERROR}")
            explainer = XAIExplainer(
                self.stock_symbol,
                self.feature_names,
                self.dataset_metadata,
            )
            payload = explainer.explain_single_split(
                trained_models=trained_models,
                tensors=tensors,
                predictions=self.predictions,
                prediction_targets=self.prediction_targets,
                y_true_aligned=self.y_true_aligned,
                quantile_predictions=self.quantile_predictions,
            )
            XAIReportWriter(self.xai_dir).write(payload, suffix="latest")
        except Exception as exc:
            print(f"  [WARN] XAI raporu olusturulamadi, atlaniyor: {exc}")

    def _run_xai_walk_forward(
        self,
        wf_predictions: dict,
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not wf_predictions:
            return
        try:
            if XAIExplainer is None or XAIReportWriter is None:
                raise ImportError(f"XAI dependency unavailable: {XAI_IMPORT_ERROR}")
            explainer = XAIExplainer(
                self.stock_symbol,
                self.feature_names,
                self.dataset_metadata,
            )
            payload = explainer.explain_walk_forward(
                wf_predictions=wf_predictions,
                wf_y_true=np.asarray(wf_y_true) if wf_y_true is not None else np.asarray([]),
                wf_backtest_inputs=wf_backtest_inputs or {},
                backtest_results=self.latest_backtest_results.get("wf", {}),
            )
            XAIReportWriter(self.xai_dir).write(payload, suffix="wf")
        except Exception as exc:
            print(f"  [WARN] Walk-forward XAI raporu olusturulamadi, atlaniyor: {exc}")

    def generate_predictions(self, trained_models: dict, tensors: dict):
        print("\n" + "=" * 60)
        print("  ADIM 5 | Tahmin Uretimi ve Inverse Transform (EvaluationManager)")
        print("=" * 60)

        seq_models = {"LSTM", "TFT", "AttentionLSTM", "DLinear", "NLinear", "PatchTST Experimental"}
        tree_models = {"XGBoost", "Random Forest", "Ridge Return", "ElasticNet Return", "LightGBM Return"}

        prev_close_test = np.asarray(tensors["prev_close_test"]).ravel()
        dates_test = np.asarray(tensors["dates_test"])
        prediction_dates_test = np.asarray(tensors.get("dates_prediction", tensors["dates_test"]))
        y_test_price = np.asarray(tensors["original_y_test_aligned"]).ravel()
        y_test_target = np.asarray(tensors["y_test"]).ravel()

        raw_preds = {}
        raw_pred_targets = {}
        raw_quantiles = {}
        self.latest_tensors = tensors

        for name, model in trained_models.items():
            try:
                if name == "Prophet":
                    preds_target = model.predict(tensors["X_test"], dates_test=tensors["dates_test"])
                elif name in tree_models:
                    preds_scaled = model.predict(tensors["X_test_s"])
                    preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
                elif name in seq_models:
                    if hasattr(model, "predict_quantiles"):
                        quantile_scaled = model.predict_quantiles(tensors["X_test_seq"])
                        quantile_target = np.column_stack([
                            tensors["scaler_y"].inverse_transform(quantile_scaled[:, idx].reshape(-1, 1)).ravel()
                            for idx in range(quantile_scaled.shape[1])
                        ])
                        preds_target = quantile_target[:, quantile_scaled.shape[1] // 2]
                        raw_quantiles[name] = quantile_target
                    else:
                        preds_scaled = model.predict(tensors["X_test_seq"])
                        preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
                else:
                    try:
                        preds_scaled = model.predict(tensors["X_test_seq"])
                    except Exception:
                        preds_scaled = model.predict(tensors["X_test_s"])
                    preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()

                preds_target = np.asarray(preds_target).ravel()
                k = min(
                    len(preds_target),
                    len(prev_close_test),
                    len(y_test_price),
                    len(y_test_target),
                    len(dates_test),
                    len(prediction_dates_test),
                )
                preds_target = preds_target[-k:]
                prev_close_aligned = prev_close_test[-k:]
                y_true_price_aligned = y_test_price[-k:]
                y_true_target_aligned = y_test_target[-k:]
                dates_aligned = dates_test[-k:]
                prediction_dates_aligned = prediction_dates_test[-k:]

                raw_pred_targets[name] = preds_target
                raw_preds[name] = self._target_to_price(preds_target, prev_close_aligned)

                if name in raw_quantiles:
                    aligned_quantiles = raw_quantiles[name][-k:]
                    raw_quantiles[name] = np.column_stack([
                        self._target_to_price(aligned_quantiles[:, idx], prev_close_aligned)
                        for idx in range(aligned_quantiles.shape[1])
                    ])

                self.single_backtest_inputs[name] = {
                    "dates": dates_aligned,
                    "prediction_dates": prediction_dates_aligned,
                    "y_true_price": y_true_price_aligned,
                    "pred_price": raw_preds[name],
                    "prev_close": prev_close_aligned,
                    "pred_target": preds_target,
                    "y_true_target": y_true_target_aligned,
                }
                if name == "Prophet":
                    self.dataset_metadata["prophet_regressors_used"] = getattr(model, "regressors_used", [])
                print(f"  [OK] {name} tahmini uretildi - {len(raw_preds[name])} adim")
            except Exception as exc:
                print(f"  [WARN] {name} tahmini basarisiz, atlaniyor: {exc}")

        if not raw_preds:
            raise RuntimeError("Hicbir model tahmin uretemedi. Egitim adimini kontrol edin.")

        min_len = min(len(v) for v in raw_preds.values())
        self.predictions = {name: preds[-min_len:] for name, preds in raw_preds.items()}
        self.prediction_targets = {name: preds[-min_len:] for name, preds in raw_pred_targets.items()}
        self.quantile_predictions = {name: preds[-min_len:] for name, preds in raw_quantiles.items()}
        self.y_true_aligned = y_test_price[-min_len:]
        self.y_true_target_aligned = y_test_target[-min_len:]
        self.prev_close_aligned = prev_close_test[-min_len:]
        self._add_single_split_ensembles()

    def _predict_single_model(
        self,
        model_name: str,
        model: Any,
        tensors: dict,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ]:
        seq_models = {"LSTM", "TFT", "AttentionLSTM", "DLinear", "NLinear", "PatchTST Experimental"}
        tree_models = {"XGBoost", "Random Forest", "Ridge Return", "ElasticNet Return", "LightGBM Return"}
        quantile_target = None

        prev_close_test = np.asarray(tensors["prev_close_test"]).ravel()
        dates_test = np.asarray(tensors["dates_test"])
        prediction_dates_test = np.asarray(tensors.get("dates_prediction", tensors["dates_test"]))
        y_test_price = np.asarray(tensors["original_y_test_aligned"]).ravel()
        y_test_target = np.asarray(tensors["y_test"]).ravel()

        if model_name == "Prophet":
            preds_target = model.predict(tensors["X_test"], dates_test=tensors["dates_test"])
            self.dataset_metadata["prophet_regressors_used"] = getattr(model, "regressors_used", [])
        elif model_name in tree_models:
            preds_scaled = model.predict(tensors["X_test_s"])
            preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
        elif model_name in seq_models:
            if hasattr(model, "predict_quantiles"):
                quantile_scaled = np.asarray(model.predict_quantiles(tensors["X_test_seq"]))
                quantile_target = np.column_stack([
                    tensors["scaler_y"].inverse_transform(quantile_scaled[:, idx].reshape(-1, 1)).ravel()
                    for idx in range(quantile_scaled.shape[1])
                ])
                preds_target = quantile_target[:, quantile_scaled.shape[1] // 2]
            else:
                preds_scaled = model.predict(tensors["X_test_seq"])
                preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
        else:
            preds_target = model.predict(tensors["X_test"])

        preds_target = np.asarray(preds_target).ravel()
        k = min(
            len(preds_target),
            len(prev_close_test),
            len(y_test_price),
            len(y_test_target),
            len(dates_test),
            len(prediction_dates_test),
        )
        preds_target = preds_target[-k:]
        prev_close_aligned = prev_close_test[-k:]
        pred_price = self._target_to_price(preds_target, prev_close_aligned)
        quantile_price = None
        if quantile_target is not None:
            quantile_target = quantile_target[-k:]
            quantile_price = np.column_stack([
                self._target_to_price(quantile_target[:, idx], prev_close_aligned)
                for idx in range(quantile_target.shape[1])
            ])
        return (
            pred_price,
            preds_target,
            y_test_price[-k:],
            y_test_target[-k:],
            prev_close_aligned,
            dates_test[-k:],
            prediction_dates_test[-k:],
            quantile_price,
        )

    def evaluate_final_holdout(self, model_name: str, model: Any, tensors: dict) -> Dict[str, Dict[str, Any]]:
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
        metrics = self._attach_leakage_guard_metadata(metrics)
        metrics = self._attach_model_family_metadata(metrics)
        metrics[model_name]["Selection_Source"] = "walk_forward_composite_score"
        metrics[model_name]["Evaluation_Set_Name"] = "untouched_final_holdout"

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

        self.registry.register(
            model_name,
            f"{self.registry_version}_final_holdout",
            self.feature_names,
            metrics[model_name],
            model_path,
            self.dataset_hash,
            final_metadata,
        )

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
            )

        report_path = os.path.join(self.outputs_dir, "metrics_report_v4_final_holdout.csv")
        save_metrics_report(metrics, report_path)

        plot_path = os.path.join(self.outputs_dir, "benchmark_comparison_v4_final_holdout.png")
        plot_comparison(
            y_true_price,
            {model_name: pred_price},
            save_path=plot_path,
            title=f"{self.stock_symbol} - Final Holdout ({model_name})",
        )
        if quantile_price is not None:
            quantile_labels = [f"Q{idx + 1}" for idx in range(quantile_price.shape[1])]
            if quantile_price.shape[1] == 3:
                quantile_labels = ["P10", "P50", "P90"]
            quantile_df = pd.DataFrame(quantile_price, columns=quantile_labels)
            quantile_df.insert(0, "Actual", y_true_price[-len(quantile_df):])
            quantile_csv = os.path.join(self.outputs_dir, f"{model_name.replace(' ', '_').lower()}_quantiles_v5_final_holdout.csv")
            quantile_paths = write_csv_and_aligned_view(quantile_df, quantile_csv)
            print(f"[OK] Final holdout quantile raporu kaydedildi -> {quantile_paths['csv']}")
            if quantile_price.shape[1] >= 3:
                interval_plot = os.path.join(
                    self.outputs_dir,
                    f"{model_name.replace(' ', '_').lower()}_prediction_interval_v5_final_holdout.png",
                )
                plot_prediction_interval(
                    y_true_price[-len(quantile_price):],
                    quantile_price[:, 1],
                    quantile_price[:, 0],
                    quantile_price[:, 2],
                    save_path=interval_plot,
                    title=f"{self.stock_symbol} - Final Holdout Tahmin Araligi ({model_name})",
                )

        self._run_backtests(
            {
                model_name: {
                    "dates": dates,
                    "prediction_dates": prediction_dates,
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

        return metrics

    def evaluate_single_split(self, trained_models: dict):
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
        metrics = self._attach_leakage_guard_metadata(metrics)
        metrics = self._attach_model_family_metadata(metrics)

        for name, model_metrics in metrics.items():
            self.tracker.log_run(name, {"validation": "single"}, model_metrics, self.feature_names, self.dataset_hash, self.dataset_metadata)

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

            self.registry.register(name, self.registry_version, self.feature_names, model_metrics, model_path, self.dataset_hash, self.dataset_metadata)

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

        report_latest = os.path.join(self.outputs_dir, "metrics_report_v4_latest.csv")
        save_metrics_report(metrics, report_latest)

        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_latest.png")
        title_str = f"{self.stock_symbol} - Model Kiyaslama (Gercek vs Tahmin)"
        plot_comparison(self.y_true_aligned, self.predictions, save_path=plot_latest, title=title_str)

        selected_plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_latest_selected.png")
        selected_title_str = f"{self.stock_symbol} - Secilen Modeller (Gercek vs Tahmin)"
        self._save_selected_models_plot(self.y_true_aligned, self.predictions, save_path=selected_plot_latest, title=selected_title_str)

        self._run_backtests(self.single_backtest_inputs, suffix="latest", model_metrics_by_model=metrics)
        self._run_xai_single_split(trained_models, tensors=self.latest_tensors)

        if "TFT" in self.quantile_predictions:
            tft_quantiles = self.quantile_predictions["TFT"]
            quantile_labels = [f"Q{idx + 1}" for idx in range(tft_quantiles.shape[1])]
            if tft_quantiles.shape[1] == 3:
                quantile_labels = ["P10", "P50", "P90"]
            quantile_df = pd.DataFrame(tft_quantiles, columns=quantile_labels)
            quantile_df.insert(0, "Actual", self.y_true_aligned[-len(quantile_df):])
            quantile_csv = os.path.join(self.outputs_dir, "tft_quantiles_v5_latest.csv")
            quantile_paths = write_csv_and_aligned_view(quantile_df, quantile_csv)
            print(f"[OK] TFT quantile raporu kaydedildi -> {quantile_paths['csv']}")
            if tft_quantiles.shape[1] >= 3:
                interval_plot = os.path.join(self.outputs_dir, "tft_prediction_interval_v5_latest.png")
                plot_prediction_interval(
                    self.y_true_aligned[-len(tft_quantiles):],
                    tft_quantiles[:, 1],
                    tft_quantiles[:, 0],
                    tft_quantiles[:, 2],
                    save_path=interval_plot,
                    title=f"{self.stock_symbol} - TFT Tahmin Araligi",
                )

    def evaluate_walk_forward(
        self,
        wf_results: dict,
        wf_predictions: dict,
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        wf_fold_metrics: Optional[Dict[str, list[Dict[str, Any]]]] = None,
    ):
        print("\n" + "=" * 60)
        print("  ADIM 7 | Degerlendirme Gosterimi (Walk-Forward)")
        print("=" * 60)

        self._add_walk_forward_ensembles(wf_results, wf_predictions, wf_y_true, wf_backtest_inputs)
        wf_results = self._attach_composite_scores(wf_results)
        enriched_fold_metrics = self._enrich_wf_fold_metrics(wf_fold_metrics or {})
        self._save_wf_fold_metric_report(enriched_fold_metrics)
        self._calibrate_signal_quality_thresholds(enriched_fold_metrics)
        wf_results = self._attach_leakage_guard_metadata(wf_results)
        wf_results = self._attach_model_family_metadata(wf_results)
        for model_name, model_metrics in wf_results.items():
            self.registry.register(
                model_name,
                f"{self.registry_version}_wf_phase6_backtest",
                self.feature_names,
                model_metrics,
                "none",
                self.dataset_hash,
                self.dataset_metadata,
            )
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

        report_latest = os.path.join(self.outputs_dir, "metrics_report_v4_wf.csv")
        save_metrics_report(wf_results, report_latest)

        if wf_y_true is None or len(wf_predictions) == 0:
            print("  [WARN] Walk-forward sonucu yok - grafik olusturulamadi.")
            return

        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_wf.png")
        title_str = f"{self.stock_symbol} - Model Kiyaslama (Gercek vs Tahmin) [Walk-Forward]"
        plot_comparison(wf_y_true, wf_predictions, save_path=plot_latest, title=title_str)
        print(f"[OK] Walk-Forward karsilastirma grafigi kaydedildi -> {route_output_path(plot_latest)}")

        selected_plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_wf_selected.png")
        selected_title_str = f"{self.stock_symbol} - Secilen Modeller (Gercek vs Tahmin) [Walk-Forward]"
        self._save_selected_models_plot(wf_y_true, wf_predictions, save_path=selected_plot_latest, title=selected_title_str)

        self._run_backtests(wf_backtest_inputs or {}, suffix="wf", model_metrics_by_model=wf_results)
        self._run_xai_walk_forward(wf_predictions, wf_y_true, wf_backtest_inputs or {})
        return best_model_name
