# -*- coding: utf-8 -*-
"""Explicit dependency-injected evaluation workflows for ``EvaluationManager``."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.evaluator import compute_metrics, plot_comparison
from src.evaluation.financial_metrics import compute_quantile_metrics
from src.forecasting.artifacts import save_forecast_artifact_package
from src.pipeline.model_result_exporter import (
    export_final_holdout_result,
    export_single_split_result,
    export_walk_forward_results,
)

_KERAS_MODELS = {"LSTM", "LSTM Lite", "AttentionLSTM v2"}
_BACKTEST_REGISTRY_FIELDS = (
    "Net_Return",
    "BuyHold_Return",
    "Max_Drawdown",
    "Trade_Count",
    "Signal_Diagnosis",
)


def _merge_backtest_metrics(model_metrics_by_model: dict, backtest_results: dict) -> None:
    backtest_metrics = (backtest_results or {}).get("metrics") or {}
    for model_name, model_metrics in model_metrics_by_model.items():
        bt_metrics = backtest_metrics.get(model_name) or {}
        for field in _BACKTEST_REGISTRY_FIELDS:
            if field in bt_metrics:
                model_metrics[field] = bt_metrics[field]


def _build_interval_calibration(owner, model_name: str) -> dict | None:
    """Walk-forward target residual'larından interval kalibrasyonu üretir.

    Kaynak: ``owner.wf_backtest_inputs[model_name]`` (out-of-sample fold tahminleri;
    final holdout KULLANILMAZ). B2 (residual σ, rejim-koşullu) + C (conformal q̂)
    birlikte hesaplanır; aktif üreteç top-level ``method`` ile belirlenir (B2).
    WF girdisi yoksa None (interval atlanır, geriye uyumlu).
    """
    inputs = getattr(owner, "wf_backtest_inputs", {}) or {}
    entry = inputs.get(model_name)
    if not entry:
        return None
    y_true = entry.get("y_true_target")
    y_pred = entry.get("pred_target")
    if y_true is None or y_pred is None or len(y_true) == 0:
        return None
    from src.forecasting.interval_calibration import (
        compute_conformal_calibration,
        compute_residual_calibration,
    )

    record = {
        "y_true_target": list(np.asarray(y_true, dtype=float).ravel()),
        "y_pred_target": list(np.asarray(y_pred, dtype=float).ravel()),
    }
    regimes = entry.get("market_regime")
    if regimes is not None and len(regimes):
        record["market_regime"] = list(np.asarray(regimes).ravel())
    fold_records = [record]
    residual = compute_residual_calibration(fold_records, levels=(0.8,), per_regime=True)
    if residual is None:
        return None
    calibration = dict(residual)  # top-level method = "residual_b2" (B2 aktif)
    conformal = compute_conformal_calibration(fold_records, level=0.9)
    if conformal is not None:
        calibration["conformal"] = conformal  # C: rapor + ileride aktifleştirme
    return calibration


def _write_forecast_artifact_sidecars(owner, *, model_name: str, model_path: str, tensors: dict, validation_mode: str) -> None:
    if not model_path or "scaler_X" not in tensors or "scaler_y" not in tensors:
        return
    metadata = {
        "model_name": model_name,
        "validation_mode": validation_mode,
        "feature_names": list(getattr(owner, "feature_names", []) or []),
        "target_mode": owner.dataset_metadata.get("target_mode", "log_return"),
        "feature_mode": owner.dataset_metadata.get("feature_mode", "stationary_features"),
        "scaling_mode": owner.dataset_metadata.get("scaling_mode", "robust_x_standard_y_clip"),
        "time_steps": owner.dataset_metadata.get("time_steps"),
        "dataset_hash": owner.dataset_hash,
        "run_id": owner.dataset_metadata.get("run_id"),
        "clip_report": tensors.get("clip_report", {}),
        "artifact_mode": "artifact_loaded",
        "forecast_strategy": "recursive_direct_target",
    }
    try:
        interval_calib = _build_interval_calibration(owner, model_name)
        save_forecast_artifact_package(
            model_path=model_path,
            scaler_X=tensors["scaler_X"],
            scaler_y=tensors["scaler_y"],
            metadata=metadata,
            interval_calib=interval_calib,
        )
    except Exception as exc:
        print(f"  [WARN] Forecast artifact sidecar yazilamadi ({model_name}): {exc}")


def _write_attention_xai_if_available(owner, *, model_name: str, model, tensors: dict, suffix: str) -> None:
    exporter = getattr(model, "export_attention_xai", None)
    if exporter is None:
        return
    x_seq = tensors.get("X_test_seq")
    if x_seq is None:
        x_seq = tensors.get("X_train_seq")
    if x_seq is None:
        return
    try:
        out_dir = os.path.join(owner.xai_dir, "csv")
        out_path = os.path.join(out_dir, f"xai_top_reasons_attention_{suffix}.csv")
        exporter(x_seq, owner.feature_names, out_path, model_name=model_name)
        print(f"  [OK] Attention XAI tablosu kaydedildi -> {out_path}")
    except Exception as exc:
        print(f"  [WARN] Attention XAI tablosu olusturulamadi ({model_name}): {exc}")


def _attach_score_metadata(svc, metrics: dict) -> dict:
    """Composite score + model-scope metadata ekler (3 run() workflow ortak)."""
    metrics = svc._attach_composite_scores(metrics)
    return svc._attach_model_scope_metadata(metrics)


def _attach_guard_metadata(svc, metrics: dict) -> dict:
    """Leakage-guard + model-family metadata ekler (3 run() workflow ortak)."""
    metrics = svc._attach_leakage_guard_metadata(metrics)
    return svc._attach_model_family_metadata(metrics)


@dataclass
class EvaluationWorkflowServices:
    prediction: object
    backtest: object
    signal_calibration: object
    metrics: object


class _EvaluationWorkflowBase:
    def __init__(self, ctx, state, services: EvaluationWorkflowServices) -> None:
        self.ctx = ctx
        self.state = state
        self.services = services

    @property
    def stock_symbol(self) -> str:
        return self.ctx.stock_symbol

    @property
    def outputs_dir(self) -> str:
        return self.ctx.outputs_dir

    @property
    def models_dir(self) -> str:
        return self.ctx.models_dir

    @property
    def tracker(self):
        return self.ctx.tracker

    @property
    def feature_names(self) -> list:
        return self.ctx.feature_names

    @property
    def dataset_hash(self) -> str:
        return self.ctx.dataset_hash

    @property
    def dataset_metadata(self) -> dict:
        return self.ctx.dataset_metadata

    @property
    def stock_db(self):
        return self.ctx.stock_db

    @property
    def xai_dir(self) -> str:
        return self.ctx.xai_dir

    @property
    def enable_signal_execution_calibration(self) -> bool:
        return self.ctx.enable_signal_execution_calibration

    @property
    def predictions(self) -> dict:
        return self.state.predictions

    @predictions.setter
    def predictions(self, value: dict) -> None:
        self.state.predictions = value

    @property
    def prediction_targets(self) -> dict:
        return self.state.prediction_targets

    @prediction_targets.setter
    def prediction_targets(self, value: dict) -> None:
        self.state.prediction_targets = value

    @property
    def quantile_predictions(self) -> dict:
        return self.state.quantile_predictions

    @quantile_predictions.setter
    def quantile_predictions(self, value: dict) -> None:
        self.state.quantile_predictions = value

    @property
    def single_backtest_inputs(self) -> dict:
        return self.state.single_backtest_inputs

    @single_backtest_inputs.setter
    def single_backtest_inputs(self, value: dict) -> None:
        self.state.single_backtest_inputs = value

    @property
    def latest_tensors(self) -> dict:
        return self.state.latest_tensors

    @property
    def latest_model_metrics(self) -> dict:
        return self.state.latest_model_metrics

    @property
    def y_true_aligned(self):
        return self.state.y_true_aligned

    @property
    def y_true_target_aligned(self):
        return self.state.y_true_target_aligned

    @property
    def prev_close_aligned(self):
        return self.state.prev_close_aligned

    @property
    def signal_threshold_calibration_summary(self) -> dict:
        return self.state.signal_threshold_calibration_summary

    def _add_walk_forward_ensembles(self, *args, **kwargs):
        return self.services.prediction._add_walk_forward_ensembles(*args, **kwargs)

    def _predict_single_model(self, model_name: str, model, tensors: dict):
        return self.services.prediction._predict_single_model(model_name, model, tensors)

    def _run_backtests(self, backtest_inputs: dict, suffix: str, model_metrics_by_model=None):
        return self.services.backtest._run_backtests(backtest_inputs, suffix, model_metrics_by_model)

    def _calibrate_signal_quality_thresholds(self, wf_fold_metrics: dict) -> None:
        return self.services.signal_calibration._calibrate_signal_quality_thresholds(wf_fold_metrics)

    def _signal_threshold_metadata(self) -> dict:
        return self.services.signal_calibration._signal_threshold_metadata()

    def _calibrate_walk_forward_signal_parameters(self, **kwargs) -> dict:
        object.__setattr__(
            self.services.signal_calibration,
            "_signal_calibration_grid",
            self.services.signal_calibration._signal_calibration_grid,
        )
        return self.services.signal_calibration._calibrate_walk_forward_signal_parameters(**kwargs)

    def _attach_composite_scores(self, metrics: dict) -> dict:
        return self.services.metrics._attach_composite_scores(metrics)

    def _attach_model_scope_metadata(self, metrics: dict) -> dict:
        return self.services.metrics._attach_model_scope_metadata(metrics)

    def _attach_leakage_guard_metadata(self, metrics: dict) -> dict:
        return self.services.metrics._attach_leakage_guard_metadata(metrics)

    def _attach_model_family_metadata(self, metrics: dict) -> dict:
        return self.services.metrics._attach_model_family_metadata(metrics)

    def _filter_reportable_models(self, data: dict, metrics_dict: dict | None = None) -> dict:
        return self.services.metrics._filter_reportable_models(data, metrics_dict)

    def _enrich_wf_fold_metrics(self, wf_fold_metrics: dict) -> dict:
        return self.services.metrics._enrich_wf_fold_metrics(wf_fold_metrics)

    def _get_wf_fold_metric_report(self, wf_fold_metrics: dict) -> dict:
        return self.services.metrics._get_wf_fold_metric_report(wf_fold_metrics)

    def _select_best_model(self, metrics_dict: dict) -> str | None:
        return self.services.metrics._select_best_model(metrics_dict)

    def _get_xai_single_split(self, trained_models: dict, tensors: dict):
        return self.services.metrics._get_xai_single_split(trained_models, tensors)

    def _get_xai_walk_forward(self, wf_predictions: dict, wf_y_true, wf_backtest_inputs=None):
        return self.services.metrics._get_xai_walk_forward(
            wf_predictions, wf_y_true, wf_backtest_inputs
        )

    def _split_walk_forward_signal_sets(
        self,
        wf_fold_metrics: dict,
        wf_backtest_inputs: dict,
    ) -> tuple[dict, dict, dict, dict]:
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
        min_eval = int(getattr(self.ctx, "min_signal_evaluation_folds", 3))
        train_ratio = float(getattr(self.ctx, "signal_calibration_train_ratio", 0.70))
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
        calibration_inputs = self._filter_backtest_inputs_by_folds(
            wf_backtest_inputs, calibration_folds
        )
        evaluation_inputs = self._filter_backtest_inputs_by_folds(
            wf_backtest_inputs, evaluation_folds
        )
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
    def _filter_backtest_inputs_by_folds(backtest_inputs: dict, selected_folds: set) -> dict:
        filtered = {}
        for model_name, payload in backtest_inputs.items():
            fold_ids = payload.get("fold_ids")
            if fold_ids is None:
                filtered[model_name] = payload
                continue
            fold_arr = np.asarray(fold_ids)
            mask = np.isin(fold_arr, list(selected_folds))
            if not np.any(mask):
                continue
            new_payload = {}
            for key, value in payload.items():
                arr = np.asarray(value)
                if arr.ndim > 0 and len(arr) == len(mask):
                    new_payload[key] = arr[mask]
                else:
                    new_payload[key] = value
            filtered[model_name] = new_payload
        return filtered


class SingleSplitEvaluationWorkflow(_EvaluationWorkflowBase):
    def run(self, trained_models: dict):
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
        metrics = _attach_score_metadata(self, metrics)
        metrics = self._filter_reportable_models(metrics, metrics)
        self.predictions = self._filter_reportable_models(self.predictions, metrics)
        self.prediction_targets = self._filter_reportable_models(self.prediction_targets, metrics)
        self.quantile_predictions = self._filter_reportable_models(self.quantile_predictions, metrics)
        self.single_backtest_inputs = self._filter_reportable_models(self.single_backtest_inputs, metrics)
        metrics = _attach_guard_metadata(self, metrics)
        backtest_results = self._run_backtests(
            self.single_backtest_inputs,
            suffix="latest",
            model_metrics_by_model=metrics,
        )
        _merge_backtest_metrics(metrics, backtest_results)
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

            model_ext = ".keras" if name in _KERAS_MODELS else ".pkl"
            model_filename = f"{name.replace(' ', '_').lower()}_model{model_ext}"
            model_path = os.path.join(self.models_dir, model_filename)

            original_model = trained_models.get(name)
            if original_model is None:
                if not name.startswith("Ensemble "):
                    print(f"  [WARN] {name} icin kayitli model bulunamadi, dosya kaydi atlaniyor.")
                model_path = ""
            else:
                original_model.save(model_path)
                _write_forecast_artifact_sidecars(
                    self,
                    model_name=name,
                    model_path=model_path,
                    tensors=self.latest_tensors,
                    validation_mode="single_split",
                )
                _write_attention_xai_if_available(
                    self,
                    model_name=name,
                    model=original_model,
                    tensors=self.latest_tensors,
                    suffix="latest",
                )

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
            export_single_split_result(
                self,
                model_name=name,
                metrics=model_metrics,
                model_path=model_path,
            )

        xai_payload = self._get_xai_single_split(trained_models, tensors=self.latest_tensors)

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
        return {
            "metrics": metrics,
            "y_true": self.y_true_aligned,
            "predictions": self.predictions,
            "backtest": backtest_results,
            "xai_payload": xai_payload,
            "quantile_predictions": self.quantile_predictions,
        }



class WalkForwardEvaluationWorkflow(_EvaluationWorkflowBase):
    def run(
        self,
        wf_results: dict,
        wf_predictions: dict,
        wf_y_true,
        wf_backtest_inputs=None,
        wf_fold_metrics=None,
    ):
        print("\n" + "=" * 60)
        print("  ADIM 7 | Degerlendirme Gosterimi (Walk-Forward)")
        print("=" * 60)

        self._add_walk_forward_ensembles(wf_results, wf_predictions, wf_y_true, wf_backtest_inputs)
        wf_results = _attach_score_metadata(self, wf_results)
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
        calibration_results = self._run_signal_calibration_steps(
            signal_calibration_fold_metrics,
            signal_calibration_backtest_inputs,
            signal_evaluation_backtest_inputs,
            signal_split_metadata,
            wf_results,
        )
        wf_results = _attach_guard_metadata(self, wf_results)
        best_model_name = self._select_best_model(wf_results)
        if best_model_name:
            print(f"\n  [INFO] Walk-forward secim modeli: {best_model_name}")

        df_wf = pd.DataFrame(wf_results).T
        if "Composite_Score" in df_wf.columns:
            df_wf = df_wf.sort_values(by=["Composite_Score", "RMSE"], ascending=[False, True])
        print("\nWalk-Forward Ortalama Metrikleri:")
        print(df_wf)

        self._attach_stability_scores(wf_results, enriched_fold_metrics)
        backtest_results = self._run_backtests(
            signal_evaluation_backtest_inputs or {},
            suffix="wf",
            model_metrics_by_model=wf_results,
        )
        _merge_backtest_metrics(wf_results, backtest_results)
        self.latest_model_metrics["wf"] = wf_results
        self._log_walk_forward_experiments(wf_results)
        export_walk_forward_results(
            self,
            metrics_by_model=wf_results,
            fold_metrics_by_model=enriched_fold_metrics,
            backtest_inputs_by_model=wf_backtest_inputs or {},
        )
        xai_payload = self._get_xai_walk_forward(wf_predictions, wf_y_true, wf_backtest_inputs or {})

        self._plot_walk_forward_predictions(wf_predictions, wf_y_true)

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



    def _run_signal_calibration_steps(
        self,
        signal_calibration_fold_metrics,
        signal_calibration_backtest_inputs,
        signal_evaluation_backtest_inputs,
        signal_split_metadata,
        wf_results,
    ):
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
            return self._calibrate_walk_forward_signal_parameters(
                wf_backtest_inputs=signal_calibration_backtest_inputs,
                wf_evaluation_backtest_inputs=signal_evaluation_backtest_inputs,
                model_metrics_by_model=wf_results,
                suffix="wf_calibration",
            )

        self.signal_threshold_calibration_summary.update({
            "execution_calibration_status": "skipped_insufficient_signal_calibration_folds",
            "signal_calibration_split": signal_split_metadata,
            "final_holdout_used": False,
        })
        self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()
        return {}

    @staticmethod
    def _compute_stability_score(fold_rows: list) -> float:
        """Fold bazlı Sharpe değerlerinden tek bir istikrar skoru hesapla.

        stability_score = positive_fold_ratio - 0.5 * std(fold_sharpe)
        Aralık teorik olarak (-inf, 1]. Değer yükseldikçe daha istikrarlı.
        """
        sharpes = []
        for row in fold_rows:
            try:
                v = float(row.get("Sharpe", float("nan")))
                if np.isfinite(v):
                    sharpes.append(v)
            except (TypeError, ValueError):
                pass
        if not sharpes:
            return float("nan")
        arr = np.asarray(sharpes, dtype=float)
        positive_ratio = float(np.mean(arr > 0.0))
        std_sharpe = float(np.std(arr, ddof=0)) if len(arr) > 1 else 0.0
        return float(positive_ratio - 0.5 * std_sharpe)

    def _attach_stability_scores(
        self, wf_results: dict, enriched_fold_metrics: dict
    ) -> None:
        for model_name, metrics in wf_results.items():
            fold_rows = enriched_fold_metrics.get(model_name, [])
            metrics["Stability_Score"] = self._compute_stability_score(fold_rows)

    def _log_walk_forward_experiments(self, wf_results: dict) -> None:
        if self.stock_db is None:
            return
        for model_name, avg_metrics in wf_results.items():
            is_prod_ensemble = model_name in {"Ensemble Inverse RMSE", "Ensemble Cash-Gated", "Ensemble Seq-Attention Inverse RMSE"}
            self.stock_db.log_experiment(
                stock_symbol=self.stock_symbol,
                model_name=model_name,
                metrics=avg_metrics,
                model_path="",
                features=self.feature_names,
                dataset_hash=self.dataset_hash,
                validation_mode="walk_forward",
                dataset_metadata=self.dataset_metadata,
                is_production_candidate=is_prod_ensemble,
                selection_source="walk_forward_production_ensemble" if is_prod_ensemble else None,
                run_id=self.dataset_metadata.get("run_id"),
            )

    def _plot_walk_forward_predictions(self, wf_predictions: dict, wf_y_true) -> None:
        try:
            wf_true = np.asarray(wf_y_true).ravel() if wf_y_true is not None else np.array([])
            wf_preds = {
                name: np.asarray(preds).ravel()
                for name, preds in wf_predictions.items()
                if np.asarray(preds).ndim <= 2
            }
            if not wf_true.size or not wf_preds:
                return
            sample_size = min(len(wf_true), min(len(values) for values in wf_preds.values()))
            plot_comparison(
                wf_true[-sample_size:],
                {name: values[-sample_size:] for name, values in wf_preds.items()},
                save_path=os.path.join(self.outputs_dir, 'predictions_wf.png'),
                title=f'{self.stock_symbol} - Gercek vs Tahmin (walk-forward)',
            )
        except Exception as exc:
            print(f'  [WARN] WF tahmin grafigi kaydedilemedi: {exc}')


class FinalHoldoutEvaluationWorkflow(_EvaluationWorkflowBase):
    def run(self, model_name: str, model, tensors: dict):
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
        metrics = _attach_score_metadata(self, metrics)
        metrics = _attach_guard_metadata(self, metrics)
        metrics[model_name]["Selection_Source"] = "walk_forward_composite_score"
        metrics[model_name]["Evaluation_Set_Name"] = "untouched_final_holdout"

        final_metadata = dict(self.dataset_metadata)
        final_metadata["validation_mode"] = "final_holdout"
        final_metadata["protocol_stage"] = "final_holdout_evaluation"
        final_metadata["selected_by"] = "walk_forward_composite_score"

        model_ext = ".keras" if model_name in _KERAS_MODELS else ".pkl"
        model_filename = f"{model_name.replace(' ', '_').lower()}_final_holdout_model{model_ext}"
        model_path = os.path.join(self.models_dir, model_filename)
        model.save(model_path)
        _write_forecast_artifact_sidecars(
            self,
            model_name=model_name,
            model_path=model_path,
            tensors=tensors,
            validation_mode="final_holdout",
        )
        _write_attention_xai_if_available(
            self,
            model_name=model_name,
            model=model,
            tensors=tensors,
            suffix="final_holdout",
        )
        export_final_holdout_result(
            self,
            model_name=model_name,
            metrics=metrics[model_name],
            model_path=model_path,
            prediction_columns={
                "date": dates,
                "prediction_date": prediction_dates,
                "y_true_price": y_true_price,
                "y_pred_price": pred_price,
                "y_true_target": y_true_target,
                "y_pred_target": pred_target,
                "prev_close": prev_close,
            },
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
        _merge_backtest_metrics(metrics, backtest_results)
        self.latest_model_metrics["final_holdout"] = metrics

        self.tracker.log_run(
            model_name,
            {"validation": "final_holdout", "selected_by": "walk_forward"},
            metrics[model_name],
            self.feature_names,
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
                is_production_candidate=bool(metrics[model_name].get("Candidate_For_Selection", False)),
                selection_source="walk_forward_composite_score",
                run_id=self.dataset_metadata.get("run_id"),
            )

        try:
            plot_comparison(
                y_true_price,
                {model_name: pred_price},
                save_path=os.path.join(self.outputs_dir, f'predictions_final_holdout_{model_name}.png'),
                title=f'{self.stock_symbol} {model_name} — Final Holdout',
            )
        except Exception as _pe:
            print(f'  [WARN] Final holdout grafiği kaydedilemedi: {_pe}')
        return {
            "metrics": metrics,
            "y_true": y_true_price,
            "predictions": {model_name: pred_price},
            "quantiles_df": quantiles_df,
            "quantile_price": quantile_price,
            "backtest": backtest_results,
            "model_name": model_name,
        }

