# -*- coding: utf-8 -*-
"""Owner-backed evaluation workflows for ``EvaluationManager``."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.evaluation.evaluator import compute_metrics, plot_comparison
from src.evaluation.financial_metrics import compute_quantile_metrics
from src.pipeline.evaluation_services import _OwnerBackedService


class SingleSplitEvaluationWorkflow(_OwnerBackedService):
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

            model_ext = ".keras" if name == "LSTM" else ".pkl"
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



class WalkForwardEvaluationWorkflow(_OwnerBackedService):
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
        calibration_results = self._run_signal_calibration_steps(
            signal_calibration_fold_metrics,
            signal_calibration_backtest_inputs,
            signal_evaluation_backtest_inputs,
            signal_split_metadata,
            wf_results,
        )
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

        self._log_walk_forward_experiments(wf_results)

        backtest_results = self._run_backtests(
            signal_evaluation_backtest_inputs or {},
            suffix="wf",
            model_metrics_by_model=wf_results,
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

    def _log_walk_forward_experiments(self, wf_results: dict) -> None:
        if self.stock_db is None:
            return
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


class FinalHoldoutEvaluationWorkflow(_OwnerBackedService):
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

        model_ext = ".keras" if model_name == "LSTM" else ".pkl"
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
        return {
            "metrics": metrics,
            "y_true": y_true_price,
            "predictions": {model_name: pred_price},
            "quantiles_df": quantiles_df,
            "quantile_price": quantile_price,
            "backtest": backtest_results,
            "model_name": model_name,
        }

