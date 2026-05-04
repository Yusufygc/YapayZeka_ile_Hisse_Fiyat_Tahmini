# -*- coding: utf-8 -*-
"""
evaluation_manager.py - Ince orkestrasyon katmani (Faz 2.1 God Object yikimi sonrasi).

Bu sinif artik yalnizca:
  - __init__(): durum ve konfigurasyon baslangici
  - evaluate_single_split(): tek bolunme degerlendirmesi
  - evaluate_walk_forward(): walk-forward degerlendirmesi
  - evaluate_final_holdout(): son holdout degerlendirmesi

Gercek is mantigi dort mixin sinifi tarafindan saglanir:
  - _PredictionEngineMixin   (prediction_engine.py)
  - _BacktestRunnerMixin     (backtest_runner.py)
  - _SignalCalibratorMixin   (signal_calibrator.py)
  - _MetricsReporterMixin    (metrics_reporter.py)
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
from src.pipeline.backtest_runner import _BacktestRunnerMixin
from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.metrics_reporter import _MetricsReporterMixin
from src.pipeline.prediction_engine import _PredictionEngineMixin
from src.pipeline.results import FinalHoldoutResult, SingleSplitResult, WalkForwardResult
from src.pipeline.signal_calibrator import _SignalCalibratorMixin


class EvaluationManager(
    _PredictionEngineMixin,
    _BacktestRunnerMixin,
    _SignalCalibratorMixin,
    _MetricsReporterMixin,
):
    """
    Degerlendirme, kayit ve raporlama icin ince orkestrasyon sinifi.

    Is mantigi mixin'lerden miras alinir; bu sinif yalnizca durum yonetimi
    ve uc kamu metodu icerir.
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
        enriched_fold_metrics = self._enrich_wf_fold_metrics(wf_fold_metrics or {})

        wf_fold_reports = self._get_wf_fold_metric_report(enriched_fold_metrics)
        self._calibrate_signal_quality_thresholds(enriched_fold_metrics)
        calibration_results = self._calibrate_walk_forward_signal_parameters(
            wf_backtest_inputs=wf_backtest_inputs or {},
            model_metrics_by_model=wf_results,
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
            wf_backtest_inputs or {},
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
