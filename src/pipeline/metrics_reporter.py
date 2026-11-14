# -*- coding: utf-8 -*-
"""
metrics_reporter.py - Metrik zenginlestirme ve XAI raporlama (Faz 2.1 Mixin).

Sorumluluklar:
  - _attach_composite_scores(): Composite_Score hesaplama
  - _attach_leakage_guard_metadata(): leakage guard metadatasi ekleme
  - _attach_model_family_metadata(): model ailesi metadatasi
  - _enrich_wf_fold_metrics(): fold bazli composite score zenginlestirme
  - _get_wf_fold_metric_report(): fold metrik raporu
  - _select_best_model(): DM-anlamlilik + RMSE<=benchmark kapisi, sonra Composite_Score/RMSE
  - _get_xai_single_split() / _get_xai_walk_forward(): XAI aciklama raporu
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.evaluation.evaluator import enrich_with_benchmark_metrics
from src.evaluation.scoring import DM_ALPHA, compute_composite_details, diebold_mariano
from src.pipeline.model_scope import (
    BENCHMARK_MODELS,
    is_selection_candidate,
    report_group,
    reportable_model_names,
)
try:
    from src.xai import XAIExplainer, XAIReportWriter
except ImportError as _xai_import_error:  # pragma: no cover
    XAIExplainer = None
    XAIReportWriter = None
    _XAI_IMPORT_ERROR = _xai_import_error


class _MetricsReporterMixin:
    """Mixin: metrik zenginlestirme, model secimi ve XAI raporlama."""

    # ------------------------------------------------------------------ #
    #  Composite score attachment                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _attach_dm_pvalues(
        enriched: Dict[str, Dict[str, Any]],
        predictions: Dict[str, Any],
        y_true: Any,
        horizon: int = 1,
    ) -> None:
        """Her modele ``DM_PValue_vs_benchmark`` (tek tarafli, H1: model daha iyi) ekler.

        ``predictions`` ve ``y_true`` AYNI uzayda (tercihen hedef/getiri uzayi)
        ve kuyruktan hizali OOS dizileridir. Benchmark'in kendisi ve tahmini olmayan
        modeller icin deger NaN'dir (secimde engel olmaz).
        """
        y = np.asarray(y_true, dtype=float).ravel()
        for model_name, row in enriched.items():
            p_value = float("nan")
            bench = row.get("Benchmark_Model")
            if bench and bench != model_name and model_name in predictions and bench in predictions:
                pm = np.asarray(predictions[model_name], dtype=float).ravel()
                pb = np.asarray(predictions[bench], dtype=float).ravel()
                # Kuyruktan hizalama: ensemble gibi daha kisa (isinma foldu dusen) seriler
                # icin cift (model, benchmark) ortak kuyruk uzerinde test edilir.
                k = min(pm.size, pb.size, y.size)
                if k >= 3:
                    _, p_value = diebold_mariano(y[-k:] - pm[-k:], y[-k:] - pb[-k:], h=horizon)
            row["DM_PValue_vs_benchmark"] = p_value

    @staticmethod
    def _attach_composite_scores(
        metrics_dict: Dict[str, Dict[str, float]],
        *,
        predictions: Optional[Dict[str, Any]] = None,
        y_true: Any = None,
        horizon: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark-relative metrikler + Composite_Score (+ opsiyonel DM p-degeri).

        ``predictions``/``y_true`` verilirse ``DM_PValue_vs_benchmark`` hesaplanir;
        verilmezse alan eklenmez (secim DM kapisini atlar, yalniz RMSE kapisi kalir).
        """
        enriched = enrich_with_benchmark_metrics(metrics_dict)
        if predictions is not None and y_true is not None:
            _MetricsReporterMixin._attach_dm_pvalues(enriched, predictions, y_true, horizon)
        for _, model_metrics in enriched.items():
            # Benchmark yoksa (goreli metrikler NaN) model lider kapisini gecemez:
            # skor 49'da kapatilir ve Eligible_For_Leader=False zorlanir.
            try:
                rmse_ratio = float(model_metrics.get("RMSE_vs_benchmark", float("nan")))
            except (TypeError, ValueError):
                rmse_ratio = float("nan")
            if not np.isfinite(rmse_ratio):
                model_metrics["Eligible_For_Leader"] = False
            details = compute_composite_details(model_metrics)
            model_metrics["Composite_Score"] = details["Composite_Score"]
            model_metrics["Composite_Incomplete"] = details["Composite_Incomplete"]
        return enriched

    # ------------------------------------------------------------------ #
    #  Leakage guard metadata                                            #
    # ------------------------------------------------------------------ #

    def _attach_leakage_guard_metadata(
        self, metrics_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        for model_metrics in metrics_dict.values():
            model_metrics["Target_Semantics"] = self.ctx.dataset_metadata.get("target_semantics", "")
            model_metrics["Execution_Lag"] = self.ctx.dataset_metadata.get("execution_lag", "")
            model_metrics["Macro_Release_Lag"] = str(self.ctx.dataset_metadata.get("macro_release_lag", {}))
            model_metrics["Transaction_Costs"] = f"commission_bps={self.ctx.commission_bps}; slippage_bps={self.ctx.slippage_bps}"
            model_metrics["Validation_Protocol"] = str(self.ctx.dataset_metadata.get("validation_config", {}))
            model_metrics["Selection_Set"] = str(self.ctx.dataset_metadata.get("selection_set", {}))
            model_metrics["Evaluation_Set"] = str(self.ctx.dataset_metadata.get("evaluation_set", {}))
            model_metrics["Final_Holdout_Used_For_Selection"] = False
            model_metrics["Corporate_Action_Adjustment"] = str(self.ctx.dataset_metadata.get("corporate_action", {}))
            model_metrics["Feature_Pruning"] = str(self.ctx.dataset_metadata.get("feature_pruning", {}))
            model_metrics["Feature_Groups"] = str(self.ctx.dataset_metadata.get("feature_groups", {}))
            model_metrics["Scaling_Clip_Report"] = str(self.ctx.dataset_metadata.get("scaling_reports", []))
            model_metrics["Threshold_Config"] = str(self.ctx.dataset_metadata.get("signal_threshold_config", {}))
            model_metrics["Market_Regime_Source"] = "Market_Regime_SMA200"
            model_metrics["Prophet_Regressors_Used"] = str(self.ctx.dataset_metadata.get("prophet_regressors_used", []))
            model_metrics["Survivorship_Bias_Check"] = str(self.ctx.dataset_metadata.get("survivorship_bias", {}))
        return metrics_dict

    # ------------------------------------------------------------------ #
    #  Model family metadata                                              #
    # ------------------------------------------------------------------ #

    def _attach_model_family_metadata(
        self, metrics_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        for model_name, model_metrics in metrics_dict.items():
            if model_name.startswith("Ensemble "):
                model_metrics["Model_Family"] = "ensemble"
                model_metrics["Ensemble_Method"] = model_name.replace("Ensemble ", "")
                model_metrics["Ensemble_Weights"] = str(self.state.ensemble_weights.get(model_name, {}))
            elif model_name in {"DLinear", "NLinear"}:
                model_metrics["Model_Family"] = "low_parameter_sequence_baseline"
            elif model_name == "LightGBM Return":
                model_metrics["Model_Family"] = "gradient_boosting_return_baseline"
            else:
                model_metrics.setdefault("Model_Family", model_name)
        return self._attach_model_scope_metadata(metrics_dict)

    def _attach_model_scope_metadata(
        self, metrics_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        candidate_models = set(self.ctx.dataset_metadata.get("candidate_models", []))
        benchmark_models = set(self.ctx.dataset_metadata.get("benchmark_models", BENCHMARK_MODELS))
        for model_name, model_metrics in metrics_dict.items():
            group = report_group(model_name, candidate_models)
            model_metrics["Candidate_For_Selection"] = bool(is_selection_candidate(model_name, candidate_models))
            model_metrics["Report_Group"] = group
            model_metrics["Benchmark_Model"] = bool(model_name in benchmark_models)
        return metrics_dict

    def _filter_reportable_models(
        self, data: Dict[str, Any],
        metrics_dict: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        candidate_models = set(self.ctx.dataset_metadata.get("candidate_models", []))
        names = set(metrics_dict or data)
        allowed = reportable_model_names(names, candidate_models)
        return {name: value for name, value in data.items() if name in allowed}

    # ------------------------------------------------------------------ #
    #  Walk-forward fold metric enrichment                               #
    # ------------------------------------------------------------------ #

    def _enrich_wf_fold_metrics(
        self, wf_fold_metrics: Dict[str, list[Dict[str, Any]]]
    ) -> Dict[str, list[Dict[str, Any]]]:
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

    def _get_wf_fold_metric_report(
        self, wf_fold_metrics: Dict[str, list[Dict[str, Any]]]
    ) -> Dict[str, pd.DataFrame]:
        rows = [row for rows in wf_fold_metrics.values() for row in rows]
        if not rows:
            return {}

        fold_df = pd.DataFrame(rows)
        fold_df.sort_values(by=["Model", "Fold"], inplace=True)

        worst_rows = []
        for model_name, model_df in fold_df.groupby("Model", sort=False):
            worst = model_df.sort_values(
                by=["Composite_Score", "RMSE", "Dir_Acc"],
                ascending=[True, False, True],
            ).iloc[0].copy()
            worst["Worst_Fold_Rule"] = "min_composite_then_max_rmse"
            worst_rows.append(worst)
        worst_df = pd.DataFrame(worst_rows)
        return {
            "fold_metrics": fold_df,
            "worst_folds": worst_df,
        }

    # ------------------------------------------------------------------ #
    #  Best model selection                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _passes_selection_gate(metrics: Dict[str, Any]) -> bool:
        """Sonlu RMSE_vs_benchmark <= 1 ve (varsa) DM p < DM_ALPHA."""
        try:
            ratio = float(metrics.get("RMSE_vs_benchmark", float("nan")))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(ratio) or ratio > 1.0:
            return False
        try:
            dm_p = float(metrics.get("DM_PValue_vs_benchmark", float("nan")))
        except (TypeError, ValueError):
            dm_p = float("nan")
        # DM p-degeri yoksa (tahmin verilmedi / hesaplanamadi) kapi uygulanmaz.
        return bool(np.isnan(dm_p) or dm_p < DM_ALPHA)

    @staticmethod
    def _select_best_model(metrics_dict: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Aday modeller arasindan en iyisini secer; anlamli aday yoksa ``None``.

        Kapi: sonlu ``RMSE_vs_benchmark <= 1`` ve mevcutsa
        ``DM_PValue_vs_benchmark < 0.10``. Kapidan gecenler arasinda
        (Composite_Score, -RMSE) maksimumu secilir (composite yalniz es-puan
        kirici). Hic aday gecmezse ``None`` doner ve aday satirlarina
        ``Selection_Status="no_significant_candidate"`` yazilir.
        """
        if not metrics_dict:
            return None
        candidates = {
            name: metrics
            for name, metrics in metrics_dict.items()
            if bool(metrics.get("Candidate_For_Selection", False))
        }
        if not candidates:
            return None
        passing = {
            name: metrics
            for name, metrics in candidates.items()
            if _MetricsReporterMixin._passes_selection_gate(metrics)
        }
        if not passing:
            for metrics in candidates.values():
                metrics["Selection_Status"] = "no_significant_candidate"
            return None
        return max(
            passing,
            key=lambda name: (
                float(passing[name].get("Composite_Score", float("-inf"))),
                -float(passing[name].get("RMSE", float("inf"))),
            ),
        )

    # ------------------------------------------------------------------ #
    #  XAI reports                                                        #
    # ------------------------------------------------------------------ #

    def _get_xai_single_split(
        self, trained_models: dict, tensors: dict
    ) -> Optional[Dict[str, Any]]:
        if not self.state.predictions:
            return None
        try:
            if XAIExplainer is None:
                raise ImportError(f"XAI dependency unavailable: {_XAI_IMPORT_ERROR}")
            explainer = XAIExplainer(
                self.ctx.stock_symbol,
                self.ctx.feature_names,
                self.ctx.dataset_metadata,
            )
            payload = explainer.explain_single_split(
                trained_models=trained_models,
                tensors=tensors,
                predictions=self.state.predictions,
                prediction_targets=self.state.prediction_targets,
                y_true_aligned=self.state.y_true_aligned,
                quantile_predictions=self.state.quantile_predictions,
            )
            self._write_xai_reports(payload, suffix="latest")
            return payload
        except Exception as exc:
            print(f"  [WARN] Single split XAI raporu olusturulamadi, atlaniyor: {exc}")
            return None

    def _get_xai_walk_forward(
        self,
        wf_predictions: dict,
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not wf_predictions:
            return None
        try:
            if XAIExplainer is None:
                raise ImportError(f"XAI dependency unavailable: {_XAI_IMPORT_ERROR}")
            explainer = XAIExplainer(
                self.ctx.stock_symbol,
                self.ctx.feature_names,
                self.ctx.dataset_metadata,
            )
            payload = explainer.explain_walk_forward(
                wf_predictions=wf_predictions,
                wf_y_true=np.asarray(wf_y_true) if wf_y_true is not None else np.asarray([]),
                wf_backtest_inputs=wf_backtest_inputs or {},
                backtest_results=self.state.latest_backtest_results.get("wf", {}),
                wf_xai_records=getattr(self.state, "wf_xai_records", {}),
            )
            self._write_xai_reports(payload, suffix="wf")
            return payload
        except Exception as exc:
            print(f"  [WARN] Walk-forward XAI raporu olusturulamadi, atlaniyor: {exc}")
            return None

    def _write_xai_reports(self, payload, suffix: str) -> None:
        """XAI payload'ını XAIReportWriter aracılığıyla diske yazar (CSV, MD, PNG)."""
        if not payload or XAIReportWriter is None:
            return
        try:
            XAIReportWriter(
                self.ctx.xai_dir,
                write_tables=bool(self.ctx.write_xai_tables),
                write_markdown=bool(self.ctx.write_markdown_reports),
            ).write(payload, suffix=suffix)
        except Exception as exc:
            print(f"  [WARN] XAI dosya yazimi basarisiz ({suffix}): {exc}")

