# -*- coding: utf-8 -*-
"""
evaluation_manager.py - Evaluation orkestrasyon katmani.

Bu sinif yalnizca:
  - __init__(): context (READ-ONLY config/identity) + state (mutable runtime) +
    servis/workflow kompozisyonu
  - generate_predictions(): tahmin uretimi (PredictionService)
  - evaluate_single_split(): tek bolunme degerlendirmesi
  - evaluate_walk_forward(): walk-forward degerlendirmesi
  - evaluate_final_holdout(): son holdout degerlendirmesi

Gercek is mantigi servis kompozisyonu tarafindan saglanir:
  - PredictionService
  - BacktestService
  - SignalCalibrationService
  - MetricsReportingService
Servisler ve workflow'lar `manager.context` / `manager.state` nesnelerini dogrudan
paylasir; manager uzerinde config/state alanlari icin forward property yoktur.
"""

from typing import Any, Dict, Optional

from src.database.stock_model_db import StockModelDB
from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.evaluation_workflows import (
    EvaluationWorkflowServices,
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
    """Degerlendirme, kayit ve raporlama icin ince orkestrasyon sinifi."""

    def __init__(
        self,
        stock_symbol: str,
        outputs_dir: str,
        models_dir: str,
        feature_names: list,
        dataset_hash: str,
        dataset_metadata: Dict[str, Any],
        exe_cfg: ExecutionConfig,
        model_cfg: ModelConfig,
        stock_db: Optional[StockModelDB] = None,
    ):
        self.context = EvaluationContext.from_config(
            stock_symbol=stock_symbol,
            outputs_dir=outputs_dir,
            models_dir=models_dir,
            feature_names=feature_names,
            dataset_hash=dataset_hash,
            dataset_metadata=dataset_metadata,
            exe_cfg=exe_cfg,
            model_cfg=model_cfg,
            stock_db=stock_db,
        )
        # Mutable runtime state; sinyal esik durumu varsayilan config ile baslar
        # (signal_threshold_source="default_config", ozet={} dataclass default'u).
        self.state = EvaluationState(signal_config=exe_cfg.signal_config)
        self._init_services()
        # Esik metadata snapshot'i servis kurulduktan sonra alinir.
        self.context.dataset_metadata["signal_threshold_config"] = (
            self.signal_calibration_service._signal_threshold_metadata()
        )

    def _init_services(self) -> None:
        self.prediction_service = PredictionService(self.context, self.state)
        self.backtest_service = BacktestService(self.context, self.state)
        self.signal_calibration_service = SignalCalibrationService(self.context, self.state)
        self.metrics_reporting_service = MetricsReportingService(self.context, self.state)
        workflow_services = EvaluationWorkflowServices(
            prediction=self.prediction_service,
            backtest=self.backtest_service,
            signal_calibration=self.signal_calibration_service,
            metrics=self.metrics_reporting_service,
        )
        self.single_split_workflow = SingleSplitEvaluationWorkflow(
            self.context, self.state, workflow_services
        )
        self.walk_forward_workflow = WalkForwardEvaluationWorkflow(
            self.context, self.state, workflow_services
        )
        self.final_holdout_workflow = FinalHoldoutEvaluationWorkflow(
            self.context, self.state, workflow_services
        )

    # ------------------------------------------------------------------ #
    #  Public evaluation methods                                          #
    # ------------------------------------------------------------------ #

    def generate_predictions(self, trained_models: dict, tensors: dict):
        return self.prediction_service.generate_predictions(trained_models, tensors)

    def evaluate_single_split(self, trained_models: dict) -> SingleSplitResult:
        return self.single_split_workflow.run(trained_models)

    def evaluate_walk_forward(
        self,
        wf_results: dict,
        wf_predictions: dict,
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        wf_fold_metrics: Optional[Dict[str, list[Dict[str, Any]]]] = None,
        wf_xai_records: Optional[Dict[str, list[Dict[str, Any]]]] = None,
    ) -> WalkForwardResult:
        # Interval kalibrasyonu (final holdout artifact sidecar'i) walk-forward
        # target residual'larini kullanir; owner'a stash et ki FinalHoldout workflow
        # (_build_interval_calibration) okuyabilsin.
        self.state.wf_backtest_inputs = wf_backtest_inputs or {}
        self.state.wf_xai_records = wf_xai_records or {}
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
        return self.final_holdout_workflow.run(model_name, model, tensors)
