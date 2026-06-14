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
        # Faz 1+2: context + (boş) state ÖNCE kurulur. READ-ONLY config/identity
        # artık EvaluationContext'te, mutable runtime state EvaluationState'te yaşar;
        # manager property'leri bunlara forward eder. Bu yüzden context/state'e yazan
        # _init_* metotlarından önce gelmeli. (Eskiden burada yapılan düz
        # self.stock_symbol = ... atamaları context constructor'a taşındı.)
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
        self._init_model_attrs()
        self._init_execution_attrs()
        # signal threshold metadata triggers an early service init (uses
        # signal_calibration_service); call order preserved intentionally.
        self._init_signal_calibration_state()
        self._init_mutable_state()
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
        # Faz 1: predictions/.../y_true_* artik EvaluationState varsayilanlarinda.
        # Faz 2: xai_dir READ-ONLY olarak EvaluationContext'e gider; bu atama
        # context-backed property setter uzerinden context'e yazilir.
        self.xai_dir = os.path.join(self.outputs_dir, "xai")

    def _init_context_and_state(self, **kwargs: Any) -> None:
        # Context yalnizca base identity/cfg kwarg'lariyla kurulur; turetilmis
        # READ-ONLY alanlar (selected_models, ensemble_enabled, backtest_enabled,
        # commission_bps, slippage_bps, initial_capital, signal_mode,
        # default_signal_config, xai_dir) asagidaki _init_* metotlarinda
        # context-backed property setter'lari uzerinden context'e yazilir.
        self.context = EvaluationContext(**kwargs)
        # Bos state; tüm mutable alanlar dataclass varsayilanlarindan gelir ve
        # manager property'leri (asagida) bu nesneye forward eder.
        self.state = EvaluationState()

    # ------------------------------------------------------------------ #
    #  READ-ONLY config/identity property forward'lari (Faz 2)            #
    #                                                                     #
    #  manager.X  <->  manager.context.X. Owner-forward servisleri/       #
    #  workflow'lari getattr ile okur; bu getter'lar okumayı context'e    #
    #  yönlendirir, böylece context tek READ-ONLY config kaynaktir.       #
    #  Faz 3'te servisler dogrudan self.ctx.X kullanacak.                 #
    # ------------------------------------------------------------------ #

    @property
    def context(self) -> EvaluationContext:
        # Lazy: state ile ayni gerekce — __new__ ile kurulan mekanizma testleri
        # ilk erisimde bos EvaluationContext alir. Gercek __init__ explicit kurar.
        ctx = self.__dict__.get("_context")
        if ctx is None:
            ctx = EvaluationContext()
            self.__dict__["_context"] = ctx
        return ctx

    @context.setter
    def context(self, value: EvaluationContext) -> None:
        self.__dict__["_context"] = value

    @property
    def stock_symbol(self) -> str:
        return self.context.stock_symbol

    @stock_symbol.setter
    def stock_symbol(self, value: str) -> None:
        self.context.stock_symbol = value

    @property
    def outputs_dir(self) -> str:
        return self.context.outputs_dir

    @outputs_dir.setter
    def outputs_dir(self, value: str) -> None:
        self.context.outputs_dir = value

    @property
    def models_dir(self) -> str:
        return self.context.models_dir

    @models_dir.setter
    def models_dir(self, value: str) -> None:
        self.context.models_dir = value

    @property
    def tracker(self) -> Any:
        return self.context.tracker

    @tracker.setter
    def tracker(self, value: Any) -> None:
        self.context.tracker = value

    @property
    def feature_names(self) -> list:
        return self.context.feature_names

    @feature_names.setter
    def feature_names(self, value: list) -> None:
        self.context.feature_names = value

    @property
    def dataset_hash(self) -> str:
        return self.context.dataset_hash

    @dataset_hash.setter
    def dataset_hash(self, value: str) -> None:
        self.context.dataset_hash = value

    @property
    def dataset_metadata(self) -> Dict[str, Any]:
        return self.context.dataset_metadata

    @dataset_metadata.setter
    def dataset_metadata(self, value: Dict[str, Any]) -> None:
        self.context.dataset_metadata = value

    @property
    def exe_cfg(self) -> Any:
        return self.context.exe_cfg

    @exe_cfg.setter
    def exe_cfg(self, value: Any) -> None:
        self.context.exe_cfg = value

    @property
    def model_cfg(self) -> Any:
        return self.context.model_cfg

    @model_cfg.setter
    def model_cfg(self, value: Any) -> None:
        self.context.model_cfg = value

    @property
    def stock_db(self) -> Any:
        return self.context.stock_db

    @stock_db.setter
    def stock_db(self, value: Any) -> None:
        self.context.stock_db = value

    @property
    def ensemble_enabled(self) -> bool:
        return self.context.ensemble_enabled

    @ensemble_enabled.setter
    def ensemble_enabled(self, value: bool) -> None:
        self.context.ensemble_enabled = value

    @property
    def selected_models(self) -> Optional[set]:
        return self.context.selected_models

    @selected_models.setter
    def selected_models(self, value: Optional[set]) -> None:
        self.context.selected_models = value

    @property
    def backtest_enabled(self) -> bool:
        return self.context.backtest_enabled

    @backtest_enabled.setter
    def backtest_enabled(self, value: bool) -> None:
        self.context.backtest_enabled = value

    @property
    def commission_bps(self) -> float:
        return self.context.commission_bps

    @commission_bps.setter
    def commission_bps(self, value: float) -> None:
        self.context.commission_bps = value

    @property
    def slippage_bps(self) -> float:
        return self.context.slippage_bps

    @slippage_bps.setter
    def slippage_bps(self, value: float) -> None:
        self.context.slippage_bps = value

    @property
    def initial_capital(self) -> float:
        return self.context.initial_capital

    @initial_capital.setter
    def initial_capital(self, value: float) -> None:
        self.context.initial_capital = value

    @property
    def signal_mode(self) -> str:
        return self.context.signal_mode

    @signal_mode.setter
    def signal_mode(self, value: str) -> None:
        self.context.signal_mode = value

    @property
    def default_signal_config(self) -> Any:
        return self.context.default_signal_config

    @default_signal_config.setter
    def default_signal_config(self, value: Any) -> None:
        self.context.default_signal_config = value

    @property
    def xai_dir(self) -> str:
        return self.context.xai_dir

    @xai_dir.setter
    def xai_dir(self, value: str) -> None:
        self.context.xai_dir = value

    # Faz 3.2: BacktestService'in okudugu exe_cfg flag'leri context'e tasindi.
    @property
    def write_trade_logs(self) -> bool:
        return self.context.write_trade_logs

    @write_trade_logs.setter
    def write_trade_logs(self, value: bool) -> None:
        self.context.write_trade_logs = value

    @property
    def signal_calibration_min_trades(self) -> int:
        return self.context.signal_calibration_min_trades

    @signal_calibration_min_trades.setter
    def signal_calibration_min_trades(self, value: int) -> None:
        self.context.signal_calibration_min_trades = value

    @property
    def signal_calibration_train_ratio(self) -> float:
        return self.context.signal_calibration_train_ratio

    @signal_calibration_train_ratio.setter
    def signal_calibration_train_ratio(self, value: float) -> None:
        self.context.signal_calibration_train_ratio = value

    @property
    def min_signal_evaluation_folds(self) -> int:
        return self.context.min_signal_evaluation_folds

    @min_signal_evaluation_folds.setter
    def min_signal_evaluation_folds(self, value: int) -> None:
        self.context.min_signal_evaluation_folds = value

    @property
    def signal_calibration_reject_behavior(self) -> str:
        return self.context.signal_calibration_reject_behavior

    @signal_calibration_reject_behavior.setter
    def signal_calibration_reject_behavior(self, value: str) -> None:
        self.context.signal_calibration_reject_behavior = value

    @property
    def auto_signal_diagnostics(self) -> bool:
        return self.context.auto_signal_diagnostics

    @auto_signal_diagnostics.setter
    def auto_signal_diagnostics(self, value: bool) -> None:
        self.context.auto_signal_diagnostics = value

    @property
    def enable_gate_diagnostics(self) -> bool:
        return self.context.enable_gate_diagnostics

    @enable_gate_diagnostics.setter
    def enable_gate_diagnostics(self, value: bool) -> None:
        self.context.enable_gate_diagnostics = value

    @property
    def enable_shadow_backtests(self) -> bool:
        return self.context.enable_shadow_backtests

    @enable_shadow_backtests.setter
    def enable_shadow_backtests(self, value: bool) -> None:
        self.context.enable_shadow_backtests = value

    # Faz 3.3: SignalCalibrationService'in okudugu exe_cfg flag'leri context'e tasindi.
    @property
    def calibration_scope(self) -> str:
        return self.context.calibration_scope

    @calibration_scope.setter
    def calibration_scope(self, value: str) -> None:
        self.context.calibration_scope = value

    @property
    def signal_calibration_require_oos_confirmation(self) -> bool:
        return self.context.signal_calibration_require_oos_confirmation

    @signal_calibration_require_oos_confirmation.setter
    def signal_calibration_require_oos_confirmation(self, value: bool) -> None:
        self.context.signal_calibration_require_oos_confirmation = value

    @property
    def signal_calibration_min_eval_excess_return(self) -> float:
        return self.context.signal_calibration_min_eval_excess_return

    @signal_calibration_min_eval_excess_return.setter
    def signal_calibration_min_eval_excess_return(self, value: float) -> None:
        self.context.signal_calibration_min_eval_excess_return = value

    @property
    def signal_calibration_min_eval_sharpe(self) -> float:
        return self.context.signal_calibration_min_eval_sharpe

    @signal_calibration_min_eval_sharpe.setter
    def signal_calibration_min_eval_sharpe(self, value: float) -> None:
        self.context.signal_calibration_min_eval_sharpe = value

    @property
    def signal_calibration_objective(self) -> str:
        return self.context.signal_calibration_objective

    @signal_calibration_objective.setter
    def signal_calibration_objective(self, value: str) -> None:
        self.context.signal_calibration_objective = value

    @property
    def signal_calibration_profile(self) -> str:
        return self.context.signal_calibration_profile

    @signal_calibration_profile.setter
    def signal_calibration_profile(self, value: str) -> None:
        self.context.signal_calibration_profile = value

    @property
    def signal_calibration_sampler(self) -> str:
        return self.context.signal_calibration_sampler

    @signal_calibration_sampler.setter
    def signal_calibration_sampler(self, value: str) -> None:
        self.context.signal_calibration_sampler = value

    @property
    def signal_calibration_seed(self) -> int:
        return self.context.signal_calibration_seed

    @signal_calibration_seed.setter
    def signal_calibration_seed(self, value: int) -> None:
        self.context.signal_calibration_seed = value

    @property
    def signal_calibration_max_trials(self) -> Optional[int]:
        return self.context.signal_calibration_max_trials

    @signal_calibration_max_trials.setter
    def signal_calibration_max_trials(self, value: Optional[int]) -> None:
        self.context.signal_calibration_max_trials = value

    # Faz 3.4: MetricsReportingService'in (XAI yazimi) okudugu exe_cfg flag'leri context'e tasindi.
    @property
    def write_xai_tables(self) -> bool:
        return self.context.write_xai_tables

    @write_xai_tables.setter
    def write_xai_tables(self, value: bool) -> None:
        self.context.write_xai_tables = value

    @property
    def write_markdown_reports(self) -> bool:
        return self.context.write_markdown_reports

    @write_markdown_reports.setter
    def write_markdown_reports(self, value: bool) -> None:
        self.context.write_markdown_reports = value

    # ------------------------------------------------------------------ #
    #  Mutable state property forward'lari (Faz 1)                        #
    #                                                                     #
    #  manager.X  <->  manager.state.X. Owner-forward servisleri/         #
    #  workflow'lari `setattr(owner, X, ...)` ile yazar; bu setter'lar    #
    #  yazımı state'e yönlendirir, böylece state tek mutable kaynaktir.   #
    #  Faz 3'te servisler dogrudan self.state.X kullanacak.               #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> EvaluationState:
        # Lazy: __init__'i atlayan (`__new__`) manuel kurulumlar veya erken
        # erişim için bir EvaluationState garanti eder. Gerçek __init__ zaten
        # _init_context_and_state'te explicit kurar (setter), lazy yol tetiklenmez.
        st = self.__dict__.get("_state")
        if st is None:
            st = EvaluationState()
            self.__dict__["_state"] = st
        return st

    @state.setter
    def state(self, value: EvaluationState) -> None:
        self.__dict__["_state"] = value

    @property
    def predictions(self) -> Dict[str, np.ndarray]:
        return self.state.predictions

    @predictions.setter
    def predictions(self, value: Dict[str, np.ndarray]) -> None:
        self.state.predictions = value

    @property
    def prediction_targets(self) -> Dict[str, np.ndarray]:
        return self.state.prediction_targets

    @prediction_targets.setter
    def prediction_targets(self, value: Dict[str, np.ndarray]) -> None:
        self.state.prediction_targets = value

    @property
    def quantile_predictions(self) -> Dict[str, np.ndarray]:
        return self.state.quantile_predictions

    @quantile_predictions.setter
    def quantile_predictions(self, value: Dict[str, np.ndarray]) -> None:
        self.state.quantile_predictions = value

    @property
    def single_backtest_inputs(self) -> Dict[str, Dict[str, Any]]:
        return self.state.single_backtest_inputs

    @single_backtest_inputs.setter
    def single_backtest_inputs(self, value: Dict[str, Dict[str, Any]]) -> None:
        self.state.single_backtest_inputs = value

    @property
    def latest_tensors(self) -> Dict[str, Any]:
        return self.state.latest_tensors

    @latest_tensors.setter
    def latest_tensors(self, value: Dict[str, Any]) -> None:
        self.state.latest_tensors = value

    @property
    def latest_backtest_results(self) -> Dict[str, Any]:
        return self.state.latest_backtest_results

    @latest_backtest_results.setter
    def latest_backtest_results(self, value: Dict[str, Any]) -> None:
        self.state.latest_backtest_results = value

    @property
    def latest_backtest_metrics(self) -> Dict[str, Any]:
        return self.state.latest_backtest_metrics

    @latest_backtest_metrics.setter
    def latest_backtest_metrics(self, value: Dict[str, Any]) -> None:
        self.state.latest_backtest_metrics = value

    @property
    def latest_model_metrics(self) -> Dict[str, Any]:
        return self.state.latest_model_metrics

    @latest_model_metrics.setter
    def latest_model_metrics(self, value: Dict[str, Any]) -> None:
        self.state.latest_model_metrics = value

    @property
    def ensemble_weights(self) -> Dict[str, Dict[str, float]]:
        return self.state.ensemble_weights

    @ensemble_weights.setter
    def ensemble_weights(self, value: Dict[str, Dict[str, float]]) -> None:
        self.state.ensemble_weights = value

    @property
    def ensemble_weight_scope(self) -> Dict[str, str]:
        return self.state.ensemble_weight_scope

    @ensemble_weight_scope.setter
    def ensemble_weight_scope(self, value: Dict[str, str]) -> None:
        self.state.ensemble_weight_scope = value

    @property
    def y_true_aligned(self) -> Optional[np.ndarray]:
        return self.state.y_true_aligned

    @y_true_aligned.setter
    def y_true_aligned(self, value: Optional[np.ndarray]) -> None:
        self.state.y_true_aligned = value

    @property
    def y_true_target_aligned(self) -> Optional[np.ndarray]:
        return self.state.y_true_target_aligned

    @y_true_target_aligned.setter
    def y_true_target_aligned(self, value: Optional[np.ndarray]) -> None:
        self.state.y_true_target_aligned = value

    @property
    def prev_close_aligned(self) -> Optional[np.ndarray]:
        return self.state.prev_close_aligned

    @prev_close_aligned.setter
    def prev_close_aligned(self, value: Optional[np.ndarray]) -> None:
        self.state.prev_close_aligned = value

    @property
    def signal_config(self) -> Any:
        return self.state.signal_config

    @signal_config.setter
    def signal_config(self, value: Any) -> None:
        self.state.signal_config = value

    @property
    def signal_threshold_source(self) -> str:
        return self.state.signal_threshold_source

    @signal_threshold_source.setter
    def signal_threshold_source(self, value: str) -> None:
        self.state.signal_threshold_source = value

    @property
    def signal_threshold_calibration_summary(self) -> Dict[str, Any]:
        return self.state.signal_threshold_calibration_summary

    @signal_threshold_calibration_summary.setter
    def signal_threshold_calibration_summary(self, value: Dict[str, Any]) -> None:
        self.state.signal_threshold_calibration_summary = value

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

    def _predict_single_model(self, model_name: str, model: Any, tensors: dict):
        self._ensure_services()
        return self.prediction_service._predict_single_model(model_name, model, tensors)

    def generate_predictions(self, trained_models: dict, tensors: dict):
        self._ensure_services()
        return self.prediction_service.generate_predictions(trained_models, tensors)

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

    @classmethod
    def _select_signal_calibration_row(cls, rows: list[Dict[str, Any]]) -> Dict[str, Any] | None:
        return SignalCalibrationService._select_signal_calibration_row(rows)

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
        wf_xai_records: Optional[Dict[str, list[Dict[str, Any]]]] = None,
    ) -> WalkForwardResult:
        self._ensure_services()
        # Interval kalibrasyonu (final holdout artifact sidecar'i) walk-forward
        # target residual'larini kullanir; owner'a stash et ki FinalHoldout workflow
        # (_build_interval_calibration) okuyabilsin.
        self.wf_backtest_inputs = wf_backtest_inputs or {}
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
        self._ensure_services()
        return self.final_holdout_workflow.run(model_name, model, tensors)
