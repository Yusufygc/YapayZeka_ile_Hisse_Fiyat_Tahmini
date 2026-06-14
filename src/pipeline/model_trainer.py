# -*- coding: utf-8 -*-
"""
model_trainer.py - Model Egitim Orkestratoru
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.experiments.experiment_tracker import ExperimentTracker
from src.pipeline import model_factory
from src.pipeline.model_scope import normalize_candidate_models
from src.pipeline.training_workflows import (
    FinalHoldoutTrainingWorkflow,
    SingleSplitTrainingWorkflow,
    WalkForwardTrainingWorkflow,
)
from src.validation.walk_forward import WalkForwardValidator

# Aktif üretim modelleri — pipeline menüsünde gösterilir
_ALL_MODELS = model_factory.ALL_MODELS

# Kabul görmüş literatüre göre isteğe bağlı; yüksek hesaplama maliyeti veya
# sınırlı ek değer sunar; yalnızca karşılaştırma amacıyla kullanılır
_OPTIONAL_MODELS = model_factory.OPTIONAL_MODELS

# Üretim kullanımı önerilmez (literatür temelli):
#   ARIMA  — X_train'i yok sayar, yalnızca y_train kullanır (Fama 1970/1991)
#   Prophet — walk-forward desteği yok; yalnızca single-split (Taylor & Letham 2018)
_LEGACY_MODELS = model_factory.LEGACY_MODELS

_BENCHMARK_MODELS = model_factory.BENCHMARK_MODEL_SET
_TREE_MODELS = model_factory.TREE_MODELS
_SEQ_MODELS = model_factory.SEQ_MODELS
XGBoostModel = model_factory.XGBoostModel
RandomForestModel = model_factory.RandomForestModel


@dataclass
class TrainingContext:
    stock_symbol: str = ""
    tracker: ExperimentTracker | None = None
    feature_names: list = field(default_factory=list)
    selected_models: list = field(default_factory=list)
    candidate_models: set = field(default_factory=set)
    benchmark_models: set = field(default_factory=set)
    dataset_hash: str = "N/A"
    dataset_metadata: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)
    deep_config: dict = field(default_factory=dict)


@dataclass
class TrainingState:
    trained_models: dict = field(default_factory=dict)
    wf_results: dict = field(default_factory=dict)
    wf_fold_metrics: dict = field(default_factory=dict)
    wf_predictions: dict = field(default_factory=dict)
    wf_backtest_inputs: dict = field(default_factory=dict)
    wf_xai_records: dict = field(default_factory=dict)
    wf_y_true: Any = None
    final_holdout_model: Any = None
    final_holdout_model_name: str | None = None


@dataclass
class TrainingHelpers:
    model_class_for_name: Callable
    make_prophet: Callable
    make_arima: Callable
    make_lstm: Callable
    make_lstm_lite: Callable
    make_attention_lstm_v2: Callable
    has_min_sequences: Callable
    wf_has_min_sequences: Callable
    skip: Callable
    wf_run: Callable
    baseline_specs: Callable
    linear_baseline_specs: Callable
    boosting_baseline_specs: Callable
    sequence_baseline_specs: Callable
    dump_feature_importances: Callable


class ModelTrainer:
    def __init__(
        self,
        stock_symbol: str,
        tracker: ExperimentTracker,
        feature_names: list,
        selected_models: list = None,
        dataset_hash: str = "N/A",
        dataset_metadata: dict | None = None,
        model_config: dict | None = None,
    ):
        dataset_metadata = dataset_metadata or {}
        model_config = model_config or dataset_metadata.get("model_config", {})
        selected_models = normalize_candidate_models(selected_models)
        self.training_context = TrainingContext(
            stock_symbol=stock_symbol,
            tracker=tracker,
            feature_names=feature_names,
            selected_models=selected_models,
            candidate_models=set(selected_models),
            benchmark_models=set(_BENCHMARK_MODELS),
            dataset_hash=dataset_hash,
            dataset_metadata=dataset_metadata,
            model_config=model_config,
            deep_config=self._build_deep_config(model_config.get("deep_learning", {})),
        )
        self.training_state = TrainingState()
        self._init_training_workflows()

    def _init_training_workflows(self) -> None:
        helpers = TrainingHelpers(
            model_class_for_name=self._model_class_for_name,
            make_prophet=self._make_prophet,
            make_arima=self._make_arima,
            make_lstm=self._make_lstm,
            make_lstm_lite=self._make_lstm_lite,
            make_attention_lstm_v2=self._make_attention_lstm_v2,
            has_min_sequences=self._has_min_sequences,
            wf_has_min_sequences=self._wf_has_min_sequences,
            skip=self._skip,
            wf_run=self._wf_run,
            baseline_specs=self._baseline_specs,
            linear_baseline_specs=self._linear_baseline_specs,
            boosting_baseline_specs=self._boosting_baseline_specs,
            sequence_baseline_specs=self._sequence_baseline_specs,
            dump_feature_importances=self._dump_feature_importances,
        )
        self.single_split_training_workflow = SingleSplitTrainingWorkflow(
            self.training_context, self.training_state, helpers
        )
        self.walk_forward_training_workflow = WalkForwardTrainingWorkflow(
            self.training_context, self.training_state, helpers
        )
        self.final_holdout_training_workflow = FinalHoldoutTrainingWorkflow(
            self.training_context, self.training_state, helpers
        )

    def _ensure_training_workflows(self) -> None:
        self._ensure_training_context()
        self._ensure_training_state()
        if not all(
            hasattr(self, attr)
            for attr in (
                "single_split_training_workflow",
                "walk_forward_training_workflow",
                "final_holdout_training_workflow",
            )
        ):
            self._init_training_workflows()

    def _ensure_training_context(self) -> TrainingContext:
        if "training_context" not in self.__dict__:
            self.__dict__["training_context"] = TrainingContext()
        return self.__dict__["training_context"]

    def _ensure_training_state(self) -> TrainingState:
        if "training_state" not in self.__dict__:
            self.__dict__["training_state"] = TrainingState()
        return self.__dict__["training_state"]

    @property
    def stock_symbol(self) -> str:
        return self._ensure_training_context().stock_symbol

    @stock_symbol.setter
    def stock_symbol(self, value: str) -> None:
        self._ensure_training_context().stock_symbol = value

    @property
    def tracker(self) -> ExperimentTracker | None:
        return self._ensure_training_context().tracker

    @tracker.setter
    def tracker(self, value: ExperimentTracker | None) -> None:
        self._ensure_training_context().tracker = value

    @property
    def feature_names(self) -> list:
        return self._ensure_training_context().feature_names

    @feature_names.setter
    def feature_names(self, value: list) -> None:
        self._ensure_training_context().feature_names = value

    @property
    def selected_models(self) -> list:
        return self._ensure_training_context().selected_models

    @selected_models.setter
    def selected_models(self, value: list) -> None:
        self._ensure_training_context().selected_models = value

    @property
    def candidate_models(self) -> set:
        return self._ensure_training_context().candidate_models

    @candidate_models.setter
    def candidate_models(self, value: set) -> None:
        self._ensure_training_context().candidate_models = value

    @property
    def benchmark_models(self) -> set:
        return self._ensure_training_context().benchmark_models

    @benchmark_models.setter
    def benchmark_models(self, value: set) -> None:
        self._ensure_training_context().benchmark_models = value

    @property
    def dataset_hash(self) -> str:
        return self._ensure_training_context().dataset_hash

    @dataset_hash.setter
    def dataset_hash(self, value: str) -> None:
        self._ensure_training_context().dataset_hash = value

    @property
    def dataset_metadata(self) -> dict:
        return self._ensure_training_context().dataset_metadata

    @dataset_metadata.setter
    def dataset_metadata(self, value: dict) -> None:
        self._ensure_training_context().dataset_metadata = value

    @property
    def model_config(self) -> dict:
        return self._ensure_training_context().model_config

    @model_config.setter
    def model_config(self, value: dict) -> None:
        self._ensure_training_context().model_config = value

    @property
    def deep_config(self) -> dict:
        return self._ensure_training_context().deep_config

    @deep_config.setter
    def deep_config(self, value: dict) -> None:
        self._ensure_training_context().deep_config = value

    @property
    def trained_models(self) -> dict:
        return self._ensure_training_state().trained_models

    @trained_models.setter
    def trained_models(self, value: dict) -> None:
        self._ensure_training_state().trained_models = value

    @property
    def wf_results(self) -> dict:
        return self._ensure_training_state().wf_results

    @wf_results.setter
    def wf_results(self, value: dict) -> None:
        self._ensure_training_state().wf_results = value

    @property
    def wf_fold_metrics(self) -> dict:
        return self._ensure_training_state().wf_fold_metrics

    @wf_fold_metrics.setter
    def wf_fold_metrics(self, value: dict) -> None:
        self._ensure_training_state().wf_fold_metrics = value

    @property
    def wf_predictions(self) -> dict:
        return self._ensure_training_state().wf_predictions

    @wf_predictions.setter
    def wf_predictions(self, value: dict) -> None:
        self._ensure_training_state().wf_predictions = value

    @property
    def wf_backtest_inputs(self) -> dict:
        return self._ensure_training_state().wf_backtest_inputs

    @wf_backtest_inputs.setter
    def wf_backtest_inputs(self, value: dict) -> None:
        self._ensure_training_state().wf_backtest_inputs = value

    @property
    def wf_xai_records(self) -> dict:
        return self._ensure_training_state().wf_xai_records

    @wf_xai_records.setter
    def wf_xai_records(self, value: dict) -> None:
        self._ensure_training_state().wf_xai_records = value

    @property
    def wf_y_true(self):
        return self._ensure_training_state().wf_y_true

    @wf_y_true.setter
    def wf_y_true(self, value) -> None:
        self._ensure_training_state().wf_y_true = value

    @property
    def final_holdout_model(self):
        return self._ensure_training_state().final_holdout_model

    @final_holdout_model.setter
    def final_holdout_model(self, value) -> None:
        self._ensure_training_state().final_holdout_model = value

    @property
    def final_holdout_model_name(self) -> str | None:
        return self._ensure_training_state().final_holdout_model_name

    @final_holdout_model_name.setter
    def final_holdout_model_name(self, value: str | None) -> None:
        self._ensure_training_state().final_holdout_model_name = value

    @staticmethod
    def _build_deep_config(config: dict) -> dict:
        return model_factory.build_deep_config(config)

    def _arima_config(self) -> dict:
        return model_factory.arima_config(self.model_config)

    def _make_prophet(self):
        return model_factory.make_prophet(self.model_config, self.feature_names)

    def _make_arima(self):
        return model_factory.make_arima(self.model_config)

    def _make_lstm(self, stage: str):
        return model_factory.make_lstm(self.deep_config, stage)

    def _make_lstm_lite(self, stage: str):
        return model_factory.make_lstm_lite(self.deep_config, stage)

    def _make_attention_lstm_v2(self, stage: str):
        return model_factory.make_attention_lstm_v2(self.deep_config, stage)

    def _min_sequence_samples_for(self, model_name: str) -> int:
        if model_name == "AttentionLSTM v2":
            return int(self.deep_config.get("attention_lstm_v2_min_sequence_samples", 252))
        if model_name == "LSTM Lite":
            return int(self.deep_config.get("lstm_lite_min_sequence_samples", 252))
        return int(self.deep_config.get("min_sequence_samples", 64))

    def _has_min_sequences(self, count: int, model_name: str, context: str) -> bool:
        min_seq = self._min_sequence_samples_for(model_name)
        if count < min_seq:
            print(f"  [WARN] {model_name} atlandi: {context} sequence sayisi {count} < {min_seq}.")
            return False
        return True

    def _wf_has_min_sequences(self, wf_splits: list, data_manager, model_name: str) -> bool:
        min_seq = self._min_sequence_samples_for(model_name)
        time_steps = getattr(data_manager, "time_steps", None) or data_manager.data_cfg.time_steps
        min_fold_seq = min(max(0, len(split["train"]) - time_steps) for split in wf_splits) if wf_splits else 0
        if min_fold_seq < min_seq:
            print(f"  [WARN] {model_name} walk-forward atlandi: en kucuk fold sequence sayisi {min_fold_seq} < {min_seq}.")
            return False
        return True

    def _skip(self, name: str) -> bool:
        if name in _BENCHMARK_MODELS:
            return False
        if name not in self.selected_models:
            print(f"  [--] {name} atlandi (secilmedi).")
            return True
        return False

    def _wf_run(
        self,
        name: str,
        factory,
        preprocessor,
        wf_splits: list,
        validators: dict,
        *,
        skip_import_err: bool = False,
    ) -> None:
        """Walk-forward dogrulamasi icin yardimci metot (DRY).

        WalkForwardValidator olusturur, calistirir ve validators sozlugune kaydeder.
        skip_import_err=True ile opsiyonel bagimlilik hatalarini sessizce atar.
        """
        try:
            validator = WalkForwardValidator(
                factory,
                preprocessor,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
                feature_names=self.feature_names,
            )
            validator.run(wf_splits)
            validators[name] = validator
        except ImportError as exc:
            if skip_import_err:
                print(f"  [WARN] {name} walk-forward atlandi: {exc}")
            else:
                raise

    def _benchmark_specs(self):
        target_mode = self.dataset_metadata.get("target_mode", "log_return")
        return model_factory.benchmark_specs(target_mode)

    def _baseline_specs(self):
        return self._benchmark_specs()

    def _linear_baseline_specs(self):
        return model_factory.linear_baseline_specs()

    def _boosting_baseline_specs(self):
        return model_factory.boosting_baseline_specs()

    def _sequence_baseline_specs(self):
        return model_factory.sequence_baseline_specs()

    def _model_class_for_name(self, model_name: str):
        target_mode = self.dataset_metadata.get("target_mode", "log_return")
        return model_factory.model_class_for_name(model_name, self.model_config, target_mode)

    def train_final_holdout_model(self, model_name: str, data_manager):
        self._ensure_training_workflows()
        return self.final_holdout_training_workflow.run(model_name, data_manager)

    def train_single_split(self, tensors: dict):
        self._ensure_training_workflows()
        return self.single_split_training_workflow.run(tensors)

    def train_walk_forward(self, wf_splits: list, data_manager):
        self._ensure_training_workflows()
        return self.walk_forward_training_workflow.run(wf_splits, data_manager)

    def _dump_feature_importances(self, validators: dict) -> None:
        import csv as _csv
        run_dir = os.path.dirname(self.tracker.log_dir)
        xai_dir = os.path.join(run_dir, "xai")
        os.makedirs(xai_dir, exist_ok=True)
        for model_name, validator in validators.items():
            fi = getattr(validator, "mean_feature_importance", None)
            if fi is None:
                continue
            if not self.feature_names or len(fi) != len(self.feature_names):
                continue
            safe_name = model_name.replace(" ", "_")
            out_path = os.path.join(xai_dir, f"feature_importance_{safe_name}_wf.csv")
            rows = sorted(zip(self.feature_names, fi.tolist()), key=lambda r: r[1], reverse=True)
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                writer = _csv.writer(fh)
                writer.writerow(["Feature", "Mean_Importance_WF"])
                writer.writerows(rows)
            print(f"  [OK] Feature importance kaydedildi -> {out_path}")
