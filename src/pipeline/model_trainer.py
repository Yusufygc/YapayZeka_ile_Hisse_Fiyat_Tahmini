# -*- coding: utf-8 -*-
"""
model_trainer.py - Model Egitim Orkestratoru
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

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

_BENCHMARK_MODELS = model_factory.BENCHMARK_MODEL_SET
_TREE_MODELS = model_factory.TREE_MODELS
XGBoostModel = model_factory.XGBoostModel


@dataclass
class TrainingContext:
    stock_symbol: str = ""
    run_dir: str | None = None
    feature_names: list = field(default_factory=list)
    selected_models: list = field(default_factory=list)
    candidate_models: set = field(default_factory=set)
    benchmark_models: set = field(default_factory=set)
    dataset_hash: str = "N/A"
    dataset_metadata: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)


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
    skip: Callable
    wf_run: Callable
    baseline_specs: Callable
    linear_baseline_specs: Callable
    boosting_baseline_specs: Callable
    dump_feature_importances: Callable


class ModelTrainer:
    def __init__(
        self,
        stock_symbol: str,
        feature_names: list,
        selected_models: list = None,
        dataset_hash: str = "N/A",
        dataset_metadata: dict | None = None,
        model_config: dict | None = None,
        run_dir: str | None = None,
    ):
        dataset_metadata = dataset_metadata or {}
        model_config = model_config or dataset_metadata.get("model_config", {})
        selected_models = normalize_candidate_models(selected_models)
        self.training_context = TrainingContext(
            stock_symbol=stock_symbol,
            run_dir=run_dir,
            feature_names=feature_names,
            selected_models=selected_models,
            candidate_models=set(selected_models),
            benchmark_models=set(_BENCHMARK_MODELS),
            dataset_hash=dataset_hash,
            dataset_metadata=dataset_metadata,
            model_config=model_config,
        )
        self.training_state = TrainingState()
        self._init_training_workflows()

    def _init_training_workflows(self) -> None:
        helpers = TrainingHelpers(
            model_class_for_name=self._model_class_for_name,
            skip=self._skip,
            wf_run=self._wf_run,
            baseline_specs=self._baseline_specs,
            linear_baseline_specs=self._linear_baseline_specs,
            boosting_baseline_specs=self._boosting_baseline_specs,
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

    def _skip(self, name: str) -> bool:
        if name in _BENCHMARK_MODELS:
            return False
        if name not in self.training_context.selected_models:
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
                target_mode=self.training_context.dataset_metadata.get("target_mode", "log_return"),
                feature_names=self.training_context.feature_names,
            )
            validator.run(wf_splits)
            validators[name] = validator
        except ImportError as exc:
            if skip_import_err:
                print(f"  [WARN] {name} walk-forward atlandi: {exc}")
            else:
                raise

    def _benchmark_specs(self):
        target_mode = self.training_context.dataset_metadata.get("target_mode", "log_return")
        return model_factory.benchmark_specs(target_mode)

    def _baseline_specs(self):
        return self._benchmark_specs()

    def _linear_baseline_specs(self):
        return model_factory.linear_baseline_specs()

    def _boosting_baseline_specs(self):
        return model_factory.boosting_baseline_specs()

    def _model_class_for_name(self, model_name: str):
        target_mode = self.training_context.dataset_metadata.get("target_mode", "log_return")
        return model_factory.model_class_for_name(model_name, self.training_context.model_config, target_mode)

    def train_final_holdout_model(self, model_name: str, data_manager):
        return self.final_holdout_training_workflow.run(model_name, data_manager)

    def train_single_split(self, tensors: dict):
        return self.single_split_training_workflow.run(tensors)

    def train_walk_forward(self, wf_splits: list, data_manager):
        return self.walk_forward_training_workflow.run(wf_splits, data_manager)

    def _dump_feature_importances(self, validators: dict) -> None:
        import csv as _csv
        if not self.training_context.run_dir:
            return
        xai_dir = os.path.join(self.training_context.run_dir, "xai")
        os.makedirs(xai_dir, exist_ok=True)
        for model_name, validator in validators.items():
            fi = getattr(validator, "mean_feature_importance", None)
            if fi is None:
                continue
            if not self.training_context.feature_names or len(fi) != len(self.training_context.feature_names):
                continue
            safe_name = model_name.replace(" ", "_")
            out_path = os.path.join(xai_dir, f"feature_importance_{safe_name}_wf.csv")
            rows = sorted(zip(self.training_context.feature_names, fi.tolist()), key=lambda r: r[1], reverse=True)
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                writer = _csv.writer(fh)
                writer.writerow(["Feature", "Mean_Importance_WF"])
                writer.writerows(rows)
            print(f"  [OK] Feature importance kaydedildi -> {out_path}")
