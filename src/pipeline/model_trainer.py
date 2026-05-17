# -*- coding: utf-8 -*-
"""
model_trainer.py - Model Egitim Orkestratoru
"""

import os

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
        self.stock_symbol = stock_symbol
        self.tracker = tracker
        self.feature_names = feature_names
        self.selected_models = normalize_candidate_models(selected_models)
        self.candidate_models = set(self.selected_models)
        self.benchmark_models = set(_BENCHMARK_MODELS)
        self.dataset_hash = dataset_hash
        self.dataset_metadata = dataset_metadata or {}
        self.model_config = model_config or self.dataset_metadata.get("model_config", {})
        self.deep_config = self._build_deep_config(self.model_config.get("deep_learning", {}))

        self.trained_models = {}
        self.wf_results = {}
        self.wf_fold_metrics = {}
        self.wf_predictions = {}
        self.wf_backtest_inputs = {}
        self.wf_y_true = None
        self.final_holdout_model = None
        self.final_holdout_model_name = None
        self._init_training_workflows()

    def _init_training_workflows(self) -> None:
        self.single_split_training_workflow = SingleSplitTrainingWorkflow(self)
        self.walk_forward_training_workflow = WalkForwardTrainingWorkflow(self)
        self.final_holdout_training_workflow = FinalHoldoutTrainingWorkflow(self)

    def _ensure_training_workflows(self) -> None:
        if not all(
            hasattr(self, attr)
            for attr in (
                "single_split_training_workflow",
                "walk_forward_training_workflow",
                "final_holdout_training_workflow",
            )
        ):
            self._init_training_workflows()

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

    def _has_min_sequences(self, count: int, model_name: str, context: str) -> bool:
        min_seq = int(self.deep_config.get("min_sequence_samples", 64))
        if count < min_seq:
            print(f"  [WARN] {model_name} atlandi: {context} sequence sayisi {count} < {min_seq}.")
            return False
        return True

    def _wf_has_min_sequences(self, wf_splits: list, data_manager, model_name: str) -> bool:
        min_seq = int(self.deep_config.get("min_sequence_samples", 64))
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
