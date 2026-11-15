# -*- coding: utf-8 -*-
"""Explicit dependency-injected training workflows for ``ModelTrainer``."""

from __future__ import annotations

import importlib
import os
import uuid

import numpy as np

from src.pipeline import model_factory

_TREE_MODELS = model_factory.TREE_MODELS


class _TrainingWorkflowBase:
    def __init__(self, ctx, state, helpers) -> None:
        self.ctx = ctx
        self.state = state
        self.helpers = helpers

    @property
    def stock_symbol(self) -> str:
        return self.ctx.stock_symbol

    @property
    def feature_names(self) -> list:
        return self.ctx.feature_names

    @property
    def selected_models(self) -> list:
        return self.ctx.selected_models

    @property
    def dataset_hash(self) -> str:
        return self.ctx.dataset_hash

    @property
    def dataset_metadata(self) -> dict:
        return self.ctx.dataset_metadata

    @property
    def trained_models(self) -> dict:
        return self.state.trained_models

    @property
    def wf_results(self) -> dict:
        return self.state.wf_results

    @property
    def wf_fold_metrics(self) -> dict:
        return self.state.wf_fold_metrics

    @property
    def wf_predictions(self) -> dict:
        return self.state.wf_predictions

    @property
    def wf_backtest_inputs(self) -> dict:
        return self.state.wf_backtest_inputs

    @property
    def wf_xai_records(self) -> dict:
        return self.state.wf_xai_records

    @property
    def wf_y_true(self):
        return self.state.wf_y_true

    @wf_y_true.setter
    def wf_y_true(self, value) -> None:
        self.state.wf_y_true = value

    @property
    def final_holdout_model(self):
        return self.state.final_holdout_model

    @final_holdout_model.setter
    def final_holdout_model(self, value) -> None:
        self.state.final_holdout_model = value

    @property
    def final_holdout_model_name(self):
        return self.state.final_holdout_model_name

    @final_holdout_model_name.setter
    def final_holdout_model_name(self, value) -> None:
        self.state.final_holdout_model_name = value

    def _model_class_for_name(self, model_name: str):
        return self.helpers.model_class_for_name(model_name)

    def _skip(self, name: str) -> bool:
        return self.helpers.skip(name)

    def _wf_run(self, *args, **kwargs) -> None:
        return self.helpers.wf_run(*args, **kwargs)

    def _baseline_specs(self):
        return self.helpers.baseline_specs()

    def _linear_baseline_specs(self):
        return self.helpers.linear_baseline_specs()

    def _boosting_baseline_specs(self):
        return self.helpers.boosting_baseline_specs()

    def _dump_feature_importances(self, validators: dict) -> None:
        return self.helpers.dump_feature_importances(validators)


class FinalHoldoutTrainingWorkflow(_TrainingWorkflowBase):
    def run(self, model_name: str, data_manager):
        if data_manager.selection_df is None or data_manager.final_holdout_df is None:
            raise ValueError("Final holdout egitimi icin selection_df ve final_holdout_df gerekir.")
        if data_manager.final_holdout_df.empty:
            raise ValueError("Final holdout seti bos.")

        tensors = data_manager.prepare_tensors(data_manager.selection_df, data_manager.final_holdout_df)
        cls = self._model_class_for_name(model_name)

        if model_name in _TREE_MODELS:
            model = cls()
            model.train(tensors["X_train_s"], tensors["y_train_s"])
        else:
            model = cls()
            model.train(tensors["X_train"], tensors["y_train"])

        self.final_holdout_model_name = model_name
        self.final_holdout_model = model
        return model, tensors



class SingleSplitTrainingWorkflow(_TrainingWorkflowBase):
    def run(self, tensors: dict):
        optuna_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "optuna",
        )
        os.makedirs(optuna_dir, exist_ok=True)
        optuna_path = os.path.join(optuna_dir, f"optuna_studies_{self.stock_symbol}.db")
        optuna_storage = f"sqlite:///{optuna_path.replace(os.sep, '/')}"
        self._train_baseline_models(tensors)
        self._train_tree_family_models(tensors)
        self._train_tuned_tree_models(tensors, optuna_storage)

    def _train_baseline_models(self, tensors: dict) -> None:
        for name, cls in self._baseline_specs():
            model = cls()
            model.train(tensors["X_train"], tensors["y_train"])
            self.trained_models[name] = model

    def _train_tree_family_models(self, tensors: dict) -> None:
        for name, cls in self._linear_baseline_specs():
            if self._skip(name):
                continue
            model = cls()
            model.train(tensors["X_train_s"], tensors["y_train_s"])
            self.trained_models[name] = model

        for name, cls in self._boosting_baseline_specs():
            if self._skip(name):
                continue
            try:
                model = cls()
                model.train(tensors["X_train_s"], tensors["y_train_s"])
                self.trained_models[name] = model
            except ImportError as exc:
                print(f"  [WARN] {name} atlandi: {exc}")

    def _train_tuned_tree_models(self, tensors: dict, optuna_storage: str) -> None:
        # Study adı çalıştırma başına benzersiz: kalıcı per-symbol DB'de önceki
        # koşuların trial'ları yeniden kullanılmasın (kayıt amaçlı storage).
        run_token = (self.dataset_metadata or {}).get("run_id") or uuid.uuid4().hex[:12]
        specs = (
            ("XGBoost", "XGBoostModel", f"xgb_{self.stock_symbol}_{run_token}"),
        )
        trainer_module = importlib.import_module("src.pipeline.model_trainer")
        for model_name, attr, study_name in specs:
            if self._skip(model_name):
                continue
            model = getattr(trainer_module, attr)()
            model.tune_and_train(
                tensors["X_train_s"], tensors["y_train_s"],
                n_trials=40, n_splits=3,
                study_storage=optuna_storage,
                study_name=study_name,
            )
            self.trained_models[model_name] = model


class WalkForwardTrainingWorkflow(_TrainingWorkflowBase):
    def run(self, wf_splits: list, data_manager):
        preprocessors = {
            "baseline": self._preprocessor(data_manager, "baseline"),
            "tree": self._preprocessor(data_manager, "tree"),
        }
        validators = {}
        self._run_standard_validators(wf_splits, preprocessors, validators)
        self._run_tuned_tree_validators(wf_splits, preprocessors["tree"], validators)
        self._materialize_validator_results(validators)
        self._dump_feature_importances(validators)
        self._stamp_walk_forward_threshold_config()

    def _preprocessor(self, data_manager, kind: str):
        def prepare(train_df, test_df):
            tensors = data_manager.prepare_tensors(train_df, test_df)
            if kind == "baseline":
                x_train, y_train = tensors["X_train"], tensors["y_train"]
                x_test, y_test, scaler_y = tensors["X_test"], tensors["y_test"], None
            else:
                x_train, y_train = tensors["X_train_s"], tensors["y_train_s"]
                x_test, y_test, scaler_y = tensors["X_test_s"], tensors["y_test_s"], tensors["scaler_y"]
            return (
                x_train, y_train,
                x_test, y_test,
                scaler_y,
                tensors["original_y_test_aligned"],
                tensors["prev_close_test"],
                tensors["dates_test"],
                tensors["dates_prediction"],
                tensors["y_test"],
                tensors.get("market_regime_test", []),
            )
        return prepare

    def _run_standard_validators(self, wf_splits, preprocessors, validators) -> None:
        for name, cls in self._baseline_specs():
            self._wf_run(name, cls, preprocessors["baseline"], wf_splits, validators)
        self._run_tabular_baselines(wf_splits, preprocessors["tree"], validators)

    def _run_tabular_baselines(self, wf_splits, preprocessor, validators) -> None:
        for name, cls in self._linear_baseline_specs():
            if not self._skip(name):
                self._wf_run(name, cls, preprocessor, wf_splits, validators)
        for name, cls in self._boosting_baseline_specs():
            if not self._skip(name):
                self._wf_run(name, cls, preprocessor, wf_splits, validators, skip_import_err=True)

    def _run_tuned_tree_validators(self, wf_splits, preprocessor, validators) -> None:
        specs = (
            ("XGBoost", "XGBoostModel", {"early_stopping_rounds": 50}),
        )
        trainer_module = importlib.import_module("src.pipeline.model_trainer")
        for model_name, attr, extra_kwargs in specs:
            if self._skip(model_name):
                continue
            self._wf_run(
                model_name,
                lambda attr=attr, extra_kwargs=extra_kwargs: getattr(trainer_module, attr)(
                    tune_on_fit=True,
                    tune_n_trials=40,
                    tune_n_splits=3,
                    **extra_kwargs,
                ),
                preprocessor,
                wf_splits,
                validators,
            )

    def _materialize_validator_results(self, validators: dict) -> None:
        for name, validator in validators.items():
            self.wf_results[name] = validator.aggregated_metrics
            self.wf_fold_metrics[name] = [
                {"Model": name, "Fold": window["split_idx"], **window["metrics"]}
                for window in validator.results
            ]
            self.wf_xai_records[name] = list(getattr(validator, "xai_records", []))
            self._store_backtest_inputs(name, validator.results)

    def _store_backtest_inputs(self, name: str, windows: list) -> None:
        all_preds, all_trues = [], []
        all_dates, all_prediction_dates, all_prev_close = [], [], []
        all_pred_target, all_true_target, all_fold_ids = [], [], []
        all_market_regime = []
        for window in windows:
            all_preds.extend(window["y_pred_price"])
            all_trues.extend(window["y_true_price"])
            all_dates.extend(window["dates"])
            all_prediction_dates.extend(window["prediction_dates"])
            all_prev_close.extend(window["prev_close"])
            all_pred_target.extend(window["y_pred_target"])
            all_true_target.extend(window["y_true_target"])
            all_market_regime.extend(window.get("market_regime", []))
            all_fold_ids.extend([window["split_idx"]] * len(window["y_true_price"]))

        self.wf_predictions[name] = np.asarray(all_preds, dtype=float)
        self.wf_y_true = np.asarray(all_trues, dtype=float)
        self.wf_backtest_inputs[name] = {
            "dates": np.asarray(all_dates),
            "prediction_dates": np.asarray(all_prediction_dates),
            "y_true_price": np.asarray(all_trues, dtype=float),
            "pred_price": np.asarray(all_preds, dtype=float),
            "prev_close": np.asarray(all_prev_close, dtype=float),
            "fold_ids": np.asarray(all_fold_ids),
            "pred_target": np.asarray(all_pred_target, dtype=float),
            "y_true_target": np.asarray(all_true_target, dtype=float),
            # Interval kalibrasyonu (B2 rejim-koşullu σ) için rejim etiketleri.
            "market_regime": np.asarray(all_market_regime),
        }

    def _stamp_walk_forward_threshold_config(self) -> None:
        for model_name, metrics in self.wf_results.items():
            metrics["Threshold_Config"] = str(self.dataset_metadata.get("signal_threshold_config", {}))

