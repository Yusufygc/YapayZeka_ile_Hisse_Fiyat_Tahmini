# -*- coding: utf-8 -*-
"""Owner-backed training workflows for ``ModelTrainer``."""

from __future__ import annotations

import importlib
import os

import numpy as np

from src.pipeline.evaluation_services import _OwnerBackedService
from src.pipeline import model_factory

_TREE_MODELS = model_factory.TREE_MODELS
_SEQ_MODELS = model_factory.SEQ_MODELS


class FinalHoldoutTrainingWorkflow(_OwnerBackedService):
    def run(self, model_name: str, data_manager):
        if data_manager.selection_df is None or data_manager.final_holdout_df is None:
            raise ValueError("Final holdout egitimi icin selection_df ve final_holdout_df gerekir.")
        if data_manager.final_holdout_df.empty:
            raise ValueError("Final holdout seti bos.")

        tensors = data_manager.prepare_tensors(data_manager.selection_df, data_manager.final_holdout_df)
        cls = self._model_class_for_name(model_name)

        if model_name == "Prophet":
            model = self._make_prophet()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
        elif model_name in _TREE_MODELS:
            model = cls()
            model.train(tensors["X_train_s"], tensors["y_train_s"])
        elif model_name in _SEQ_MODELS:
            if not self._has_min_sequences(len(tensors["X_train_seq"]), model_name, "final holdout train"):
                raise ValueError(f"{model_name} final holdout egitimi icin sequence sayisi yetersiz.")
            if model_name == "LSTM":
                model = self._make_lstm("final")
            elif model_name == "LSTM Lite":
                model = self._make_lstm_lite("final")
            elif model_name == "AttentionLSTM v2":
                model = self._make_attention_lstm_v2("final")
            else:
                model = cls()
            model.train(tensors["X_train_seq"], tensors["y_train_seq"])
        else:
            model = cls()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])

        self.final_holdout_model_name = model_name
        self.final_holdout_model = model
        return model, tensors



class SingleSplitTrainingWorkflow(_OwnerBackedService):
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
        self._train_sequence_baselines(tensors)
        self._train_prophet_if_selected(tensors)
        self._train_tuned_tree_models(tensors, optuna_storage)
        self._train_deep_models(tensors)

    def _train_baseline_models(self, tensors: dict) -> None:
        for name, cls in self._baseline_specs():
            model = cls()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
            self.trained_models[name] = model

        if not self._skip("ARIMA"):
            model = self._make_arima()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
            self.trained_models["ARIMA"] = model

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

    def _train_sequence_baselines(self, tensors: dict) -> None:
        for name, cls in self._sequence_baseline_specs():
            if self._skip(name):
                continue
            if self._has_min_sequences(len(tensors["X_train_seq"]), name, "train"):
                model = cls()
                model.train(tensors["X_train_seq"], tensors["y_train_seq"])
                self.trained_models[name] = model

    def _train_prophet_if_selected(self, tensors: dict) -> None:
        if self._skip("Prophet") or "Prophet" not in self.selected_models:
            return
        try:
            prophet = self._make_prophet()
            prophet.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
            self.trained_models["Prophet"] = prophet
        except Exception as exc:
            print(f"  [WARN] Prophet egitimi basarisiz, atlaniyor: {exc}")

    def _train_tuned_tree_models(self, tensors: dict, optuna_storage: str) -> None:
        specs = (
            ("XGBoost", "XGBoostModel", f"xgb_{self.stock_symbol}"),
            ("Random Forest", "RandomForestModel", f"rf_{self.stock_symbol}"),
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

    def _train_deep_models(self, tensors: dict) -> None:
        specs = (
            ("LSTM", lambda: self._make_lstm("single")),
            ("LSTM Lite", lambda: self._make_lstm_lite("single")),
            ("AttentionLSTM v2", lambda: self._make_attention_lstm_v2("single")),
        )
        for model_name, factory in specs:
            if self._skip(model_name):
                continue
            if self._has_min_sequences(len(tensors["X_train_seq"]), model_name, "train"):
                model = factory()
                model.train(tensors["X_train_seq"], tensors["y_train_seq"])
                self.trained_models[model_name] = model



class WalkForwardTrainingWorkflow(_OwnerBackedService):
    def run(self, wf_splits: list, data_manager):
        preprocessors = {
            "baseline": self._preprocessor(data_manager, "baseline"),
            "tree": self._preprocessor(data_manager, "tree"),
            "seq": self._preprocessor(data_manager, "seq"),
        }
        validators = {}
        self._run_standard_validators(wf_splits, data_manager, preprocessors, validators)
        self._run_tuned_tree_validators(wf_splits, preprocessors["tree"], validators)
        self._run_deep_validators(wf_splits, data_manager, preprocessors["seq"], validators)
        self._materialize_validator_results(validators)
        self._dump_feature_importances(validators)
        self._log_walk_forward_runs()

    def _preprocessor(self, data_manager, kind: str):
        def prepare(train_df, test_df, context_df=None):
            tensors = data_manager.prepare_tensors(train_df, test_df, context_df=context_df)
            if kind == "baseline":
                x_train, y_train = tensors["X_train"], tensors["y_train"]
                x_test, y_test, scaler_y = tensors["X_test"], tensors["y_test"], None
            elif kind == "tree":
                x_train, y_train = tensors["X_train_s"], tensors["y_train_s"]
                x_test, y_test, scaler_y = tensors["X_test_s"], tensors["y_test_s"], tensors["scaler_y"]
            else:
                x_train, y_train = tensors["X_train_seq"], tensors["y_train_seq"]
                x_test, y_test, scaler_y = tensors["X_test_seq"], tensors["y_test_seq"], tensors["scaler_y"]
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

    def _run_standard_validators(self, wf_splits, data_manager, preprocessors, validators) -> None:
        for name, cls in self._baseline_specs():
            self._wf_run(name, cls, preprocessors["baseline"], wf_splits, validators)
        self._run_prophet_and_arima(wf_splits, preprocessors["baseline"], validators)
        self._run_tabular_baselines(wf_splits, preprocessors["tree"], validators)
        self._run_sequence_baselines(wf_splits, data_manager, preprocessors["seq"], validators)

    def _run_prophet_and_arima(self, wf_splits, preprocessor, validators) -> None:
        if not self._skip("Prophet"):
            try:
                self._wf_run("Prophet", self._make_prophet, preprocessor, wf_splits, validators, skip_import_err=True)
            except Exception as exc:
                print(f"  [WARN] Prophet walk-forward calistirilamadi: {exc}")
        if not self._skip("ARIMA"):
            self._wf_run("ARIMA", self._make_arima, preprocessor, wf_splits, validators)

    def _run_tabular_baselines(self, wf_splits, preprocessor, validators) -> None:
        for name, cls in self._linear_baseline_specs():
            if not self._skip(name):
                self._wf_run(name, cls, preprocessor, wf_splits, validators)
        for name, cls in self._boosting_baseline_specs():
            if not self._skip(name):
                self._wf_run(name, cls, preprocessor, wf_splits, validators, skip_import_err=True)

    def _run_sequence_baselines(self, wf_splits, data_manager, preprocessor, validators) -> None:
        for name, cls in self._sequence_baseline_specs():
            if self._skip(name):
                continue
            if self._wf_has_min_sequences(wf_splits, data_manager, name):
                self._wf_run(name, cls, preprocessor, wf_splits, validators)

    def _run_tuned_tree_validators(self, wf_splits, preprocessor, validators) -> None:
        specs = (
            ("XGBoost", "XGBoostModel", {"early_stopping_rounds": 50}),
            ("Random Forest", "RandomForestModel", {}),
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

    def _run_deep_validators(self, wf_splits, data_manager, preprocessor, validators) -> None:
        specs = (
            ("LSTM", lambda: self._make_lstm("wf")),
            ("LSTM Lite", lambda: self._make_lstm_lite("wf")),
            ("AttentionLSTM v2", lambda: self._make_attention_lstm_v2("wf")),
        )
        for model_name, factory in specs:
            if self._skip(model_name):
                continue
            if self._wf_has_min_sequences(wf_splits, data_manager, model_name):
                self._wf_run(model_name, factory, preprocessor, wf_splits, validators)

    def _materialize_validator_results(self, validators: dict) -> None:
        for name, validator in validators.items():
            self.wf_results[name] = validator.aggregated_metrics
            self.wf_fold_metrics[name] = [
                {"Model": name, "Fold": window["split_idx"], **window["metrics"]}
                for window in validator.results
            ]
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

    def _log_walk_forward_runs(self) -> None:
        for model_name, metrics in self.wf_results.items():
            metrics["Threshold_Config"] = str(self.dataset_metadata.get("signal_threshold_config", {}))
            self.tracker.log_run(
                model_name,
                {"validation": "walk_forward"},
                metrics,
                self.feature_names,
                self.dataset_hash,
                self.dataset_metadata,
            )

