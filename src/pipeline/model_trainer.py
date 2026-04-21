# -*- coding: utf-8 -*-
"""
model_trainer.py - Model Egitim Orkestratoru
"""

import numpy as np

from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import AttentionLSTMModel
from src.models.naive_model import NaiveDriftModel, NaiveLastValueModel, NaiveZeroReturnModel
from src.models.prophet_model import ProphetModel
from src.models.random_forest_model import RandomForestModel
from src.models.tft_model import TFTModel
from src.models.xgboost_model import XGBoostModel
from src.validation.walk_forward import WalkForwardValidator

_ALL_MODELS = ["Prophet", "XGBoost", "Random Forest", "LSTM", "TFT"]
_BASELINE_MODELS = ["Naive Last Value", "Naive Zero Return", "Naive Drift", "ARIMA"]


class ModelTrainer:
    def __init__(
        self,
        stock_symbol: str,
        tracker: ExperimentTracker,
        registry: ModelRegistry,
        feature_names: list,
        selected_models: list = None,
        dataset_hash: str = "N/A",
        dataset_metadata: dict | None = None,
        registry_version: str = "v5",
    ):
        self.stock_symbol = stock_symbol
        self.tracker = tracker
        self.registry = registry
        self.feature_names = feature_names
        self.selected_models = set(selected_models) if selected_models else set(_ALL_MODELS)
        self.dataset_hash = dataset_hash
        self.dataset_metadata = dataset_metadata or {}
        self.registry_version = registry_version

        self.trained_models = {}
        self.wf_results = {}
        self.wf_predictions = {}
        self.wf_backtest_inputs = {}
        self.wf_y_true = None

    def _skip(self, name: str) -> bool:
        if name in _BASELINE_MODELS:
            return False
        if name not in self.selected_models:
            print(f"  [--] {name} atlandi (secilmedi).")
            return True
        return False

    def _baseline_specs(self):
        target_mode = self.dataset_metadata.get("target_mode", "log_return")
        specs = [
            ("Naive Last Value", NaiveLastValueModel),
            ("Naive Drift", NaiveDriftModel),
            ("ARIMA", ARIMAModel),
        ]
        if target_mode in {"return", "log_return"}:
            specs.insert(1, ("Naive Zero Return", NaiveZeroReturnModel))
        return specs

    def train_single_split(self, tensors: dict):
        for name, cls in self._baseline_specs():
            model = cls()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
            self.trained_models[name] = model

        if not self._skip("Prophet"):
            try:
                prophet = ProphetModel(yearly_seasonality=True, weekly_seasonality=True)
                prophet.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
                self.trained_models["Prophet"] = prophet
            except Exception as exc:
                print(f"  [WARN] Prophet egitimi basarisiz, atlaniyor: {exc}")

        if not self._skip("XGBoost"):
            xgb = XGBoostModel()
            xgb.tune_and_train(tensors["X_train_s"], tensors["y_train_s"], n_trials=5, n_splits=3)
            self.trained_models["XGBoost"] = xgb

        if not self._skip("Random Forest"):
            rf = RandomForestModel()
            rf.tune_and_train(tensors["X_train_s"], tensors["y_train_s"], n_trials=5, n_splits=3)
            self.trained_models["Random Forest"] = rf

        if not self._skip("LSTM"):
            lstm = AttentionLSTMModel(epochs=80)
            lstm.train(tensors["X_train_seq"], tensors["y_train_seq"])
            self.trained_models["LSTM"] = lstm

        if not self._skip("TFT"):
            tft = TFTModel(epochs=80, patience=15)
            tft.train(tensors["X_train_seq"], tensors["y_train_seq"])
            self.trained_models["TFT"] = tft

    def train_walk_forward(self, wf_splits: list, data_manager):
        def preprocessor_baseline(train_df, test_df):
            t = data_manager.prepare_tensors(train_df, test_df)
            return (
                t["X_train"], t["y_train"],
                t["X_test"], t["y_test"],
                None,
                t["original_y_test_aligned"],
                t["prev_close_test"],
                t["dates_test"],
            )

        def preprocessor_tree(train_df, test_df):
            t = data_manager.prepare_tensors(train_df, test_df)
            return (
                t["X_train_s"], t["y_train_s"],
                t["X_test_s"], t["y_test_s"],
                t["scaler_y"],
                t["original_y_test_aligned"],
                t["prev_close_test"],
                t["dates_test"],
            )

        def preprocessor_seq(train_df, test_df):
            t = data_manager.prepare_tensors(train_df, test_df)
            return (
                t["X_train_seq"], t["y_train_seq"],
                t["X_test_seq"], t["y_test_seq"],
                t["scaler_y"],
                t["original_y_test_aligned"],
                t["prev_close_test"],
                t["dates_test"],
            )

        if "Prophet" in self.selected_models:
            print("  [WARN] Prophet, walk-forward modunda desteklenmiyor. Atlaniyor.")

        validators = {}

        for name, cls in self._baseline_specs():
            validator = WalkForwardValidator(
                cls,
                preprocessor_baseline,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            validator.run(wf_splits)
            validators[name] = validator

        if not self._skip("XGBoost"):
            validator = WalkForwardValidator(
                XGBoostModel,
                preprocessor_tree,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            validator.run(wf_splits)
            validators["XGBoost"] = validator

        if not self._skip("Random Forest"):
            validator = WalkForwardValidator(
                RandomForestModel,
                preprocessor_tree,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            validator.run(wf_splits)
            validators["Random Forest"] = validator

        if not self._skip("LSTM"):
            validator = WalkForwardValidator(
                lambda: AttentionLSTMModel(epochs=50),
                preprocessor_seq,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            validator.run(wf_splits)
            validators["LSTM"] = validator

        if not self._skip("TFT"):
            validator = WalkForwardValidator(
                lambda: TFTModel(epochs=50, patience=12),
                preprocessor_seq,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            validator.run(wf_splits)
            validators["TFT"] = validator

        for name, validator in validators.items():
            self.wf_results[name] = validator.aggregated_metrics

            all_preds, all_trues = [], []
            all_dates, all_prev_close, all_pred_target = [], [], []
            for window in validator.results:
                all_preds.extend(window["y_pred_price"])
                all_trues.extend(window["y_true_price"])
                all_dates.extend(window["dates"])
                all_prev_close.extend(window["prev_close"])
                all_pred_target.extend(window["y_pred_target"])

            self.wf_predictions[name] = np.asarray(all_preds, dtype=float)
            self.wf_y_true = np.asarray(all_trues, dtype=float)
            self.wf_backtest_inputs[name] = {
                "dates": np.asarray(all_dates),
                "y_true_price": np.asarray(all_trues, dtype=float),
                "pred_price": np.asarray(all_preds, dtype=float),
                "prev_close": np.asarray(all_prev_close, dtype=float),
                "pred_target": np.asarray(all_pred_target, dtype=float),
            }

        for model_name, metrics in self.wf_results.items():
            self.tracker.log_run(
                model_name,
                {"validation": "walk_forward"},
                metrics,
                self.feature_names,
                self.dataset_hash,
                self.dataset_metadata,
            )
            self.registry.register(
                model_name,
                f"{self.registry_version}_wf",
                self.feature_names,
                metrics,
                "none",
                self.dataset_hash,
                self.dataset_metadata,
            )
