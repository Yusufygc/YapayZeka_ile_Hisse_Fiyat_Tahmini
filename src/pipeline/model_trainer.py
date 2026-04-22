# -*- coding: utf-8 -*-
"""
model_trainer.py - Model Egitim Orkestratoru
"""

import numpy as np

from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry
from src.models.arima_model import ARIMAModel
from src.models.gradient_boosting_model import LightGBMReturnModel
from src.models.linear_model import ElasticNetReturnModel, RidgeReturnModel
from src.models.linear_sequence_model import DLinearSequenceModel, NLinearSequenceModel, PatchTSTExperimentalModel
from src.models.lstm_model import AttentionLSTMModel
from src.models.naive_model import NaiveDriftModel, NaiveLastValueModel, NaiveZeroReturnModel
from src.models.prophet_model import ProphetModel
from src.models.random_forest_model import RandomForestModel
from src.models.tft_model import TFTModel
from src.models.xgboost_model import XGBoostModel
from src.validation.walk_forward import WalkForwardValidator

_ALL_MODELS = ["Prophet", "XGBoost", "Random Forest", "LightGBM Return", "LSTM", "TFT", "DLinear", "NLinear", "PatchTST Experimental"]
_BASELINE_MODELS = [
    "Naive Last Value",
    "Naive Zero Return",
    "Naive Drift",
    "ARIMA",
    "Ridge Return",
    "ElasticNet Return",
    "LightGBM Return",
    "DLinear",
    "NLinear",
    "PatchTST Experimental",
]
_TREE_MODELS = {"XGBoost", "Random Forest", "Ridge Return", "ElasticNet Return", "LightGBM Return"}
_SEQ_MODELS = {"LSTM", "TFT", "DLinear", "NLinear", "PatchTST Experimental"}


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
        model_config: dict | None = None,
    ):
        self.stock_symbol = stock_symbol
        self.tracker = tracker
        self.registry = registry
        self.feature_names = feature_names
        self.selected_models = set(selected_models) if selected_models else set(_ALL_MODELS)
        self.dataset_hash = dataset_hash
        self.dataset_metadata = dataset_metadata or {}
        self.registry_version = registry_version
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

    @staticmethod
    def _build_deep_config(config: dict) -> dict:
        default = {
            "min_sequence_samples": 64,
            "validation_ratio": 0.1,
            "min_validation_samples": 32,
            "lstm": {
                "epochs_single": 80,
                "epochs_wf": 50,
                "epochs_final": 50,
                "patience": 15,
                "lr_patience": 5,
                "dropout": 0.2,
                "batch_size": 32,
            },
            "tft": {
                "epochs_single": 80,
                "epochs_wf": 50,
                "epochs_final": 50,
                "patience_single": 15,
                "patience_wf": 12,
                "patience_final": 12,
                "lr_patience": 5,
                "dropout": 0.3,
                "batch_size": 32,
            },
            "patchtst": {"patch_length": 16, "stride": 8, "alpha": 1.0},
        }
        merged = dict(default)
        merged.update({key: value for key, value in config.items() if key not in {"lstm", "tft"}})
        for section in ("lstm", "tft"):
            section_cfg = dict(default[section])
            section_cfg.update(config.get(section, {}))
            merged[section] = section_cfg
        return merged

    def _arima_config(self) -> dict:
        return self.model_config.get("arima", {})

    def _make_prophet(self) -> ProphetModel:
        cfg = self.model_config.get("prophet", {})
        return ProphetModel(
            yearly_seasonality=True,
            weekly_seasonality=True,
            use_regressors=bool(cfg.get("use_regressors", False)),
            regressor_names=cfg.get("regressor_names"),
            feature_names=self.feature_names,
        )

    def _make_arima(self) -> ARIMAModel:
        cfg = self._arima_config()
        return ARIMAModel(
            order=tuple(cfg.get("order", (1, 0, 0))),
            auto_order=bool(cfg.get("auto_order", False)),
            candidate_orders=[tuple(order) for order in cfg.get("candidate_orders", [])] or None,
        )

    def _make_lstm(self, stage: str) -> AttentionLSTMModel:
        cfg = self.deep_config["lstm"]
        return AttentionLSTMModel(
            epochs=int(cfg.get(f"epochs_{stage}", cfg.get("epochs_single", 80))),
            patience=int(cfg.get("patience", 15)),
            dropout_rate=float(cfg.get("dropout", 0.2)),
            batch_size=int(cfg.get("batch_size", 32)),
            lr_patience=int(cfg.get("lr_patience", 5)),
            validation_ratio=float(self.deep_config.get("validation_ratio", 0.1)),
            min_val_samples=int(self.deep_config.get("min_validation_samples", 32)),
        )

    def _make_tft(self, stage: str) -> TFTModel:
        cfg = self.deep_config["tft"]
        return TFTModel(
            epochs=int(cfg.get(f"epochs_{stage}", cfg.get("epochs_single", 80))),
            patience=int(cfg.get(f"patience_{stage}", cfg.get("patience_single", 15))),
            dropout=float(cfg.get("dropout", 0.3)),
            batch_size=int(cfg.get("batch_size", 32)),
            lr_patience=int(cfg.get("lr_patience", 5)),
            validation_ratio=float(self.deep_config.get("validation_ratio", 0.1)),
            min_val_samples=int(self.deep_config.get("min_validation_samples", 32)),
        )

    def _has_min_sequences(self, count: int, model_name: str, context: str) -> bool:
        min_seq = int(self.deep_config.get("min_sequence_samples", 64))
        if count < min_seq:
            print(f"  [WARN] {model_name} atlandi: {context} sequence sayisi {count} < {min_seq}.")
            return False
        return True

    def _wf_has_min_sequences(self, wf_splits: list, data_manager, model_name: str) -> bool:
        min_seq = int(self.deep_config.get("min_sequence_samples", 64))
        min_fold_seq = min(max(0, len(split["train"]) - data_manager.time_steps) for split in wf_splits) if wf_splits else 0
        if min_fold_seq < min_seq:
            print(f"  [WARN] {model_name} walk-forward atlandi: en kucuk fold sequence sayisi {min_fold_seq} < {min_seq}.")
            return False
        return True

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
            ("ARIMA", self._make_arima),
        ]
        if target_mode in {"return", "log_return"}:
            specs.insert(1, ("Naive Zero Return", NaiveZeroReturnModel))
        return specs

    def _linear_baseline_specs(self):
        return [
            ("Ridge Return", RidgeReturnModel),
            ("ElasticNet Return", ElasticNetReturnModel),
        ]

    def _boosting_baseline_specs(self):
        return [("LightGBM Return", LightGBMReturnModel)]

    def _sequence_baseline_specs(self):
        patch_cfg = self.model_config.get("experimental_sequence_baselines", {}).get("patchtst_config", {})
        return [
            ("DLinear", DLinearSequenceModel),
            ("NLinear", NLinearSequenceModel),
            (
                "PatchTST Experimental",
                lambda: PatchTSTExperimentalModel(
                    alpha=float(patch_cfg.get("alpha", 1.0)),
                    patch_length=int(patch_cfg.get("patch_length", 16)),
                    stride=int(patch_cfg.get("stride", 8)),
                ),
            ),
        ]

    def _model_class_for_name(self, model_name: str):
        mapping = {name: cls for name, cls in self._baseline_specs()}
        mapping.update({name: cls for name, cls in self._linear_baseline_specs()})
        mapping.update({name: cls for name, cls in self._boosting_baseline_specs()})
        mapping.update({name: cls for name, cls in self._sequence_baseline_specs()})
        mapping.update({
            "Prophet": ProphetModel,
            "XGBoost": XGBoostModel,
            "Random Forest": RandomForestModel,
            "LSTM": AttentionLSTMModel,
            "TFT": TFTModel,
        })
        if model_name not in mapping:
            raise KeyError(f"Bilinmeyen model adi: {model_name}")
        return mapping[model_name]

    def train_final_holdout_model(self, model_name: str, data_manager):
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
            elif model_name == "TFT":
                model = self._make_tft("final")
            else:
                model = cls()
            model.train(tensors["X_train_seq"], tensors["y_train_seq"])
        else:
            model = cls()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])

        self.final_holdout_model_name = model_name
        self.final_holdout_model = model
        return model, tensors

    def train_single_split(self, tensors: dict):
        for name, cls in self._baseline_specs():
            model = cls()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
            self.trained_models[name] = model

        for name, cls in self._linear_baseline_specs():
            model = cls()
            model.train(tensors["X_train_s"], tensors["y_train_s"])
            self.trained_models[name] = model

        for name, cls in self._boosting_baseline_specs():
            try:
                model = cls()
                model.train(tensors["X_train_s"], tensors["y_train_s"])
                self.trained_models[name] = model
            except ImportError as exc:
                print(f"  [WARN] {name} atlandi: {exc}")

        for name, cls in self._sequence_baseline_specs():
            if self._has_min_sequences(len(tensors["X_train_seq"]), name, "train"):
                model = cls()
                model.train(tensors["X_train_seq"], tensors["y_train_seq"])
                self.trained_models[name] = model

        if not self._skip("Prophet"):
            try:
                prophet = self._make_prophet()
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
            if self._has_min_sequences(len(tensors["X_train_seq"]), "LSTM", "train"):
                lstm = self._make_lstm("single")
                lstm.train(tensors["X_train_seq"], tensors["y_train_seq"])
                self.trained_models["LSTM"] = lstm

        if not self._skip("TFT"):
            if self._has_min_sequences(len(tensors["X_train_seq"]), "TFT", "train"):
                tft = self._make_tft("single")
                tft.train(tensors["X_train_seq"], tensors["y_train_seq"])
                self.trained_models["TFT"] = tft

    def train_walk_forward(self, wf_splits: list, data_manager):
        def preprocessor_baseline(train_df, test_df, context_df=None):
            t = data_manager.prepare_tensors(train_df, test_df, context_df=context_df)
            return (
                t["X_train"], t["y_train"],
                t["X_test"], t["y_test"],
                None,
                t["original_y_test_aligned"],
                t["prev_close_test"],
                t["dates_test"],
                t["dates_prediction"],
                t["y_test"],
            )

        def preprocessor_tree(train_df, test_df, context_df=None):
            t = data_manager.prepare_tensors(train_df, test_df, context_df=context_df)
            return (
                t["X_train_s"], t["y_train_s"],
                t["X_test_s"], t["y_test_s"],
                t["scaler_y"],
                t["original_y_test_aligned"],
                t["prev_close_test"],
                t["dates_test"],
                t["dates_prediction"],
                t["y_test"],
            )

        def preprocessor_seq(train_df, test_df, context_df=None):
            t = data_manager.prepare_tensors(train_df, test_df, context_df=context_df)
            return (
                t["X_train_seq"], t["y_train_seq"],
                t["X_test_seq"], t["y_test_seq"],
                t["scaler_y"],
                t["original_y_test_aligned"],
                t["prev_close_test"],
                t["dates_test"],
                t["dates_prediction"],
                t["y_test"],
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

        for name, cls in self._linear_baseline_specs():
            validator = WalkForwardValidator(
                cls,
                preprocessor_tree,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            validator.run(wf_splits)
            validators[name] = validator

        for name, cls in self._boosting_baseline_specs():
            try:
                validator = WalkForwardValidator(
                    cls,
                    preprocessor_tree,
                    target_mode=self.dataset_metadata.get("target_mode", "log_return"),
                )
                validator.run(wf_splits)
                validators[name] = validator
            except ImportError as exc:
                print(f"  [WARN] {name} walk-forward atlandi: {exc}")

        for name, cls in self._sequence_baseline_specs():
            if self._wf_has_min_sequences(wf_splits, data_manager, name):
                validator = WalkForwardValidator(
                    cls,
                    preprocessor_seq,
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
            if self._wf_has_min_sequences(wf_splits, data_manager, "LSTM"):
                validator = WalkForwardValidator(
                    lambda: self._make_lstm("wf"),
                    preprocessor_seq,
                    target_mode=self.dataset_metadata.get("target_mode", "log_return"),
                )
                validator.run(wf_splits)
                validators["LSTM"] = validator

        if not self._skip("TFT"):
            if self._wf_has_min_sequences(wf_splits, data_manager, "TFT"):
                validator = WalkForwardValidator(
                    lambda: self._make_tft("wf"),
                    preprocessor_seq,
                    target_mode=self.dataset_metadata.get("target_mode", "log_return"),
                )
                validator.run(wf_splits)
                validators["TFT"] = validator

        for name, validator in validators.items():
            self.wf_results[name] = validator.aggregated_metrics
            self.wf_fold_metrics[name] = [
                {
                    "Model": name,
                    "Fold": window["split_idx"],
                    **window["metrics"],
                }
                for window in validator.results
            ]

            all_preds, all_trues = [], []
            all_dates, all_prediction_dates, all_prev_close, all_pred_target, all_true_target, all_fold_ids = [], [], [], [], [], []
            for window in validator.results:
                all_preds.extend(window["y_pred_price"])
                all_trues.extend(window["y_true_price"])
                all_dates.extend(window["dates"])
                all_prediction_dates.extend(window["prediction_dates"])
                all_prev_close.extend(window["prev_close"])
                all_pred_target.extend(window["y_pred_target"])
                all_true_target.extend(window["y_true_target"])
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
            }

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
            self.registry.register(
                model_name,
                f"{self.registry_version}_wf",
                self.feature_names,
                metrics,
                "none",
                self.dataset_hash,
                self.dataset_metadata,
            )
