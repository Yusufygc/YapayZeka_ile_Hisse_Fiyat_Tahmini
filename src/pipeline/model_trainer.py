# -*- coding: utf-8 -*-
"""
model_trainer.py — Model Eğitim Orkestratörü
SRP: Yalnızca modelleri başlatmak ve "single" veya "walk-forward" stratejisine göre eğitmekten sorumludur.
"""

from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import AttentionLSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.tft_model import TFTModel
from src.models.naive_model import NaiveLastValueModel, NaiveZeroReturnModel, NaiveDriftModel
from src.models.arima_model import ARIMAModel

from src.validation.walk_forward import WalkForwardValidator
from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry

_ALL_MODELS = ["Prophet", "XGBoost", "Random Forest", "LSTM", "TFT"]
_BASELINE_MODELS = ["Naive Last Value", "Naive Zero Return", "Naive Drift", "ARIMA"]


class ModelTrainer:
    def __init__(self, stock_symbol: str, tracker: ExperimentTracker, registry: ModelRegistry,
                 feature_names: list, selected_models: list = None,
                 dataset_hash: str = "N/A", dataset_metadata: dict | None = None,
                 registry_version: str = "v5"):
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
        self.wf_y_true = None

    def _skip(self, name: str) -> bool:
        if name in _BASELINE_MODELS:
            return False
        if name not in self.selected_models:
            print(f"  [--] {name} atlandı (seçilmedi).")
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
        baseline_specs = self._baseline_specs()
        for name, cls in baseline_specs:
            model = cls()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
            self.trained_models[name] = model

        if not self._skip("Prophet"):
            try:
                prophet = ProphetModel(yearly_seasonality=True, weekly_seasonality=True)
                prophet.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
                self.trained_models["Prophet"] = prophet
            except Exception as exc:
                print(f"  [WARN] Prophet eğitimi başarısız, atlanıyor: {exc}")

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
        # v2 (H1): preprocessor sözleşmesi 7'li tuple döner:
        #   (X_train, y_train_logret_s, X_test, y_test_logret_s,
        #    scaler_y, y_test_price, prev_close_test)
        def preprocessor_tree(train_df, test_df):
            t = data_manager.prepare_tensors(train_df, test_df)
            return (
                t["X_train_s"], t["y_train_s"],
                t["X_test_s"],  t["y_test_s"],
                t["scaler_y"],
                t["original_y_test_aligned"],
                t["prev_close_test"],
            )

        def preprocessor_seq(train_df, test_df):
            t = data_manager.prepare_tensors(train_df, test_df)
            return (
                t["X_train_seq"], t["y_train_seq"],
                t["X_test_seq"],  t["y_test_seq"],
                t["scaler_y"],
                t["original_y_test_aligned"],
                t["prev_close_test"],
            )

        if "Prophet" in self.selected_models:
            print("  [WARN] Prophet, walk-forward modunda desteklenmiyor (log-getiri pipeline'ıyla uyumsuz). Atlanıyor.")

        validators = {}

        for name, cls in self._baseline_specs():
            wf_model = WalkForwardValidator(
                cls,
                preprocessor_tree,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            wf_model.run(wf_splits)
            validators[name] = wf_model

        if not self._skip("XGBoost"):
            wf_xgb = WalkForwardValidator(
                XGBoostModel,
                preprocessor_tree,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            wf_xgb.run(wf_splits)
            validators["XGBoost"] = wf_xgb

        if not self._skip("Random Forest"):
            wf_rf = WalkForwardValidator(
                RandomForestModel,
                preprocessor_tree,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            wf_rf.run(wf_splits)
            validators["Random Forest"] = wf_rf

        if not self._skip("LSTM"):
            wf_lstm = WalkForwardValidator(
                lambda: AttentionLSTMModel(epochs=50),
                preprocessor_seq,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            wf_lstm.run(wf_splits)
            validators["LSTM"] = wf_lstm

        if not self._skip("TFT"):
            wf_tft = WalkForwardValidator(
                lambda: TFTModel(epochs=50, patience=12),
                preprocessor_seq,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            wf_tft.run(wf_splits)
            validators["TFT"] = wf_tft

        import numpy as np

        for name, validator in validators.items():
            self.wf_results[name] = validator.aggregated_metrics

            all_preds, all_trues = [], []
            for window in validator.results:
                all_preds.extend(window["y_pred"])
                all_trues.extend(window["y_true"])

            self.wf_predictions[name] = np.array(all_preds)
            self.wf_y_true = np.array(all_trues)

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
