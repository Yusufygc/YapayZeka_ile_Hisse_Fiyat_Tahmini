# -*- coding: utf-8 -*-
"""
model_trainer.py - Model Egitim Orkestratoru
"""

import os

import numpy as np

from src.experiments.experiment_tracker import ExperimentTracker
from src.pipeline import model_factory
from src.pipeline.model_scope import normalize_candidate_models
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

    def _make_tft(self, stage: str):
        return model_factory.make_tft(self.deep_config, stage)

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
        _optuna_storage = f"sqlite:///optuna_studies_{self.stock_symbol}.db"

        for name, cls in self._baseline_specs():
            model = cls()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
            self.trained_models[name] = model

        if not self._skip("ARIMA"):
            model = self._make_arima()
            model.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
            self.trained_models["ARIMA"] = model

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

        for name, cls in self._sequence_baseline_specs():
            if self._skip(name):
                continue
            if self._has_min_sequences(len(tensors["X_train_seq"]), name, "train"):
                model = cls()
                model.train(tensors["X_train_seq"], tensors["y_train_seq"])
                self.trained_models[name] = model

        if not self._skip("Prophet") and "Prophet" in self.selected_models:
            # Prophet yalnızca _LEGACY_MODELS içindedir; kullanıcı açıkça seçmişse çalıştır
            try:
                prophet = self._make_prophet()
                prophet.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
                self.trained_models["Prophet"] = prophet
            except Exception as exc:
                print(f"  [WARN] Prophet egitimi basarisiz, atlaniyor: {exc}")

        if not self._skip("XGBoost"):
            xgb = XGBoostModel()
            xgb.tune_and_train(
                tensors["X_train_s"], tensors["y_train_s"],
                n_trials=30, n_splits=3,
                study_storage=_optuna_storage,
                study_name=f"xgb_{self.stock_symbol}",
            )
            self.trained_models["XGBoost"] = xgb

        if not self._skip("Random Forest"):
            rf = RandomForestModel()
            rf.tune_and_train(
                tensors["X_train_s"], tensors["y_train_s"],
                n_trials=30, n_splits=3,
                study_storage=_optuna_storage,
                study_name=f"rf_{self.stock_symbol}",
            )
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
                t.get("market_regime_test", []),
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
                t.get("market_regime_test", []),
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
                t.get("market_regime_test", []),
            )

        validators = {}

        for name, cls in self._baseline_specs():
            self._wf_run(name, cls, preprocessor_baseline, wf_splits, validators)

        if not self._skip("Prophet"):
            try:
                self._wf_run(
                    "Prophet",
                    self._make_prophet,
                    preprocessor_baseline,
                    wf_splits,
                    validators,
                    skip_import_err=True,
                )
            except Exception as exc:
                print(f"  [WARN] Prophet walk-forward calistirilamadi: {exc}")

        if not self._skip("ARIMA"):
            self._wf_run("ARIMA", self._make_arima, preprocessor_baseline, wf_splits, validators)

        for name, cls in self._linear_baseline_specs():
            if self._skip(name):
                continue
            self._wf_run(name, cls, preprocessor_tree, wf_splits, validators)

        for name, cls in self._boosting_baseline_specs():
            if self._skip(name):
                continue
            self._wf_run(name, cls, preprocessor_tree, wf_splits, validators, skip_import_err=True)

        for name, cls in self._sequence_baseline_specs():
            if self._skip(name):
                continue
            if self._wf_has_min_sequences(wf_splits, data_manager, name):
                self._wf_run(name, cls, preprocessor_seq, wf_splits, validators)

        if not self._skip("XGBoost"):
            self._wf_run(
                "XGBoost",
                lambda: XGBoostModel(
                    tune_on_fit=True,
                    tune_n_trials=30,
                    tune_n_splits=3,
                    early_stopping_rounds=50,
                ),
                preprocessor_tree,
                wf_splits,
                validators,
            )

        if not self._skip("Random Forest"):
            self._wf_run(
                "Random Forest",
                lambda: RandomForestModel(
                    tune_on_fit=True,
                    tune_n_trials=30,
                    tune_n_splits=3,
                ),
                preprocessor_tree,
                wf_splits,
                validators,
            )

        if not self._skip("LSTM"):
            if self._wf_has_min_sequences(wf_splits, data_manager, "LSTM"):
                self._wf_run("LSTM", lambda: self._make_lstm("wf"), preprocessor_seq, wf_splits, validators)

        if not self._skip("TFT"):
            if self._wf_has_min_sequences(wf_splits, data_manager, "TFT"):
                self._wf_run("TFT", lambda: self._make_tft("wf"), preprocessor_seq, wf_splits, validators)

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

        self._dump_feature_importances(validators)

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
            # registry.register kaldırıldı (Faz 1.3)

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
