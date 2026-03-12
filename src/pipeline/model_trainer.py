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

from src.validation.walk_forward import WalkForwardValidator
from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry

class ModelTrainer:
    def __init__(self, stock_symbol: str, tracker: ExperimentTracker, registry: ModelRegistry, feature_names: list):
        self.stock_symbol = stock_symbol
        self.tracker = tracker
        self.registry = registry
        self.feature_names = feature_names
        
        self.trained_models = {}
        self.wf_results = {}

    def train_single_split(self, tensors: dict):
        # Prophet
        prophet = ProphetModel(yearly_seasonality=True, weekly_seasonality=True)
        prophet.train(tensors["X_train"], tensors["y_train"], dates_train=tensors["dates_train"])
        self.trained_models["Prophet"] = prophet
        
        # XGBoost
        xgb = XGBoostModel()
        xgb_params = xgb.tune_and_train(tensors["X_train_s"], tensors["y_train_s"], n_trials=5, n_splits=3)
        self.trained_models["XGBoost"] = xgb
        
        # Random Forest
        rf = RandomForestModel()
        rf_params = rf.tune_and_train(tensors["X_train_s"], tensors["y_train_s"], n_trials=5, n_splits=3)
        self.trained_models["Random Forest"] = rf
        
        # LSTM
        lstm = AttentionLSTMModel(epochs=20)
        lstm.train(tensors["X_train_seq"], tensors["y_train_seq"])
        self.trained_models["LSTM"] = lstm
        
        # TFT
        tft = TFTModel(epochs=20)
        tft.train(tensors["X_train_seq"], tensors["y_train_seq"])
        self.trained_models["TFT"] = tft

    def train_walk_forward(self, wf_splits: list, data_manager):
        def preprocessor_tree(train_df, test_df):
            t = data_manager.prepare_tensors(train_df, test_df)
            return t["X_train_s"], t["y_train_s"], t["X_test_s"], t["y_test_s"], t["scaler_y"], test_df["Close"].values
            
        def preprocessor_seq(train_df, test_df):
            t = data_manager.prepare_tensors(train_df, test_df)
            return t["X_train_seq"], t["y_train_seq"], t["X_test_seq"], t["y_test_seq"], t["scaler_y"], t["original_y_test_aligned"]
            
        wf_xgb = WalkForwardValidator(XGBoostModel, preprocessor_tree)
        wf_xgb.run(wf_splits)
        
        wf_rf = WalkForwardValidator(RandomForestModel, preprocessor_tree)
        wf_rf.run(wf_splits)
        
        wf_lstm = WalkForwardValidator(lambda: AttentionLSTMModel(epochs=5), preprocessor_seq)
        wf_lstm.run(wf_splits)
        
        wf_tft = WalkForwardValidator(lambda: TFTModel(epochs=5), preprocessor_seq)
        wf_tft.run(wf_splits)
        
        self.wf_results = {}
        self.wf_predictions = {}
        self.wf_y_true = None
        
        import numpy as np
        
        validators = {
            "XGBoost": wf_xgb,
            "Random Forest": wf_rf,
            "LSTM": wf_lstm,
            "TFT": wf_tft
        }
        
        for name, validator in validators.items():
            self.wf_results[name] = validator.aggregated_metrics
            
            all_preds = []
            all_trues = []
            for window in validator.results:
                all_preds.extend(window["y_pred"])
                all_trues.extend(window["y_true"])
                
            self.wf_predictions[name] = np.array(all_preds)
            self.wf_y_true = np.array(all_trues)
        
        # Log and Register average results
        for model_name, metrics in self.wf_results.items():
            dataset_hash = str(hash(self.stock_symbol))
            self.tracker.log_run(model_name, {"validation": "walk_forward"}, metrics, self.feature_names, dataset_hash)
            self.registry.register(model_name, "v4_wf", self.feature_names, metrics, "none", dataset_hash)
