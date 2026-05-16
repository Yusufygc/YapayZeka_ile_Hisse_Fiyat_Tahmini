# -*- coding: utf-8 -*-
"""Model construction helpers used by ``ModelTrainer``.

The trainer remains the orchestration facade; this module owns the factory
mapping and stage-specific model initialization details.
"""

from __future__ import annotations

from src.models.arima_model import ARIMAModel
from src.models.gradient_boosting_model import LightGBMReturnModel
from src.models.linear_model import ElasticNetReturnModel, RidgeReturnModel
from src.models.linear_sequence_model import DLinearSequenceModel, NLinearSequenceModel
from src.models.lstm_model import AttentionLSTMModel
from src.models.naive_model import NaiveDriftModel, NaiveLastValueModel, NaiveZeroReturnModel
from src.models.random_forest_model import RandomForestModel
from src.models.tft_v2 import TFTModel
from src.models.xgboost_model import XGBoostModel
from src.pipeline.model_scope import BENCHMARK_MODELS, DEFAULT_CANDIDATE_MODELS

ALL_MODELS = list(DEFAULT_CANDIDATE_MODELS)
OPTIONAL_MODELS = ["Random Forest", "LightGBM Return"]
LEGACY_MODELS = ["ARIMA", "Prophet"]
BENCHMARK_MODEL_SET = set(BENCHMARK_MODELS)
TREE_MODELS = {"XGBoost", "Random Forest", "Ridge Return", "ElasticNet Return", "LightGBM Return"}
SEQ_MODELS = {"LSTM", "TFT", "DLinear", "NLinear"}


def build_deep_config(config: dict) -> dict:
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
    }
    merged = dict(default)
    merged.update({key: value for key, value in config.items() if key not in {"lstm", "tft"}})
    for section in ("lstm", "tft"):
        section_cfg = dict(default[section])
        section_cfg.update(config.get(section, {}))
        merged[section] = section_cfg
    return merged


def arima_config(model_config: dict) -> dict:
    return model_config.get("arima", {})


def make_prophet(model_config: dict, feature_names: list):
    from src.models.prophet_model import ProphetModel  # optional dependency

    cfg = model_config.get("prophet", {})
    return ProphetModel(
        yearly_seasonality=True,
        weekly_seasonality=True,
        use_regressors=bool(cfg.get("use_regressors", False)),
        regressor_names=cfg.get("regressor_names"),
        feature_names=feature_names,
    )


def make_arima(model_config: dict) -> ARIMAModel:
    cfg = arima_config(model_config)
    return ARIMAModel(
        order=tuple(cfg.get("order", (1, 0, 0))),
        auto_order=bool(cfg.get("auto_order", False)),
        candidate_orders=[tuple(order) for order in cfg.get("candidate_orders", [])] or None,
    )


def make_lstm(deep_config: dict, stage: str) -> AttentionLSTMModel:
    cfg = deep_config["lstm"]
    return AttentionLSTMModel(
        epochs=int(cfg.get(f"epochs_{stage}", cfg.get("epochs_single", 80))),
        patience=int(cfg.get("patience", 15)),
        dropout_rate=float(cfg.get("dropout", 0.2)),
        batch_size=int(cfg.get("batch_size", 32)),
        lr_patience=int(cfg.get("lr_patience", 5)),
        validation_ratio=float(deep_config.get("validation_ratio", 0.1)),
        min_val_samples=int(deep_config.get("min_validation_samples", 32)),
    )


def make_tft(deep_config: dict, stage: str) -> TFTModel:
    cfg = deep_config["tft"]
    return TFTModel(
        epochs=int(cfg.get(f"epochs_{stage}", cfg.get("epochs_single", 80))),
        patience=int(cfg.get(f"patience_{stage}", cfg.get("patience_single", 15))),
        dropout=float(cfg.get("dropout", 0.3)),
        batch_size=int(cfg.get("batch_size", 32)),
        lr_patience=int(cfg.get("lr_patience", 5)),
        validation_ratio=float(deep_config.get("validation_ratio", 0.1)),
        min_val_samples=int(deep_config.get("min_validation_samples", 32)),
    )


def benchmark_specs(target_mode: str):
    specs = [
        ("Naive Last Value", NaiveLastValueModel),
        ("Naive Drift", NaiveDriftModel),
    ]
    if target_mode in {"return", "log_return"}:
        specs.insert(1, ("Naive Zero Return", NaiveZeroReturnModel))
    return specs


def linear_baseline_specs():
    return [
        ("Ridge Return", RidgeReturnModel),
        ("ElasticNet Return", ElasticNetReturnModel),
    ]


def boosting_baseline_specs():
    return [("LightGBM Return", LightGBMReturnModel)]


def sequence_baseline_specs():
    return [
        ("DLinear", DLinearSequenceModel),
        ("NLinear", NLinearSequenceModel),
    ]


def model_class_for_name(model_name: str, model_config: dict, target_mode: str):
    mapping = {name: cls for name, cls in benchmark_specs(target_mode)}
    mapping["ARIMA"] = lambda: make_arima(model_config)
    mapping.update({name: cls for name, cls in linear_baseline_specs()})
    mapping.update({name: cls for name, cls in boosting_baseline_specs()})
    mapping.update({name: cls for name, cls in sequence_baseline_specs()})
    mapping.update({
        "XGBoost": XGBoostModel,
        "Random Forest": RandomForestModel,
        "LSTM": AttentionLSTMModel,
        "TFT": TFTModel,
    })
    try:
        from src.models.prophet_model import ProphetModel

        mapping["Prophet"] = ProphetModel
    except ImportError:
        pass
    if model_name not in mapping:
        raise KeyError(f"Bilinmeyen model adi: {model_name}")
    return mapping[model_name]
