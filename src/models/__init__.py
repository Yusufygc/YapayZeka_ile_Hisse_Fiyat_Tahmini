# -*- coding: utf-8 -*-
"""
Lightweight model package exports.

Model implementations are imported lazily so optional dependencies such as
Prophet, TensorFlow, PyTorch, LightGBM, or joblib do not break unrelated tests
or baseline-only workflows during package import.
"""

_EXPORTS = {
    "ProphetModel": ("prophet_model", "ProphetModel"),
    "XGBoostModel": ("xgboost_model", "XGBoostModel"),
    "AttentionLSTMModel": ("lstm_model", "AttentionLSTMModel"),
    "RandomForestModel": ("random_forest_model", "RandomForestModel"),
    "NaiveLastValueModel": ("naive_model", "NaiveLastValueModel"),
    "NaiveZeroReturnModel": ("naive_model", "NaiveZeroReturnModel"),
    "NaiveDriftModel": ("naive_model", "NaiveDriftModel"),
    "ARIMAModel": ("arima_model", "ARIMAModel"),
    "RidgeReturnModel": ("linear_model", "RidgeReturnModel"),
    "ElasticNetReturnModel": ("linear_model", "ElasticNetReturnModel"),
    "LightGBMReturnModel": ("gradient_boosting_model", "LightGBMReturnModel"),
    "DLinearSequenceModel": ("linear_sequence_model", "DLinearSequenceModel"),
    "NLinearSequenceModel": ("linear_sequence_model", "NLinearSequenceModel"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
