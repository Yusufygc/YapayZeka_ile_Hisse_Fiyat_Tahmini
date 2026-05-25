# -*- coding: utf-8 -*-
"""Lightweight model package exports.

Model implementations are imported lazily so optional dependencies such as
Prophet, TensorFlow, PyTorch, LightGBM, or joblib do not break unrelated tests
or baseline-only workflows during package import.
"""

_EXPORTS = {
    "ProphetModel": ("prophet_model", "ProphetModel"),
    "ProphetHybridModel": ("prophet_hybrid_model", "ProphetHybridModel"),
    "XGBoostModel": ("xgboost_model", "XGBoostModel"),
    "AttentionLSTMModel": ("lstm_model", "AttentionLSTMModel"),
    "AttentionLSTMV2Model": ("attention_lstm_v2_model", "AttentionLSTMV2Model"),
    "LSTMLiteModel": ("lstm_lite_model", "LSTMLiteModel"),
    "RandomForestModel": ("random_forest_model", "RandomForestModel"),
    "NaiveLastValueModel": ("naive_model", "NaiveLastValueModel"),
    "NaiveZeroReturnModel": ("naive_model", "NaiveZeroReturnModel"),
    "NaiveDriftModel": ("naive_model", "NaiveDriftModel"),
    "ARIMAModel": ("arima_model", "ARIMAModel"),
    "RidgeReturnModel": ("linear_model", "RidgeReturnModel"),
    "ElasticNetReturnModel": ("linear_model", "ElasticNetReturnModel"),
    "LightGBMReturnModel": ("gradient_boosting_model", "LightGBMReturnModel"),
    "QuantileLightGBMModel": ("quantile_lightgbm_model", "QuantileLightGBMModel"),
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


# --- Plug-in kesfi (Faz 1 - additive) ------------------------------------
# `model_registry.ensure_loaded()` tarafindan bir kez cagrilir.
# Her *_model.py (ve linear_sequence_model) import edilince
# iclerindeki `register_model(ModelSpec(...))` cagrilari registry'i doldurur.
_DISCOVERED = False


def _discover_models() -> None:
    """src/models icindeki model modullerini import et -> registry tetiklensin.

    Idempotent. Optional dep eksikse modul atlanır, warning verilir.
    Mevcut lazy `_EXPORTS` davranisi bozulmaz; bu sadece registry icin ek import.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    import importlib
    import pkgutil
    import warnings

    # Discover kapsami: bu paketin dogrudan modulleri/alt paketleri.
    for mod_info in pkgutil.iter_modules(__path__):
        name = mod_info.name
        # Sadece model dosyalari ve bilinen alt paketler.
        if not (
            name.endswith("_model")
            or name == "linear_sequence_model"
        ):
            continue
        # base_model abstrakttir, kayit gerekmez.
        if name == "base_model":
            continue
        try:
            importlib.import_module(f"{__name__}.{name}")
        except ImportError as exc:
            warnings.warn(f"Model modulu atlandi ({name}): {exc}")
        except Exception as exc:  # pragma: no cover - kayit hatasi nadir
            warnings.warn(f"Model modulu import sirasinda hata ({name}): {exc}")
