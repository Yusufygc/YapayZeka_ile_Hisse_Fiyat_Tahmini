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


# --- Plug-in keşfi (Faz 1 — additive) ------------------------------------
# `model_registry.ensure_loaded()` tarafından bir kez çağrılır.
# Her *_model.py (ve linear_sequence_model, tft_v2 paketi) import edilince
# içlerindeki `register_model(ModelSpec(...))` çağrıları registry'i doldurur.
_DISCOVERED = False


def _discover_models() -> None:
    """src/models içindeki model modüllerini import et → registry tetiklensin.

    Idempotent. Optional dep eksikse modül atlanır, warning verilir.
    Mevcut lazy `_EXPORTS` davranışı bozulmaz; bu sadece registry için ek import.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    import importlib
    import pkgutil
    import warnings

    # Discover kapsamı: bu paketin doğrudan modülleri/alt paketleri.
    for mod_info in pkgutil.iter_modules(__path__):
        name = mod_info.name
        # Sadece model dosyaları ve bilinen alt paketler.
        if not (
            name.endswith("_model")
            or name == "linear_sequence_model"
            or name == "tft_v2"
        ):
            continue
        # base_model abstrakttır, kayıt gerekmez.
        if name == "base_model":
            continue
        try:
            importlib.import_module(f"{__name__}.{name}")
        except ImportError as exc:
            warnings.warn(f"Model modülü atlandı ({name}): {exc}")
        except Exception as exc:  # pragma: no cover - kayıt hatası nadir
            warnings.warn(f"Model modülü import sırasında hata ({name}): {exc}")
