# -*- coding: utf-8 -*-
"""Model construction helpers used by ``ModelTrainer``.

Faz 3: spec helper'ları (`benchmark_specs`, `linear_baseline_specs`,
`boosting_baseline_specs`, `sequence_baseline_specs`) ve set sabitleri
(`TREE_MODELS`, `SEQ_MODELS`, `BENCHMARK_MODEL_SET`, `ALL_MODELS`) registry
üzerinden türetilir. Modül-seviyesi `__getattr__` sayesinde eski isimler
import edilebilir; dinamik değerler ensure_loaded() üzerinden çözülür.

`make_arima`, `make_lstm`, `make_prophet`, `build_deep_config`,
`arima_config` helper'ları korunur — `ModelTrainer` ve `ForecastRunner`
bu yardımcılarla dedike workflow çağırır.
"""

from __future__ import annotations

from typing import Callable

from src.models.arima_model import ARIMAModel
from src.models.lstm_model import AttentionLSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel

# Faz 3: TREE_MODELS / SEQ_MODELS / BENCHMARK_MODEL_SET / ALL_MODELS artık
# `__getattr__` ile registry'den çözülür (aşağıda). Eski sabitler bu modülde
# tanımlı değil — `from src.pipeline.model_factory import TREE_MODELS` hâlâ
# çalışır (PEP 562).

OPTIONAL_MODELS = ["Random Forest", "LightGBM Return"]
LEGACY_MODELS = ["ARIMA", "Prophet"]

# Hangi registry kategorileri "tree-family" (scaled tabular) sayılır.
_TREE_CATEGORIES: frozenset[str] = frozenset({"tree", "linear_shrinkage"})
# Hangi registry kategorileri "sequence-family" sayılır.
_SEQ_CATEGORIES: frozenset[str] = frozenset({"seq", "linear_decomp"})


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
    }
    merged = dict(default)
    merged.update({key: value for key, value in config.items() if key not in {"lstm"}})
    for section in ("lstm",):
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


# --- Spec helper'ları (Faz 3 — registry-derived) ------------------------

def _spec_pair(name: str) -> tuple[str, Callable[[], object]]:
    """Registry spec → ``(name, zero_arg_factory)`` tuple."""
    from src.pipeline.model_registry import get_spec

    return (name, get_spec(name).factory)


def benchmark_specs(target_mode: str):
    """Naive benchmark sırası: Last Value, [Zero Return,] Drift.

    `Naive Zero Return` yalnızca `return`/`log_return` modlarında dahil edilir;
    `price` hedefinde target_modes filtresi onu dışarıda bırakır.
    """
    from src.pipeline.model_registry import get_spec, has_spec

    specs: list[tuple[str, Callable[[], object]]] = []
    last_spec = get_spec("Naive Last Value")
    drift_spec = get_spec("Naive Drift")
    if target_mode in last_spec.target_modes:
        specs.append(("Naive Last Value", last_spec.factory))
    if (
        target_mode in {"return", "log_return"}
        and has_spec("Naive Zero Return")
        and target_mode in get_spec("Naive Zero Return").target_modes
    ):
        specs.append(("Naive Zero Return", get_spec("Naive Zero Return").factory))
    if target_mode in drift_spec.target_modes:
        specs.append(("Naive Drift", drift_spec.factory))
    return specs


def linear_baseline_specs():
    """Lineer shrinkage modelleri (Ridge, ElasticNet)."""
    from src.pipeline.model_registry import all_specs

    return [
        (s.name, s.factory)
        for s in all_specs(category="linear_shrinkage", role="candidate")
    ]


def boosting_baseline_specs():
    """LightGBM ailesi — tree kategorisinin boosting alt-kümesi.

    Mevcut tasarımda yalnızca `LightGBM Return` boosting baseline'ı olarak
    raporlanır; XGBoost / Random Forest ayrı (tuned) yolda çalışır.
    """
    from src.pipeline.model_registry import get_spec, has_spec

    if has_spec("LightGBM Return"):
        return [("LightGBM Return", get_spec("LightGBM Return").factory)]
    return []


def sequence_baseline_specs():
    """DLinear + NLinear (linear_decomp kategorisi)."""
    from src.pipeline.model_registry import all_specs

    # Tarihi sıralama: DLinear önce, NLinear sonra.
    by_name = {s.name: s for s in all_specs(category="linear_decomp")}
    ordered = []
    for name in ("DLinear", "NLinear"):
        if name in by_name:
            ordered.append((name, by_name[name].factory))
    # Beklenmedik plug-in linear_decomp modelleri sonuna eklenir.
    for name, spec in by_name.items():
        if name not in ("DLinear", "NLinear"):
            ordered.append((name, spec.factory))
    return ordered


def model_class_for_name(model_name: str, model_config: dict, target_mode: str):
    """Registry-proxy (Faz 2).

    Davranış:
      * Tree/linear/sequence/naive modeller için sıfır-arg ``cls()`` çağrılabilir.
      * ARIMA için config-aware factory ``arima`` alt-sözlüğünü unpack eder.
      * Bilinmeyen model adında ``KeyError`` (orijinal mesaj).
      * `target_modes` desteklemeyen modeller yine ``KeyError`` döndürür.
    """
    from src.pipeline.model_registry import ensure_loaded, get_spec, has_spec

    ensure_loaded()
    if not has_spec(model_name):
        raise KeyError(f"Bilinmeyen model adi: {model_name}")
    spec = get_spec(model_name)
    if target_mode not in spec.target_modes:
        raise KeyError(f"Bilinmeyen model adi: {model_name}")

    kwargs = {k: model_config.get(k, {}) for k in spec.needs_config_keys}
    if kwargs:
        return lambda: spec.factory(**kwargs)
    return lambda: spec.factory()


# --- Dinamik set sabitleri (Faz 3) --------------------------------------

def _tree_model_set() -> set[str]:
    from src.pipeline.model_registry import all_specs

    return {s.name for s in all_specs() if s.category in _TREE_CATEGORIES}


def _seq_model_set() -> set[str]:
    from src.pipeline.model_registry import all_specs

    return {s.name for s in all_specs() if s.category in _SEQ_CATEGORIES}


def _benchmark_model_set() -> set[str]:
    from src.pipeline.model_registry import all_specs

    return {s.name for s in all_specs(role="benchmark")}


def _all_models_list() -> list[str]:
    from src.pipeline.model_scope import default_candidate_models

    return list(default_candidate_models())


_DYNAMIC_NAMES = {
    "TREE_MODELS": _tree_model_set,
    "SEQ_MODELS": _seq_model_set,
    "BENCHMARK_MODEL_SET": _benchmark_model_set,
    "ALL_MODELS": _all_models_list,
}


def __getattr__(name: str):
    resolver = _DYNAMIC_NAMES.get(name)
    if resolver is not None:
        return resolver()
    raise AttributeError(f"module 'src.pipeline.model_factory' has no attribute {name!r}")
