# -*- coding: utf-8 -*-
"""Model construction helpers used by ``ModelTrainer``.

Faz 3: spec helper'ları (`benchmark_specs`, `linear_baseline_specs`,
`boosting_baseline_specs`) ve set sabitleri (`TREE_MODELS`,
`BENCHMARK_MODEL_SET`, `ALL_MODELS`) registry üzerinden türetilir.
Modül-seviyesi `__getattr__` sayesinde eski isimler import edilebilir;
dinamik değerler ensure_loaded() üzerinden çözülür.

Sequence/deep-learning, Prophet ve ARIMA modelleri kaldirildi; per-symbol yol
yalnizca tabular modelleri ve naive benchmark'lari icerir.
"""

from __future__ import annotations

from typing import Callable

from src.models.xgboost_model import XGBoostModel

# Faz 3: TREE_MODELS / BENCHMARK_MODEL_SET / ALL_MODELS artık
# `__getattr__` ile registry'den çözülür (aşağıda). Eski sabitler bu modülde
# tanımlı değil — `from src.pipeline.model_factory import TREE_MODELS` hâlâ
# çalışır (PEP 562).

OPTIONAL_MODELS = ["LightGBM Return"]

# Hangi registry kategorileri "tree-family" (scaled tabular) sayılır.
_TREE_CATEGORIES: frozenset[str] = frozenset({"tree", "linear_shrinkage"})


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
    """Lineer shrinkage modelleri (Ridge)."""
    from src.pipeline.model_registry import all_specs

    return [
        (s.name, s.factory)
        for s in all_specs(category="linear_shrinkage", role="candidate")
    ]


def boosting_baseline_specs():
    """LightGBM ailesi — tree kategorisinin boosting alt-kümesi.

    Mevcut tasarımda yalnızca `LightGBM Return` boosting baseline'ı olarak
    raporlanır; XGBoost ayrı (tuned) yolda çalışır.
    """
    from src.pipeline.model_registry import get_spec, has_spec

    if has_spec("LightGBM Return"):
        return [("LightGBM Return", get_spec("LightGBM Return").factory)]
    return []


def model_class_for_name(model_name: str, model_config: dict, target_mode: str):
    """Registry-proxy (Faz 2).

    Davranış:
      * Tree/linear/naive modeller için sıfır-arg ``cls()`` çağrılabilir.
      * ``needs_config_keys`` tanimli modeller icin config alt-sozlukleri unpack edilir.
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


def _benchmark_model_set() -> set[str]:
    from src.pipeline.model_registry import all_specs

    return {s.name for s in all_specs(role="benchmark")}


def _all_models_list() -> list[str]:
    from src.pipeline.model_scope import default_candidate_models

    return list(default_candidate_models())


_DYNAMIC_NAMES = {
    "TREE_MODELS": _tree_model_set,
    "BENCHMARK_MODEL_SET": _benchmark_model_set,
    "ALL_MODELS": _all_models_list,
}


def __getattr__(name: str):
    resolver = _DYNAMIC_NAMES.get(name)
    if resolver is not None:
        return resolver()
    raise AttributeError(f"module 'src.pipeline.model_factory' has no attribute {name!r}")
