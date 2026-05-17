# -*- coding: utf-8 -*-
"""Faz 1 — model_registry birim testleri."""
from __future__ import annotations

import pytest

from src.pipeline import model_registry as reg
from src.pipeline.model_registry import (
    ModelSpec,
    all_specs,
    ensure_loaded,
    get_spec,
    has_spec,
    is_available,
    register_model,
    unregister,
)


def test_register_and_get_spec():
    name = "__TestDummyModel"
    unregister(name)
    spec = register_model(ModelSpec(
        name=name,
        factory=lambda **kw: object(),
        category="tree",
        role="candidate",
    ))
    assert spec.name == name
    assert get_spec(name) is spec
    assert has_spec(name)
    unregister(name)
    assert not has_spec(name)


def test_register_duplicate_raises():
    name = "__TestDupModel"
    unregister(name)
    register_model(ModelSpec(
        name=name,
        factory=lambda **kw: object(),
        category="tree",
    ))
    with pytest.raises(ValueError):
        register_model(ModelSpec(
            name=name,
            factory=lambda **kw: object(),
            category="tree",
        ))
    unregister(name)


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_spec("__DefinitelyNotRegistered__")


def test_ensure_loaded_idempotent():
    ensure_loaded()
    count1 = len(all_specs())
    ensure_loaded()
    count2 = len(all_specs())
    assert count1 == count2
    assert count1 > 0


def test_discovery_populates_core_models():
    """Faz 1: src/models içindeki tüm core modeller registry'de olmalı."""
    ensure_loaded()
    names = {s.name for s in all_specs()}
    # Optional dep gerektirmeyen core modeller mutlaka var olmalı.
    for required in ("Random Forest", "Ridge Return", "ElasticNet Return",
                      "DLinear", "NLinear",
                      "Naive Last Value", "Naive Zero Return", "Naive Drift"):
        assert required in names, f"{required} registry'de yok; mevcut: {sorted(names)}"


def test_filter_by_role():
    ensure_loaded()
    benchmarks = all_specs(role="benchmark")
    assert {s.name for s in benchmarks} == {
        "Naive Last Value", "Naive Zero Return", "Naive Drift",
    }
    for s in benchmarks:
        assert s.ensemble_eligible is False


def test_filter_by_category():
    ensure_loaded()
    tree_names = {s.name for s in all_specs(category="tree")}
    # En azından RF orada olmalı.
    assert "Random Forest" in tree_names


def test_filter_ensemble_only_excludes_benchmarks():
    ensure_loaded()
    eligible = {s.name for s in all_specs(ensemble_only=True)}
    assert "Naive Zero Return" not in eligible
    assert "Random Forest" in eligible


def test_filter_target_mode():
    ensure_loaded()
    log_return_names = {s.name for s in all_specs(target_mode="log_return")}
    assert "Naive Zero Return" in log_return_names
    price_names = {s.name for s in all_specs(target_mode="price")}
    # Naive Zero Return target_modes sadece return/log_return → price filtresinde olmamalı.
    assert "Naive Zero Return" not in price_names


def test_is_available_no_requires_ok():
    spec = ModelSpec(name="__NoDep", factory=lambda **kw: None, category="tree")
    ok, reason = is_available(spec)
    assert ok is True
    assert reason == ""


def test_is_available_missing_dep_false():
    spec = ModelSpec(
        name="__MissingDep",
        factory=lambda **kw: None,
        category="tree",
        requires=("__definitely_not_a_real_package__",),
    )
    ok, reason = is_available(spec)
    assert ok is False
    assert "__definitely_not_a_real_package__" in reason


def test_factory_invokable_for_naive():
    """Naive model factory'leri parametre olmadan instance üretebilmeli."""
    ensure_loaded()
    spec = get_spec("Naive Zero Return")
    instance = spec.factory()
    assert instance is not None
    # BaseModel sözleşmesi — predict çağrılabilir olmalı.
    import numpy as np
    instance.train(np.zeros((10, 3)), np.zeros(10))
    out = instance.predict(np.zeros((5, 3)))
    assert len(out) == 5


def test_spec_fields_frozen():
    """ModelSpec dataclass frozen olmalı — kazara mutasyon engellenir."""
    spec = ModelSpec(name="__Frozen", factory=lambda **kw: None, category="tree")
    with pytest.raises(Exception):
        spec.name = "Other"  # type: ignore[misc]
