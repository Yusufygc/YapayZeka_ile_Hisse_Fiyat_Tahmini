# -*- coding: utf-8 -*-
"""Faz 4 — CLI tak-çıkar yardımcıları + resolve_candidates."""
from __future__ import annotations

import pytest

from src.cli._model_filters import (
    expand_filters,
    list_models_table,
    resolve_disabled,
    resolve_selected,
)
from src.pipeline.model_scope import resolve_candidates


def test_expand_filters_by_category():
    names = expand_filters(category="tree")
    assert "LightGBM Return" in names
    assert "XGBoost" in names
    # linear_shrinkage tree değil — orada olmamalı
    assert "Ridge Return" not in names


def test_expand_filters_multi_category():
    names = expand_filters(category="tree,linear_shrinkage")
    assert {"XGBoost", "Ridge Return"}.issubset(names)


def test_expand_filters_by_role_benchmark():
    names = expand_filters(role="benchmark")
    assert names == {"Naive Last Value", "Naive Zero Return", "Naive Drift"}


def test_expand_filters_empty_returns_empty():
    """Hiçbir bayrak yoksa boş set döner."""
    assert expand_filters() == set()


def test_resolve_selected_combines_enable_and_category():
    out = resolve_selected(enable="XGBoost", category="linear_shrinkage")
    assert "XGBoost" in out
    assert "Ridge Return" in out


def test_resolve_selected_returns_none_when_no_input():
    assert resolve_selected() is None


def test_resolve_disabled():
    assert resolve_disabled(disable="XGBoost, LightGBM Return") == ["XGBoost", "LightGBM Return"]
    assert resolve_disabled(disable=None) == []


def test_resolve_candidates_filters_disabled():
    result = resolve_candidates(
        selected=["LightGBM Return", "XGBoost", "Ridge Return"],
        disabled=["XGBoost"],
    )
    assert "LightGBM Return" in result
    assert "Ridge Return" in result
    assert "XGBoost" not in result


def test_resolve_candidates_default_when_no_selected():
    """selected boşsa default candidate kümesi döner."""
    result = resolve_candidates(selected=None, disabled=[])
    assert result == {"XGBoost", "Ridge Return", "LightGBM Return"}


def test_resolve_candidates_disabled_overrides_default():
    """Default candidate olsa bile disabled listesindekiler düşer."""
    result = resolve_candidates(selected=None, disabled=["Ridge Return", "XGBoost"])
    assert "Ridge Return" not in result
    assert "XGBoost" not in result
    assert "LightGBM Return" in result


def test_resolve_candidates_require_available_drops_missing_deps():
    """require_available=True ile optional dep eksik modeller düşer."""
    result = resolve_candidates(
        selected=["Ridge Return", "XGBoost"],
        disabled=[],
        require_available=True,
    )
    assert {"Ridge Return", "XGBoost"}.issubset(result)


def test_resolve_candidates_rejects_removed_models():
    """Silinmis modeller (LSTM vb.) artik sessizce dusmez: acik ValueError (2026-11-16)."""
    with pytest.raises(ValueError, match="LSTM"):
        resolve_candidates(
            selected=["Ridge Return", "ElasticNet Return", "Random Forest", "XGBoost", "LSTM"],
            disabled=[],
            require_available=True,
        )


def test_list_models_table_contains_all_models():
    out = list_models_table()
    for required in (
        "Ridge Return",
        "Naive Zero Return", "XGBoost",
        "LightGBM Return",
    ):
        assert required in out, f"{required} listede yok"
    for removed in ("DLinear", "NLinear", "ARIMA", "LSTM", "Prophet", "ElasticNet Return", "Random Forest"):
        assert removed not in out, f"{removed} listede olmamali"
    # Başlıklar görünmeli.
    for header in ("Name", "Category", "Role", "Avail", "Requires"):
        assert header in out


def test_list_models_table_includes_availability_column():
    out = list_models_table()
    # Y veya N bayrağı her satırda olmalı.
    assert "Y" in out
    assert "N" in out


def test_modelconfig_new_fields_default():
    from src.pipeline.config import ModelConfig
    cfg = ModelConfig()
    assert cfg.disabled_models == []
    assert cfg.require_available is False
    assert cfg.ensemble_eligibility_overrides == {}
    for removed_key in ("deep_learning", "arima", "prophet"):
        assert removed_key not in cfg.model_settings


def test_modelconfig_accepts_disabled_and_require_available():
    from src.pipeline.config import ModelConfig
    cfg = ModelConfig(disabled_models=["XGBoost"], require_available=True)
    assert cfg.disabled_models == ["XGBoost"]
    assert cfg.require_available is True
