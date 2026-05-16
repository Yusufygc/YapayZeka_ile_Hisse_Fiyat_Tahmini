# -*- coding: utf-8 -*-
"""Correlation-based feature pruning strategy."""

from __future__ import annotations

from typing import Any

import pandas as pd


def prune_correlated_features(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    threshold: float,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    numeric_features = [name for name in feature_names if pd.api.types.is_numeric_dtype(df[name])]
    if len(numeric_features) < 2:
        return df, feature_names, _report(threshold, [])

    corr = df[numeric_features].corr().abs()
    feature_order = {name: idx for idx, name in enumerate(feature_names)}
    adjacency = _correlation_adjacency(corr, numeric_features, threshold)
    dropped = _cluster_drop_decisions(corr, numeric_features, feature_order, adjacency)
    drop_names = [item["feature"] for item in dropped]
    if drop_names:
        df = df.drop(columns=drop_names)
        feature_names = [name for name in feature_names if name not in drop_names]
    return df, feature_names, _report(threshold, dropped)


def _correlation_adjacency(
    corr: pd.DataFrame,
    numeric_features: list[str],
    threshold: float,
) -> dict[str, set[str]]:
    adjacency = {name: set() for name in numeric_features}
    for idx, left in enumerate(numeric_features):
        for right in numeric_features[idx + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value) and value > threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
    return adjacency


def _cluster_drop_decisions(
    corr: pd.DataFrame,
    numeric_features: list[str],
    feature_order: dict[str, int],
    adjacency: dict[str, set[str]],
) -> list[dict[str, Any]]:
    dropped: list[dict[str, Any]] = []
    visited: set[str] = set()
    for feature in numeric_features:
        if feature in visited:
            continue
        component = _connected_component(feature, adjacency, visited)
        if len(component) >= 2:
            dropped.extend(_component_drop_decisions(corr, component, feature_order))
    return dropped


def _connected_component(
    start: str,
    adjacency: dict[str, set[str]],
    visited: set[str],
) -> list[str]:
    stack = [start]
    component = []
    visited.add(start)
    while stack:
        current = stack.pop()
        component.append(current)
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return component


def _component_drop_decisions(
    corr: pd.DataFrame,
    component: list[str],
    feature_order: dict[str, int],
) -> list[dict[str, Any]]:
    component = sorted(component, key=lambda name: feature_order.get(name, 10**9))
    mean_corr = {
        name: float(corr.loc[name, [other for other in component if other != name]].mean())
        for name in component
    }
    kept = min(component, key=lambda name: (mean_corr[name], feature_order.get(name, 10**9)))
    dropped = []
    for name in component:
        if name == kept:
            continue
        peers = [other for other in component if other != name]
        correlated_with = max(peers, key=lambda other: float(corr.loc[name, other]))
        dropped.append({
            "feature": name,
            "kept_feature": kept,
            "correlated_with": str(correlated_with),
            "abs_corr": float(corr.loc[name, correlated_with]),
            "mean_abs_corr": mean_corr[name],
            "kept_mean_abs_corr": mean_corr[kept],
            "method": "mean_abs_correlation_cluster",
        })
    return dropped


def _report(threshold: float, dropped: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "enabled": True,
        "threshold": threshold,
        "method": "mean_abs_correlation_cluster",
        "dropped_features": dropped,
    }
