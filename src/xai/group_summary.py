# -*- coding: utf-8 -*-
"""Group-level XAI summaries for product and peer explanations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.xai.feature_dictionary import feature_group_label, feature_group_reason, has_readable_label


@dataclass
class XaiGroupSummaryData:
    feature_group: str
    group_label: str
    total_importance: float
    net_contribution: float
    direction: str
    top_features: list[str]
    reason: str
    approximate_ratio: float


def build_group_summaries(
    rows: Iterable[Mapping[str, Any]],
    *,
    context: str = "forecast",
    top_n_features: int = 3,
) -> list[XaiGroupSummaryData]:
    """Aggregate XAI rows into stable product-facing group summaries."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        feature = str(_value(raw, "feature_name", "Feature") or "").strip()
        if not feature or not has_readable_label(feature):
            continue
        group = str(_value(raw, "feature_group", "Feature_Group") or "").strip() or "other"
        importance = _float(_value(raw, "importance", "Importance"))
        if importance <= 0:
            continue
        contribution = _optional_float(_value(raw, "contribution", "Contribution"))
        approximate = _bool(_value(raw, "approximate", "Approximate"))
        buckets.setdefault(group, []).append(
            {
                "feature": feature,
                "importance": importance,
                "contribution": contribution,
                "approximate": approximate,
            }
        )

    summaries: list[XaiGroupSummaryData] = []
    for group, items in buckets.items():
        total_importance = sum(float(item["importance"]) for item in items)
        contributions = [item["contribution"] for item in items if item["contribution"] is not None]
        net_contribution = float(sum(contributions)) if contributions else 0.0
        direction = _direction(net_contribution, has_contribution=bool(contributions))
        approx_count = sum(1 for item in items if item["approximate"] is True)
        ranked = sorted(items, key=lambda item: float(item["importance"]), reverse=True)
        top_features = [str(item["feature"]) for item in ranked[:top_n_features]]
        summaries.append(
            XaiGroupSummaryData(
                feature_group=group,
                group_label=feature_group_label(group),
                total_importance=float(total_importance),
                net_contribution=net_contribution,
                direction=direction,
                top_features=top_features,
                reason=feature_group_reason(group, direction, context=context),
                approximate_ratio=float(approx_count / len(items)) if items else 0.0,
            )
        )
    summaries.sort(key=lambda item: item.total_importance, reverse=True)
    return summaries


def group_summaries_to_dicts(summaries: Iterable[XaiGroupSummaryData]) -> list[dict[str, Any]]:
    return [
        {
            "feature_group": item.feature_group,
            "group_label": item.group_label,
            "total_importance": item.total_importance,
            "net_contribution": item.net_contribution,
            "direction": item.direction,
            "top_features": list(item.top_features),
            "reason": item.reason,
            "approximate_ratio": item.approximate_ratio,
        }
        for item in summaries
    ]


def _value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _float(value: Any) -> float:
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "evet"}:
        return True
    if text in {"false", "0", "no", "hayir", "hayır"}:
        return False
    return None


def _direction(net_contribution: float, *, has_contribution: bool) -> str:
    if not has_contribution:
        return "dikkat"
    if net_contribution > 0:
        return "yukari"
    if net_contribution < 0:
        return "asagi"
    return "notr"
