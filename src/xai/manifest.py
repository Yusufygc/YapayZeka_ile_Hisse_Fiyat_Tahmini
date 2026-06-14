# -*- coding: utf-8 -*-
"""XAI manifest and quality metadata helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from src.xai.feature_dictionary import compute_dictionary_coverage


def build_xai_manifest(
    payload: Mapping[str, Any],
    *,
    suffix: str,
    output_dir: str,
) -> dict[str, Any]:
    top_reasons = payload.get("top_reasons")
    daily_reasons = payload.get("daily_reasons")
    top_df = top_reasons if isinstance(top_reasons, pd.DataFrame) else pd.DataFrame()
    daily_df = daily_reasons if isinstance(daily_reasons, pd.DataFrame) else pd.DataFrame()
    methods = _method_counts(top_df, daily_df)
    fallback_rows = _fallback_rows(top_df, daily_df)
    total_rows = int(len(top_df) + len(daily_df))
    approximate_rows = _approximate_rows(top_df, daily_df)
    approximate_ratio = float(approximate_rows / total_rows) if total_rows else 0.0
    feature_names = _feature_names(top_df, daily_df)
    coverage = compute_dictionary_coverage(feature_names)
    background_scope = str(payload.get("background_scope") or _first_non_empty(top_df, "Background_Scope") or "unavailable")
    run_id = payload.get("run_id") or _metadata_value(payload, "run_id")
    validation_mode = payload.get("validation_mode") or _metadata_value(payload, "validation_mode")
    model_names = sorted({str(v) for v in top_df.get("Model", pd.Series(dtype=str)).dropna().tolist()})
    return {
        "schema_version": 1,
        "suffix": suffix,
        "run_id": run_id,
        "validation_mode": validation_mode,
        "model_name": ",".join(model_names) if len(model_names) == 1 else None,
        "model_names": model_names,
        "method": _dominant_method(methods),
        "method_counts": methods,
        "method_detail": _method_detail(methods),
        "approximate": bool(approximate_rows > 0),
        "approximate_ratio": approximate_ratio,
        "fallback_reason": "fallback_rows_present" if fallback_rows else "",
        "fallback_rows": fallback_rows,
        "background_scope": background_scope,
        "explained_rows": total_rows,
        "feature_count": len(feature_names),
        "top_feature_stability": dict(payload.get("feature_stability_top") or {}),
        "dictionary_coverage": coverage,
        "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "output_dir": os.path.abspath(output_dir),
    }


def write_xai_manifest(
    payload: Mapping[str, Any],
    *,
    suffix: str,
    output_dir: str,
) -> dict[str, Any]:
    manifest = build_xai_manifest(payload, suffix=suffix, output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)
    suffixed = os.path.join(output_dir, f"xai_manifest_{suffix}.json")
    alias = os.path.join(output_dir, "xai_manifest.json")
    for path in (suffixed, alias):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return manifest


def read_xai_manifest(xai_dir: str) -> dict[str, Any]:
    candidates = [os.path.join(xai_dir, "xai_manifest.json")]
    if os.path.isdir(xai_dir):
        candidates.extend(
            sorted(
                [
                    os.path.join(xai_dir, name)
                    for name in os.listdir(xai_dir)
                    if name.startswith("xai_manifest_") and name.endswith(".json")
                ],
                reverse=True,
            )
        )
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                parsed = json.load(handle)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            continue
    return {}


def _metadata_value(payload: Mapping[str, Any], key: str) -> Any:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return None


def _method_counts(*frames: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        if frame.empty or "Method" not in frame.columns:
            continue
        for method in frame["Method"].dropna().astype(str):
            method = method.strip()
            if method:
                counts[method] = counts.get(method, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _dominant_method(methods: Mapping[str, int]) -> str:
    return next(iter(methods), "")


def _method_detail(methods: Mapping[str, int]) -> str:
    if not methods:
        return ""
    if len(methods) == 1:
        return next(iter(methods))
    return "mixed:" + ",".join(f"{name}={count}" for name, count in methods.items())


def _fallback_rows(*frames: pd.DataFrame) -> int:
    total = 0
    for frame in frames:
        if frame.empty or "Method" not in frame.columns:
            continue
        total += int(frame["Method"].astype(str).str.contains("fallback|unavailable|coefficients", case=False, regex=True).sum())
    return total


def _approximate_rows(*frames: pd.DataFrame) -> int:
    total = 0
    for frame in frames:
        if frame.empty or "Approximate" not in frame.columns:
            continue
        total += int(frame["Approximate"].map(_truthy).sum())
    return total


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "evet"}


def _feature_names(*frames: pd.DataFrame) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        if frame.empty or "Feature" not in frame.columns:
            continue
        for raw in frame["Feature"].dropna().astype(str):
            if raw and raw not in seen:
                seen.add(raw)
                names.append(raw)
    return names


def _first_non_empty(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame.columns:
        return None
    for raw in frame[column].dropna().astype(str):
        raw = raw.strip()
        if raw:
            return raw
    return None
