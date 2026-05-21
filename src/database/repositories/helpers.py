# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional

PRODUCTION_ENSEMBLE_METHODS = {"Inverse RMSE", "Cash-Gated", "Seq-Attention Inverse RMSE"}


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in (None, ""):
        return None
    try:
        return json.loads(str(value).replace("'", '"'))
    except Exception:
        return value


def _optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> Optional[int]:
    float_value = _optional_float(value)
    return None if float_value is None else int(float_value)


def _optional_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _ensemble_metadata_for(
    model_name: str,
    metrics: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not str(model_name).startswith("Ensemble "):
        return None
    method = str(metrics.get("Ensemble_Method") or str(model_name).replace("Ensemble ", ""))
    weights = _parse_jsonish(metrics.get("Ensemble_Weights"))
    source_ids = _parse_jsonish(metrics.get("Ensemble_Source_Experiment_IDs"))
    return {
        "type": "ensemble",
        "method": method,
        "production_method": method in PRODUCTION_ENSEMBLE_METHODS,
        "members": list(weights.keys()) if isinstance(weights, dict) else [],
        "weights": weights if isinstance(weights, dict) else {},
        "source_experiment_ids": source_ids if isinstance(source_ids, list) else [],
        "source_run_ids": _parse_jsonish(metrics.get("Ensemble_Source_Run_IDs")) or [],
        "run_id": dataset_metadata.get("run_id"),
    }
