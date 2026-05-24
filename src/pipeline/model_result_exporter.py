# -*- coding: utf-8 -*-
"""Per-model run result exports for easier manual inspection."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


def model_result_slug(model_name: str) -> str:
    """Return a stable folder name for a model display name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(model_name)).strip("_").lower()
    return slug or "model"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _as_1d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return None
    return arr.ravel()


def _write_aligned_csv(path: str, columns: Mapping[str, Any]) -> bool:
    arrays = {name: _as_1d(value) for name, value in columns.items()}
    arrays = {name: arr for name, arr in arrays.items() if arr is not None and len(arr) > 0}
    if not arrays:
        return False

    min_len = min(len(arr) for arr in arrays.values())
    if min_len <= 0:
        return False

    frame = pd.DataFrame({name: arr[-min_len:] for name, arr in arrays.items()})
    frame.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    return True


def _result_dir(outputs_dir: str, model_name: str) -> str:
    path = os.path.join(outputs_dir, "model_results", model_result_slug(model_name))
    os.makedirs(path, exist_ok=True)
    return path


def export_single_split_result(
    owner: Any,
    *,
    model_name: str,
    metrics: Mapping[str, Any],
    model_path: str,
) -> None:
    """Write model-scoped single-split artifacts inside the current run."""
    out_dir = _result_dir(owner.outputs_dir, model_name)
    _write_json(os.path.join(out_dir, "metrics_single_split.json"), dict(metrics))
    _write_json(
        os.path.join(out_dir, "artifact_manifest.json"),
        {
            "model_name": model_name,
            "stage": "single_split",
            "run_id": owner.dataset_metadata.get("run_id"),
            "model_path": model_path,
            "forecast_sidecars_next_to_model": bool(model_path),
        },
    )
    _write_aligned_csv(
        os.path.join(out_dir, "predictions_single_split.csv"),
        {
            "date": owner.latest_tensors.get("dates_test"),
            "prediction_date": owner.latest_tensors.get("dates_prediction"),
            "y_true_price": owner.y_true_aligned,
            "y_pred_price": owner.predictions.get(model_name),
            "y_true_target": owner.y_true_target_aligned,
            "y_pred_target": owner.prediction_targets.get(model_name),
            "prev_close": owner.prev_close_aligned,
        },
    )


def export_walk_forward_results(
    owner: Any,
    *,
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    fold_metrics_by_model: Mapping[str, list[Mapping[str, Any]]],
    backtest_inputs_by_model: Mapping[str, Mapping[str, Any]],
) -> None:
    """Write model-scoped walk-forward summaries and prediction rows."""
    for model_name, metrics in metrics_by_model.items():
        out_dir = _result_dir(owner.outputs_dir, model_name)
        _write_json(os.path.join(out_dir, "metrics_walk_forward.json"), dict(metrics))
        _write_json(
            os.path.join(out_dir, "artifact_manifest.json"),
            {
                "model_name": model_name,
                "stage": "walk_forward",
                "run_id": owner.dataset_metadata.get("run_id"),
                "model_path": "",
                "model_file_note": "Walk-forward fold models are evaluated in-memory; only final holdout persists a selected model.",
            },
        )

        fold_rows = list(fold_metrics_by_model.get(model_name, []))
        if fold_rows:
            pd.DataFrame(fold_rows).to_csv(
                os.path.join(out_dir, "fold_metrics_walk_forward.csv"),
                index=False,
                sep=";",
                encoding="utf-8-sig",
            )

        payload = backtest_inputs_by_model.get(model_name, {})
        _write_aligned_csv(
            os.path.join(out_dir, "predictions_walk_forward.csv"),
            {
                "date": payload.get("dates"),
                "prediction_date": payload.get("prediction_dates"),
                "fold": payload.get("fold_ids"),
                "y_true_price": payload.get("y_true_price"),
                "y_pred_price": payload.get("pred_price"),
                "y_true_target": payload.get("y_true_target"),
                "y_pred_target": payload.get("pred_target"),
                "prev_close": payload.get("prev_close"),
            },
        )


def export_final_holdout_result(
    owner: Any,
    *,
    model_name: str,
    metrics: Mapping[str, Any],
    model_path: str,
    prediction_columns: Mapping[str, Any],
) -> None:
    """Write model-scoped final-holdout artifacts for the selected WF model."""
    out_dir = _result_dir(owner.outputs_dir, model_name)
    _write_json(os.path.join(out_dir, "metrics_final_holdout.json"), dict(metrics))
    _write_json(
        os.path.join(out_dir, "artifact_manifest_final_holdout.json"),
        {
            "model_name": model_name,
            "stage": "final_holdout",
            "run_id": owner.dataset_metadata.get("run_id"),
            "model_path": model_path,
            "forecast_sidecars_next_to_model": bool(model_path),
        },
    )
    _write_aligned_csv(
        os.path.join(out_dir, "predictions_final_holdout.csv"),
        prediction_columns,
    )
