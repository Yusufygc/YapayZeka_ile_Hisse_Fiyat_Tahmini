"""Üretim forecast artifact paketleme/yükleme.

Sorumluluklar:
  - ForecastArtifactPackage: model + scaler + metadata sidecar paketi.
  - ForecastArtifactError: eksik/bozuk artifact paketinde fırlatılır.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

import joblib


class ForecastArtifactError(RuntimeError):
    """Raised when a production forecast artifact package is missing or invalid."""


@dataclass
class ForecastArtifactPackage:
    model: Any
    scaler_X: Any
    scaler_y: Any
    metadata: Dict[str, Any]


def artifact_sidecar_paths(model_path: str) -> Dict[str, str]:
    base, _ = os.path.splitext(model_path)
    return {
        "metadata": f"{base}.forecast_metadata.json",
        "scaler_X": f"{base}.scaler_X.pkl",
        "scaler_y": f"{base}.scaler_y.pkl",
    }


def save_forecast_artifact_package(
    *,
    model_path: str,
    scaler_X: Any,
    scaler_y: Any,
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    if not model_path:
        raise ForecastArtifactError("model_path is required for artifact metadata")
    paths = artifact_sidecar_paths(model_path)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(scaler_X, paths["scaler_X"])
    joblib.dump(scaler_y, paths["scaler_y"])
    payload = dict(metadata)
    payload.setdefault("artifact_version", "forecast_artifact_v1")
    payload["model_path"] = model_path
    payload["scaler_X_path"] = paths["scaler_X"]
    payload["scaler_y_path"] = paths["scaler_y"]
    with open(paths["metadata"], "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return paths


def load_forecast_artifact_package(
    *,
    model_name: str,
    model_path: str,
    model_factory: Callable[[str], Any],
) -> ForecastArtifactPackage:
    if not model_path or not os.path.isfile(model_path):
        raise ForecastArtifactError(f"{model_name} artifact model file not found: {model_path}")
    paths = artifact_sidecar_paths(model_path)
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    if missing:
        raise ForecastArtifactError(
            f"{model_name} artifact sidecars missing: {', '.join(missing)}"
        )

    model = model_factory(model_name)
    if not hasattr(model, "load"):
        raise ForecastArtifactError(f"{model_name} does not support artifact loading")
    model.load(model_path)

    with open(paths["metadata"], "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return ForecastArtifactPackage(
        model=model,
        scaler_X=joblib.load(paths["scaler_X"]),
        scaler_y=joblib.load(paths["scaler_y"]),
        metadata=metadata,
    )
