# -*- coding: utf-8 -*-
"""
model_registry.py - Hafif JSON Kayit Sistemi (Uyumluluk Katmani)
=================================================================
Faz 1.3'te kaldirilan tam JSON kaydinin yerine gecen minimal implementasyon.
Uretim kayit islevi artik StockModelDB (SQLite) uzerindedir; bu sinif yalnizca
mevcut testlerin (test_phase7_acceptance.py) beklentisini karsilamak icin
korunmaktadir.

Kullanim:
    registry = ModelRegistry("/path/to/registry_dir")
    registry.register("XGBoost", "v1", ["feat_a"], {"RMSE": 0.5}, "model.pkl",
                      dataset_hash="abc123",
                      dataset_metadata={"validation_config": {"mode": "single"}})
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """Minimal JSON tabanli model kayit defteri (uyumluluk katmani)."""

    REGISTRY_FILE = "registry.json"

    def __init__(self, registry_dir: str) -> None:
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self._registry_path = os.path.join(registry_dir, self.REGISTRY_FILE)
        if not os.path.exists(self._registry_path):
            self._write([])

    def _read(self) -> List[Dict[str, Any]]:
        with open(self._registry_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, entries: List[Dict[str, Any]]) -> None:
        with open(self._registry_path, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False, indent=2)

    def register(
        self,
        model_name: str,
        version: str,
        features: List[str],
        metrics: Dict[str, Any],
        model_path: str,
        dataset_hash: Optional[str] = None,
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Yeni bir model girisini kayit defterine ekler ve giris sozlugunu doner."""
        entry: Dict[str, Any] = {
            "model_name": model_name,
            "version": version,
            "features": features,
            "metrics": metrics,
            "model_path": model_path,
            "dataset_hash": dataset_hash,
            "dataset_metadata": dataset_metadata or {},
            "registered_at": datetime.utcnow().isoformat(),
        }
        entries = self._read()
        entries.append(entry)
        self._write(entries)
        return entry

    def list_entries(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Tum girisleri doner; istege bagli olarak model_name'e gore filtreler."""
        entries = self._read()
        if model_name is not None:
            entries = [e for e in entries if e["model_name"] == model_name]
        return entries
