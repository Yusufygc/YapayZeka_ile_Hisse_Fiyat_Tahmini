# -*- coding: utf-8 -*-
"""
model_registry.py — Model Versioning and Registration.
Tracks models via a JSON manifest and stores their metadata.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List

class ModelRegistry:
    def __init__(self, registry_dir: str):
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(self.registry_dir, "registry.json")
        os.makedirs(self.registry_dir, exist_ok=True)
        
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def _read_registry(self) -> List[Dict]:
        with open(self.registry_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_registry(self, data: List[Dict]):
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register(self, model_name: str, version: str, features: List[str], metrics: Dict[str, float], model_path: str, dataset_hash: str = "N/A"):
        """
        Registers a new model version into the JSON registry.
        """
        registry_data = self._read_registry()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = {
            "model_name": model_name,
            "version": version,
            "trained_at": timestamp,
            "features": features,
            "metrics": metrics,
            "model_path": model_path,
            "dataset_hash": dataset_hash
        }
        
        registry_data.append(entry)
        self._write_registry(registry_data)
        print(f"  [INFO] Registered {model_name} (v{version}) in model registry.")
