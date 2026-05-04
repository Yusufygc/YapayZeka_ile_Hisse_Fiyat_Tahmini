# -*- coding: utf-8 -*-
"""
experiment_tracker.py -- Experiment Logging Module.

Faz 1.4 — Normalizasyon:
  Dataset_Metadata blobu CSV'den kaldirildi.
  Metadata artik run_metadata/{dataset_hash}.json dosyasinda
  hash basina bir kez saklanir; CSV'de yalnizca Dataset_Hash kalir.
  Bu sayede her satirdaki 2KB tekrarli blob ortadan kalkar.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional


class ExperimentTracker:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "experiment_log.csv")
        self.metadata_dir = os.path.join(self.log_dir, "run_metadata")
        os.makedirs(self.log_dir, exist_ok=True)

    # ── Metadata kalici depolama ──────────────────────────────────────────────

    def _save_metadata_blob(self, dataset_hash: str, dataset_metadata: Dict[str, Any]) -> None:
        """
        Metadata'yi run_metadata/{dataset_hash}.json dosyasina yazar.
        Ayni hash icin sadece bir kez yazilir (idempotent).
        """
        if dataset_hash in ("N/A", "", None):
            return
        os.makedirs(self.metadata_dir, exist_ok=True)
        meta_path = os.path.join(self.metadata_dir, f"{dataset_hash}.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(dataset_metadata, f, ensure_ascii=False, indent=2, sort_keys=True)

    def get_metadata(self, dataset_hash: str) -> Optional[Dict[str, Any]]:
        """Hash'e gore metadata blob'unu yukler. Bulunamazsa None doner."""
        meta_path = os.path.join(self.metadata_dir, f"{dataset_hash}.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Ana log metodu ────────────────────────────────────────────────────────

    def log_run(
        self,
        model_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        features: List[str],
        dataset_hash: str = "N/A",
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Tek bir deney calismasini CSV'ye kaydeder.

        Dataset_Metadata artik CSV satirina yazilmaz; bunun yerine
        run_metadata/{dataset_hash}.json dosyasina hash basina bir kez
        yazilir. Boylece N satir x 2KB blob yerine 1 JSON dosyasi kalir.
        """
        dataset_metadata = dataset_metadata or {}

        # Metadata'yi hash'e gore bir kez kaydet
        self._save_metadata_blob(dataset_hash, dataset_metadata)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        run_data = {
            "Timestamp": timestamp,
            "Model": model_name,
            "Parameters": str(parameters),
            "Metrics": str(metrics),
            "Features_Count": len(features),
            "Dataset_Hash": dataset_hash,
            "Target_Mode": dataset_metadata.get("target_mode", "N/A"),
            "Feature_Mode": dataset_metadata.get("feature_mode", "N/A"),
            "Scaling_Mode": dataset_metadata.get("scaling_mode", "N/A"),
            "Date_Range": dataset_metadata.get("date_range", "N/A"),
            "Validation_Mode": dataset_metadata.get("validation_mode", "N/A"),
            # Dataset_Metadata blob kaldırıldı — run_metadata/{hash}.json kullanın
        }

        df_new = pd.DataFrame([run_data])

        if os.path.exists(self.log_file):
            df_existing = pd.read_csv(self.log_file)
            # Eski CSV'lerde Dataset_Metadata sutunu varsa uyumluluk icin kaldir
            df_existing = df_existing.drop(columns=["Dataset_Metadata"], errors="ignore")
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(self.log_file, index=False)
        print(f"  [INFO] Logged experiment run for {model_name} to {self.log_file}")
