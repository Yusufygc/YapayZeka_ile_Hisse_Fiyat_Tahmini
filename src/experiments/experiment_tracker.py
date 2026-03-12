# -*- coding: utf-8 -*-
"""
experiment_tracker.py — Experiment Logging Module.
Saves hyperparameters, metrics, and metadata to a centralized CSV.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

class ExperimentTracker:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "experiment_log.csv")
        os.makedirs(self.log_dir, exist_ok=True)

    def log_run(self, model_name: str, parameters: Dict[str, Any], metrics: Dict[str, float], features: List[str], dataset_hash: str = "N/A"):
        """
        Logs a single experiment run to the tracking CSV.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        run_data = {
            "Timestamp": timestamp,
            "Model": model_name,
            "Parameters": str(parameters),
            "Metrics": str(metrics),
            "Features_Count": len(features),
            "Dataset_Hash": dataset_hash
        }
        
        df_new = pd.DataFrame([run_data])
        
        if os.path.exists(self.log_file):
            df_existing = pd.read_csv(self.log_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        df_combined.to_csv(self.log_file, index=False)
        print(f"  [INFO] Logged experiment run for {model_name} to {self.log_file}")
