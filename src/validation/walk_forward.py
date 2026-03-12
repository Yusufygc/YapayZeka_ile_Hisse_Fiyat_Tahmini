# -*- coding: utf-8 -*-
"""
walk_forward.py — Walk-Forward Validation Engine
Trains models sequentially across time splits and tracks windowed metrics.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.evaluation.financial_metrics import compute_financial_metrics

class WalkForwardValidator:
    """
    Orchestrates the backtesting of models across multiple chronological splits.
    """
    
    def __init__(self, model_initializer: callable, preprocessor_fn: callable):
        """
        :param model_initializer: A function that returns a fresh instance of the BaseModel to be trained.
        :param preprocessor_fn: A function that scales and prepares data (creates sequences if needed).
                                Signature: preprocessor_fn(train_df, test_df) -> X_train, y_train, X_test, y_test, scaler_y
        """
        self.model_initializer = model_initializer
        self.preprocessor = preprocessor_fn
        self.results = []
        self.aggregated_metrics = {}

    def run(self, splits: List[Dict], verbose: bool = True) -> Dict[str, Any]:
        """
        Runs the walk-forward validation across the provided splits.
        """
        self.results = []
        
        all_metrics = []
        
        for idx, split in enumerate(splits):
            if verbose:
                print(f"\n  [INFO] Walk-Forward Window {idx + 1}/{len(splits)} (Split Index: {split['split_idx']})")
                print(f"         Train points: {len(split['train'])}, Test points: {len(split['test'])}")
                
            train_df = split["train"]
            test_df = split["test"]
            
            # 1. Preprocess specific to this window
            X_train, y_train, X_test, y_test, scaler_y, original_y_test_aligned = self.preprocessor(train_df, test_df)
            
            # 2. Initialize fresh model (prevents data leakage across iterations)
            model = self.model_initializer()
            
            # 3. Train
            # Prophet requires 'dates_train', we'll rely on kwargs passing if necessary or handle it inside the model wrapper
            dates_train = train_df["Date"].values if "Date" in train_df.columns else None
            dates_test = test_df["Date"].values if "Date" in test_df.columns else None
            
            model.train(X_train, y_train, dates_train=dates_train)
            
            # 4. Predict
            preds = model.predict(X_test, dates_test=dates_test)
            
            # If scaling was applied, inverse transform
            if scaler_y is not None and preds.ndim > 0:
                preds_original = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
            else:
                preds_original = preds

            # Handle sequence prediction length mismatch
            min_len = min(len(preds_original), len(original_y_test_aligned))
            preds_final = preds_original[-min_len:]
            y_true_final = original_y_test_aligned[-min_len:]
            
            # 5. Evaluate
            metrics = compute_financial_metrics(y_true_final, preds_final)
            all_metrics.append(metrics)
            
            self.results.append({
                "split_idx": split["split_idx"],
                "y_true": y_true_final.tolist(),
                "y_pred": preds_final.tolist(),
                "metrics": metrics
            })
            
        # 6. Aggregate Metrics (Averaging over all splits)
        if len(all_metrics) > 0:
            avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
            self.aggregated_metrics = avg_metrics
            if verbose:
                print("\n  [INFO] Walk-Forward Complete. Average Metrics:")
                for k, v in avg_metrics.items():
                    print(f"         {k}: {v:.4f}")
                    
        return {
            "window_results": self.results,
            "average_metrics": self.aggregated_metrics
        }
