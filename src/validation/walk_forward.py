# -*- coding: utf-8 -*-
"""
walk_forward.py — Walk-Forward Validation Engine
Trains models sequentially across time splits and tracks windowed metrics.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.evaluation.financial_metrics import compute_financial_metrics
from src.preprocessor import reconstruct_prices_from_logret, reconstruct_prices_from_return

class WalkForwardValidator:
    """
    Orchestrates the backtesting of models across multiple chronological splits.
    """
    
    def __init__(self, model_initializer: callable, preprocessor_fn: callable, target_mode: str = "log_return"):
        """
        :param model_initializer: A function that returns a fresh instance of the BaseModel to be trained.
        :param preprocessor_fn: A function that scales and prepares data (creates sequences if needed).
                                Signature (v2, H1 düzeltmesi):
                                  preprocessor_fn(train_df, test_df) ->
                                    X_train, y_train_logret_s,
                                    X_test,  y_test_logret_s,
                                    scaler_y,
                                    y_test_price,     # gerçek kapanışlar (fiyat-uzayı kıyas)
                                    prev_close_test   # fiyat inşası için t-1 kapanışları

                                y_* artık **ölçekli log-getiri**. Tahminler log-getiri
                                uzayında inverse_transform edilir, ardından
                                reconstruct_prices_from_logret ile fiyata çevrilir.
        """
        self.model_initializer = model_initializer
        self.preprocessor = preprocessor_fn
        self.target_mode = target_mode
        self.results = []
        self.aggregated_metrics = {}

    def _target_to_price(self, preds_target: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        if self.target_mode == "log_return":
            return reconstruct_prices_from_logret(preds_target, prev_close)
        if self.target_mode == "return":
            return reconstruct_prices_from_return(preds_target, prev_close)
        if self.target_mode == "price":
            return np.asarray(preds_target).ravel()
        raise ValueError(f"Desteklenmeyen target_mode: {self.target_mode}")

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

            # 1. Preprocess specific to this window (yeni 7'li sözleşme)
            (X_train, y_train, X_test, y_test,
             scaler_y, y_test_price, prev_close_test) = self.preprocessor(train_df, test_df)

            # 2. Initialize fresh model (prevents data leakage across iterations)
            model = self.model_initializer()

            # 3. Train
            # Prophet requires 'dates_train', we'll rely on kwargs passing if necessary or handle it inside the model wrapper
            dates_train = train_df["Date"].values if "Date" in train_df.columns else None
            dates_test = test_df["Date"].values if "Date" in test_df.columns else None

            model.train(X_train, y_train, dates_train=dates_train)

            # 4. Predict (log-getiri uzayı, ölçekli)
            preds = model.predict(X_test, dates_test=dates_test)

            # 5. Inverse transform → log-getiri
            if scaler_y is not None and preds.ndim > 0:
                preds_logret = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
            else:
                # scaler_y yoksa (ör. Prophet) preds zaten log-getiri
                preds_logret = np.asarray(preds).ravel()

            # 6. Sequence uzunluk uyuşmasını hizala (trailing align)
            min_len = min(len(preds_logret), len(y_test_price), len(prev_close_test))
            preds_logret_aligned = preds_logret[-min_len:]
            prev_close_aligned   = prev_close_test[-min_len:]

            # 7. Log-getiriyi fiyata çevir
            preds_final = self._target_to_price(preds_logret_aligned, prev_close_aligned)
            y_true_final = y_test_price[-min_len:]
            
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
