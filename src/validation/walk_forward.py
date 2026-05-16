# -*- coding: utf-8 -*-
"""
walk_forward.py - Walk-Forward Validation Engine
Trains models sequentially across time splits and tracks windowed metrics.
"""

from typing import Any, Dict, List

import numpy as np

from src.evaluation.financial_metrics import compute_financial_metrics
from src.data.preprocessor import reconstruct_prices_from_logret, reconstruct_prices_from_return


class WalkForwardValidator:
    """
    Orchestrates model evaluation across chronological splits without leakage.
    """

    def __init__(self, model_initializer: callable, preprocessor_fn: callable, target_mode: str = "log_return"):
        self.model_initializer = model_initializer
        self.preprocessor = preprocessor_fn
        self.target_mode = target_mode
        self.results = []
        self.aggregated_metrics = {}
        self.feature_importances: List[np.ndarray] = []
        self.mean_feature_importance: np.ndarray | None = None

    def _target_to_price(self, preds_target: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        if self.target_mode == "log_return":
            return reconstruct_prices_from_logret(preds_target, prev_close)
        if self.target_mode == "return":
            return reconstruct_prices_from_return(preds_target, prev_close)
        if self.target_mode == "price":
            return np.asarray(preds_target).ravel()
        raise ValueError(f"Desteklenmeyen target_mode: {self.target_mode}")

    def run(self, splits: List[Dict], verbose: bool = True) -> Dict[str, Any]:
        self.results = []
        all_metrics = []

        for idx, split in enumerate(splits):
            if verbose:
                print(f"\n  [INFO] Walk-Forward Window {idx + 1}/{len(splits)} (Split Index: {split['split_idx']})")
                print(f"         Train points: {len(split['train'])}, Test points: {len(split['test'])}")
                print(
                    f"         Train dates : {split.get('train_date_start')} -> {split.get('train_date_end')} | "
                    f"Test dates: {split.get('test_date_start')} -> {split.get('test_date_end')}"
                )

            train_df = split["train"]
            test_df = split["test"]
            context_df = split.get("embargo_context")

            (
                X_train,
                y_train,
                X_test,
                y_test,
                scaler_y,
                y_test_price,
                prev_close_test,
                dates_test,
                prediction_dates_test,
                y_test_target,
                market_regime_test,
            ) = self.preprocessor(train_df, test_df, context_df=context_df)

            model = self.model_initializer()
            dates_train = train_df["Date"].values if "Date" in train_df.columns else None
            model.train(X_train, y_train, dates_train=dates_train)

            dates_test_raw = test_df["Date"].values if "Date" in test_df.columns else None
            preds = model.predict(X_test, dates_test=dates_test_raw)

            inner = getattr(model, "model", model)
            fi = getattr(inner, "feature_importances_", None)
            if fi is not None:
                self.feature_importances.append(np.asarray(fi, dtype=float))

            if scaler_y is not None and np.asarray(preds).ndim > 0:
                preds_target = scaler_y.inverse_transform(np.asarray(preds).reshape(-1, 1)).ravel()
            else:
                preds_target = np.asarray(preds).ravel()

            min_len = min(
                len(preds_target),
                len(y_test_price),
                len(prev_close_test),
                len(dates_test),
                len(prediction_dates_test),
                len(y_test_target),
                len(market_regime_test),
            )
            preds_target_aligned = preds_target[-min_len:]
            prev_close_aligned = np.asarray(prev_close_test).ravel()[-min_len:]
            y_true_final = np.asarray(y_test_price).ravel()[-min_len:]
            y_true_target_aligned = np.asarray(y_test_target).ravel()[-min_len:]
            dates_aligned = np.asarray(dates_test)[-min_len:]
            prediction_dates_aligned = np.asarray(prediction_dates_test)[-min_len:]
            market_regime_aligned = np.asarray(market_regime_test).ravel()[-min_len:]
            preds_final = self._target_to_price(preds_target_aligned, prev_close_aligned)

            metrics = compute_financial_metrics(
                y_true_final,
                preds_final,
                y_true_target=y_true_target_aligned,
                y_pred_target=preds_target_aligned,
                prev_close=prev_close_aligned,
                target_mode=self.target_mode,
            )
            all_metrics.append(metrics)

            self.results.append({
                "split_idx": split["split_idx"],
                "dates": dates_aligned.tolist(),
                "prediction_dates": prediction_dates_aligned.tolist(),
                "market_regime": market_regime_aligned.tolist(),
                "prev_close": prev_close_aligned.tolist(),
                "y_true_price": y_true_final.tolist(),
                "y_true_target": y_true_target_aligned.tolist(),
                "y_pred_price": preds_final.tolist(),
                "y_pred_target": preds_target_aligned.tolist(),
                "metrics": metrics,
                "y_true": y_true_final.tolist(),
                "y_pred": preds_final.tolist(),
            })

        if self.feature_importances:
            shapes = {arr.shape[0] for arr in self.feature_importances}
            if len(shapes) == 1:
                self.mean_feature_importance = np.vstack(self.feature_importances).mean(axis=0)

        if all_metrics:
            avg_metrics = {key: float(np.mean([metric[key] for metric in all_metrics])) for key in all_metrics[0].keys()}
            self.aggregated_metrics = avg_metrics
            if verbose:
                print("\n  [INFO] Walk-Forward Complete. Average Metrics:")
                for key, value in avg_metrics.items():
                    print(f"         {key}: {value:.4f}")

        return {
            "window_results": self.results,
            "average_metrics": self.aggregated_metrics,
        }
