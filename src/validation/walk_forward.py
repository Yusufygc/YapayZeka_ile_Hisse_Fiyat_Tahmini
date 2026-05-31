# -*- coding: utf-8 -*-
"""
walk_forward.py - Walk-Forward Validation Engine
Trains models sequentially across time splits and tracks windowed metrics.

Sprint 3 (2026-05-25) Plan A3.3:
  Concat-Sharpe aggregation. Tum OOS fold'larin gunluk strateji getirilerini
  birlestirip tek bir Sharpe ratio hesaplar (fold-mean Sharpe yerine).
  Ek olarak bootstrap %95 CI uretir. Sharpe_Concat ve Sharpe_CI_95_*
  alanlari `aggregated_metrics` icine eklenir.
"""

from typing import Any, Dict, List

import numpy as np

from src.evaluation.financial_metrics import (
    _annualized_sharpe,
    _price_to_simple_returns,
    _target_to_simple_returns,
    compute_financial_metrics,
)
from src.utils.risk_free_rate import get_current_risk_free_rate

# Sprint 3 (2026-05-25): preprocessor lazy-import. `joblib` agir bagimliligini
# minimum test ortaminda yuklemekten kacinmak icin ic kullanim yerine
# tasiniyor. Asagidaki yardimcilar (concat-Sharpe, strategy returns) bagimsiz.


_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_SEED = 20260525


def _bootstrap_sharpe_ci(
    returns: np.ndarray,
    risk_free_annual: float | None,
    n_resamples: int = _BOOTSTRAP_RESAMPLES,
    ci: float = 0.95,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap (low, high) Sharpe CI on concatenated returns."""
    returns = np.asarray(returns, dtype=float).ravel()
    if returns.size < 2 or risk_free_annual is None:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = returns.size
    sharpe_samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sharpe_samples[i] = _annualized_sharpe(returns[idx], risk_free_annual)
    alpha = (1.0 - ci) / 2.0
    low = float(np.nanquantile(sharpe_samples, alpha))
    high = float(np.nanquantile(sharpe_samples, 1.0 - alpha))
    return low, high


def _compute_strategy_returns(
    y_true_target: np.ndarray,
    y_pred_target: np.ndarray,
    y_true_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    """Sign-of-pred * realized simple return — match financial_metrics path."""
    if target_mode in ("log_return", "return"):
        realized = _target_to_simple_returns(
            np.asarray(y_true_target, dtype=float).ravel(), target_mode
        )
        signal_source = np.asarray(y_pred_target, dtype=float).ravel()
    else:
        realized = _price_to_simple_returns(
            np.asarray(y_true_price), np.asarray(prev_close)
        )
        signal_source = realized.copy()
    k = min(len(realized), len(signal_source))
    if k == 0:
        return np.asarray([], dtype=float)
    signs = np.sign(signal_source[-k:])
    return signs * realized[-k:]


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
        # Lazy-import: joblib bagimliligi run() ortaminda gerekir; modul yuklemede degil.
        from src.data.preprocessor import (
            reconstruct_prices_from_logret,
            reconstruct_prices_from_return,
        )
        if self.target_mode == "log_return":
            return reconstruct_prices_from_logret(preds_target, prev_close)
        if self.target_mode == "return":
            return reconstruct_prices_from_return(preds_target, prev_close)
        if self.target_mode == "price":
            return np.asarray(preds_target).ravel()
        raise ValueError(f"Desteklenmeyen target_mode: {self.target_mode}")

    def run(self, splits: List[Dict], verbose: bool = True) -> Dict[str, Any]:
        """Walk-forward pencerelerini sırayla eğitir/değerlendirir.

        Her pencere için scaler yalnızca o fold'un train dilimine fit edilir,
        model sıfırdan kurulur ve test diliminde tahmin üretilir; fold metrikleri
        ve strateji getirileri toplanır (concat-Sharpe için).

        Args:
            splits: `train`/`test` (+ opsiyonel `embargo_context`) içeren pencere
                sözlüklerinin kronolojik listesi.
            verbose: True ise her pencere için tarih/uzunluk özetini yazdırır.

        Returns:
            Toplu fold metrikleri ve birleştirilmiş strateji getirilerini içeren
            sözlük.
        """
        self.results = []
        all_metrics = []
        all_strategy_returns: List[np.ndarray] = []

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

            # Sprint 3 A3.3: fold strategy returns concat-Sharpe icin biriktir.
            fold_strategy_returns = _compute_strategy_returns(
                y_true_target_aligned,
                preds_target_aligned,
                y_true_final,
                prev_close_aligned,
                self.target_mode,
            )
            if fold_strategy_returns.size > 0:
                all_strategy_returns.append(fold_strategy_returns)

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
            # Numeric-only mean (skip None, booleans, strings).
            def _safe_mean(values):
                arr = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if not arr:
                    return float("nan")
                return float(np.mean(arr))

            avg_metrics = {key: _safe_mean([m.get(key) for m in all_metrics]) for key in all_metrics[0].keys()}

            # Sprint 3 A3.3: concat-Sharpe + bootstrap %95 CI.
            if all_strategy_returns:
                concat_returns = np.concatenate(all_strategy_returns)
                # Risk-free rate fetched once for concat (Sprint 1 A1.1 fail-loud uyumlu).
                try:
                    rf = get_current_risk_free_rate()
                except Exception:
                    rf = None
                if rf is not None and concat_returns.size > 0:
                    sharpe_concat = _annualized_sharpe(concat_returns, rf)
                    ci_low, ci_high = _bootstrap_sharpe_ci(concat_returns, rf)
                else:
                    sharpe_concat = float("nan")
                    ci_low = float("nan")
                    ci_high = float("nan")
                avg_metrics["Sharpe_Concat"] = sharpe_concat
                avg_metrics["Sharpe_CI_95_Low"] = ci_low
                avg_metrics["Sharpe_CI_95_High"] = ci_high
                avg_metrics["Concat_Returns_N"] = int(concat_returns.size)
            else:
                avg_metrics["Sharpe_Concat"] = float("nan")
                avg_metrics["Sharpe_CI_95_Low"] = float("nan")
                avg_metrics["Sharpe_CI_95_High"] = float("nan")
                avg_metrics["Concat_Returns_N"] = 0

            self.aggregated_metrics = avg_metrics
            if verbose:
                print("\n  [INFO] Walk-Forward Complete. Average Metrics:")
                for key, value in avg_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"         {key}: {value:.4f}")
                    else:
                        print(f"         {key}: {value}")

        return {
            "window_results": self.results,
            "average_metrics": self.aggregated_metrics,
        }
