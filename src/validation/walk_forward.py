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

from src.xai.background import XAIBackgroundProvider
from src.xai.strategies import TabularContributionStrategy
from src.evaluation.financial_metrics import (
    _annualized_sharpe,
    _price_to_simple_returns,
    _target_to_simple_returns,
    compute_financial_metrics,
)

# Sprint 3 (2026-05-25): preprocessor lazy-import. `joblib` agir bagimliligini
# minimum test ortaminda yuklemekten kacinmak icin ic kullanim yerine
# tasiniyor. Asagidaki yardimcilar (concat-Sharpe, strategy returns) bagimsiz.


_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_SEED = 20260525


_BOOTSTRAP_MEAN_BLOCK = 5


def _stationary_bootstrap_indices(
    n: int,
    rng: np.random.Generator,
    mean_block: float = _BOOTSTRAP_MEAN_BLOCK,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indeksleri (uzunluk n).

    Her adimda p = 1/mean_block olasiligiyla yeni rastgele baslangic, aksi
    halde bir onceki indeks + 1 (dairesel). Blok uzunluklari geometrik dagilir.
    """
    p = 1.0 / max(float(mean_block), 1.0)
    restart = rng.random(n) < p
    restart[0] = True
    starts = rng.integers(0, n, size=n)
    positions = np.arange(n)
    # Her pozisyon icin icinde bulundugu blogun baslangic pozisyonu.
    block_begin = np.maximum.accumulate(np.where(restart, positions, 0))
    return (starts[block_begin] + (positions - block_begin)) % n


def _bootstrap_sharpe_ci(
    returns: np.ndarray,
    risk_free_annual: float | None,
    n_resamples: int = _BOOTSTRAP_RESAMPLES,
    ci: float = 0.95,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Stationary (Politis-Romano) block-bootstrap (low, high) Sharpe CI.

    Getiriler seri korelasyonlu/volatilite-kumelenmeli olabilir; iid yeniden
    ornekleme CI'yi olduğundan dar verir. Ortalama blok uzunlugu 5, sabit seed
    (tekrarlanabilir).
    """
    returns = np.asarray(returns, dtype=float).ravel()
    if returns.size < 2 or risk_free_annual is None:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = returns.size
    sharpe_samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = _stationary_bootstrap_indices(n, rng)
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
    *,
    y_pred_price: np.ndarray | None = None,
) -> np.ndarray:
    """Sign-of-pred * realized simple return — match financial_metrics path.

    Fiyat modunda sinyal tahmin edilen fiyattan türetilir
    (`y_pred_price` / `prev_close` - 1); gerçekleşen getiriden TÜRETİLMEZ
    (aksi halde |realized| = mükemmel öngörü).
    """
    if target_mode in ("log_return", "return"):
        realized = _target_to_simple_returns(
            np.asarray(y_true_target, dtype=float).ravel(), target_mode
        )
        signal_source = np.asarray(y_pred_target, dtype=float).ravel()
    else:
        realized = _price_to_simple_returns(
            np.asarray(y_true_price), np.asarray(prev_close)
        )
        if y_pred_price is None:
            raise ValueError(
                "target_mode='price' için y_pred_price zorunludur "
                "(sinyal tahmin edilen fiyattan türetilir)."
            )
        signal_source = _price_to_simple_returns(
            np.asarray(y_pred_price), np.asarray(prev_close)
        )
    k = min(len(realized), len(signal_source))
    if k == 0:
        return np.asarray([], dtype=float)
    signs = np.sign(signal_source[-k:])
    return signs * realized[-k:]


class WalkForwardValidator:
    """
    Orchestrates model evaluation across chronological splits without leakage.
    """

    def __init__(
        self,
        model_initializer: callable,
        preprocessor_fn: callable,
        target_mode: str = "log_return",
        feature_names: list[str] | None = None,
        xai_max_rows: int = 80,
    ):
        self.model_initializer = model_initializer
        self.preprocessor = preprocessor_fn
        self.target_mode = target_mode
        self.feature_names = list(feature_names or [])
        self.xai_max_rows = int(xai_max_rows)
        self.results = []
        self.aggregated_metrics = {}
        self.feature_importances: List[np.ndarray] = []
        self.mean_feature_importance: np.ndarray | None = None
        self.xai_records: List[Dict[str, Any]] = []

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
        self.xai_records = []
        all_metrics = []
        all_strategy_returns: List[np.ndarray] = []

        for idx, split in enumerate(splits):
            metrics, fold_strategy_returns, record = self._run_single_fold(
                split, idx, len(splits), verbose
            )
            all_metrics.append(metrics)
            if fold_strategy_returns.size > 0:
                all_strategy_returns.append(fold_strategy_returns)
            self.results.append(record)

        if self.feature_importances:
            shapes = {arr.shape[0] for arr in self.feature_importances}
            if len(shapes) == 1:
                self.mean_feature_importance = np.vstack(self.feature_importances).mean(axis=0)

        if all_metrics:
            self.aggregated_metrics = self._aggregate_metrics(
                all_metrics, all_strategy_returns, verbose
            )

        return {
            "window_results": self.results,
            "average_metrics": self.aggregated_metrics,
        }

    def _run_single_fold(self, split: Dict, idx: int, total: int, verbose: bool):
        """Tek walk-forward penceresini eğitir/değerlendirir (leakage-free).

        Scaler fit yalnızca train dilimine; model sıfırdan kurulur. Feature
        importance side-effect olarak `self.feature_importances`'a eklenir.

        Returns:
            (metrics, fold_strategy_returns, result_record) üçlüsü.
        """
        if verbose:
            print(f"\n  [INFO] Walk-Forward Window {idx + 1}/{total} (Split Index: {split['split_idx']})")
            print(f"         Train points: {len(split['train'])}, Test points: {len(split['test'])}")
            print(
                f"         Train dates : {split.get('train_date_start')} -> {split.get('train_date_end')} | "
                f"Test dates: {split.get('test_date_start')} -> {split.get('test_date_end')}"
            )

        train_df = split["train"]
        test_df = split["test"]

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
        ) = self.preprocessor(train_df, test_df)

        model = self.model_initializer()
        model.train(X_train, y_train)
        preds = model.predict(X_test)

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

        # Sprint 3 A3.3: fold strategy returns concat-Sharpe icin biriktir.
        fold_strategy_returns = _compute_strategy_returns(
            y_true_target_aligned,
            preds_target_aligned,
            y_true_final,
            prev_close_aligned,
            self.target_mode,
            y_pred_price=preds_final,
        )

        record = {
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
        }
        xai_record = self._build_xai_record(
            model=model,
            X_train=X_train,
            X_test=X_test,
            fold_id=split["split_idx"],
            dates=dates_aligned,
            pred_target=preds_target_aligned,
            pred_price=preds_final,
            y_true_price=y_true_final,
        )
        if xai_record:
            self.xai_records.append(xai_record)
            record["xai_method"] = xai_record.get("method")
            record["xai_approximate"] = xai_record.get("approximate")
        return metrics, fold_strategy_returns, record

    def _build_xai_record(
        self,
        *,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        fold_id: Any,
        dates: np.ndarray,
        pred_target: np.ndarray,
        pred_price: np.ndarray,
        y_true_price: np.ndarray,
    ) -> Dict[str, Any]:
        if not self.feature_names:
            return {}
        try:
            X_train_arr = np.asarray(X_train, dtype=float)
            X_test_arr = np.asarray(X_test, dtype=float)
            if X_test_arr.ndim == 2:
                background = XAIBackgroundProvider.from_arrays(X_train=X_train_arr)
                strategy = TabularContributionStrategy(
                    self.feature_names,
                    max_rows=self.xai_max_rows,
                    background_provider=background,
                )
                estimator = getattr(model, "model", model)
                if hasattr(estimator, "coef_"):
                    contribs, method, approximate = strategy.linear_contributions(estimator, X_test_arr)
                else:
                    contribs, method, approximate = strategy.tree_contributions(estimator, X_test_arr)
            else:
                return {}
            contribs = np.asarray(contribs, dtype=float)
            if contribs.ndim == 1:
                contribs = contribs.reshape(1, -1)
            if contribs.shape[1] != len(self.feature_names):
                return {}
            min_len = min(len(contribs), len(dates), len(pred_target), len(pred_price), len(y_true_price))
            if min_len <= 0:
                return {}
            contribs = contribs[-min_len:]
            fold_importance = {
                feature: float(value)
                for feature, value in zip(self.feature_names, np.mean(np.abs(contribs), axis=0))
            }
            return {
                "fold": fold_id,
                "method": method,
                "approximate": bool(approximate),
                "background_scope": background.scope,
                "contributions": contribs.tolist(),
                "fold_importance": fold_importance,
                "dates": np.asarray(dates)[-min_len:].tolist(),
                "pred_target": np.asarray(pred_target, dtype=float)[-min_len:].tolist(),
                "pred_price": np.asarray(pred_price, dtype=float)[-min_len:].tolist(),
                "y_true_price": np.asarray(y_true_price, dtype=float)[-min_len:].tolist(),
            }
        except Exception as exc:
            return {
                "fold": fold_id,
                "method": "xai_error",
                "approximate": True,
                "background_scope": "unavailable",
                "error": type(exc).__name__,
            }

    def _aggregate_metrics(
        self, all_metrics: List[Dict], all_strategy_returns: List[np.ndarray], verbose: bool
    ) -> Dict[str, Any]:
        """Fold metriklerini sayısal-ortalar + concat-Sharpe & bootstrap %95 CI ekler."""

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
            # sign(pred) * getiri long/short, kendi kendini finanse eden bir seridir;
            # risksiz faiz fazlasi kazanmaz -> rf=0 (financial_metrics ile tutarli).
            rf = 0.0
            if concat_returns.size > 0:
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

        if verbose:
            print("\n  [INFO] Walk-Forward Complete. Average Metrics:")
            for key, value in avg_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"         {key}: {value:.4f}")
                else:
                    print(f"         {key}: {value}")

        return avg_metrics
