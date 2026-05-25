# -*- coding: utf-8 -*-
"""
financial_metrics.py - Forecast and financial evaluation metrics.

Price-space errors are still reported for comparability, but directional and
trading metrics are computed in the target/return space whenever those arrays
are provided by the pipeline.
"""

from typing import Dict

try:
    from src.utils.risk_free_rate import get_current_risk_free_rate as _get_rf
except ImportError:
    _get_rf = None

import numpy as np
try:
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
except ImportError:  # pragma: no cover - fallback for minimal validation runtimes
    def mean_absolute_error(y_true, y_pred):
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    def mean_squared_error(y_true, y_pred):
        err = np.asarray(y_true) - np.asarray(y_pred)
        return float(np.mean(err * err))

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))))


def _daily_risk_free_rate(risk_free_annual: float) -> float:
    return float((1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0)


def _annualized_sharpe(returns: np.ndarray, risk_free_annual: float | None = 0.0) -> float:
    """
    Sprint 1 (2026-05-25) Plan A1.1: risk_free_annual None ise Sharpe NaN
    doner; cagiran fonksiyon `risk_free_unavailable` uyarisini metric
    sozlugune ekler. Macro cache + env yoksa fail-loud.
    """
    if risk_free_annual is None:
        return float("nan")
    returns = np.asarray(returns, dtype=float).ravel()
    if returns.size == 0:
        return 0.0

    excess = returns - _daily_risk_free_rate(risk_free_annual)
    std_excess = np.std(excess)
    if std_excess <= 0:
        return 0.0

    return float((np.mean(excess) / std_excess) * np.sqrt(252))


def _price_to_simple_returns(y_true: np.ndarray, prev_close: np.ndarray | None = None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float).ravel()
    if prev_close is not None:
        prev_close = np.asarray(prev_close, dtype=float).ravel()
        k = min(len(y_true), len(prev_close))
        if k == 0:
            return np.asarray([], dtype=float)
        return (y_true[-k:] / np.maximum(prev_close[-k:], 1e-12)) - 1.0
    if y_true.size < 2:
        return np.asarray([], dtype=float)
    return (y_true[1:] / np.maximum(y_true[:-1], 1e-12)) - 1.0


def _target_to_simple_returns(target_values: np.ndarray, target_mode: str) -> np.ndarray:
    target_values = np.asarray(target_values, dtype=float).ravel()
    if target_mode == "log_return":
        return np.expm1(target_values)
    if target_mode == "return":
        return target_values
    return target_values


def _price_to_target_returns(
    values: np.ndarray,
    target_mode: str,
    prev_close: np.ndarray | None = None,
) -> np.ndarray:
    if target_mode == "price":
        return _price_to_simple_returns(values, prev_close)

    simple_returns = _price_to_simple_returns(values, prev_close)
    if target_mode == "log_return":
        return np.log1p(np.clip(simple_returns, -0.999999, None))
    if target_mode == "return":
        return simple_returns
    raise ValueError(f"Desteklenmeyen target_mode: {target_mode}")


def compute_buy_hold_sharpe(
    y_true: np.ndarray,
    prev_close: np.ndarray | None = None,
    risk_free_annual: float | None = None,
) -> float:
    """
    Compute buy-and-hold Sharpe from daily simple returns, not price differences.

    Sprint 1 (2026-05-25) Plan A1.1: risk_free_annual None ise macro cache'ten
    dinamik okunur; cache + env yoksa fail-loud — Sharpe NaN doner ve cagiran
    katmana `risk_free_unavailable` durumu sinyallenir.
    """
    if risk_free_annual is None:
        risk_free_annual = _get_rf() if _get_rf is not None else None
    return _annualized_sharpe(_price_to_simple_returns(y_true, prev_close), risk_free_annual)


def compute_financial_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_true_target: np.ndarray | None = None,
    y_pred_target: np.ndarray | None = None,
    prev_close: np.ndarray | None = None,
    target_mode: str = "price",
    risk_free_annual: float | None = None,
) -> Dict[str, float]:
    """
    Compute price-space forecast errors and return-space financial metrics.

    Sprint 1 (2026-05-25) Plan A1.1: risk_free_annual None ise macro cache'ten
    dinamik okunur. Cache + env yoksa fail-loud: Sharpe + BuyHold_Sharpe
    NaN doner ve sozluge `Risk_Free_Unavailable=True` +
    `Sharpe_Warning="risk_free_unavailable"` eklenir. Bu uyari ileride
    confidence.warnings zincirine baglanir (Sprint 8).
    """
    risk_free_unavailable = False
    if risk_free_annual is None:
        risk_free_annual = _get_rf() if _get_rf is not None else None
        if risk_free_annual is None:
            risk_free_unavailable = True
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    k_price = min(len(y_true), len(y_pred))
    y_true = y_true[-k_price:]
    y_pred = y_pred[-k_price:]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    safe_y_true = np.where(y_true == 0, 1e-10, y_true)
    mape = mean_absolute_percentage_error(safe_y_true, y_pred)

    if target_mode == "price":
        true_target = _price_to_simple_returns(
            y_true if y_true_target is None else y_true_target,
            prev_close,
        )
        pred_target = _price_to_simple_returns(
            y_pred if y_pred_target is None else y_pred_target,
            prev_close,
        )
    else:
        true_target = (
            _price_to_target_returns(y_true, target_mode, prev_close)
            if y_true_target is None
            else np.asarray(y_true_target, dtype=float).ravel()
        )
        pred_target = (
            _price_to_target_returns(y_pred, target_mode, prev_close)
            if y_pred_target is None
            else np.asarray(y_pred_target, dtype=float).ravel()
        )

    k_target = min(len(true_target), len(pred_target))
    true_target = true_target[-k_target:] if k_target else np.asarray([], dtype=float)
    pred_target = pred_target[-k_target:] if k_target else np.asarray([], dtype=float)

    if k_target:
        dir_acc = float(np.mean(np.sign(true_target) == np.sign(pred_target)) * 100)
        return_mae = float(mean_absolute_error(true_target, pred_target))
        return_rmse = float(np.sqrt(mean_squared_error(true_target, pred_target)))
    else:
        dir_acc = 0.0
        return_mae = 0.0
        return_rmse = 0.0

    buy_hold_sharpe = compute_buy_hold_sharpe(y_true, prev_close, risk_free_annual)

    realized_simple_returns = (
        _target_to_simple_returns(true_target, target_mode)
        if target_mode in {"log_return", "return"}
        else _price_to_simple_returns(y_true, prev_close)
    )
    predicted_signal_source = (
        pred_target
        if target_mode in {"log_return", "return"}
        else _price_to_target_returns(y_pred, "return", prev_close)
    )

    k_strategy = min(len(realized_simple_returns), len(predicted_signal_source))
    if k_strategy:
        signals = np.sign(predicted_signal_source[-k_strategy:])
        strategy_returns = signals * realized_simple_returns[-k_strategy:]
        sharpe = _annualized_sharpe(strategy_returns, risk_free_annual)

        active_mask = signals != 0
        hit_rate = float(np.mean(strategy_returns[active_mask] > 0) * 100) if np.any(active_mask) else 0.0
        neutral_rate = float(np.mean(~active_mask) * 100)
    else:
        sharpe = 0.0
        hit_rate = 0.0
        neutral_rate = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Return_MAE": return_mae,
        "Return_RMSE": return_rmse,
        "Dir_Acc": dir_acc,
        "Sharpe": sharpe,
        "Hit_Rate": hit_rate,
        "Neutral_Rate": neutral_rate,
        "BuyHold_Sharpe": buy_hold_sharpe,
        # Sprint 1 A1.1: rf yoksa downstream confidence chain'e bayrak.
        "Risk_Free_Unavailable": bool(risk_free_unavailable),
        "Risk_Free_Annual_Used": None if risk_free_unavailable else float(risk_free_annual),
        "Sharpe_Warning": "risk_free_unavailable" if risk_free_unavailable else "",
    }


def compute_quantile_metrics(
    y_true: np.ndarray,
    quantile_predictions: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    q_preds = np.asarray(quantile_predictions, dtype=float)
    if q_preds.ndim != 2 or q_preds.shape[1] < 2:
        return {}

    k = min(len(y_true), len(q_preds))
    y_true = y_true[-k:]
    q_preds = q_preds[-k:]
    q = tuple(quantiles[: q_preds.shape[1]])

    pinball_losses = []
    for idx, quantile in enumerate(q):
        err = y_true - q_preds[:, idx]
        pinball_losses.append(np.maximum(quantile * err, (quantile - 1.0) * err))

    metrics = {
        "Pinball_Loss": float(np.mean(np.column_stack(pinball_losses))),
        "Median_Pinball_Loss": float(np.mean(pinball_losses[len(pinball_losses) // 2])),
    }

    if q_preds.shape[1] >= 3:
        lower = q_preds[:, 0]
        upper = q_preds[:, -1]
        in_interval = (y_true >= lower) & (y_true <= upper)
        interval_width = np.maximum(upper - lower, 0.0)
        alpha = 1.0 - (q[-1] - q[0])
        below = np.maximum(lower - y_true, 0.0)
        above = np.maximum(y_true - upper, 0.0)
        winkler = interval_width + (2.0 / max(alpha, 1e-12)) * (below + above)
        metrics.update({
            "Interval_Coverage": float(np.mean(in_interval) * 100.0),
            "P10_P90_Coverage": float(np.mean(in_interval) * 100.0),
            "Avg_Interval_Width": float(np.mean(interval_width)),
            "Winkler_Score": float(np.mean(winkler)),
        })

    return metrics
