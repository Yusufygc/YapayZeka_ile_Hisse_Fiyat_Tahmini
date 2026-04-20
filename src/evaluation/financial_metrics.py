# -*- coding: utf-8 -*-
"""
financial_metrics.py — Advanced Financial Evaluation Metrics
Extends standard MSE/MAE with quantitative trading metrics like Sharpe Ratio and Directional Accuracy.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from typing import Dict


def _annualized_sharpe(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float).ravel()
    if returns.size == 0:
        return 0.0

    std_returns = np.std(returns)
    if std_returns <= 0:
        return 0.0

    return float((np.mean(returns) / std_returns) * np.sqrt(252))


def compute_buy_hold_sharpe(y_true: np.ndarray) -> float:
    """
    Basit buy-and-hold referansı için fiyat farklarından yıllıklandırılmış Sharpe üretir.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    if y_true.size < 2:
        return 0.0

    return _annualized_sharpe(np.diff(y_true))

def compute_financial_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes both standard regression metrics and financial evaluation metrics.
    """
    # Standard regression
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Handle zeros in true values for MAPE
    safe_y_true = np.where(y_true == 0, 1e-10, y_true)
    mape = mean_absolute_percentage_error(safe_y_true, y_pred)
    
    # Financial metrics
    # Directional Accuracy (Did we predict the sign of the return correctly?)
    diff_true = np.diff(y_true)
    diff_pred = np.diff(y_pred)
    
    if len(diff_true) > 0:
        dir_acc = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100
    else:
        dir_acc = 0.0

    buy_hold_sharpe = compute_buy_hold_sharpe(y_true)

    # Approximated Sharpe Ratio (simulating buying on predicted up days and shorting down days)
    # Neutral signals stay neutral; they are not coerced into long positions.
    if len(diff_true) > 0:
        signals = np.sign(diff_pred)
        strategy_returns = signals * diff_true
        sharpe = _annualized_sharpe(strategy_returns)

        active_mask = signals != 0
        if np.any(active_mask):
            hit_rate = float(np.mean(strategy_returns[active_mask] > 0) * 100)
        else:
            hit_rate = 0.0

        neutral_rate = float(np.mean(~active_mask) * 100)
    else:
        sharpe = 0.0
        hit_rate = 0.0
        neutral_rate = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Dir_Acc": dir_acc,
        "Sharpe": sharpe,
        "Hit_Rate": hit_rate,
        "Neutral_Rate": neutral_rate,
        "BuyHold_Sharpe": buy_hold_sharpe,
    }
