# -*- coding: utf-8 -*-
"""
financial_metrics.py — Advanced Financial Evaluation Metrics
Extends standard MSE/MAE with quantitative trading metrics like Sharpe Ratio and Directional Accuracy.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from typing import Dict

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

    # Approximated Sharpe Ratio (simulating buying on predicted up days and shorting down days)
    # Return = P_{t} - P_{t-1}
    # If predicted diff > 0, we hold long. If < 0, short.
    if len(diff_true) > 0:
        signals = np.sign(diff_pred)
        signals = np.where(signals == 0, 1, signals) # DEFAULT to long
        strategy_returns = signals * diff_true
        
        mean_strat_rtn = np.mean(strategy_returns)
        std_strat_rtn = np.std(strategy_returns)
        
        if std_strat_rtn > 0:
            # Annualize assuming approx 252 trading days (using single sample periods, rough scaling)
            # Normally we divide percentage return, but doing price diff ratio:
            sharpe = (mean_strat_rtn / std_strat_rtn) * np.sqrt(252) # Approximation
        else:
            sharpe = 0.0
            
        hit_rate = np.mean(strategy_returns > 0) * 100
    else:
        sharpe = 0.0
        hit_rate = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Dir_Acc": dir_acc,
        "Sharpe": sharpe,
        "Hit_Rate": hit_rate
    }
