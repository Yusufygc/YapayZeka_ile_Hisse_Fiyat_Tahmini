# -*- coding: utf-8 -*-
"""Equity curve frame construction helpers for backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_FRAME_COLUMNS = [
    "Decision",
    "Recommendation",
    "Recommendation_TR",
    "Expected_Return",
    "Base_Entry_Threshold",
    "Entry_Threshold",
    "Exit_Threshold",
    "Signal_Strength",
    "Quality_Gate_Mode",
    "Quality_Threshold_Multiplier",
    "Quality_Gate_Reason",
    "Market_Regime_SMA200",
    "Regime_Threshold_Multiplier",
    "Volatility_Regime",
    "Volatility_Threshold_Multiplier",
    "Final_Threshold_Multiplier",
    "Rolling_Volatility",
    "Holding_Bars",
    "Trade_Return",
    "Take_Profit_Return",
    "Stop_Loss_Return",
    "Cooldown_Remaining",
    "Risk_State",
    "Signal_Reason",
]


def build_equity_curve(
    *,
    prediction_dates: pd.Index,
    dates: pd.Index,
    equity: np.ndarray,
    buy_hold_equity: np.ndarray,
    execution_positions: np.ndarray,
    decision_positions: np.ndarray,
    signals: np.ndarray,
    gross_strategy_returns: np.ndarray,
    net_strategy_returns: np.ndarray,
    transaction_costs: np.ndarray,
    commission_costs: np.ndarray,
    slippage_costs: np.ndarray,
    entry_transaction_costs: np.ndarray,
    exit_transaction_costs: np.ndarray,
    entry_events: np.ndarray,
    exit_events_for_cost: np.ndarray,
    realized_returns: np.ndarray,
    observed_returns: np.ndarray,
    pred_price: np.ndarray,
    prices: np.ndarray,
    prev_close: np.ndarray,
    fold_id_arr: np.ndarray,
    pred_target_arr: np.ndarray | None,
    signal_frame: pd.DataFrame,
) -> pd.DataFrame:
    equity_curve = pd.DataFrame({
        "Prediction_Date": prediction_dates,
        "Date": dates,
        "Execution_Date": dates,
        "Realized_Return_Date": dates,
        "Equity": equity,
        "BuyHold_Equity": buy_hold_equity,
        "Position": execution_positions,
        "Desired_Position": decision_positions,
        "Signal": signals,
        "Gross_Return": gross_strategy_returns,
        "Net_Return": net_strategy_returns,
        "Transaction_Cost": transaction_costs,
        "Commission_Cost": commission_costs,
        "Slippage_Cost": slippage_costs,
        "Entry_Transaction_Cost": entry_transaction_costs,
        "Exit_Transaction_Cost": exit_transaction_costs,
        "Entry_Event": entry_events,
        "Exit_Event": exit_events_for_cost,
        "Realized_Return": realized_returns,
        "Observed_Return_At_Decision": observed_returns,
        "Predicted_Price": pred_price,
        "Actual_Price": prices,
        "Prev_Close": prev_close,
        "Fold": fold_id_arr,
    })
    if pred_target_arr is not None:
        equity_curve["Predicted_Target"] = pred_target_arr
    attach_signal_columns(equity_curve, signal_frame)
    return equity_curve


def attach_signal_columns(equity_curve: pd.DataFrame, signal_frame: pd.DataFrame) -> None:
    for column in SIGNAL_FRAME_COLUMNS:
        if column in signal_frame.columns:
            equity_curve[column] = signal_frame[column].to_numpy()
