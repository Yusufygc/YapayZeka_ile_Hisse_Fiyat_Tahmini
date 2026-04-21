# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.backtesting.signals import generate_long_flat_signals


def run_backtest(
    *,
    dates,
    y_true_price,
    pred_price,
    prev_close,
    model_name: str,
    validation_mode: str,
    target_mode: str,
    pred_target=None,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> Dict[str, Any]:
    prices = np.asarray(y_true_price, dtype=float).ravel()
    pred_price = np.asarray(pred_price, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    pred_target_arr = None if pred_target is None else np.asarray(pred_target, dtype=float).ravel()

    if dates is None:
        dates = pd.RangeIndex(start=0, stop=len(prices), step=1)
    dates = pd.to_datetime(pd.Index(dates))

    lengths = [len(prices), len(pred_price), len(prev_close), len(dates)]
    if pred_target_arr is not None:
        lengths.append(len(pred_target_arr))
    n = min(lengths) if lengths else 0
    if n == 0:
        return {
            "model_name": model_name,
            "validation_mode": validation_mode,
            "equity_curve": pd.DataFrame(columns=["Date", "Equity", "BuyHold_Equity", "Position", "Signal", "Net_Return"]),
            "trades": pd.DataFrame(columns=["Model", "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price", "Gross_Return", "Net_Return", "Holding_Period"]),
            "series": {},
        }

    prices = prices[-n:]
    pred_price = pred_price[-n:]
    prev_close = prev_close[-n:]
    dates = dates[-n:]
    if pred_target_arr is not None:
        pred_target_arr = pred_target_arr[-n:]

    realized_returns = (prices / np.maximum(prev_close, 1e-12)) - 1.0
    buy_hold_equity = np.cumprod(1.0 + realized_returns)

    signals = generate_long_flat_signals(
        pred_target=pred_target_arr,
        pred_price=pred_price,
        prev_close=prev_close,
        target_mode=target_mode,
    )
    positions = signals.copy()
    position_changes = np.abs(np.diff(np.concatenate(([0.0], positions))))
    total_cost_rate = (commission_bps + slippage_bps) / 10000.0
    transaction_costs = position_changes * total_cost_rate
    gross_strategy_returns = positions * realized_returns
    net_strategy_returns = gross_strategy_returns - transaction_costs
    equity = np.cumprod(1.0 + net_strategy_returns)

    trade_rows = []
    entry_idx = None
    for idx in range(n):
        current_position = positions[idx]
        prev_position = positions[idx - 1] if idx > 0 else 0.0
        opened = prev_position == 0.0 and current_position == 1.0
        closed = prev_position == 1.0 and current_position == 0.0

        if opened:
            entry_idx = idx
        if closed and entry_idx is not None:
            gross_trade_return = float(np.prod(1.0 + realized_returns[entry_idx:idx]) - 1.0)
            net_trade_return = float(np.prod(1.0 + net_strategy_returns[entry_idx:idx]) - 1.0)
            trade_rows.append({
                "Model": model_name,
                "Entry_Date": dates[entry_idx],
                "Exit_Date": dates[idx],
                "Entry_Price": float(prev_close[entry_idx]),
                "Exit_Price": float(prev_close[idx]),
                "Gross_Return": gross_trade_return,
                "Net_Return": net_trade_return,
                "Holding_Period": int(idx - entry_idx),
            })
            entry_idx = None

    if entry_idx is not None:
        gross_trade_return = float(np.prod(1.0 + realized_returns[entry_idx:]) - 1.0)
        net_trade_return = float(np.prod(1.0 + net_strategy_returns[entry_idx:]) - 1.0)
        trade_rows.append({
            "Model": model_name,
            "Entry_Date": dates[entry_idx],
            "Exit_Date": dates[-1],
            "Entry_Price": float(prev_close[entry_idx]),
            "Exit_Price": float(prices[-1]),
            "Gross_Return": gross_trade_return,
            "Net_Return": net_trade_return,
            "Holding_Period": int((n - 1) - entry_idx),
        })

    equity_curve = pd.DataFrame({
        "Date": dates,
        "Equity": equity,
        "BuyHold_Equity": buy_hold_equity,
        "Position": positions,
        "Signal": signals,
        "Gross_Return": gross_strategy_returns,
        "Net_Return": net_strategy_returns,
        "Transaction_Cost": transaction_costs,
        "Realized_Return": realized_returns,
        "Predicted_Price": pred_price,
        "Actual_Price": prices,
        "Prev_Close": prev_close,
    })
    if pred_target_arr is not None:
        equity_curve["Predicted_Target"] = pred_target_arr

    return {
        "model_name": model_name,
        "validation_mode": validation_mode,
        "equity_curve": equity_curve,
        "trades": pd.DataFrame(trade_rows),
        "series": {
            "positions": positions,
            "position_changes": position_changes,
            "transaction_costs": transaction_costs,
            "strategy_returns": net_strategy_returns,
            "buy_hold_returns": realized_returns,
        },
    }
