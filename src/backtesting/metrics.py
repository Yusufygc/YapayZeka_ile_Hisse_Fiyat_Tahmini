# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _annualized_sharpe(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float).ravel()
    if returns.size == 0:
        return 0.0
    std_returns = np.std(returns)
    if std_returns <= 0:
        return 0.0
    return float((np.mean(returns) / std_returns) * np.sqrt(252))


def _max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float).ravel()
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity / np.maximum(running_max, 1e-12)) - 1.0
    return float(drawdowns.min())


def summarize_backtest(
    backtest_result: Dict[str, Any],
    initial_capital: float = 100000.0,
) -> Dict[str, float | str | bool]:
    equity_curve: pd.DataFrame = backtest_result["equity_curve"]
    trades: pd.DataFrame = backtest_result["trades"]
    model_name = backtest_result["model_name"]

    if equity_curve.empty:
        return {
            "Model": model_name,
            "Net_Return": 0.0,
            "Annualized_Return": 0.0,
            "Volatility": 0.0,
            "Sharpe": 0.0,
            "Max_Drawdown": 0.0,
            "Calmar": 0.0,
            "Exposure": 0.0,
            "Active_Bars": 0,
            "Signal_Count": 0,
            "Days_In_Market": 0,
            "Trade_Count": 0,
            "Turnover": 0.0,
            "Win_Rate": 0.0,
            "Avg_Trade_Return": 0.0,
            "Initial_Capital": round(initial_capital, 2),
            "End_Capital": round(initial_capital, 2),
            "Profit_TL": 0.0,
            "BuyHold_Return": 0.0,
            "BuyHold_End_Capital": round(initial_capital, 2),
            "BuyHold_Profit_TL": 0.0,
            "BuyHold_Sharpe": 0.0,
            "Beats_BuyHold_NetReturn": False,
        }

    strategy_returns = equity_curve["Net_Return"].to_numpy(dtype=float)
    buy_hold_returns = equity_curve["Realized_Return"].to_numpy(dtype=float)
    equity = equity_curve["Equity"].to_numpy(dtype=float)
    buy_hold_equity = equity_curve["BuyHold_Equity"].to_numpy(dtype=float)
    periods = len(equity_curve)

    net_return = float(equity[-1] - 1.0)
    buy_hold_return = float(buy_hold_equity[-1] - 1.0)
    annualized_return = float((equity[-1] ** (252.0 / max(periods, 1))) - 1.0) if periods > 0 else 0.0
    volatility = float(np.std(strategy_returns) * np.sqrt(252)) if periods > 1 else 0.0
    sharpe = _annualized_sharpe(strategy_returns)
    buy_hold_sharpe = _annualized_sharpe(buy_hold_returns)
    max_drawdown = _max_drawdown(equity)
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    exposure = float(np.mean(equity_curve["Position"].to_numpy(dtype=float)) * 100.0)
    active_bars = int(np.sum(equity_curve["Position"].to_numpy(dtype=float) > 0))
    signal_count = int(np.sum(equity_curve["Signal"].to_numpy(dtype=float) > 0))
    days_in_market = active_bars
    turnover = float(np.sum(np.abs(np.diff(np.concatenate(([0.0], equity_curve["Position"].to_numpy(dtype=float)))))))
    trade_count = int(len(trades))
    win_rate = float((trades["Net_Return"] > 0).mean() * 100.0) if not trades.empty else 0.0
    avg_trade_return = float(trades["Net_Return"].mean()) if not trades.empty else 0.0
    end_capital = float(initial_capital * equity[-1])
    buy_hold_end_capital = float(initial_capital * buy_hold_equity[-1])
    profit_tl = float(end_capital - initial_capital)
    buy_hold_profit_tl = float(buy_hold_end_capital - initial_capital)

    return {
        "Model": model_name,
        "Net_Return": round(net_return, 6),
        "Annualized_Return": round(annualized_return, 6),
        "Volatility": round(volatility, 6),
        "Sharpe": round(sharpe, 6),
        "Max_Drawdown": round(max_drawdown, 6),
        "Calmar": round(calmar, 6),
        "Exposure": round(exposure, 4),
        "Active_Bars": active_bars,
        "Signal_Count": signal_count,
        "Days_In_Market": days_in_market,
        "Trade_Count": trade_count,
        "Turnover": round(turnover, 6),
        "Win_Rate": round(win_rate, 4),
        "Avg_Trade_Return": round(avg_trade_return, 6),
        "Initial_Capital": round(initial_capital, 2),
        "End_Capital": round(end_capital, 2),
        "Profit_TL": round(profit_tl, 2),
        "BuyHold_Return": round(buy_hold_return, 6),
        "BuyHold_End_Capital": round(buy_hold_end_capital, 2),
        "BuyHold_Profit_TL": round(buy_hold_profit_tl, 2),
        "BuyHold_Sharpe": round(buy_hold_sharpe, 6),
        "Beats_BuyHold_NetReturn": net_return > buy_hold_return,
    }
