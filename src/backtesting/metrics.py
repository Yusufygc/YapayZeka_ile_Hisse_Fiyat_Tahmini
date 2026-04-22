# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _annualized_sharpe(returns: np.ndarray, risk_free_annual: float = 0.40) -> float:
    rf_daily = (1 + risk_free_annual) ** (1/252) - 1
    excess = returns - rf_daily
    std_excess = np.std(excess)
    if std_excess <= 0:
        return 0.0
    return float(np.mean(excess) / std_excess * np.sqrt(252))


def _annualized_sortino(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float).ravel()
    if returns.size == 0:
        return 0.0
    downside = returns[returns < 0]
    if downside.size == 0:
        return 0.0
    downside_std = np.std(downside)
    if downside_std <= 0:
        return 0.0
    return float((np.mean(returns) / downside_std) * np.sqrt(252))


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
            "CAGR": 0.0,
            "Volatility": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
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
            "Avg_Holding_Period": 0.0,
            "Profit_Factor": 0.0,
            "Avg_Win": 0.0,
            "Avg_Loss": 0.0,
            "Expectancy": 0.0,
            "Cost_Drag": 0.0,
            "Commission_Drag": 0.0,
            "Slippage_Drag": 0.0,
            "Entry_Cost_Drag": 0.0,
            "Exit_Cost_Drag": 0.0,
            "Trade_Efficiency": 0.0,
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
    cagr = annualized_return
    volatility = float(np.std(strategy_returns) * np.sqrt(252)) if periods > 1 else 0.0
    sharpe = _annualized_sharpe(strategy_returns)
    sortino = _annualized_sortino(strategy_returns)
    buy_hold_sharpe = _annualized_sharpe(buy_hold_returns)
    max_drawdown = _max_drawdown(equity)
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    exposure = float(np.mean(equity_curve["Position"].to_numpy(dtype=float)) * 100.0)
    active_bars = int(np.sum(equity_curve["Position"].to_numpy(dtype=float) > 0))
    signal_count = int(np.sum(equity_curve["Signal"].to_numpy(dtype=float) > 0))
    days_in_market = active_bars
    if {"Entry_Event", "Exit_Event"}.issubset(equity_curve.columns):
        turnover = float(equity_curve["Entry_Event"].sum() + equity_curve["Exit_Event"].sum())
    else:
        turnover = float(np.sum(np.abs(np.diff(np.concatenate(([0.0], equity_curve["Position"].to_numpy(dtype=float)))))))
    trade_count = int(len(trades))
    win_rate = float((trades["Net_Return"] > 0).mean() * 100.0) if not trades.empty else 0.0
    avg_trade_return = float(trades["Net_Return"].mean()) if not trades.empty else 0.0
    avg_holding_period = float(trades["Holding_Period"].mean()) if not trades.empty and "Holding_Period" in trades.columns else 0.0
    profit_factor, avg_win, avg_loss, expectancy = _trade_quality_metrics(trades)
    cost_drag = float(equity_curve["Transaction_Cost"].sum()) if "Transaction_Cost" in equity_curve.columns else 0.0
    commission_drag = float(equity_curve["Commission_Cost"].sum()) if "Commission_Cost" in equity_curve.columns else 0.0
    slippage_drag = float(equity_curve["Slippage_Cost"].sum()) if "Slippage_Cost" in equity_curve.columns else 0.0
    entry_cost_drag = float(equity_curve["Entry_Transaction_Cost"].sum()) if "Entry_Transaction_Cost" in equity_curve.columns else 0.0
    exit_cost_drag = float(equity_curve["Exit_Transaction_Cost"].sum()) if "Exit_Transaction_Cost" in equity_curve.columns else 0.0
    gross_return = float(np.prod(1.0 + equity_curve["Gross_Return"].to_numpy(dtype=float)) - 1.0) if "Gross_Return" in equity_curve.columns else net_return
    trade_efficiency = _trade_efficiency(net_return, gross_return, cost_drag, max_drawdown)
    end_capital = float(initial_capital * equity[-1])
    buy_hold_end_capital = float(initial_capital * buy_hold_equity[-1])
    profit_tl = float(end_capital - initial_capital)
    buy_hold_profit_tl = float(buy_hold_end_capital - initial_capital)

    return {
        "Model": model_name,
        "Net_Return": round(net_return, 6),
        "Annualized_Return": round(annualized_return, 6),
        "CAGR": round(cagr, 6),
        "Volatility": round(volatility, 6),
        "Sharpe": round(sharpe, 6),
        "Sortino": round(sortino, 6),
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
        "Avg_Holding_Period": round(avg_holding_period, 4),
        "Profit_Factor": round(profit_factor, 6),
        "Avg_Win": round(avg_win, 6),
        "Avg_Loss": round(avg_loss, 6),
        "Expectancy": round(expectancy, 6),
        "Cost_Drag": round(cost_drag, 6),
        "Commission_Drag": round(commission_drag, 6),
        "Slippage_Drag": round(slippage_drag, 6),
        "Entry_Cost_Drag": round(entry_cost_drag, 6),
        "Exit_Cost_Drag": round(exit_cost_drag, 6),
        "Trade_Efficiency": round(trade_efficiency, 6),
        "Initial_Capital": round(initial_capital, 2),
        "End_Capital": round(end_capital, 2),
        "Profit_TL": round(profit_tl, 2),
        "BuyHold_Return": round(buy_hold_return, 6),
        "BuyHold_End_Capital": round(buy_hold_end_capital, 2),
        "BuyHold_Profit_TL": round(buy_hold_profit_tl, 2),
        "BuyHold_Sharpe": round(buy_hold_sharpe, 6),
        "Beats_BuyHold_NetReturn": net_return > buy_hold_return,
    }


def _trade_quality_metrics(trades: pd.DataFrame) -> tuple[float, float, float, float]:
    if trades.empty or "Net_Return" not in trades.columns:
        return 0.0, 0.0, 0.0, 0.0

    returns = trades["Net_Return"].to_numpy(dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    win_probability = len(wins) / len(returns) if len(returns) else 0.0
    loss_probability = len(losses) / len(returns) if len(returns) else 0.0
    expectancy = (win_probability * avg_win) + (loss_probability * avg_loss)
    return float(profit_factor), avg_win, avg_loss, float(expectancy)


def _trade_efficiency(net_return: float, gross_return: float, cost_drag: float, max_drawdown: float) -> float:
    if gross_return == 0 and cost_drag == 0 and max_drawdown == 0:
        return 0.0
    denominator = abs(gross_return) + cost_drag + abs(max_drawdown)
    if denominator <= 0:
        return 0.0
    return float(net_return / denominator)
