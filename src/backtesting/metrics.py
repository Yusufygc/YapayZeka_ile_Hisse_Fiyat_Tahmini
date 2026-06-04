# -*- coding: utf-8 -*-
"""Backtest performans metrikleri.

Sorumluluklar:
  - summarize_backtest(): getiri, drawdown, Sharpe/Sortino, hit-rate gibi özet
    metrikleri hesaplar.
  - Risk-free oran yoksa Sharpe/Sortino NaN döner ve Risk_Free_Unavailable
    bayrağı yükselir (sessiz fallback yok).
"""

from __future__ import annotations

try:
    from src.utils.risk_free_rate import get_current_risk_free_rate as _get_rf
except ImportError:
    _get_rf = None

from typing import Any, Dict

import math
import numpy as np
import pandas as pd


def _daily_risk_free_rate(risk_free_annual: float) -> float:
    return float((1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0)


def _annualized_sharpe(returns: np.ndarray, risk_free_annual: float | None = 0.0) -> float:
    """
    Sprint 1 (2026-05-25): risk_free_annual None ise Sharpe NaN doner.
    Cagiran fonksiyon bu NaN'i `risk_free_unavailable` uyarisi ile
    metric sozlugune yansitir.
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
    return float(np.mean(excess) / std_excess * np.sqrt(252))


def _annualized_sortino(returns: np.ndarray, risk_free_annual: float | None = 0.0) -> float:
    """Sprint 1: risk_free_annual None ise Sortino NaN doner."""
    if risk_free_annual is None:
        return float("nan")
    returns = np.asarray(returns, dtype=float).ravel()
    if returns.size == 0:
        return 0.0
    excess = returns - _daily_risk_free_rate(risk_free_annual)
    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0
    downside_std = np.std(downside)
    if downside_std <= 0:
        return 0.0
    return float((np.mean(excess) / downside_std) * np.sqrt(252))


def _max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float).ravel()
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity / np.maximum(running_max, 1e-12)) - 1.0
    return float(drawdowns.min())


def _var_cvar(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    returns = np.asarray(returns, dtype=float).ravel()
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return 0.0, 0.0

    var = float(np.quantile(returns, 1.0 - confidence))
    tail = returns[returns <= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _deflated_sharpe(
    sharpe: float,
    returns: np.ndarray,
    trial_count: int = 1,
) -> tuple[float, float]:
    returns = np.asarray(returns, dtype=float).ravel()
    returns = returns[np.isfinite(returns)]
    if returns.size < 3 or not np.isfinite(sharpe):
        return 0.0, 0.0

    trials = max(1, int(trial_count))
    std_error = math.sqrt(max(1e-12, (1.0 + 0.5 * sharpe * sharpe) / returns.size))
    multiple_testing_penalty = math.sqrt(2.0 * math.log(trials)) * std_error if trials > 1 else 0.0
    deflated = float(sharpe - multiple_testing_penalty)
    probabilistic_score = float(_normal_cdf(deflated / std_error) * 100.0)
    return deflated, probabilistic_score


def _resolve_risk_free(risk_free_annual: float | None) -> tuple[float | None, bool]:
    """risk_free_annual None ise macro cache + env'den çözer (fail-loud).

    Returns:
        (oran, unavailable) — bulunamazsa (None, True).
    """
    if risk_free_annual is not None:
        return risk_free_annual, False
    resolved = _get_rf() if _get_rf is not None else None
    return resolved, (resolved is None)


def _empty_metrics(
    model_name: str,
    initial_capital: float,
    risk_free_unavailable: bool,
    risk_free_annual: float | None,
) -> Dict[str, float | str | bool]:
    """equity_curve boşken döndürülen sıfır-metrik sözlüğü."""
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
        "VaR_95": 0.0,
        "CVaR_95": 0.0,
        "Deflated_Sharpe": 0.0,
        "Sharpe_Probabilistic_Score": 0.0,
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
        "BuyHold_VaR_95": 0.0,
        "BuyHold_CVaR_95": 0.0,
        "BuyHold_End_Capital": round(initial_capital, 2),
        "BuyHold_Profit_TL": 0.0,
        "BuyHold_Sharpe": 0.0,
        "Beats_BuyHold_NetReturn": False,
        "Omega_Ratio": 0.0,
        "Recovery_Factor": 0.0,
        "Max_Consecutive_Loss": 0,
        "Information_Ratio": 0.0,
        "Risk_Free_Unavailable": bool(risk_free_unavailable),
        "Risk_Free_Annual_Used": None if risk_free_unavailable else float(risk_free_annual),
        "Sharpe_Warning": "risk_free_unavailable" if risk_free_unavailable else "",
    }


def _exposure_and_turnover(equity_curve: pd.DataFrame) -> tuple[float, int, int, int, float]:
    """(exposure%, active_bars, signal_count, days_in_market, turnover)."""
    position = equity_curve["Position"].to_numpy(dtype=float)
    exposure = float(np.mean(position) * 100.0)
    active_bars = int(np.sum(position > 0))
    signal_count = int(np.sum(equity_curve["Signal"].to_numpy(dtype=float) > 0))
    days_in_market = active_bars
    if {"Entry_Event", "Exit_Event"}.issubset(equity_curve.columns):
        turnover = float(equity_curve["Entry_Event"].sum() + equity_curve["Exit_Event"].sum())
    else:
        turnover = float(np.sum(np.abs(np.diff(np.concatenate(([0.0], position))))))
    return exposure, active_bars, signal_count, days_in_market, turnover


def _cost_drags(
    equity_curve: pd.DataFrame, net_return: float
) -> tuple[float, float, float, float, float, float]:
    """(cost, commission, slippage, entry, exit, gross_return) — eksik kolon 0.0,
    Gross_Return yoksa gross_return net_return'e düşer."""
    def _col_sum(name: str) -> float:
        return float(equity_curve[name].sum()) if name in equity_curve.columns else 0.0

    cost_drag = _col_sum("Transaction_Cost")
    commission_drag = _col_sum("Commission_Cost")
    slippage_drag = _col_sum("Slippage_Cost")
    entry_cost_drag = _col_sum("Entry_Transaction_Cost")
    exit_cost_drag = _col_sum("Exit_Transaction_Cost")
    if "Gross_Return" in equity_curve.columns:
        gross_return = float(np.prod(1.0 + equity_curve["Gross_Return"].to_numpy(dtype=float)) - 1.0)
    else:
        gross_return = net_return
    return cost_drag, commission_drag, slippage_drag, entry_cost_drag, exit_cost_drag, gross_return


def _basic_trade_stats(trades: pd.DataFrame) -> tuple[int, float, float, float]:
    """(trade_count, win_rate%, avg_trade_return, avg_holding_period); boşsa sıfır."""
    trade_count = int(len(trades))
    if trades.empty:
        return trade_count, 0.0, 0.0, 0.0
    win_rate = float((trades["Net_Return"] > 0).mean() * 100.0)
    avg_trade_return = float(trades["Net_Return"].mean())
    avg_holding_period = (
        float(trades["Holding_Period"].mean()) if "Holding_Period" in trades.columns else 0.0
    )
    return trade_count, win_rate, avg_trade_return, avg_holding_period


def summarize_backtest(
    backtest_result: Dict[str, Any],
    initial_capital: float = 100000.0,
    risk_free_annual: float | None = None,
    trial_count: int = 1,
) -> Dict[str, float | str | bool]:
    """Backtest sonucundan özet performans metriklerini hesaplar.

    Net getiri, drawdown, hit-rate, Sharpe/Sortino gibi metrikleri üretir.
    Risk-free oran None ise macro cache + env'den çözülmeye çalışılır; yine
    bulunamazsa Sharpe/Sortino NaN döner ve `Risk_Free_Unavailable` bayrağı
    eklenir (sessiz fallback yok). `trial_count` çoklu deneme deflated-Sharpe
    düzeltmesi için kullanılır.

    Args:
        backtest_result: `equity_curve` ve işlem/pozisyon alanlarını içeren sonuç.
        initial_capital: Başlangıç sermayesi.
        risk_free_annual: Yıllık risk-free oran; None ise otomatik çözülür.
        trial_count: Deneme sayısı (deflated-Sharpe için).

    Returns:
        Metrik adı -> değer sözlüğü (`Risk_Free_Unavailable` bayrağı dahil).
    """
    # Sprint 1 (2026-05-25) Plan A1.1: risk_free None ise fail-loud — macro cache
    # + env yoksa Sharpe/Sortino NaN gelir, "Risk_Free_Unavailable" bayragi eklenir.
    risk_free_annual, risk_free_unavailable = _resolve_risk_free(risk_free_annual)
    equity_curve: pd.DataFrame = backtest_result["equity_curve"]
    trades: pd.DataFrame = backtest_result["trades"]
    model_name = backtest_result["model_name"]

    if equity_curve.empty:
        return _empty_metrics(model_name, initial_capital, risk_free_unavailable, risk_free_annual)

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
    sharpe = _annualized_sharpe(strategy_returns, risk_free_annual)
    sortino = _annualized_sortino(strategy_returns, risk_free_annual)
    buy_hold_sharpe = _annualized_sharpe(buy_hold_returns, risk_free_annual)
    var_95, cvar_95 = _var_cvar(strategy_returns, confidence=0.95)
    buy_hold_var_95, buy_hold_cvar_95 = _var_cvar(buy_hold_returns, confidence=0.95)
    deflated_sharpe, sharpe_probabilistic_score = _deflated_sharpe(
        sharpe,
        strategy_returns,
        trial_count=trial_count,
    )
    max_drawdown = _max_drawdown(equity)
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else float("inf")
    # Sprint 1: rf yoksa daily_rf 0.0 alinir; omega/recovery rf'siz cikar.
    daily_rf = 0.0 if risk_free_unavailable else _daily_risk_free_rate(risk_free_annual)
    omega = _omega_ratio(strategy_returns, threshold=daily_rf)
    recovery_factor = _recovery_factor(net_return, max_drawdown)
    max_consec_loss = _max_consecutive_loss(trades)
    information_ratio = _information_ratio(strategy_returns, buy_hold_returns)
    exposure, active_bars, signal_count, days_in_market, turnover = _exposure_and_turnover(equity_curve)
    trade_count, win_rate, avg_trade_return, avg_holding_period = _basic_trade_stats(trades)
    profit_factor, avg_win, avg_loss, expectancy = _trade_quality_metrics(trades)
    (
        cost_drag,
        commission_drag,
        slippage_drag,
        entry_cost_drag,
        exit_cost_drag,
        gross_return,
    ) = _cost_drags(equity_curve, net_return)
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
        "VaR_95": round(var_95, 6),
        "CVaR_95": round(cvar_95, 6),
        "Deflated_Sharpe": round(deflated_sharpe, 6),
        "Sharpe_Probabilistic_Score": round(sharpe_probabilistic_score, 6),
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
        "BuyHold_VaR_95": round(buy_hold_var_95, 6),
        "BuyHold_CVaR_95": round(buy_hold_cvar_95, 6),
        "BuyHold_End_Capital": round(buy_hold_end_capital, 2),
        "BuyHold_Profit_TL": round(buy_hold_profit_tl, 2),
        "BuyHold_Sharpe": round(buy_hold_sharpe, 6),
        "Beats_BuyHold_NetReturn": net_return > buy_hold_return,
        "Omega_Ratio": round(omega, 6) if np.isfinite(omega) else float("inf"),
        "Recovery_Factor": round(recovery_factor, 6) if np.isfinite(recovery_factor) else float("inf"),
        "Max_Consecutive_Loss": max_consec_loss,
        "Information_Ratio": round(information_ratio, 6),
        "Risk_Free_Unavailable": bool(risk_free_unavailable),
        "Risk_Free_Annual_Used": None if risk_free_unavailable else float(risk_free_annual),
        "Sharpe_Warning": "risk_free_unavailable" if risk_free_unavailable else "",
    }


def _omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega Ratio: kazanc dagiliminin kayip dagilimina orani (threshold uzerinde/altinda).
    threshold: gunluk risk-free orani olarak kullanilir.
    """
    excess = returns - threshold
    gains = np.sum(excess[excess > 0])
    losses = np.sum(np.abs(excess[excess < 0]))
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return float(gains / losses)


def _recovery_factor(net_return: float, max_drawdown: float) -> float:
    """
    Recovery Factor: Net getiri / Maksimum drawdown (mutlak deger).
    Yuksek deger = drawdown hizla geri kazanildi.
    """
    if max_drawdown == 0:
        return float("inf") if net_return > 0 else 0.0
    return float(net_return / abs(max_drawdown))


def _max_consecutive_loss(trades: pd.DataFrame) -> int:
    """
    Maksimum ardisik kaybeden islem sayisi.
    """
    if trades.empty or "Net_Return" not in trades.columns:
        return 0
    returns = trades["Net_Return"].to_numpy(dtype=float)
    max_consec = 0
    current = 0
    for r in returns:
        if r < 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    return int(max_consec)


def _information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """
    Information Ratio: aktif getiri / tracking error (yillandirilmis).
    IR > 0.5 iyi, > 1.0 mukemmel kabul edilir.
    """
    active = strategy_returns - benchmark_returns
    tracking_error = float(np.std(active) * np.sqrt(252))
    if tracking_error == 0:
        return 0.0
    active_return = float(np.mean(active) * 252)
    return float(active_return / tracking_error)


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
