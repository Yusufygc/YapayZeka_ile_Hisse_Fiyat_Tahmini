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

try:
    from src.utils.risk_free_rate import get_risk_free_rate_series as _get_rf_series
except ImportError:
    _get_rf_series = None

from typing import Any, Dict

import math
import numpy as np
import pandas as pd

# Annualisation yapilabilmesi icin gereken asgari bar sayisi (yarim yil). Daha
# kisa orneklemde (ornegin 30 barlik fold) CAGR/Calmar ekstrapolasyonu anlamsiz
# (absurt) degerler uretir -> NaN.
MIN_BARS_FOR_ANNUALIZATION = 126

_EPS = 1e-14


def _daily_risk_free_rate(risk_free_annual: float) -> float:
    return float((1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0)


def _excess_returns(
    returns: np.ndarray,
    risk_free_annual: float,
    exposure: np.ndarray | None = None,
    rf_daily: np.ndarray | None = None,
) -> np.ndarray:
    """Excess return serisi.

    ``exposure`` verilmezse ``returns - rf`` (eski davranis). Verilirse
    ``returns - rf * exposure``: nakit (flat) barlar rf kazanir, dolayisiyla
    dusuk-exposure stratejiler her bar rf kesintisi yuzunden yapisal olarak
    negatif Sharpe almaz. ``rf_daily`` verilirse (bara hizali, point-in-time
    gunluk rf) skaler rf yerine kullanilir; NaN girdiler skaler rf'ye dusur.
    Bu duzeltme YALNIZCA excess-return oranlari icindir; equity / Net_Return
    saf strateji getirisidir (rf yield eklenmez).
    """
    returns = np.asarray(returns, dtype=float).ravel()
    daily_scalar = _daily_risk_free_rate(risk_free_annual)
    if rf_daily is None:
        rf_vec: np.ndarray | float = daily_scalar
    else:
        rf_vec = np.asarray(rf_daily, dtype=float).ravel()
        if rf_vec.size != returns.size:
            raise ValueError("rf_daily uzunlugu returns ile ayni olmali")
        rf_vec = np.where(np.isfinite(rf_vec), rf_vec, daily_scalar)
    if exposure is None:
        return returns - rf_vec
    exp = np.asarray(exposure, dtype=float).ravel()
    if exp.size != returns.size:
        raise ValueError("exposure uzunlugu returns ile ayni olmali")
    exp = np.clip(np.nan_to_num(exp, nan=0.0), 0.0, 1.0)
    return returns - rf_vec * exp


def _annualized_sharpe(
    returns: np.ndarray,
    risk_free_annual: float | None = 0.0,
    exposure: np.ndarray | None = None,
    rf_daily: np.ndarray | None = None,
) -> float:
    """Yillandirilmis Sharpe (ddof=1).

    risk_free_annual None ise NaN doner (cagiran `risk_free_unavailable`
    uyarisini metric sozlugune yansitir). ``exposure`` / ``rf_daily`` icin
    bkz. :func:`_excess_returns`.
    """
    if risk_free_annual is None:
        return float("nan")
    returns = np.asarray(returns, dtype=float).ravel()
    if returns.size < 2:
        return 0.0
    excess = _excess_returns(returns, risk_free_annual, exposure, rf_daily)
    std_excess = np.std(excess, ddof=1)
    if not std_excess > _EPS:
        return 0.0
    return float(np.mean(excess) / std_excess * np.sqrt(252))


def _annualized_sortino(
    returns: np.ndarray,
    risk_free_annual: float | None = 0.0,
    exposure: np.ndarray | None = None,
    rf_daily: np.ndarray | None = None,
) -> float:
    """Yillandirilmis Sortino: hedef (=rf) yarı-sapması, TUM N gozlem uzerinden.

    downside deviation = sqrt(mean(min(excess, 0)**2)). Hic kayip yoksa
    (downside deviation == 0) NaN doner (0.0 donmek en iyi stratejiyi vasat
    olanlarin altina siraliyordu). Bos girdi -> NaN. risk_free_annual None ->
    NaN.
    """
    if risk_free_annual is None:
        return float("nan")
    returns = np.asarray(returns, dtype=float).ravel()
    if returns.size == 0:
        return float("nan")
    excess = _excess_returns(returns, risk_free_annual, exposure, rf_daily)
    downside_dev = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    if not downside_dev > _EPS:
        return float("nan")
    return float(np.mean(excess) / downside_dev * np.sqrt(252))


def _max_drawdown(equity: np.ndarray) -> float:
    """Maksimum drawdown; baslangic equity'si (1.0) dahil — ilk bar kaybi sayilir."""
    equity = np.asarray(equity, dtype=float).ravel()
    if equity.size == 0:
        return 0.0
    equity = np.concatenate(([1.0], equity))
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


def _normal_ppf(p: float) -> float:
    """Standart normal quantile (scipy varsa o, yoksa Acklam rasyonel yaklasimi)."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except ImportError:
        pass
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


_EULER_GAMMA = 0.5772156649015329


def _deflated_sharpe(
    sharpe: float,
    returns: np.ndarray,
    trial_count: int = 1,
    excess_returns: np.ndarray | None = None,
) -> tuple[float, float]:
    """Probabilistic / Deflated Sharpe (Bailey & Lopez de Prado).

    Hesap PER-PERIOD (gunluk) Sharpe uzerinde yapilir; ``sharpe`` (yillik)
    parametresi yalnizca geriye uyumluluk ve gecerlilik kontrolu icindir
    (sonlu degilse (0, 0) doner), deger ``returns``'ten yeniden hesaplanir.
    Excess seri icin ``excess_returns`` verilebilir (rf / exposure
    duzeltmeli); verilmezse ``returns`` excess kabul edilir.

        sr   = mean(x) / std(x, ddof=1)
        SE   = sqrt((1 - skew*sr + (kurt-1)/4 * sr^2) / (N-1))   (kurt: non-excess)
        sr0  = SE * ((1-g) Phi^-1(1-1/T) + g Phi^-1(1-1/(T e)))   (T>1; aksi 0)
        PSR  = Phi((sr - sr0) / SE)

    Returns:
        (deflated_sharpe_annualized = (sr - sr0) * sqrt(252), PSR yuzdesi).
    """
    base = excess_returns if excess_returns is not None else returns
    x = np.asarray(base, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 3 or not np.isfinite(sharpe):
        return 0.0, 0.0

    std = float(np.std(x, ddof=1))
    if not std > _EPS:
        return 0.0, 0.0
    n = x.size
    sr = float(np.mean(x) / std)
    centred = x - np.mean(x)
    m2 = float(np.mean(centred ** 2))
    skew = float(np.mean(centred ** 3) / m2 ** 1.5)
    kurt = float(np.mean(centred ** 4) / m2 ** 2)

    variance = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (n - 1)
    std_error = math.sqrt(max(variance, 1e-12))

    trials = max(1, int(trial_count))
    if trials > 1:
        expected_max = std_error * (
            (1.0 - _EULER_GAMMA) * _normal_ppf(1.0 - 1.0 / trials)
            + _EULER_GAMMA * _normal_ppf(1.0 - 1.0 / (trials * math.e))
        )
    else:
        expected_max = 0.0

    deflated = float((sr - expected_max) * math.sqrt(252.0))
    probabilistic_score = float(_normal_cdf((sr - expected_max) / std_error) * 100.0)
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
        "Risk_Free_Point_In_Time": False,
        "Risk_Free_Cash_Adjusted": False,
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

    rf düzeltmeleri (yalnızca excess-return oranları: Sharpe, Sortino, Omega,
    Deflated Sharpe; equity / Net_Return saf strateji getirisidir):
      * Nakit barlar rf kazanır: ``equity_curve["Position"]`` exposure olarak
        kullanılır, excess = getiri - rf * exposure.
      * ``risk_free_annual`` otomatik çözüldüğünde ve ``equity_curve["Date"]``
        varsa rf point-in-time (yayın tarihi = ay sonu + 1 gün) gunluk seri
        olarak uygulanır; yoksa son oran tüm geçmişe uygulanır.
    CAGR/Annualized_Return/Calmar, 126 barın altında NaN'dır.

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
    rf_auto_resolved = risk_free_annual is None
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

    exposure_series = _exposure_series(backtest_result, equity_curve)
    rf_daily_series = (
        _point_in_time_daily_rf(equity_curve)
        if (rf_auto_resolved and not risk_free_unavailable)
        else None
    )

    net_return = float(equity[-1] - 1.0)
    buy_hold_return = float(buy_hold_equity[-1] - 1.0)
    long_enough = periods >= MIN_BARS_FOR_ANNUALIZATION
    annualized_return = (
        float((equity[-1] ** (252.0 / periods)) - 1.0) if long_enough else float("nan")
    )
    cagr = annualized_return
    volatility = float(np.std(strategy_returns, ddof=1) * np.sqrt(252)) if periods > 1 else 0.0
    sharpe = _annualized_sharpe(
        strategy_returns, risk_free_annual, exposure=exposure_series, rf_daily=rf_daily_series
    )
    sortino = _annualized_sortino(
        strategy_returns, risk_free_annual, exposure=exposure_series, rf_daily=rf_daily_series
    )
    buy_hold_sharpe = _annualized_sharpe(buy_hold_returns, risk_free_annual, rf_daily=rf_daily_series)
    var_95, cvar_95 = _var_cvar(strategy_returns, confidence=0.95)
    buy_hold_var_95, buy_hold_cvar_95 = _var_cvar(buy_hold_returns, confidence=0.95)
    # rf yoksa excess hesaplanamaz; omega raw getiriyle (esik 0) hesaplanir.
    if risk_free_unavailable:
        excess_returns = None
    else:
        excess_returns = _excess_returns(
            strategy_returns, risk_free_annual, exposure=exposure_series, rf_daily=rf_daily_series
        )
    deflated_sharpe, sharpe_probabilistic_score = _deflated_sharpe(
        sharpe,
        strategy_returns,
        trial_count=trial_count,
        excess_returns=excess_returns,
    )
    max_drawdown = _max_drawdown(equity)
    if long_enough and max_drawdown < 0:
        calmar = float(annualized_return / abs(max_drawdown))
    else:
        calmar = float("nan")
    omega = _omega_ratio(excess_returns if excess_returns is not None else strategy_returns, threshold=0.0)
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
        "Risk_Free_Point_In_Time": bool(rf_daily_series is not None),
        "Risk_Free_Cash_Adjusted": bool(exposure_series is not None),
    }


def _exposure_series(backtest_result: Dict[str, Any], equity_curve: pd.DataFrame) -> np.ndarray | None:
    """Bar basina exposure (0..1): once equity_curve["Position"], yoksa series.positions."""
    n = len(equity_curve)
    if "Position" in equity_curve.columns:
        return np.clip(equity_curve["Position"].to_numpy(dtype=float), 0.0, 1.0)
    series = backtest_result.get("series") or {}
    positions = series.get("positions")
    if positions is not None and len(positions) == n:
        return np.clip(np.asarray(positions, dtype=float), 0.0, 1.0)
    return None


def _point_in_time_daily_rf(equity_curve: pd.DataFrame) -> np.ndarray | None:
    """Tarihe hizali point-in-time gunluk rf; tarih/seri yoksa None (skaler rf).

    Ilk yayin tarihinden onceki barlar serinin ilk bilinen degerini alir;
    hicbir deger yoksa None doner (skaler fallback).
    """
    if _get_rf_series is None or "Date" not in equity_curve.columns:
        return None
    dates = pd.to_datetime(equity_curve["Date"], errors="coerce")
    series = _get_rf_series(pd.DatetimeIndex(dates))
    if series is None:
        return None
    values = pd.Series(np.asarray(series, dtype=float)).bfill().to_numpy(dtype=float)
    if values.size != len(equity_curve) or not np.isfinite(values).any():
        return None
    return values


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
    if active.size < 2:
        return 0.0
    tracking_error = float(np.std(active, ddof=1) * np.sqrt(252))
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
