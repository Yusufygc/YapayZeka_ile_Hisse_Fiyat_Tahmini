# -*- coding: utf-8 -*-
"""Selection and summary helpers for signal calibration trials."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def risk_adjusted_score(
    *,
    mean_net_return: float,
    mean_excess_return: float,
    mean_sharpe: float,
    mean_max_drawdown: float,
) -> float:
    sharpe_normalized = float(np.clip(mean_sharpe / 3.0, -1.0, 1.0))
    drawdown_score = 1.0 + mean_max_drawdown
    return float(
        0.35 * mean_net_return
        + 0.25 * mean_excess_return
        + 0.25 * sharpe_normalized
        + 0.15 * drawdown_score
    )


def calibration_constraint_passed(row: Dict[str, Any]) -> bool:
    return (
        bool(row.get("Meets_Min_Trade_Count", False))
        and safe_float(row.get("Mean_Excess_Return"), -1.0) > 0.0
    )


def select_confirmed_row(rows: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if calibration_constraint_passed(row)
        and bool(row.get("OOS_Constraint_Passed", False))
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            safe_float(row.get("Risk_Adjusted_Score"), -1e9),
            safe_float(row.get("Eval_Excess_Return"), -1e9),
            safe_float(row.get("Eval_Sharpe"), -1e9),
            safe_float(row.get("Mean_Max_Drawdown"), -1.0),
        ),
    )


def summarize_trial(
    *,
    trial_idx: int,
    params: Dict[str, float | int],
    summaries: list[Dict[str, Any]],
    min_trade_count: int,
) -> Dict[str, Any]:
    valid = [row for row in summaries if np.isfinite(float(row.get("Net_Return", np.nan)))]
    if not valid:
        return empty_trial_summary(trial_idx=trial_idx, params=params)

    net_returns = np.asarray([float(row.get("Net_Return", 0.0)) for row in valid], dtype=float)
    buy_hold_returns = np.asarray([float(row.get("BuyHold_Return", 0.0)) for row in valid], dtype=float)
    excess_returns = net_returns - buy_hold_returns
    drawdowns = np.asarray([float(row.get("Max_Drawdown", 0.0)) for row in valid], dtype=float)
    sharpes = np.asarray([float(row.get("Sharpe", 0.0)) for row in valid], dtype=float)
    calmars = np.asarray([safe_float(row.get("Calmar"), 0.0) for row in valid], dtype=float)
    trade_count = int(sum(int(float(row.get("Trade_Count", 0) or 0)) for row in valid))
    beats_buy_hold_count = int(sum(bool(row.get("Beats_BuyHold_NetReturn", False)) for row in valid))
    mean_net = float(np.nanmean(net_returns))
    mean_buy_hold = float(np.nanmean(buy_hold_returns))
    mean_excess = float(np.nanmean(excess_returns))
    mean_drawdown = float(np.nanmean(drawdowns))
    mean_sharpe = float(np.nanmean(sharpes))
    risk_score = risk_adjusted_score(
        mean_net_return=mean_net,
        mean_excess_return=mean_excess,
        mean_sharpe=mean_sharpe,
        mean_max_drawdown=mean_drawdown,
    )
    return {
        "Trial": trial_idx,
        **params,
        "Model_Count": int(len(valid)),
        "Mean_Net_Return": round(mean_net, 6),
        "Mean_BuyHold_Return": round(mean_buy_hold, 6),
        "Mean_Excess_Return": round(mean_excess, 6),
        "Risk_Adjusted_Score": round(risk_score, 6),
        "Beats_BuyHold_Count": beats_buy_hold_count,
        "Median_Net_Return": round(float(np.nanmedian(net_returns)), 6),
        "Mean_Max_Drawdown": round(mean_drawdown, 6),
        "Total_Trade_Count": trade_count,
        "Min_Trade_Count": int(min_trade_count),
        "Meets_Min_Trade_Count": bool(trade_count >= min_trade_count),
        "Mean_Sharpe": round(mean_sharpe, 6),
        "Mean_Calmar": round(float(np.nanmean(calmars)), 6),
        "Selection_Rank": None,
        "Positive_Net_Return": bool(mean_net > 0.0),
        "Status": "ok",
    }


def empty_trial_summary(*, trial_idx: int, params: Dict[str, float | int]) -> Dict[str, Any]:
    return {
        "Trial": trial_idx,
        **params,
        "Model_Count": 0,
        "Mean_Net_Return": np.nan,
        "Mean_BuyHold_Return": np.nan,
        "Mean_Excess_Return": np.nan,
        "Risk_Adjusted_Score": np.nan,
        "Beats_BuyHold_Count": 0,
        "Median_Net_Return": np.nan,
        "Mean_Max_Drawdown": np.nan,
        "Total_Trade_Count": 0,
        "Meets_Min_Trade_Count": False,
        "Mean_Sharpe": np.nan,
        "Mean_Calmar": np.nan,
        "Selection_Rank": None,
        "Positive_Net_Return": False,
        "Status": "failed_all_models",
    }


def sort_key(row: Dict[str, Any]) -> tuple:
    mean_net = safe_float(row.get("Mean_Net_Return"), -1e9)
    mean_excess = safe_float(row.get("Mean_Excess_Return"), mean_net)
    risk_score = safe_float(
        row.get("Risk_Adjusted_Score"),
        risk_adjusted_score(
            mean_net_return=mean_net,
            mean_excess_return=mean_excess,
            mean_sharpe=safe_float(row.get("Mean_Sharpe"), 0.0),
            mean_max_drawdown=safe_float(row.get("Mean_Max_Drawdown"), -1.0),
        ),
    )
    mean_drawdown = safe_float(row.get("Mean_Max_Drawdown"), -1.0)
    trade_count = int(row.get("Total_Trade_Count", 0) or 0)
    return (
        bool(row.get("Meets_Min_Trade_Count", False)),
        risk_score,
        mean_excess,
        mean_net,
        mean_drawdown,
        trade_count,
    )


def select_row(rows: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not rows:
        return None
    meeting = [row for row in rows if bool(row.get("Meets_Min_Trade_Count", False))]
    if meeting:
        return max(meeting, key=sort_key)

    traded = [row for row in rows if int(row.get("Total_Trade_Count", 0) or 0) > 0]
    if traded:
        return max(
            traded,
            key=lambda row: (
                int(row.get("Total_Trade_Count", 0) or 0),
                safe_float(row.get("Risk_Adjusted_Score"), -1e9),
                safe_float(row.get("Mean_Excess_Return"), -1e9),
                safe_float(row.get("Mean_Net_Return"), -1e9),
                safe_float(row.get("Mean_Max_Drawdown"), -1.0),
            ),
        )
    return max(rows, key=sort_key)
