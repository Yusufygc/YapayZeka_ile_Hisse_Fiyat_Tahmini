# -*- coding: utf-8 -*-
"""Execution arrays, cost arrays, and order annotation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def execution_arrays(decision_positions: np.ndarray) -> Dict[str, np.ndarray]:
    positions = np.asarray(decision_positions, dtype=float).copy()
    previous_positions = np.concatenate(([0.0], positions[:-1]))
    entry_events = ((previous_positions == 0.0) & (positions == 1.0)).astype(float)
    exit_events = ((previous_positions == 1.0) & (positions == 0.0)).astype(float)
    forced_exit_events = np.zeros(len(positions), dtype=float)
    if len(positions) and positions[-1] == 1.0:
        forced_exit_events[-1] = 1.0

    exit_events_for_cost = exit_events + forced_exit_events
    return {
        "positions": positions,
        "previous_positions": previous_positions,
        "entry_events": entry_events,
        "exit_events": exit_events,
        "exit_events_for_cost": exit_events_for_cost,
        "position_changes": entry_events + exit_events_for_cost,
    }


def cost_arrays(
    *,
    entry_events: np.ndarray,
    exit_events_for_cost: np.ndarray,
    commission_bps: float,
    slippage_bps: float,
) -> Dict[str, np.ndarray]:
    commission_rate = commission_bps / 10000.0
    slippage_rate = slippage_bps / 10000.0
    entry_commission_costs = entry_events * commission_rate
    exit_commission_costs = exit_events_for_cost * commission_rate
    entry_slippage_costs = entry_events * slippage_rate
    exit_slippage_costs = exit_events_for_cost * slippage_rate
    commission_costs = entry_commission_costs + exit_commission_costs
    slippage_costs = entry_slippage_costs + exit_slippage_costs
    entry_transaction_costs = entry_commission_costs + entry_slippage_costs
    exit_transaction_costs = exit_commission_costs + exit_slippage_costs
    return {
        "transaction_costs": entry_transaction_costs + exit_transaction_costs,
        "commission_costs": commission_costs,
        "slippage_costs": slippage_costs,
        "entry_transaction_costs": entry_transaction_costs,
        "exit_transaction_costs": exit_transaction_costs,
    }


def attach_executable_orders(
    equity_curve: pd.DataFrame,
    *,
    previous_execution_positions: np.ndarray,
    execution_positions: np.ndarray,
    entry_events: np.ndarray,
    exit_events: np.ndarray,
) -> None:
    order_en = np.full(len(equity_curve), "HOLD", dtype=object)
    order_tr = np.full(len(equity_curve), "TUT", dtype=object)
    order_en[entry_events > 0.0] = "BUY"
    order_tr[entry_events > 0.0] = "AL"
    order_en[exit_events > 0.0] = "SELL"
    order_tr[exit_events > 0.0] = "SAT"

    equity_curve["Previous_Position"] = previous_execution_positions
    equity_curve["New_Position"] = execution_positions
    equity_curve["Executable_Order"] = order_en
    equity_curve["Executable_Order_TR"] = order_tr
    if "Signal_Reason" in equity_curve.columns:
        equity_curve["Order_Reason"] = equity_curve["Signal_Reason"]
    else:
        equity_curve["Order_Reason"] = np.where(
            entry_events > 0.0,
            "Pozisyon acildi.",
            np.where(exit_events > 0.0, "Pozisyon kapatildi.", "Pozisyon korundu."),
        )
