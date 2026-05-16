# -*- coding: utf-8 -*-
"""Trade log extraction helpers for long/flat backtests."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def trade_rows(
    *,
    n: int,
    model_name: str,
    fold_id_arr: np.ndarray,
    dates: pd.Index,
    prediction_dates: pd.Index,
    prev_close: np.ndarray,
    prices: np.ndarray,
    realized_returns: np.ndarray,
    net_strategy_returns: np.ndarray,
    execution_positions: np.ndarray,
    signal_frame: pd.DataFrame,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    entry_idx = None
    for idx in range(n):
        current_position = execution_positions[idx]
        prev_position = execution_positions[idx - 1] if idx > 0 else 0.0
        opened = prev_position == 0.0 and current_position == 1.0
        closed = prev_position == 1.0 and current_position == 0.0

        if opened:
            entry_idx = idx
        if closed and entry_idx is not None:
            rows.append(
                closed_trade_row(
                    model_name=model_name,
                    fold_id=fold_id_arr[entry_idx],
                    entry_idx=entry_idx,
                    exit_idx=idx,
                    dates=dates,
                    prediction_dates=prediction_dates,
                    prev_close=prev_close,
                    realized_returns=realized_returns,
                    net_strategy_returns=net_strategy_returns,
                    signal_frame=signal_frame,
                )
            )
            entry_idx = None

    if entry_idx is not None:
        rows.append(
            terminal_trade_row(
                model_name=model_name,
                fold_id=fold_id_arr[entry_idx],
                entry_idx=entry_idx,
                n=n,
                dates=dates,
                prediction_dates=prediction_dates,
                prev_close=prev_close,
                prices=prices,
                realized_returns=realized_returns,
                net_strategy_returns=net_strategy_returns,
                signal_frame=signal_frame,
            )
        )
    return rows


def closed_trade_row(
    *,
    model_name: str,
    fold_id: object,
    entry_idx: int,
    exit_idx: int,
    dates: pd.Index,
    prediction_dates: pd.Index,
    prev_close: np.ndarray,
    realized_returns: np.ndarray,
    net_strategy_returns: np.ndarray,
    signal_frame: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "Model": model_name,
        "Fold": fold_id,
        "Entry_Prediction_Date": prediction_dates[entry_idx],
        "Entry_Date": dates[entry_idx],
        "Exit_Prediction_Date": prediction_dates[exit_idx],
        "Exit_Date": dates[exit_idx],
        "Entry_Price": float(prev_close[entry_idx]),
        "Exit_Price": float(prev_close[exit_idx]),
        "Gross_Return": float(np.prod(1.0 + realized_returns[entry_idx:exit_idx]) - 1.0),
        "Net_Return": float(np.prod(1.0 + net_strategy_returns[entry_idx: exit_idx + 1]) - 1.0),
        "Holding_Period": int(exit_idx - entry_idx),
        "Entry_Reason": signal_value(signal_frame, entry_idx, "Signal_Reason"),
        "Exit_Reason": signal_value(signal_frame, exit_idx, "Signal_Reason"),
    }


def terminal_trade_row(
    *,
    model_name: str,
    fold_id: object,
    entry_idx: int,
    n: int,
    dates: pd.Index,
    prediction_dates: pd.Index,
    prev_close: np.ndarray,
    prices: np.ndarray,
    realized_returns: np.ndarray,
    net_strategy_returns: np.ndarray,
    signal_frame: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "Model": model_name,
        "Fold": fold_id,
        "Entry_Prediction_Date": prediction_dates[entry_idx],
        "Entry_Date": dates[entry_idx],
        "Exit_Prediction_Date": prediction_dates[-1],
        "Exit_Date": dates[-1],
        "Entry_Price": float(prev_close[entry_idx]),
        "Exit_Price": float(prices[-1]),
        "Gross_Return": float(np.prod(1.0 + realized_returns[entry_idx:]) - 1.0),
        "Net_Return": float(np.prod(1.0 + net_strategy_returns[entry_idx:]) - 1.0),
        "Holding_Period": int((n - 1) - entry_idx),
        "Entry_Reason": signal_value(signal_frame, entry_idx, "Signal_Reason"),
        "Exit_Reason": "Test donemi sonunda acik pozisyon kapatildi.",
    }


def signal_value(signal_frame: pd.DataFrame, idx: int, column: str) -> object:
    if column not in signal_frame.columns or idx < 0 or idx >= len(signal_frame):
        return ""
    return signal_frame.iloc[idx][column]
