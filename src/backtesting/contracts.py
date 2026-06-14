# -*- coding: utf-8 -*-
"""Backtest data-contract helpers.

This module keeps the public backtest API stable while making the internal
alignment, return, and execution contracts testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.backtesting import execution as execution_helpers


@dataclass(frozen=True)
class BacktestInputFrame:
    dates: pd.Index
    prediction_dates: pd.Index
    prices: np.ndarray
    pred_price: np.ndarray
    prev_close: np.ndarray
    pred_target: np.ndarray | None
    fold_ids: np.ndarray
    market_regime: np.ndarray
    n: int


@dataclass(frozen=True)
class BacktestReturnFrame:
    realized_returns: np.ndarray
    observed_returns: np.ndarray
    buy_hold_equity: np.ndarray


@dataclass(frozen=True)
class BacktestExecutionFrame:
    execution: Dict[str, np.ndarray]
    costs: Dict[str, np.ndarray]
    gross_strategy_returns: np.ndarray
    net_strategy_returns: np.ndarray
    equity: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return self.execution["positions"]

    @property
    def previous_positions(self) -> np.ndarray:
        return self.execution["previous_positions"]

    @property
    def entry_events(self) -> np.ndarray:
        return self.execution["entry_events"]

    @property
    def exit_events(self) -> np.ndarray:
        return self.execution["exit_events"]

    @property
    def exit_events_for_cost(self) -> np.ndarray:
        return self.execution["exit_events_for_cost"]

    @property
    def position_changes(self) -> np.ndarray:
        return self.execution["position_changes"]

    @property
    def transaction_costs(self) -> np.ndarray:
        return self.costs["transaction_costs"]

    @property
    def commission_costs(self) -> np.ndarray:
        return self.costs["commission_costs"]

    @property
    def slippage_costs(self) -> np.ndarray:
        return self.costs["slippage_costs"]

    @property
    def entry_transaction_costs(self) -> np.ndarray:
        return self.costs["entry_transaction_costs"]

    @property
    def exit_transaction_costs(self) -> np.ndarray:
        return self.costs["exit_transaction_costs"]


def prepare_backtest_inputs(
    *,
    dates,
    prediction_dates=None,
    y_true_price,
    pred_price,
    prev_close,
    pred_target=None,
    fold_ids=None,
    market_regime=None,
) -> BacktestInputFrame:
    prices = np.asarray(y_true_price, dtype=float).ravel()
    pred_price_arr = np.asarray(pred_price, dtype=float).ravel()
    prev_close_arr = np.asarray(prev_close, dtype=float).ravel()
    pred_target_arr = None if pred_target is None else np.asarray(pred_target, dtype=float).ravel()
    fold_id_arr = None if fold_ids is None else np.asarray(fold_ids).ravel()
    market_regime_arr = None if market_regime is None else np.asarray(market_regime, dtype=float).ravel()

    if dates is None:
        dates = pd.RangeIndex(start=0, stop=len(prices), step=1)
    date_idx = pd.to_datetime(pd.Index(dates))
    if prediction_dates is None:
        prediction_dates = date_idx
    prediction_date_idx = pd.to_datetime(pd.Index(prediction_dates))

    lengths = [
        len(prices),
        len(pred_price_arr),
        len(prev_close_arr),
        len(date_idx),
        len(prediction_date_idx),
    ]
    if pred_target_arr is not None:
        lengths.append(len(pred_target_arr))
    if fold_id_arr is not None:
        lengths.append(len(fold_id_arr))
    if market_regime_arr is not None:
        lengths.append(len(market_regime_arr))
    n = min(lengths) if lengths else 0

    if n == 0:
        return BacktestInputFrame(
            dates=date_idx[:0],
            prediction_dates=prediction_date_idx[:0],
            prices=prices[:0],
            pred_price=pred_price_arr[:0],
            prev_close=prev_close_arr[:0],
            pred_target=None if pred_target_arr is None else pred_target_arr[:0],
            fold_ids=np.asarray([], dtype=object),
            market_regime=np.asarray([], dtype=float),
            n=0,
        )

    if fold_id_arr is None:
        fold_id_arr = np.full(n, "all", dtype=object)
    else:
        fold_id_arr = fold_id_arr[-n:]
    if market_regime_arr is None:
        market_regime_arr = np.zeros(n, dtype=float)
    else:
        market_regime_arr = market_regime_arr[-n:]

    return BacktestInputFrame(
        dates=date_idx[-n:],
        prediction_dates=prediction_date_idx[-n:],
        prices=prices[-n:],
        pred_price=pred_price_arr[-n:],
        prev_close=prev_close_arr[-n:],
        pred_target=None if pred_target_arr is None else pred_target_arr[-n:],
        fold_ids=fold_id_arr,
        market_regime=market_regime_arr,
        n=n,
    )


def compute_return_frame(prices: np.ndarray, prev_close: np.ndarray) -> BacktestReturnFrame:
    realized_returns = (prices / np.maximum(prev_close, 1e-12)) - 1.0
    observed_returns = np.concatenate(([0.0], realized_returns[:-1]))
    buy_hold_equity = np.cumprod(1.0 + realized_returns)
    return BacktestReturnFrame(
        realized_returns=realized_returns,
        observed_returns=observed_returns,
        buy_hold_equity=buy_hold_equity,
    )


def build_execution_frame(
    *,
    decision_positions: np.ndarray,
    realized_returns: np.ndarray,
    commission_bps: float,
    slippage_bps: float,
) -> BacktestExecutionFrame:
    execution = execution_helpers.execution_arrays(decision_positions)
    costs = execution_helpers.cost_arrays(
        entry_events=execution["entry_events"],
        exit_events_for_cost=execution["exit_events_for_cost"],
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    gross_strategy_returns = execution["positions"] * realized_returns
    net_strategy_returns = gross_strategy_returns - costs["transaction_costs"]
    equity = np.cumprod(1.0 + net_strategy_returns)
    return BacktestExecutionFrame(
        execution=execution,
        costs=costs,
        gross_strategy_returns=gross_strategy_returns,
        net_strategy_returns=net_strategy_returns,
        equity=equity,
    )


def empty_backtest_result(model_name: str, validation_mode: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "validation_mode": validation_mode,
        "equity_curve": pd.DataFrame(
            columns=["Date", "Equity", "BuyHold_Equity", "Position", "Desired_Position", "Signal", "Net_Return"]
        ),
        "trades": pd.DataFrame(
            columns=[
                "Model",
                "Fold",
                "Entry_Date",
                "Exit_Date",
                "Entry_Price",
                "Exit_Price",
                "Gross_Return",
                "Net_Return",
                "Holding_Period",
            ]
        ),
        "series": {},
    }
