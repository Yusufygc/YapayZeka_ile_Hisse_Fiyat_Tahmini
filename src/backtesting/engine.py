# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.backtesting.signals import SignalConfig, generate_long_flat_signals, generate_professional_signals


def run_backtest(
    *,
    dates,
    prediction_dates=None,
    y_true_price,
    pred_price,
    prev_close,
    fold_ids=None,
    model_name: str,
    validation_mode: str,
    target_mode: str,
    pred_target=None,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    signal_mode: str = "legacy",
    signal_config: SignalConfig | None = None,
    model_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prices = np.asarray(y_true_price, dtype=float).ravel()
    pred_price = np.asarray(pred_price, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    pred_target_arr = None if pred_target is None else np.asarray(pred_target, dtype=float).ravel()
    fold_id_arr = None if fold_ids is None else np.asarray(fold_ids).ravel()

    if dates is None:
        dates = pd.RangeIndex(start=0, stop=len(prices), step=1)
    dates = pd.to_datetime(pd.Index(dates))
    if prediction_dates is None:
        prediction_dates = dates
    prediction_dates = pd.to_datetime(pd.Index(prediction_dates))

    lengths = [len(prices), len(pred_price), len(prev_close), len(dates), len(prediction_dates)]
    if pred_target_arr is not None:
        lengths.append(len(pred_target_arr))
    if fold_id_arr is not None:
        lengths.append(len(fold_id_arr))
    n = min(lengths) if lengths else 0
    if n == 0:
        return {
            "model_name": model_name,
            "validation_mode": validation_mode,
            "equity_curve": pd.DataFrame(columns=["Date", "Equity", "BuyHold_Equity", "Position", "Desired_Position", "Signal", "Net_Return"]),
            "trades": pd.DataFrame(columns=["Model", "Fold", "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price", "Gross_Return", "Net_Return", "Holding_Period"]),
            "series": {},
        }

    prices = prices[-n:]
    pred_price = pred_price[-n:]
    prev_close = prev_close[-n:]
    dates = dates[-n:]
    prediction_dates = prediction_dates[-n:]
    if pred_target_arr is not None:
        pred_target_arr = pred_target_arr[-n:]
    if fold_id_arr is not None:
        fold_id_arr = fold_id_arr[-n:]
    else:
        fold_id_arr = np.full(n, "all", dtype=object)

    realized_returns = (prices / np.maximum(prev_close, 1e-12)) - 1.0
    observed_returns = np.concatenate(([0.0], realized_returns[:-1]))
    buy_hold_equity = np.cumprod(1.0 + realized_returns)

    signal_mode = signal_mode.lower()
    if signal_mode == "legacy":
        signals = generate_long_flat_signals(
            pred_target=pred_target_arr,
            pred_price=pred_price,
            prev_close=prev_close,
            target_mode=target_mode,
        )
        decision_positions = signals.copy()
        signal_frame = _legacy_signal_frame(signals, n)
    elif signal_mode == "professional":
        cfg = signal_config or SignalConfig()
        block_reason, block_state = _professional_trade_block(model_name, model_metrics or {}, cfg)
        if block_reason:
            signal_frame = _blocked_signal_frame(n, block_reason, block_state)
        else:
            signal_frame = generate_professional_signals(
                pred_target=pred_target_arr,
                pred_price=pred_price,
                prev_close=prev_close,
                target_mode=target_mode,
                observed_returns=observed_returns,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                config=cfg,
            ).reset_index(drop=True)
        decision_positions = signal_frame["Position"].to_numpy(dtype=float)
        signals = (signal_frame["Decision"].isin(["BUY", "HOLD"]).to_numpy(dtype=float))
    else:
        raise ValueError(f"Desteklenmeyen signal_mode: {signal_mode}. Beklenen: legacy, professional")

    execution_positions = np.concatenate(([0.0], decision_positions[:-1]))
    previous_execution_positions = np.concatenate(([0.0], execution_positions[:-1]))
    entry_events = ((previous_execution_positions == 0.0) & (execution_positions == 1.0)).astype(float)
    exit_events = ((previous_execution_positions == 1.0) & (execution_positions == 0.0)).astype(float)

    forced_exit_events = np.zeros(n, dtype=float)
    if execution_positions[-1] == 1.0:
        forced_exit_events[-1] = 1.0
    exit_events_for_cost = exit_events + forced_exit_events

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
    transaction_costs = entry_transaction_costs + exit_transaction_costs
    position_changes = entry_events + exit_events_for_cost

    gross_strategy_returns = execution_positions * realized_returns
    net_strategy_returns = gross_strategy_returns - transaction_costs
    equity = np.cumprod(1.0 + net_strategy_returns)

    trade_rows = []
    entry_idx = None
    for idx in range(n):
        current_position = execution_positions[idx]
        prev_position = execution_positions[idx - 1] if idx > 0 else 0.0
        opened = prev_position == 0.0 and current_position == 1.0
        closed = prev_position == 1.0 and current_position == 0.0

        if opened:
            entry_idx = idx
        if closed and entry_idx is not None:
            gross_trade_return = float(np.prod(1.0 + realized_returns[entry_idx:idx]) - 1.0)
            net_trade_return = float(np.prod(1.0 + net_strategy_returns[entry_idx:idx]) - 1.0)
            trade_rows.append({
                "Model": model_name,
                "Fold": fold_id_arr[entry_idx],
                "Entry_Prediction_Date": prediction_dates[entry_idx],
                "Entry_Date": dates[entry_idx],
                "Exit_Prediction_Date": prediction_dates[idx],
                "Exit_Date": dates[idx],
                "Entry_Price": float(prev_close[entry_idx]),
                "Exit_Price": float(prev_close[idx]),
                "Gross_Return": gross_trade_return,
                "Net_Return": net_trade_return,
                "Holding_Period": int(idx - entry_idx),
                "Entry_Reason": _signal_value(signal_frame, entry_idx, "Signal_Reason"),
                "Exit_Reason": _signal_value(signal_frame, idx, "Signal_Reason"),
            })
            entry_idx = None

    if entry_idx is not None:
        gross_trade_return = float(np.prod(1.0 + realized_returns[entry_idx:]) - 1.0)
        net_trade_return = float(np.prod(1.0 + net_strategy_returns[entry_idx:]) - 1.0)
        trade_rows.append({
            "Model": model_name,
            "Fold": fold_id_arr[entry_idx],
            "Entry_Prediction_Date": prediction_dates[entry_idx],
            "Entry_Date": dates[entry_idx],
            "Exit_Prediction_Date": prediction_dates[-1],
            "Exit_Date": dates[-1],
            "Entry_Price": float(prev_close[entry_idx]),
            "Exit_Price": float(prices[-1]),
            "Gross_Return": gross_trade_return,
            "Net_Return": net_trade_return,
            "Holding_Period": int((n - 1) - entry_idx),
            "Entry_Reason": _signal_value(signal_frame, entry_idx, "Signal_Reason"),
            "Exit_Reason": "Test donemi sonunda acik pozisyon kapatildi.",
        })

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

    for column in [
        "Decision",
        "Expected_Return",
        "Entry_Threshold",
        "Exit_Threshold",
        "Signal_Strength",
        "Rolling_Volatility",
        "Holding_Bars",
        "Trade_Return",
        "Take_Profit_Return",
        "Stop_Loss_Return",
        "Cooldown_Remaining",
        "Risk_State",
        "Signal_Reason",
    ]:
        if column in signal_frame.columns:
            equity_curve[column] = signal_frame[column].to_numpy()

    return {
        "model_name": model_name,
        "validation_mode": validation_mode,
        "signal_mode": signal_mode,
        "equity_curve": equity_curve,
        "trades": pd.DataFrame(trade_rows),
        "series": {
            "positions": execution_positions,
            "desired_positions": decision_positions,
            "execution_positions": execution_positions,
            "signals": signals,
            "position_changes": position_changes,
            "transaction_costs": transaction_costs,
            "strategy_returns": net_strategy_returns,
            "buy_hold_returns": realized_returns,
            "signal_frame": signal_frame,
        },
    }


def _legacy_signal_frame(signals: np.ndarray, n: int) -> pd.DataFrame:
    signals = np.asarray(signals, dtype=float).ravel()[-n:]
    decisions = np.where(signals > 0, "BUY", "NO_TRADE")
    return pd.DataFrame({
        "Decision": decisions,
        "Position": signals,
        "Expected_Return": np.nan,
        "Entry_Threshold": np.nan,
        "Exit_Threshold": np.nan,
        "Signal_Strength": np.nan,
        "Risk_State": np.where(signals > 0, "legacy_long", "legacy_flat"),
        "Signal_Reason": np.where(
            signals > 0,
            "Legacy mod: tahmin pozitif oldugu icin long pozisyon.",
            "Legacy mod: tahmin pozitif olmadigi icin pozisyon yok.",
        ),
    })


def _blocked_signal_frame(n: int, reason: str, risk_state: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Decision": ["NO_TRADE"] * n,
        "Position": np.zeros(n, dtype=float),
        "Expected_Return": np.full(n, np.nan, dtype=float),
        "Entry_Threshold": np.full(n, np.nan, dtype=float),
        "Exit_Threshold": np.full(n, np.nan, dtype=float),
        "Signal_Strength": np.full(n, np.nan, dtype=float),
        "Rolling_Volatility": np.full(n, np.nan, dtype=float),
        "Holding_Bars": np.zeros(n, dtype=int),
        "Trade_Return": np.zeros(n, dtype=float),
        "Take_Profit_Return": np.full(n, np.nan, dtype=float),
        "Stop_Loss_Return": np.full(n, np.nan, dtype=float),
        "Cooldown_Remaining": np.zeros(n, dtype=int),
        "Risk_State": [risk_state] * n,
        "Signal_Reason": [reason] * n,
    })


def _professional_trade_block(
    model_name: str,
    model_metrics: Dict[str, Any],
    config: SignalConfig,
) -> tuple[str, str]:
    if model_name in config.benchmark_only_models:
        return (
            f"{model_name} sadece benchmark olarak kullanildigi icin profesyonel modda islem acilmadi.",
            "benchmark_only",
        )

    dir_acc = _metric_float(model_metrics.get("Dir_Acc"))
    if np.isfinite(dir_acc) and dir_acc < config.min_directional_accuracy:
        return (
            f"Modelin yon dogrulugu %{dir_acc:.2f}; islem icin gereken %{config.min_directional_accuracy:.2f} esiginin altinda.",
            "quality_dir_acc",
        )

    rmse_vs_benchmark = _metric_float(model_metrics.get("RMSE_vs_benchmark"))
    if np.isfinite(rmse_vs_benchmark) and rmse_vs_benchmark > config.max_rmse_vs_benchmark:
        return (
            f"Modelin RMSE benchmark orani {rmse_vs_benchmark:.4f}; izin verilen {config.max_rmse_vs_benchmark:.4f} esiginin uzerinde.",
            "quality_rmse",
        )

    composite_score = _metric_float(model_metrics.get("Composite_Score"))
    if np.isfinite(composite_score) and composite_score < config.min_composite_score:
        return (
            f"Model kalite skoru {composite_score:.4f}; islem icin gereken {config.min_composite_score:.4f} esiginin altinda.",
            "quality_composite",
        )

    return "", ""


def _metric_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _signal_value(signal_frame: pd.DataFrame, idx: int, column: str) -> object:
    if column not in signal_frame.columns or idx < 0 or idx >= len(signal_frame):
        return ""
    return signal_frame.iloc[idx][column]
