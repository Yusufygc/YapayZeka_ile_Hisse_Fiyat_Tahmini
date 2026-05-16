# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.backtesting import equity as equity_helpers
from src.backtesting import execution as execution_helpers
from src.backtesting import trades as trade_helpers
from src.backtesting.signals import (
    SignalConfig,
    generate_long_flat_signals,
    generate_professional_signals,
    generate_simple_signals,
)

SIGNAL_FRAME_COLUMNS = equity_helpers.SIGNAL_FRAME_COLUMNS


def run_backtest(
    *,
    dates,
    prediction_dates=None,
    y_true_price,
    pred_price,
    prev_close,
    fold_ids=None,
    market_regime=None,
    model_name: str,
    validation_mode: str,
    target_mode: str,
    pred_target=None,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    signal_mode: str = "simple",
    signal_config: SignalConfig | None = None,
    model_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prices = np.asarray(y_true_price, dtype=float).ravel()
    pred_price = np.asarray(pred_price, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    pred_target_arr = None if pred_target is None else np.asarray(pred_target, dtype=float).ravel()
    fold_id_arr = None if fold_ids is None else np.asarray(fold_ids).ravel()
    market_regime_arr = None if market_regime is None else np.asarray(market_regime, dtype=float).ravel()

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
    if market_regime_arr is not None:
        lengths.append(len(market_regime_arr))
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
    if market_regime_arr is not None:
        market_regime_arr = market_regime_arr[-n:]
    else:
        market_regime_arr = np.zeros(n, dtype=float)

    realized_returns = (prices / np.maximum(prev_close, 1e-12)) - 1.0
    observed_returns = np.concatenate(([0.0], realized_returns[:-1]))
    buy_hold_equity = np.cumprod(1.0 + realized_returns)

    signal_mode, signal_frame, decision_positions, signals = _build_signal_frame(
        signal_mode=signal_mode,
        n=n,
        pred_target_arr=pred_target_arr,
        pred_price=pred_price,
        prev_close=prev_close,
        target_mode=target_mode,
        observed_returns=observed_returns,
        market_regime_arr=market_regime_arr,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        signal_config=signal_config,
        model_name=model_name,
        model_metrics=model_metrics or {},
    )

    execution = _execution_arrays(decision_positions)
    costs = _cost_arrays(
        entry_events=execution["entry_events"],
        exit_events_for_cost=execution["exit_events_for_cost"],
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    gross_strategy_returns = execution["positions"] * realized_returns
    net_strategy_returns = gross_strategy_returns - costs["transaction_costs"]
    equity = np.cumprod(1.0 + net_strategy_returns)

    execution_positions = execution["positions"]
    previous_execution_positions = execution["previous_positions"]
    entry_events = execution["entry_events"]
    exit_events = execution["exit_events"]
    exit_events_for_cost = execution["exit_events_for_cost"]
    position_changes = execution["position_changes"]
    transaction_costs = costs["transaction_costs"]
    commission_costs = costs["commission_costs"]
    slippage_costs = costs["slippage_costs"]
    entry_transaction_costs = costs["entry_transaction_costs"]
    exit_transaction_costs = costs["exit_transaction_costs"]

    trade_rows = _trade_rows(
        n=n,
        model_name=model_name,
        fold_id_arr=fold_id_arr,
        dates=dates,
        prediction_dates=prediction_dates,
        prev_close=prev_close,
        prices=prices,
        realized_returns=realized_returns,
        net_strategy_returns=net_strategy_returns,
        execution_positions=execution_positions,
        signal_frame=signal_frame,
    )

    equity_curve = _build_equity_curve(
        prediction_dates=prediction_dates,
        dates=dates,
        equity=equity,
        buy_hold_equity=buy_hold_equity,
        execution_positions=execution_positions,
        decision_positions=decision_positions,
        signals=signals,
        gross_strategy_returns=gross_strategy_returns,
        net_strategy_returns=net_strategy_returns,
        transaction_costs=transaction_costs,
        commission_costs=commission_costs,
        slippage_costs=slippage_costs,
        entry_transaction_costs=entry_transaction_costs,
        exit_transaction_costs=exit_transaction_costs,
        entry_events=entry_events,
        exit_events_for_cost=exit_events_for_cost,
        realized_returns=realized_returns,
        observed_returns=observed_returns,
        pred_price=pred_price,
        prices=prices,
        prev_close=prev_close,
        fold_id_arr=fold_id_arr,
        pred_target_arr=pred_target_arr,
        signal_frame=signal_frame,
    )

    _attach_executable_orders(
        equity_curve,
        previous_execution_positions=previous_execution_positions,
        execution_positions=execution_positions,
        entry_events=entry_events,
        exit_events=exit_events,
    )

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


def _build_equity_curve(
    *,
    prediction_dates: pd.Index,
    dates: pd.Index,
    equity: np.ndarray,
    buy_hold_equity: np.ndarray,
    execution_positions: np.ndarray,
    decision_positions: np.ndarray,
    signals: np.ndarray,
    gross_strategy_returns: np.ndarray,
    net_strategy_returns: np.ndarray,
    transaction_costs: np.ndarray,
    commission_costs: np.ndarray,
    slippage_costs: np.ndarray,
    entry_transaction_costs: np.ndarray,
    exit_transaction_costs: np.ndarray,
    entry_events: np.ndarray,
    exit_events_for_cost: np.ndarray,
    realized_returns: np.ndarray,
    observed_returns: np.ndarray,
    pred_price: np.ndarray,
    prices: np.ndarray,
    prev_close: np.ndarray,
    fold_id_arr: np.ndarray,
    pred_target_arr: np.ndarray | None,
    signal_frame: pd.DataFrame,
) -> pd.DataFrame:
    return equity_helpers.build_equity_curve(
        prediction_dates=prediction_dates,
        dates=dates,
        equity=equity,
        buy_hold_equity=buy_hold_equity,
        execution_positions=execution_positions,
        decision_positions=decision_positions,
        signals=signals,
        gross_strategy_returns=gross_strategy_returns,
        net_strategy_returns=net_strategy_returns,
        transaction_costs=transaction_costs,
        commission_costs=commission_costs,
        slippage_costs=slippage_costs,
        entry_transaction_costs=entry_transaction_costs,
        exit_transaction_costs=exit_transaction_costs,
        entry_events=entry_events,
        exit_events_for_cost=exit_events_for_cost,
        realized_returns=realized_returns,
        observed_returns=observed_returns,
        pred_price=pred_price,
        prices=prices,
        prev_close=prev_close,
        fold_id_arr=fold_id_arr,
        pred_target_arr=pred_target_arr,
        signal_frame=signal_frame,
    )


def _attach_signal_columns(equity_curve: pd.DataFrame, signal_frame: pd.DataFrame) -> None:
    equity_helpers.attach_signal_columns(equity_curve, signal_frame)


def _execution_arrays(decision_positions: np.ndarray) -> Dict[str, np.ndarray]:
    return execution_helpers.execution_arrays(decision_positions)


def _cost_arrays(
    *,
    entry_events: np.ndarray,
    exit_events_for_cost: np.ndarray,
    commission_bps: float,
    slippage_bps: float,
) -> Dict[str, np.ndarray]:
    return execution_helpers.cost_arrays(
        entry_events=entry_events,
        exit_events_for_cost=exit_events_for_cost,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )


def _trade_rows(
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
    return trade_helpers.trade_rows(
        n=n,
        model_name=model_name,
        fold_id_arr=fold_id_arr,
        dates=dates,
        prediction_dates=prediction_dates,
        prev_close=prev_close,
        prices=prices,
        realized_returns=realized_returns,
        net_strategy_returns=net_strategy_returns,
        execution_positions=execution_positions,
        signal_frame=signal_frame,
    )


def _closed_trade_row(
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
    return trade_helpers.closed_trade_row(
        model_name=model_name,
        fold_id=fold_id,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        dates=dates,
        prediction_dates=prediction_dates,
        prev_close=prev_close,
        realized_returns=realized_returns,
        net_strategy_returns=net_strategy_returns,
        signal_frame=signal_frame,
    )


def _terminal_trade_row(
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
    return trade_helpers.terminal_trade_row(
        model_name=model_name,
        fold_id=fold_id,
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


def _build_signal_frame(
    *,
    signal_mode: str,
    n: int,
    pred_target_arr: np.ndarray | None,
    pred_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
    observed_returns: np.ndarray,
    market_regime_arr: np.ndarray,
    commission_bps: float,
    slippage_bps: float,
    signal_config: SignalConfig | None,
    model_name: str,
    model_metrics: Dict[str, Any],
) -> tuple[str, pd.DataFrame, np.ndarray, np.ndarray]:
    normalized_mode = signal_mode.lower()
    if normalized_mode == "simple":
        signal_frame = generate_simple_signals(
            pred_target=pred_target_arr,
            pred_price=pred_price,
            prev_close=prev_close,
            target_mode=target_mode,
            config=signal_config,
        ).reset_index(drop=True)
        decision_positions = signal_frame["Position"].to_numpy(dtype=float)
        signals = (decision_positions > 0.0).astype(float)
        return normalized_mode, signal_frame, decision_positions, signals

    if normalized_mode == "legacy":
        signals = generate_long_flat_signals(
            pred_target=pred_target_arr,
            pred_price=pred_price,
            prev_close=prev_close,
            target_mode=target_mode,
        )
        decision_positions = signals.copy()
        signal_frame = _legacy_signal_frame(signals, n)
        return normalized_mode, signal_frame, decision_positions, signals

    if normalized_mode == "professional":
        signal_frame = _professional_signal_frame(
            n=n,
            pred_target_arr=pred_target_arr,
            pred_price=pred_price,
            prev_close=prev_close,
            target_mode=target_mode,
            observed_returns=observed_returns,
            market_regime_arr=market_regime_arr,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            signal_config=signal_config,
            model_name=model_name,
            model_metrics=model_metrics,
        )
        decision_positions = signal_frame["Position"].to_numpy(dtype=float)
        signals = signal_frame["Decision"].isin(["BUY", "HOLD"]).to_numpy(dtype=float)
        return normalized_mode, signal_frame, decision_positions, signals

    if normalized_mode in {"rejected_no_trade", "no_trade"}:
        signal_frame = _blocked_signal_frame(
            n,
            "Walk-forward OOS confirmation gate rejected this model; no production trade is allowed.",
            "rejected_no_trade",
        )
        decision_positions = signal_frame["Position"].to_numpy(dtype=float)
        signals = np.zeros(n, dtype=float)
        return normalized_mode, signal_frame, decision_positions, signals

    raise ValueError(
        f"Desteklenmeyen signal_mode: {signal_mode}. "
        "Beklenen: simple, legacy, professional, rejected_no_trade"
    )


def _professional_signal_frame(
    *,
    n: int,
    pred_target_arr: np.ndarray | None,
    pred_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
    observed_returns: np.ndarray,
    market_regime_arr: np.ndarray,
    commission_bps: float,
    slippage_bps: float,
    signal_config: SignalConfig | None,
    model_name: str,
    model_metrics: Dict[str, Any],
) -> pd.DataFrame:
    cfg = signal_config or SignalConfig()
    block_reason, block_state = _professional_trade_block(model_name, model_metrics, cfg)
    if block_reason:
        return _blocked_signal_frame(n, block_reason, block_state)

    cfg, quality_reason = _apply_soft_quality_gate(model_metrics, cfg)
    signal_frame = generate_professional_signals(
        pred_target=pred_target_arr,
        pred_price=pred_price,
        prev_close=prev_close,
        target_mode=target_mode,
        observed_returns=observed_returns,
        market_regime=market_regime_arr,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        config=cfg,
    ).reset_index(drop=True)
    if quality_reason:
        signal_frame["Quality_Gate_Reason"] = quality_reason
    return signal_frame


def _legacy_signal_frame(signals: np.ndarray, n: int) -> pd.DataFrame:
    signals = np.asarray(signals, dtype=float).ravel()[-n:]
    decisions = np.where(signals > 0, "BUY", "NO_TRADE")
    recommendations = np.where(signals > 0, "BUY", "HOLD")
    recommendations_tr = np.where(signals > 0, "AL", "TUT")
    return pd.DataFrame({
        "Decision": decisions,
        "Recommendation": recommendations,
        "Recommendation_TR": recommendations_tr,
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


def _attach_executable_orders(
    equity_curve: pd.DataFrame,
    *,
    previous_execution_positions: np.ndarray,
    execution_positions: np.ndarray,
    entry_events: np.ndarray,
    exit_events: np.ndarray,
) -> None:
    execution_helpers.attach_executable_orders(
        equity_curve,
        previous_execution_positions=previous_execution_positions,
        execution_positions=execution_positions,
        entry_events=entry_events,
        exit_events=exit_events,
    )


def _blocked_signal_frame(n: int, reason: str, risk_state: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Decision": ["NO_TRADE"] * n,
        "Recommendation": ["HOLD"] * n,
        "Recommendation_TR": ["TUT"] * n,
        "Position": np.zeros(n, dtype=float),
        "Expected_Return": np.full(n, np.nan, dtype=float),
        "Base_Entry_Threshold": np.full(n, np.nan, dtype=float),
        "Entry_Threshold": np.full(n, np.nan, dtype=float),
        "Exit_Threshold": np.full(n, np.nan, dtype=float),
        "Signal_Strength": np.full(n, np.nan, dtype=float),
        "Quality_Gate_Mode": [risk_state] * n,
        "Quality_Threshold_Multiplier": np.full(n, np.nan, dtype=float),
        "Quality_Gate_Reason": [reason] * n,
        "Market_Regime_SMA200": np.zeros(n, dtype=float),
        "Regime_Threshold_Multiplier": np.full(n, np.nan, dtype=float),
        "Volatility_Regime": ["blocked"] * n,
        "Volatility_Threshold_Multiplier": np.full(n, np.nan, dtype=float),
        "Final_Threshold_Multiplier": np.full(n, np.nan, dtype=float),
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

    if config.quality_gate_mode in {"soft", "off"}:
        return "", ""

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


def _apply_soft_quality_gate(
    model_metrics: Dict[str, Any],
    config: SignalConfig,
) -> tuple[SignalConfig, str]:
    if config.quality_gate_mode != "soft":
        return config, ""

    multiplier = float(config.entry_threshold_multiplier)
    reasons: list[str] = []

    dir_acc = _metric_float(model_metrics.get("Dir_Acc"))
    if np.isfinite(dir_acc):
        if dir_acc < config.soft_dir_acc_low:
            multiplier = max(multiplier, config.soft_entry_threshold_multiplier_low)
            reasons.append(
                f"Dir_Acc %{dir_acc:.2f} < %{config.soft_dir_acc_low:.2f}; yalnizca cok guclu sinyaller kabul edilecek."
            )
        elif dir_acc < config.min_directional_accuracy:
            multiplier = max(multiplier, config.soft_entry_threshold_multiplier_mid)
            reasons.append(
                f"Dir_Acc %{dir_acc:.2f} < %{config.min_directional_accuracy:.2f}; entry threshold artirildi."
            )

    rmse_vs_benchmark = _metric_float(model_metrics.get("RMSE_vs_benchmark"))
    if np.isfinite(rmse_vs_benchmark) and rmse_vs_benchmark > config.max_rmse_vs_benchmark:
        if rmse_vs_benchmark >= config.soft_rmse_penalty_full:
            multiplier = max(multiplier, config.soft_entry_threshold_multiplier_low)
        else:
            multiplier = max(multiplier, config.soft_entry_threshold_multiplier_mid)
        reasons.append(
            f"RMSE benchmark orani {rmse_vs_benchmark:.4f} > {config.max_rmse_vs_benchmark:.4f}; entry threshold artirildi."
        )

    composite_score = _metric_float(model_metrics.get("Composite_Score"))
    if np.isfinite(composite_score) and composite_score < config.min_composite_score:
        if composite_score < config.soft_composite_low:
            multiplier = max(multiplier, config.soft_entry_threshold_multiplier_low)
        else:
            multiplier = max(multiplier, config.soft_entry_threshold_multiplier_mid)
        reasons.append(
            f"Composite score {composite_score:.4f} < {config.min_composite_score:.4f}; entry threshold artirildi."
        )

    if multiplier == config.entry_threshold_multiplier:
        return config, "Soft kalite filtresi ek sikilastirma uygulamadi."

    adjusted = replace(config, entry_threshold_multiplier=round(multiplier, 6))
    return adjusted, " ".join(reasons)


def _metric_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _signal_value(signal_frame: pd.DataFrame, idx: int, column: str) -> object:
    return trade_helpers.signal_value(signal_frame, idx, column)
