# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.backtesting import signal_math
from src.backtesting.signal_validation import validate_signal_config


@dataclass(frozen=True)
class SignalConfig:
    """
    Long/flat signal settings.

    The simple mode uses buy_threshold and sell_threshold. Professional mode
    keeps the conservative transaction-cost and volatility buffers.
    """

    buy_threshold: float = 0.0
    sell_threshold: float = 0.0
    entry_cost_multiplier: float = 2.0
    exit_cost_multiplier: float = 1.0
    volatility_window: int = 20
    volatility_multiplier: float = 0.25
    exit_volatility_multiplier: float = 0.05
    min_holding_bars: int = 3
    max_holding_bars: int = 20
    take_profit_vol_multiplier: float = 1.5
    stop_loss_vol_multiplier: float = 1.0
    cooldown_bars: int = 2
    min_entry_threshold: float = 0.0
    benchmark_only_models: Tuple[str, ...] = ("Naive Last Value", "Naive Zero Return", "Naive Drift")
    quality_gate_mode: str = "soft"
    min_directional_accuracy: float = 52.0
    max_rmse_vs_benchmark: float = 1.05
    min_composite_score: float = 50.0
    entry_threshold_multiplier: float = 1.0
    soft_dir_acc_low: float = 48.0
    soft_entry_threshold_multiplier_mid: float = 1.25
    soft_entry_threshold_multiplier_low: float = 1.75
    soft_rmse_penalty_full: float = 1.10
    soft_composite_low: float = 45.0
    regime_bull_entry_multiplier: float = 0.85
    regime_neutral_entry_multiplier: float = 1.0
    regime_bear_entry_multiplier: float = 1.25
    volatility_low_entry_multiplier: float = 0.85
    volatility_normal_entry_multiplier: float = 1.0
    volatility_high_entry_multiplier: float = 1.25
    volatility_low_quantile: float = 0.33
    volatility_high_quantile: float = 0.67
    emergency_stop_overrides_min_hold: bool = True


def generate_long_flat_signals(
    pred_target: np.ndarray | None,
    pred_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    pred_price = np.asarray(pred_price, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()

    if pred_target is not None and target_mode in {"log_return", "return"}:
        signal_source = np.asarray(pred_target, dtype=float).ravel()
    elif target_mode == "price":
        signal_source = pred_price - prev_close
    elif pred_target is not None:
        signal_source = np.asarray(pred_target, dtype=float).ravel()
    else:
        signal_source = pred_price - prev_close

    k = min(len(signal_source), len(pred_price), len(prev_close))
    signal_source = signal_source[-k:]
    signals = np.zeros(k, dtype=float)
    signals[signal_source > 0] = 1.0
    return signals


def generate_simple_signals(
    pred_target: np.ndarray | None,
    pred_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """
    Generate plain AL/SAT/TUT-compatible long/flat decisions.

    BUY opens a long position, EXIT closes an existing long position, and
    HOLD/NO_TRADE preserve the current state. SELL never opens a short.
    """
    cfg = config or SignalConfig()
    _validate_config(cfg)

    pred_price = np.asarray(pred_price, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    expected_return = _expected_return(pred_target, pred_price, prev_close, target_mode)

    n = min(len(expected_return), len(pred_price), len(prev_close))
    expected_return = expected_return[-n:]

    decisions: list[str] = []
    recommendations: list[str] = []
    recommendations_tr: list[str] = []
    risk_states: list[str] = []
    reasons: list[str] = []
    positions = np.zeros(n, dtype=float)

    in_position = False
    buy_threshold = float(cfg.buy_threshold)
    sell_threshold = float(cfg.sell_threshold)
    exit_threshold = -abs(sell_threshold)

    for idx in range(n):
        exp_ret = float(expected_return[idx])

        if not in_position and exp_ret > buy_threshold:
            decision = "BUY"
            recommendation = "BUY"
            recommendation_tr = "AL"
            risk_state = "simple_entry"
            reason = (
                f"Beklenen getiri {exp_ret:.6f}, alis esigi {buy_threshold:.6f} "
                "uzerinde oldugu icin AL sinyali uretildi."
            )
            in_position = True
        elif in_position and exp_ret < exit_threshold:
            decision = "EXIT"
            recommendation = "SELL"
            recommendation_tr = "SAT"
            risk_state = "simple_exit"
            reason = (
                f"Beklenen getiri {exp_ret:.6f}, satis esigi {exit_threshold:.6f} "
                "altinda oldugu icin SAT sinyali uretildi."
            )
            in_position = False
        elif in_position:
            decision = "HOLD"
            recommendation = "HOLD"
            recommendation_tr = "TUT"
            risk_state = "simple_hold"
            reason = "Pozisyon acik ve satis esigi tetiklenmedigi icin TUT sinyali uretildi."
        else:
            decision = "NO_TRADE"
            recommendation = "HOLD"
            recommendation_tr = "TUT"
            risk_state = "simple_flat"
            reason = "Pozisyon yok ve alis esigi tetiklenmedigi icin TUT sinyali uretildi."

        positions[idx] = 1.0 if in_position else 0.0
        decisions.append(decision)
        recommendations.append(recommendation)
        recommendations_tr.append(recommendation_tr)
        risk_states.append(risk_state)
        reasons.append(reason)

    return pd.DataFrame({
        "Decision": decisions,
        "Recommendation": recommendations,
        "Recommendation_TR": recommendations_tr,
        "Position": positions,
        "Expected_Return": expected_return,
        "Base_Entry_Threshold": np.full(n, buy_threshold, dtype=float),
        "Entry_Threshold": np.full(n, buy_threshold, dtype=float),
        "Exit_Threshold": np.full(n, exit_threshold, dtype=float),
        "Signal_Strength": expected_return.copy(),
        "Quality_Gate_Mode": ["simple"] * n,
        "Quality_Threshold_Multiplier": np.ones(n, dtype=float),
        "Market_Regime_SMA200": np.zeros(n, dtype=float),
        "Regime_Threshold_Multiplier": np.ones(n, dtype=float),
        "Volatility_Regime": ["simple"] * n,
        "Volatility_Threshold_Multiplier": np.ones(n, dtype=float),
        "Final_Threshold_Multiplier": np.ones(n, dtype=float),
        "Rolling_Volatility": np.full(n, np.nan, dtype=float),
        "Holding_Bars": np.zeros(n, dtype=int),
        "Trade_Return": np.zeros(n, dtype=float),
        "Take_Profit_Return": np.full(n, np.nan, dtype=float),
        "Stop_Loss_Return": np.full(n, np.nan, dtype=float),
        "Cooldown_Remaining": np.zeros(n, dtype=int),
        "Risk_State": risk_states,
        "Signal_Reason": reasons,
    })


def generate_professional_signals(
    pred_target: np.ndarray | None,
    pred_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
    observed_returns: np.ndarray | None = None,
    realized_price: np.ndarray | None = None,
    market_regime: np.ndarray | None = None,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """
    Generate professional BUY/HOLD/EXIT/NO_TRADE decisions for long/flat trading.

    Decisions are leakage-safe: realized_price is accepted for backward
    compatibility but is not used. Volatility, stop-loss and take-profit state
    are based on returns known before each decision.
    """
    _ = realized_price  # Compatibility-only; do not use realized prices in signal decisions.
    cfg = config or SignalConfig()
    _validate_config(cfg)
    pred_price = np.asarray(pred_price, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    expected_return = _expected_return(pred_target, pred_price, prev_close, target_mode)

    lengths = [len(expected_return), len(pred_price), len(prev_close)]
    observed_arr = None
    if observed_returns is not None:
        observed_arr = np.asarray(observed_returns, dtype=float).ravel()
        lengths.append(len(observed_arr))
    n = min(lengths) if lengths else 0

    expected_return = expected_return[-n:]
    pred_price = pred_price[-n:]
    prev_close = prev_close[-n:]
    if observed_arr is not None:
        observed_arr = observed_arr[-n:]
    else:
        observed_arr = _expected_return_to_simple_return(expected_return, target_mode)
    if market_regime is not None:
        market_regime_arr = np.asarray(market_regime, dtype=float).ravel()[-n:]
    else:
        market_regime_arr = np.zeros(n, dtype=float)

    rolling_vol = _rolling_volatility(observed_arr, cfg.volatility_window)

    total_cost = (commission_bps + slippage_bps) / 10000.0
    entry_threshold = np.maximum(
        total_cost * cfg.entry_cost_multiplier,
        rolling_vol * cfg.volatility_multiplier,
    )
    entry_threshold = np.maximum(entry_threshold, cfg.min_entry_threshold)
    base_entry_threshold = entry_threshold.copy()
    regime_multiplier = _regime_entry_multiplier(market_regime_arr, cfg)
    volatility_regime, volatility_multiplier = _volatility_entry_multiplier(rolling_vol, cfg)
    combined_threshold_multiplier = (
        cfg.entry_threshold_multiplier
        * regime_multiplier
        * volatility_multiplier
    )
    entry_threshold = entry_threshold * combined_threshold_multiplier
    exit_threshold = np.maximum(
        total_cost * cfg.exit_cost_multiplier,
        rolling_vol * cfg.exit_volatility_multiplier,
    )

    decisions: list[str] = []
    recommendations: list[str] = []
    recommendations_tr: list[str] = []
    positions = np.zeros(n, dtype=float)
    signal_strength = np.zeros(n, dtype=float)
    risk_states: list[str] = []
    reasons: list[str] = []
    holding_bars_values = np.zeros(n, dtype=int)
    trade_return_values = np.zeros(n, dtype=float)
    take_profit_values = np.full(n, np.nan, dtype=float)
    stop_loss_values = np.full(n, np.nan, dtype=float)
    cooldown_values = np.zeros(n, dtype=int)

    in_position = False
    entry_price = 0.0
    entry_idx = -1
    cooldown_remaining = 0

    for idx in range(n):
        exp_ret = float(expected_return[idx])
        vol = float(rolling_vol[idx])
        cooldown_values[idx] = cooldown_remaining
        signal_strength[idx] = exp_ret / max(float(entry_threshold[idx]), 1e-12)

        if in_position:
            holding_bars = max(0, idx - entry_idx)
            held_returns = observed_arr[entry_idx + 1 : idx + 1]
            current_return = float(np.prod(1.0 + held_returns) - 1.0) if len(held_returns) else 0.0
            take_profit = max(cfg.take_profit_vol_multiplier * vol, total_cost * cfg.entry_cost_multiplier)
            stop_loss = -max(cfg.stop_loss_vol_multiplier * vol, total_cost * cfg.entry_cost_multiplier)
            holding_bars_values[idx] = holding_bars
            trade_return_values[idx] = current_return
            take_profit_values[idx] = take_profit
            stop_loss_values[idx] = stop_loss

            should_exit = False
            risk_state = "in_position"
            reason = "Pozisyon korunuyor; beklenen getiri cikis esiginin uzerinde."

            emergency_stop = current_return <= stop_loss
            if emergency_stop and cfg.emergency_stop_overrides_min_hold:
                should_exit = True
                risk_state = "stop_loss"
                reason = "Acil zarar-kes bariyeri tetiklendigi icin minimum bekleme suresi beklenmeden cikis sinyali uretildi."
            elif holding_bars < cfg.min_holding_bars:
                decision = "HOLD"
                risk_state = "min_hold"
                reason = "Minimum elde tutma suresi dolmadigi icin pozisyon korunuyor."
            elif current_return >= take_profit:
                should_exit = True
                risk_state = "take_profit"
                reason = "Kar-al bariyeri tetiklendigi icin pozisyondan cikis sinyali uretildi."
            elif emergency_stop:
                should_exit = True
                risk_state = "stop_loss"
                reason = "Zarar-kes bariyeri tetiklendigi icin pozisyondan cikis sinyali uretildi."
            elif holding_bars >= cfg.max_holding_bars:
                should_exit = True
                risk_state = "max_hold"
                reason = "Maksimum elde tutma suresi doldugu icin pozisyondan cikis sinyali uretildi."
            elif exp_ret < float(exit_threshold[idx]):
                should_exit = True
                risk_state = "weak_signal"
                reason = "Beklenen getiri cikis esiginin altina dustugu icin pozisyondan cikis sinyali uretildi."
            else:
                decision = "HOLD"

            if should_exit:
                decision = "EXIT"
                in_position = False
                cooldown_remaining = cfg.cooldown_bars
                positions[idx] = 0.0
            else:
                positions[idx] = 1.0

        else:
            if cooldown_remaining > 0:
                decision = "NO_TRADE"
                risk_state = "cooldown"
                reason = "Son cikistan sonra bekleme suresi devam ettigi icin yeni pozisyon acilmadi."
                cooldown_remaining -= 1
            elif exp_ret > float(entry_threshold[idx]):
                decision = "BUY"
                risk_state = "entry_ok"
                reason = "Beklenen getiri maliyet ve volatilite esigini astigi icin al sinyali uretildi."
                in_position = True
                entry_price = float(prev_close[idx])
                entry_idx = idx
                positions[idx] = 1.0
                holding_bars_values[idx] = 0
                trade_return_values[idx] = 0.0
                take_profit_values[idx] = max(cfg.take_profit_vol_multiplier * vol, total_cost * cfg.entry_cost_multiplier)
                stop_loss_values[idx] = -max(cfg.stop_loss_vol_multiplier * vol, total_cost * cfg.entry_cost_multiplier)
            else:
                decision = "NO_TRADE"
                risk_state = "below_threshold"
                reason = "Beklenen getiri maliyet ve volatilite esigini asmadigi icin islem acilmadi."

        decisions.append(decision)
        recommendation, recommendation_tr = _recommendation_from_decision(
            decision,
            exp_ret,
            float(entry_threshold[idx]),
        )
        recommendations.append(recommendation)
        recommendations_tr.append(recommendation_tr)
        risk_states.append(risk_state)
        reasons.append(reason)

    return pd.DataFrame({
        "Decision": decisions,
        "Recommendation": recommendations,
        "Recommendation_TR": recommendations_tr,
        "Position": positions,
        "Expected_Return": expected_return,
        "Base_Entry_Threshold": base_entry_threshold,
        "Entry_Threshold": entry_threshold,
        "Exit_Threshold": exit_threshold,
        "Signal_Strength": signal_strength,
        "Quality_Gate_Mode": [cfg.quality_gate_mode] * n,
        "Quality_Threshold_Multiplier": np.full(n, cfg.entry_threshold_multiplier, dtype=float),
        "Market_Regime_SMA200": market_regime_arr,
        "Regime_Threshold_Multiplier": regime_multiplier,
        "Volatility_Regime": volatility_regime,
        "Volatility_Threshold_Multiplier": volatility_multiplier,
        "Final_Threshold_Multiplier": combined_threshold_multiplier,
        "Rolling_Volatility": rolling_vol,
        "Holding_Bars": holding_bars_values,
        "Trade_Return": trade_return_values,
        "Take_Profit_Return": take_profit_values,
        "Stop_Loss_Return": stop_loss_values,
        "Cooldown_Remaining": cooldown_values,
        "Risk_State": risk_states,
        "Signal_Reason": reasons,
    })


def _validate_config(config: SignalConfig) -> None:
    validate_signal_config(config)


def _expected_return(
    pred_target: np.ndarray | None,
    pred_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    return signal_math.expected_return(pred_target, pred_price, prev_close, target_mode)


def _expected_return_to_simple_return(expected_return: np.ndarray, target_mode: str) -> np.ndarray:
    return signal_math.expected_return_to_simple_return(expected_return, target_mode)


def _recommendation_from_decision(
    decision: str,
    expected_return: float,
    entry_threshold: float,
) -> tuple[str, str]:
    return signal_math.recommendation_from_decision(decision, expected_return, entry_threshold)


def _rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    return signal_math.rolling_volatility(returns, window)


def _regime_entry_multiplier(market_regime: np.ndarray, config: SignalConfig) -> np.ndarray:
    return signal_math.regime_entry_multiplier(market_regime, config)


def _volatility_entry_multiplier(rolling_vol: np.ndarray, config: SignalConfig) -> tuple[np.ndarray, np.ndarray]:
    return signal_math.volatility_entry_multiplier(rolling_vol, config)
