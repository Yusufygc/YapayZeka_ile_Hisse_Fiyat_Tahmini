# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    """
    Professional long/flat signal settings.

    The defaults are intentionally conservative and include transaction-cost and
    volatility buffers before opening a position.
    """

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


def generate_professional_signals(
    pred_target: np.ndarray | None,
    pred_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
    observed_returns: np.ndarray | None = None,
    realized_price: np.ndarray | None = None,
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

    rolling_vol = _rolling_volatility(observed_arr, cfg.volatility_window)

    total_cost = (commission_bps + slippage_bps) / 10000.0
    entry_threshold = np.maximum(
        total_cost * cfg.entry_cost_multiplier,
        rolling_vol * cfg.volatility_multiplier,
    )
    entry_threshold = np.maximum(entry_threshold, cfg.min_entry_threshold)
    base_entry_threshold = entry_threshold.copy()
    entry_threshold = entry_threshold * cfg.entry_threshold_multiplier
    exit_threshold = np.maximum(
        total_cost * cfg.exit_cost_multiplier,
        rolling_vol * cfg.exit_volatility_multiplier,
    )

    decisions: list[str] = []
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
            holding_bars = idx - entry_idx + 1
            current_return = float(np.prod(1.0 + observed_arr[entry_idx : idx + 1]) - 1.0)
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
            elif holding_bars <= cfg.min_holding_bars:
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
                holding_bars_values[idx] = 1
                trade_return_values[idx] = 0.0
                take_profit_values[idx] = max(cfg.take_profit_vol_multiplier * vol, total_cost * cfg.entry_cost_multiplier)
                stop_loss_values[idx] = -max(cfg.stop_loss_vol_multiplier * vol, total_cost * cfg.entry_cost_multiplier)
            else:
                decision = "NO_TRADE"
                risk_state = "below_threshold"
                reason = "Beklenen getiri maliyet ve volatilite esigini asmadigi icin islem acilmadi."

        decisions.append(decision)
        risk_states.append(risk_state)
        reasons.append(reason)

    return pd.DataFrame({
        "Decision": decisions,
        "Position": positions,
        "Expected_Return": expected_return,
        "Base_Entry_Threshold": base_entry_threshold,
        "Entry_Threshold": entry_threshold,
        "Exit_Threshold": exit_threshold,
        "Signal_Strength": signal_strength,
        "Quality_Gate_Mode": [cfg.quality_gate_mode] * n,
        "Quality_Threshold_Multiplier": np.full(n, cfg.entry_threshold_multiplier, dtype=float),
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
    if config.entry_cost_multiplier <= 0:
        raise ValueError("entry_cost_multiplier pozitif olmalidir.")
    if config.exit_cost_multiplier < 0:
        raise ValueError("exit_cost_multiplier negatif olamaz.")
    if config.volatility_window < 2:
        raise ValueError("volatility_window en az 2 olmalidir.")
    if config.volatility_multiplier < 0 or config.exit_volatility_multiplier < 0:
        raise ValueError("volatilite carpani negatif olamaz.")
    if config.min_holding_bars < 1:
        raise ValueError("min_holding_bars en az 1 olmalidir.")
    if config.max_holding_bars < config.min_holding_bars:
        raise ValueError("max_holding_bars, min_holding_bars degerinden kucuk olamaz.")
    if config.take_profit_vol_multiplier <= 0:
        raise ValueError("take_profit_vol_multiplier pozitif olmalidir.")
    if config.stop_loss_vol_multiplier <= 0:
        raise ValueError("stop_loss_vol_multiplier pozitif olmalidir.")
    if config.cooldown_bars < 0:
        raise ValueError("cooldown_bars negatif olamaz.")
    if config.quality_gate_mode not in {"hard", "soft", "off"}:
        raise ValueError("quality_gate_mode 'hard', 'soft' veya 'off' olmalidir.")
    if config.min_directional_accuracy < 0 or config.min_directional_accuracy > 100:
        raise ValueError("min_directional_accuracy 0-100 arasinda olmalidir.")
    if config.max_rmse_vs_benchmark <= 0:
        raise ValueError("max_rmse_vs_benchmark pozitif olmalidir.")
    if config.min_composite_score < 0:
        raise ValueError("min_composite_score negatif olamaz.")
    if config.entry_threshold_multiplier < 1.0:
        raise ValueError("entry_threshold_multiplier 1.0 veya daha buyuk olmalidir.")
    if config.soft_dir_acc_low < 0 or config.soft_dir_acc_low > 100:
        raise ValueError("soft_dir_acc_low 0-100 arasinda olmalidir.")
    if config.soft_entry_threshold_multiplier_mid < 1.0:
        raise ValueError("soft_entry_threshold_multiplier_mid 1.0 veya daha buyuk olmalidir.")
    if config.soft_entry_threshold_multiplier_low < config.soft_entry_threshold_multiplier_mid:
        raise ValueError("soft_entry_threshold_multiplier_low mid carpandan kucuk olamaz.")
    if config.soft_rmse_penalty_full <= 0:
        raise ValueError("soft_rmse_penalty_full pozitif olmalidir.")
    if config.soft_composite_low < 0:
        raise ValueError("soft_composite_low negatif olamaz.")


def _expected_return(
    pred_target: np.ndarray | None,
    pred_price: np.ndarray,
    prev_close: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    if pred_target is not None and target_mode in {"log_return", "return"}:
        return np.asarray(pred_target, dtype=float).ravel()
    if target_mode == "price":
        return (pred_price / np.maximum(prev_close, 1e-12)) - 1.0
    if pred_target is not None:
        return np.asarray(pred_target, dtype=float).ravel()
    return (pred_price / np.maximum(prev_close, 1e-12)) - 1.0


def _expected_return_to_simple_return(expected_return: np.ndarray, target_mode: str) -> np.ndarray:
    expected_return = np.asarray(expected_return, dtype=float).ravel()
    if target_mode == "log_return":
        return np.expm1(expected_return)
    if target_mode == "return":
        return expected_return
    return expected_return


def _rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=float).ravel()
    if len(returns) == 0:
        return returns

    window = max(2, int(window))
    vol = pd.Series(returns).rolling(window=window, min_periods=2).std()
    fallback = 1e-6
    vol = vol.fillna(fallback).replace(0.0, fallback)
    return vol.to_numpy(dtype=float)
