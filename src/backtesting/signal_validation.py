# -*- coding: utf-8 -*-
"""Validation helpers for long/flat signal configuration objects."""

from __future__ import annotations

from typing import Any


def validate_signal_config(config: Any) -> None:
    _validate_thresholds(config)
    _validate_cost_and_volatility(config)
    _validate_holding_rules(config)
    _validate_quality_gate(config)
    _validate_soft_gate(config)
    _validate_positive_multipliers(config)
    _validate_volatility_quantiles(config)


def _validate_thresholds(config: Any) -> None:
    if config.buy_threshold < 0:
        raise ValueError("buy_threshold negatif olamaz.")
    if config.sell_threshold < 0:
        raise ValueError("sell_threshold negatif olamaz.")
    if config.entry_threshold_multiplier < 1.0:
        raise ValueError("entry_threshold_multiplier 1.0 veya daha buyuk olmalidir.")


def _validate_cost_and_volatility(config: Any) -> None:
    if config.entry_cost_multiplier <= 0:
        raise ValueError("entry_cost_multiplier pozitif olmalidir.")
    if config.exit_cost_multiplier < 0:
        raise ValueError("exit_cost_multiplier negatif olamaz.")
    if config.volatility_window < 2:
        raise ValueError("volatility_window en az 2 olmalidir.")
    if config.volatility_multiplier < 0 or config.exit_volatility_multiplier < 0:
        raise ValueError("volatilite carpani negatif olamaz.")


def _validate_holding_rules(config: Any) -> None:
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


def _validate_quality_gate(config: Any) -> None:
    if config.quality_gate_mode not in {"hard", "soft", "off"}:
        raise ValueError("quality_gate_mode 'hard', 'soft' veya 'off' olmalidir.")
    if config.min_directional_accuracy < 0 or config.min_directional_accuracy > 100:
        raise ValueError("min_directional_accuracy 0-100 arasinda olmalidir.")
    if config.max_rmse_vs_benchmark <= 0:
        raise ValueError("max_rmse_vs_benchmark pozitif olmalidir.")
    if config.min_composite_score < 0:
        raise ValueError("min_composite_score negatif olamaz.")


def _validate_soft_gate(config: Any) -> None:
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


def _validate_positive_multipliers(config: Any) -> None:
    for name in [
        "regime_bull_entry_multiplier",
        "regime_neutral_entry_multiplier",
        "regime_bear_entry_multiplier",
        "volatility_low_entry_multiplier",
        "volatility_normal_entry_multiplier",
        "volatility_high_entry_multiplier",
    ]:
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} pozitif olmalidir.")


def _validate_volatility_quantiles(config: Any) -> None:
    if not 0 <= config.volatility_low_quantile <= 1:
        raise ValueError("volatility_low_quantile 0-1 arasinda olmalidir.")
    if not 0 <= config.volatility_high_quantile <= 1:
        raise ValueError("volatility_high_quantile 0-1 arasinda olmalidir.")
    if config.volatility_low_quantile >= config.volatility_high_quantile:
        raise ValueError("volatility_low_quantile high quantile degerinden kucuk olmalidir.")
