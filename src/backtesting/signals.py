# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np


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
