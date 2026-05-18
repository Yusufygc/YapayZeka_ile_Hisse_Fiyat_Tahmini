# -*- coding: utf-8 -*-
"""Rolling-holdout değerlendirme (Adim 2.1).

Son N yıldaki veriyi step_size barlık adımlarla window_size barlık
pencerelerle değerlendirir. Her pencere için model tahminleri ve
gerçek değerlerden net_return hesaplanır.

Sonuç: median_net_return, positive_window_ratio, iqr_net_return
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


def rolling_holdout_evaluate(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    window_size: int = 60,
    step_size: int = 20,
    min_windows: int = 3,
) -> Dict[str, Any]:
    """Kayan pencereli holdout değerlendirmesi.

    Parameters
    ----------
    predict_fn:
        Eğitilmiş model çağrısı; X[window] → y_pred array döner.
    X:
        Feature matrix (n_samples, n_features).
    y:
        Gerçek return değerleri (n_samples,).
    window_size:
        Her penceredeki bar sayısı (varsayılan 60).
    step_size:
        Pencere adım boyutu (varsayılan 20).
    min_windows:
        Minimum pencere sayısı; bu sayıdan az pencere varsa boş sonuç döner.

    Returns
    -------
    dict:
        median_net_return, positive_window_ratio, iqr_net_return,
        window_returns (list), n_windows.
    """
    y_flat = np.asarray(y, dtype=float).ravel()
    n = len(y_flat)

    if n < window_size:
        return _empty_result()

    starts = list(range(0, n - window_size + 1, step_size))
    if len(starts) < min_windows:
        return _empty_result()

    window_returns: List[float] = []
    for start in starts:
        end = start + window_size
        x_win = X[start:end]
        y_win = y_flat[start:end]
        try:
            preds = np.asarray(predict_fn(x_win), dtype=float).ravel()
            signals = np.sign(preds)
            bar_returns = signals * y_win
            net_return = float(np.sum(bar_returns))
        except Exception:
            continue
        window_returns.append(net_return)

    if len(window_returns) < min_windows:
        return _empty_result()

    arr = np.array(window_returns, dtype=float)
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return {
        "median_net_return": float(np.median(arr)),
        "positive_window_ratio": float(np.mean(arr > 0.0)),
        "iqr_net_return": q75 - q25,
        "window_returns": window_returns,
        "n_windows": len(window_returns),
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "median_net_return": None,
        "positive_window_ratio": None,
        "iqr_net_return": None,
        "window_returns": [],
        "n_windows": 0,
    }
