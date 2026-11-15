# -*- coding: utf-8 -*-
"""Background data helpers for XAI strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class XAIBackgroundProvider:
    """Train-slice background summary used by SHAP/LIME/permutation paths."""

    tabular_background: np.ndarray | None = None
    tabular_median: np.ndarray | None = None
    scope: str = "unavailable"

    @classmethod
    def from_arrays(
        cls,
        *,
        X_train: Any | None = None,
        max_background_rows: int = 200,
    ) -> "XAIBackgroundProvider":
        tabular = _sample_rows(X_train, max_background_rows, expected_ndim=2)
        tabular_median = _nanmedian(tabular, axis=0) if tabular is not None else None
        scope = "train_slice_tabular" if tabular is not None else "unavailable"
        return cls(
            tabular_background=tabular,
            tabular_median=tabular_median,
            scope=scope,
        )

    @classmethod
    def unavailable(cls) -> "XAIBackgroundProvider":
        return cls()

    def tabular_mask_value(self, X: np.ndarray, feature_idx: int) -> float:
        if self.tabular_median is not None and feature_idx < len(self.tabular_median):
            value = float(self.tabular_median[feature_idx])
            if np.isfinite(value):
                return value
        col = np.asarray(X[:, feature_idx], dtype=float)
        value = float(np.nanmedian(col))
        return value if np.isfinite(value) else 0.0


def _sample_rows(values: Any | None, max_rows: int, *, expected_ndim: int = 2) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != expected_ndim or len(arr) == 0:
        return None
    if len(arr) <= max_rows:
        return arr.copy()
    return arr[-max_rows:].copy()


def _nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        out = np.nanmedian(values, axis=axis)
    return np.asarray(out, dtype=float)
