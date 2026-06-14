# -*- coding: utf-8 -*-
"""Float32 matrix helpers for pooled cross-sectional models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def as_float32_matrix(values) -> np.ndarray:
    """Return a finite, C-contiguous float32 2D matrix without unnecessary upcast."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"2D feature matrix beklenir, geldi: {arr.shape}")
    arr = np.ascontiguousarray(arr)
    if not np.isfinite(arr).all():
        arr = arr.copy()
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def as_float32_vector(values) -> np.ndarray:
    """Return a finite, C-contiguous float32 1D target/prediction vector."""
    arr = np.asarray(values, dtype=np.float32).ravel()
    arr = np.ascontiguousarray(arr)
    if not np.isfinite(arr).all():
        arr = arr.copy()
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def pooled_feature_matrix(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    row_mask=None,
) -> np.ndarray:
    """Build a C-contiguous float32 feature matrix preserving feature order."""
    cols = list(feature_cols)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"panel'de eksik ozellik kolonlari: {missing[:5]}")
    subset = frame.loc[row_mask, cols] if row_mask is not None else frame.loc[:, cols]
    values = subset.to_numpy(dtype=np.float32, copy=True)
    return as_float32_matrix(values)


def pooled_target_array(
    frame: pd.DataFrame,
    target_col: str,
    row_mask=None,
) -> np.ndarray:
    """Build a C-contiguous float32 target array."""
    if target_col not in frame.columns:
        raise ValueError(f"panel'de hedef kolonu yok: {target_col}")
    subset = frame.loc[row_mask, target_col] if row_mask is not None else frame.loc[:, target_col]
    values = subset.to_numpy(dtype=np.float32, copy=True)
    return as_float32_vector(values)
