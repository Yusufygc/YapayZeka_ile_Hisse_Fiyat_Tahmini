# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from src.data.pooled_matrix import (
    as_float32_matrix,
    as_float32_vector,
    pooled_feature_matrix,
    pooled_target_array,
)


def test_pooled_feature_matrix_preserves_order_and_returns_contiguous_float32():
    frame = pd.DataFrame({
        "a": [1.0, 2.0, np.nan],
        "b": [10.0, np.inf, 30.0],
        "c": [100.0, 200.0, 300.0],
    })
    mask = np.array([True, False, True])

    out = pooled_feature_matrix(frame, ["c", "a"], mask)

    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(out, np.array([[100.0, 1.0], [300.0, 0.0]], dtype=np.float32))


def test_pooled_target_array_returns_finite_float32_vector():
    frame = pd.DataFrame({"target": [1.0, np.nan, np.inf, -np.inf]})

    out = pooled_target_array(frame, "target")

    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(out, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_missing_feature_and_bad_matrix_shape_fail_loud():
    with pytest.raises(ValueError, match="eksik ozellik"):
        pooled_feature_matrix(pd.DataFrame({"a": [1.0]}), ["missing"])
    with pytest.raises(ValueError, match="2D feature matrix"):
        as_float32_matrix(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="hedef kolonu"):
        pooled_target_array(pd.DataFrame({"a": [1.0]}), "target")


def test_as_float32_vector_downcasts_without_upcast():
    out = as_float32_vector(np.array([1.0, 2.0], dtype=np.float64))

    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
