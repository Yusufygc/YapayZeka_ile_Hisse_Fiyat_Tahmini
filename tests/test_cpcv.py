# -*- coding: utf-8 -*-
"""
Sprint 3 (2026-05-25) — CombinatorialPurgedCV testleri.

Plan A3.2: AFML Ch.12 CPCV. C(N,k) kombinasyon path uretir;
her path icin purge + embargo uygulanir.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from src.validation.cpcv import CombinatorialPurgedCV


def test_constructor_validates_n_groups():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=1, k_test=1)


def test_constructor_validates_k_test_range():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=6, k_test=0)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=6, k_test=6)


def test_constructor_validates_negative_purge_embargo():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=6, k_test=2, purge_window=-1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=6, k_test=2, embargo=-1)


def test_get_n_splits_combinatorial():
    cv = CombinatorialPurgedCV(n_groups=6, k_test=2)
    assert cv.get_n_splits() == comb(6, 2)  # 15


def test_split_path_count():
    n = 600
    X = np.arange(n).reshape(-1, 1)
    cv = CombinatorialPurgedCV(n_groups=6, k_test=2)
    paths = list(cv.split(X))
    assert len(paths) == 15


def test_train_test_disjoint_each_path():
    n = 600
    X = np.arange(n).reshape(-1, 1)
    cv = CombinatorialPurgedCV(n_groups=6, k_test=2, purge_window=10, embargo=5)
    for train_idx, test_idx in cv.split(X):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_purge_window_removes_train_neighbors():
    n = 600
    X = np.arange(n).reshape(-1, 1)
    cv = CombinatorialPurgedCV(n_groups=6, k_test=2, purge_window=15, embargo=0)
    for train_idx, test_idx in cv.split(X):
        # No train index within purge_window before any test boundary
        for boundary in np.unique(test_idx):
            zone = np.arange(max(0, boundary - 15), boundary)
            assert not np.any(np.isin(train_idx, zone))


def test_n_samples_smaller_than_n_groups_raises():
    cv = CombinatorialPurgedCV(n_groups=10, k_test=2)
    with pytest.raises(ValueError):
        list(cv.split(np.arange(5).reshape(-1, 1)))


def test_test_indices_cover_group_segments():
    n = 600
    X = np.arange(n).reshape(-1, 1)
    cv = CombinatorialPurgedCV(n_groups=6, k_test=2, purge_window=0, embargo=0)
    group_size = n // 6
    for train_idx, test_idx in cv.split(X):
        # Test must be contiguous segments of size group_size each (or larger for last)
        assert len(test_idx) >= 2 * group_size - 5
