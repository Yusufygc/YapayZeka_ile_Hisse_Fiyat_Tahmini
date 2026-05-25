# -*- coding: utf-8 -*-
"""
Sprint 3 (2026-05-25) — PurgedKFold testleri.

Plan A3.1: AFML Ch.7 PurgedKFold. Test penceresi etrafindaki purge_window
+ embargo train ornekleri atilir.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.purged_kfold import PurgedKFold


def test_constructor_validates_n_splits():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)


def test_constructor_validates_purge_window():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, purge_window=-1)


def test_constructor_validates_embargo():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, purge_window=10, embargo=-1)


def test_get_n_splits():
    cv = PurgedKFold(n_splits=5, purge_window=10)
    assert cv.get_n_splits() == 5


def test_split_basic_partition():
    n = 100
    X = np.arange(n).reshape(-1, 1)
    cv = PurgedKFold(n_splits=5, purge_window=0, embargo=0)
    folds = list(cv.split(X))
    assert len(folds) == 5
    # Test indices cover whole range and disjoint
    all_test = np.concatenate([t for _, t in folds])
    assert np.array_equal(np.sort(all_test), np.arange(n))


def test_split_purge_window_removes_neighbors():
    n = 100
    X = np.arange(n).reshape(-1, 1)
    cv = PurgedKFold(n_splits=5, purge_window=5, embargo=0)
    for train_idx, test_idx in cv.split(X):
        test_min, test_max = test_idx.min(), test_idx.max()
        # Train must not contain indices within purge_window before test_min
        purge_zone = np.arange(max(0, test_min - 5), test_min)
        assert not np.any(np.isin(train_idx, purge_zone))


def test_split_embargo_removes_post_test():
    n = 100
    X = np.arange(n).reshape(-1, 1)
    cv = PurgedKFold(n_splits=5, purge_window=0, embargo=8)
    for train_idx, test_idx in cv.split(X):
        test_max = test_idx.max()
        embargo_zone = np.arange(test_max + 1, min(n, test_max + 1 + 8))
        assert not np.any(np.isin(train_idx, embargo_zone))


def test_train_and_test_disjoint():
    n = 200
    X = np.arange(n).reshape(-1, 1)
    cv = PurgedKFold(n_splits=4, purge_window=10, embargo=5)
    for train_idx, test_idx in cv.split(X):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_n_samples_smaller_than_n_splits_raises():
    cv = PurgedKFold(n_splits=10)
    with pytest.raises(ValueError):
        list(cv.split(np.arange(5).reshape(-1, 1)))
