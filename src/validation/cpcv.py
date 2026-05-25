# -*- coding: utf-8 -*-
"""
cpcv.py - Combinatorial Purged Cross-Validation.

Sprint 3 (2026-05-25) Plan A3.2:
  Lopez de Prado, "Advances in Financial Machine Learning" Ch. 12.

Klasik KFold tek bir backtest path uretir. CPCV birden cok path uretir:
  - Veri N grup'a (group) bolunur.
  - Her kombinasyon icin k grup test, kalan (N-k) grup train.
  - Toplam fold sayisi = C(N, k) = N! / (k! (N-k)!).
  - Train fold'larindan, test gruplarinin etrafindaki purge_window+embargo
    ornekleri atilir (PurgedKFold ile ayni leakage onlemi).

CPCV avantaji:
  - C(N,k) tane path => Sharpe icin empirical confidence interval.
  - Tek-path KFold'un overfitting riskini sayisal olarak gosterir.

Ornek (N=6, k=2):
  C(6,2) = 15 path. Her path 2 test grubu + 4 train grubu.

Kullanim:
    cv = CombinatorialPurgedCV(n_groups=6, k_test=2,
                                purge_window=200, embargo=10)
    for train_idx, test_idx in cv.split(X):
        ...

Output:
  Iterator[Tuple[np.ndarray, np.ndarray]]  -> (train_indices, test_indices).
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator, Tuple

import numpy as np


class CombinatorialPurgedCV:
    """CPCV per Lopez de Prado AFML Ch. 12.

    Args:
        n_groups: Veri kac grup'a bolunecek (>=2).
        k_test: Her kombinasyonda kac grup test (1 <= k_test < n_groups).
        purge_window: Test gruplarinin etrafinda train'den atilacak ornek.
            Onerilen: max(rolling_feature_window, time_steps).
        embargo: Test gruplarinin hemen sonrasinda train'den atilacak ek
            ornek (overlapping label leakage'i icin).

    Raises:
        ValueError: Parametre kisitlarini ihlal ederse.
    """

    def __init__(
        self,
        n_groups: int = 6,
        k_test: int = 2,
        purge_window: int = 0,
        embargo: int = 0,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups >= 2 gerekli, alindi: {n_groups}")
        if k_test < 1 or k_test >= n_groups:
            raise ValueError(
                f"1 <= k_test < n_groups gerekli, alindi: k_test={k_test}, n_groups={n_groups}"
            )
        if purge_window < 0:
            raise ValueError(f"purge_window >= 0 gerekli, alindi: {purge_window}")
        if embargo < 0:
            raise ValueError(f"embargo >= 0 gerekli, alindi: {embargo}")
        self.n_groups = int(n_groups)
        self.k_test = int(k_test)
        self.purge_window = int(purge_window)
        self.embargo = int(embargo)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """C(n_groups, k_test) kombinasyon."""
        from math import comb

        return comb(self.n_groups, self.k_test)

    def _group_bounds(self, n: int) -> list[Tuple[int, int]]:
        """Veri uzunlugu n icin n_groups bolme: [(start, end), ...]."""
        if n < self.n_groups:
            raise ValueError(f"n_samples ({n}) < n_groups ({self.n_groups})")
        group_size = n // self.n_groups
        bounds = []
        for g in range(self.n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < self.n_groups - 1 else n
            bounds.append((start, end))
        return bounds

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) for each combinatorial path."""
        n = len(X) if hasattr(X, "__len__") else int(X.shape[0])
        bounds = self._group_bounds(n)
        indices = np.arange(n)

        for combo in combinations(range(self.n_groups), self.k_test):
            test_mask = np.zeros(n, dtype=bool)
            for g in combo:
                gs, ge = bounds[g]
                test_mask[gs:ge] = True

            # Purge + embargo: her test grubunun etrafinda
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_mask] = False
            for g in combo:
                gs, ge = bounds[g]
                purge_start = max(0, gs - self.purge_window)
                embargo_end = min(n, ge + self.embargo)
                train_mask[purge_start:embargo_end] = False

            train_idx = indices[train_mask]
            test_idx = indices[test_mask]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx
