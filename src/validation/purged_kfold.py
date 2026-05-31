# -*- coding: utf-8 -*-
"""
purged_kfold.py - Purged K-Fold Cross-Validation.

Sprint 3 (2026-05-25) Plan A3.1:
  Lopez de Prado, "Advances in Financial Machine Learning" Ch. 7.

Klasik KFold zaman serisi icin guvensizdir:
  - Test penceresi train'in ortasinda olabilir => future leakage.
  - Rolling feature'lar (SMA_200, RollStd_30, vb.) test penceresi yakinindaki
    train ornekleri uzerinden test bilgisini gorur => purge gerekir.

PurgedKFold farki:
  - Fold'lar kronolojik gruplar uzerinden uretilir (shuffle YOK).
  - Her test fold'unun etrafinda `purge_window` kadar train ornek atilir.
  - Test'in hemen sonrasinda `embargo` kadar train ornek atilir (label-leakage
    onlemi; ardisik bar'larin overlapping etiketleri varsa kritik).

Kullanim:
    cv = PurgedKFold(n_splits=5, purge_window=200, embargo=10)
    for train_idx, test_idx in cv.split(X):
        ...

Output:
  Iterator[Tuple[np.ndarray, np.ndarray]]  -> (train_indices, test_indices).
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


class PurgedKFold:
    """K-Fold CV with purge window + embargo for time series.

    Args:
        n_splits: Fold sayisi (>=2).
        purge_window: Test'in etrafinda train'den atilacak ornek sayisi.
            Onerilen: `max(rolling_feature_window, time_steps)`. Plan default
            `max(200, time_steps)`.
        embargo: Test'in hemen sonrasinda train'den atilacak ek ornek.
            Overlapping label leakage'i icin (sequence model'lerde h gun
            forward etiket varsa h kadar olmali).

    Raises:
        ValueError: n_splits < 2 veya purge_window/embargo negatifse.
    """

    def __init__(self, n_splits: int = 5, purge_window: int = 0, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits >= 2 gerekli, alindi: {n_splits}")
        if purge_window < 0:
            raise ValueError(f"purge_window >= 0 gerekli, alindi: {purge_window}")
        if embargo < 0:
            raise ValueError(f"embargo >= 0 gerekli, alindi: {embargo}")
        self.n_splits = int(n_splits)
        self.purge_window = int(purge_window)
        self.embargo = int(embargo)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Fold sayısını döner (scikit-learn splitter arayüzü uyumu)."""
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) for each fold.

        Test fold'lari kronolojik sirali (shuffle yok). Her fold:
          test_idx = X[fold_start : fold_end]
          train_idx = {0..n-1} \\ test_idx \\ purge(test) \\ embargo(test)
        """
        n = len(X) if hasattr(X, "__len__") else int(X.shape[0])
        if n < self.n_splits:
            raise ValueError(f"n_samples ({n}) < n_splits ({self.n_splits})")

        indices = np.arange(n)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            test_start = k * fold_size
            # Son fold kalan ornekleri de alir
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]

            # Purge: test'in onunde purge_window
            purge_start = max(0, test_start - self.purge_window)
            # Embargo: test'in arkasinda embargo
            embargo_end = min(n, test_end + self.embargo)

            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            train_idx = indices[train_mask]

            if len(train_idx) == 0:
                # Ekstrem kucuk veri seti; bu fold'u atla
                continue

            yield train_idx, test_idx
