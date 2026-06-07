# -*- coding: utf-8 -*-
"""Date-based purged walk-forward CV for the pooled (panel) global model.

E2 Faz 2. Tasarim: docs/wiki/e2-faz2-pooled-cv-design.md.

Mevcut `TimeSeriesSplitter.walk_forward_splits` satir-indeksli ve tek-sembol
oldugu icin panele uymaz (ayni takvim tarihi bir sembolde train, baskasinda test
olur -> capraz-sembol leakage). Bu modul split'i **global takvim tarihi** ekseninde
yapar: bir test penceresindeki tum semboller ayni tarih araligini paylasir.

Leakage onlemleri:
  - capraz-sembol same-date: sinir tek global tarih `a_k`.
  - horizon (purge): train, hedefi test'e uzanan satirlari icermez. Panelde
    `target_date` kolonu varsa kesin purge (`target_date < a_k`); yoksa global
    embargo bosluguyla (E = target_horizon + embargo_buffer takvim-pozisyonu).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PooledCVConfig:
    target_horizon: int = 5
    embargo_buffer: int = 5          # h uzerine ek bosluk (pozisyon)
    window_len: int = 63             # her test penceresi ~3 islem ayi
    n_windows: int = 6               # secilebilir rolling OOS pencere sayisi
    step: int | None = None          # None -> window_len (ortusmez)
    max_train_days: int | None = None  # None=expanding, int=sliding (takvim gunu)
    min_train_days: int = 504        # bu kadar takvim gununden kisa fold atlanir
    final_holdout: bool = True       # en yeni pencere ayrilir, secimde kullanilmaz
    date_col: str = "Date"
    target_date_col: str = "target_date"  # varsa kesin purge icin


@dataclass(frozen=True)
class PooledFold:
    index: int
    train_mask: np.ndarray
    test_mask: np.ndarray
    test_date_start: pd.Timestamp
    test_date_end: pd.Timestamp
    embargo_positions: int
    is_final_holdout: bool
    n_train: int = field(default=0)
    n_test: int = field(default=0)


class PooledPurgedWalkForward:
    """Tarih-bazli purged + embargo'lu coklu-pencere walk-forward."""

    def __init__(self, cfg: PooledCVConfig | None = None) -> None:
        self.cfg = cfg or PooledCVConfig()

    def split(self, panel: pd.DataFrame) -> list[PooledFold]:
        cfg = self.cfg
        if cfg.date_col not in panel.columns:
            raise ValueError(f"panel'de '{cfg.date_col}' kolonu yok")
        dates = pd.to_datetime(panel[cfg.date_col]).to_numpy()
        uniq = np.array(sorted(pd.unique(dates)))
        m = len(uniq)
        E = max(1, int(cfg.target_horizon) + int(cfg.embargo_buffer))
        step = int(cfg.step) if cfg.step else int(cfg.window_len)
        w = int(cfg.window_len)

        has_tdate = cfg.target_date_col in panel.columns
        tdate = (
            pd.to_datetime(panel[cfg.target_date_col]).to_numpy()
            if has_tdate else None
        )

        folds: list[PooledFold] = []
        # En yeni pencereyi final-holdout olarak ayir.
        selectable_end = m - (w if cfg.final_holdout else 0)

        def _make_fold(idx: int, b_start: int, b_end: int, is_holdout: bool) -> PooledFold | None:
            if b_start < 0 or b_end > m or b_start >= b_end:
                return None
            a_k = uniq[b_start]               # ilk test tarihi
            test_last = uniq[b_end - 1]
            cutoff_pos = b_start - E          # train son tarih pozisyonu (exclusive)
            if cutoff_pos <= 0:
                return None
            cutoff_date = uniq[cutoff_pos - 1]
            # train alt sinir (expanding=basa, sliding=max_train_days geri)
            if cfg.max_train_days is not None:
                lo_date = cutoff_date - pd.Timedelta(days=int(cfg.max_train_days))
            else:
                lo_date = uniq[0]
            span_days = (pd.Timestamp(cutoff_date) - pd.Timestamp(lo_date)).days
            if span_days < cfg.min_train_days:
                return None

            test_mask = (dates >= a_k) & (dates <= test_last)
            train_mask = (dates <= cutoff_date) & (dates >= lo_date)
            # Kesin purge: hedefi test penceresine (veya sonrasina) uzanan train
            # satirlarini dusur. target_date yoksa E boslugu zaten saglar.
            if has_tdate:
                train_mask = train_mask & (tdate < a_k)
            if not test_mask.any() or not train_mask.any():
                return None
            return PooledFold(
                index=idx,
                train_mask=train_mask,
                test_mask=test_mask,
                test_date_start=pd.Timestamp(a_k),
                test_date_end=pd.Timestamp(test_last),
                embargo_positions=E,
                is_final_holdout=is_holdout,
                n_train=int(train_mask.sum()),
                n_test=int(test_mask.sum()),
            )

        # Secilebilir pencereler: selectable_end'den geriye dogru.
        idx = 0
        for k in range(int(cfg.n_windows)):
            b_end = selectable_end - k * step
            b_start = b_end - w
            fold = _make_fold(idx, b_start, b_end, is_holdout=False)
            if fold is not None:
                folds.append(fold)
                idx += 1

        folds.sort(key=lambda f: f.test_date_start)
        # Final holdout (en yeni pencere) en sona, secimde kullanilmaz.
        if cfg.final_holdout:
            hold = _make_fold(idx, m - w, m, is_holdout=True)
            if hold is not None:
                folds.append(hold)
        return folds
