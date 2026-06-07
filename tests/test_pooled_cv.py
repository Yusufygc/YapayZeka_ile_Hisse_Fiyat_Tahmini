# -*- coding: utf-8 -*-
"""Leakage + structure tests for the pooled date-based purged walk-forward CV."""

import numpy as np
import pandas as pd

from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward


def _panel(n_dates: int = 200, symbols=("AAA", "BBB", "CCC"), h: int = 5) -> pd.DataFrame:
    """Sentetik panel; CCC erken 'delisted' (survivorship testi). target_date
    her sembolun KENDI ileri tarihidir (gap'lerde kesin purge testi)."""
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for s in symbols:
        sd = dates if s != "CCC" else dates[:120]
        for i, d in enumerate(sd):
            td = sd[i + h] if i + h < len(sd) else pd.NaT
            rows.append({"symbol": s, "Date": d, "feat": np.sin(i / 10.0),
                         "target": 0.01 * np.cos(i / 7.0), "target_date": td})
    df = pd.DataFrame(rows).dropna(subset=["target_date"]).reset_index(drop=True)
    return df


def _cfg() -> PooledCVConfig:
    return PooledCVConfig(
        target_horizon=5, embargo_buffer=2, window_len=10,
        n_windows=3, min_train_days=20, final_holdout=True,
    )


def test_split_produces_selectable_folds_plus_one_holdout():
    folds = PooledPurgedWalkForward(_cfg()).split(_panel())
    holds = [f for f in folds if f.is_final_holdout]
    sel = [f for f in folds if not f.is_final_holdout]
    assert len(holds) == 1
    assert len(sel) >= 1
    # holdout en yeni: tum secilebilir pencerelerden sonra
    assert holds[0].test_date_start > max(f.test_date_end for f in sel)


def test_no_cross_symbol_or_horizon_leak():
    panel = _panel()
    dates = pd.to_datetime(panel["Date"]).to_numpy()
    tdate = pd.to_datetime(panel["target_date"]).to_numpy()
    for f in PooledPurgedWalkForward(_cfg()).split(panel):
        a_k = np.datetime64(f.test_date_start)
        # 1) capraz-sembol/zaman: tum train tarihleri test baslangicindan once
        assert dates[f.train_mask].max() < a_k
        # 2) horizon purge: hicbir train satirinin hedef tarihi test'e uzanmaz
        assert tdate[f.train_mask].max() < a_k


def test_train_test_masks_disjoint():
    panel = _panel()
    for f in PooledPurgedWalkForward(_cfg()).split(panel):
        assert not np.any(f.train_mask & f.test_mask)
        assert f.n_train > 0 and f.n_test > 0


def test_selectable_test_windows_non_overlapping():
    sel = [f for f in PooledPurgedWalkForward(_cfg()).split(_panel()) if not f.is_final_holdout]
    sel = sorted(sel, key=lambda f: f.test_date_start)
    for a, b in zip(sel, sel[1:]):
        assert a.test_date_end < b.test_date_start


def test_deterministic():
    panel = _panel()
    f1 = PooledPurgedWalkForward(_cfg()).split(panel)
    f2 = PooledPurgedWalkForward(_cfg()).split(panel)
    assert len(f1) == len(f2)
    for a, b in zip(f1, f2):
        assert a.test_date_start == b.test_date_start
        assert np.array_equal(a.train_mask, b.train_mask)
        assert np.array_equal(a.test_mask, b.test_mask)


def test_delisted_symbol_absent_from_late_test_windows():
    panel = _panel()
    # CCC son tarihi
    ccc_last = panel.loc[panel.symbol == "CCC", "Date"].max()
    for f in PooledPurgedWalkForward(_cfg()).split(panel):
        if f.test_date_start > ccc_last:
            test_syms = set(panel.loc[f.test_mask, "symbol"])
            assert "CCC" not in test_syms


def test_expanding_train_grows_with_window():
    sel = [f for f in PooledPurgedWalkForward(_cfg()).split(_panel()) if not f.is_final_holdout]
    sel = sorted(sel, key=lambda f: f.test_date_start)
    # expanding: daha gec pencere >= train satiri
    for a, b in zip(sel, sel[1:]):
        assert b.n_train >= a.n_train
