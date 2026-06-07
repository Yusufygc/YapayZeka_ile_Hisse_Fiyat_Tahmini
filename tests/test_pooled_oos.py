# -*- coding: utf-8 -*-
"""Per-symbol OOS aggregation harness tests (deterministic dummy models)."""

import numpy as np
import pandas as pd

from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward
from src.validation.pooled_oos import (
    PerSymbolOOSConfig,
    daily_cross_sectional_ic,
    evaluate_per_symbol,
)


def _panel(n_dates: int = 200, symbols=("AAA", "BBB", "CCC"), h: int = 5) -> pd.DataFrame:
    """Sentetik panel; feat = ileri getirisin gurultulu sinyali (ogrenilebilir)."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for s in symbols:
        sd = dates if s != "CCC" else dates[:120]
        close = 10.0 + np.cumsum(rng.normal(0, 0.2, len(sd)))
        close = np.clip(close, 1.0, None)
        for i, d in enumerate(sd):
            if i + h >= len(sd):
                continue
            tgt = float(np.log(close[i + h] / close[i]))
            # feat: hedefin gurultulu hali -> model ogrenebilsin
            feat = tgt + rng.normal(0, 0.01)
            rows.append({"symbol": s, "Date": d, "feat": feat,
                         "target": tgt, "target_date": sd[i + h]})
    return pd.DataFrame(rows).reset_index(drop=True)


def _cfg_cv() -> PooledCVConfig:
    return PooledCVConfig(
        target_horizon=5, embargo_buffer=2, window_len=10,
        n_windows=3, min_train_days=20, final_holdout=True,
    )


class _LinModel:
    """feat -> target lineer fit (deterministik, sklearn-vari)."""

    def fit(self, X, y):
        self.b = float(np.polyfit(X[:, 0], y, 1)[0])
        self.a = float(np.polyfit(X[:, 0], y, 1)[1])
        return self

    def predict(self, X):
        return self.a + self.b * X[:, 0]


class _ConstUp:
    """Hep kucuk pozitif tahmin (base-rate kontrolu icin)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), 1e-3)


def _folds(panel):
    return PooledPurgedWalkForward(_cfg_cv()).split(panel)


def test_result_schema_and_columns():
    panel = _panel()
    res = evaluate_per_symbol(panel, _folds(panel), _LinModel)
    for col in ["symbol", "n_oos", "n_folds", "dir_acc", "rmse",
                "base_rate", "edge", "positive_fold_ratio", "reliable"]:
        assert col in res.per_symbol.columns
    for col in ["symbol", "fold", "n", "dir_acc", "rmse", "base_rate"]:
        assert col in res.per_fold.columns
    assert set(res.per_symbol["symbol"]) <= {"AAA", "BBB", "CCC"}
    assert res.n_folds_used >= 1


def test_holdout_excluded_by_default():
    panel = _panel()
    folds = _folds(panel)
    sel = [f for f in folds if not f.is_final_holdout]
    res = evaluate_per_symbol(panel, folds, _LinModel)
    assert res.n_folds_used == len(sel)
    res2 = evaluate_per_symbol(panel, folds, _LinModel,
                               PerSymbolOOSConfig(include_holdout=True))
    assert res2.n_folds_used == len(folds)


def test_learnable_signal_beats_base_rate():
    panel = _panel()
    res = evaluate_per_symbol(panel, _folds(panel), _LinModel)
    # feat hedefi neredeyse birebir tasiyor -> dir_acc yuksek, edge pozitif
    for _, r in res.per_symbol.iterrows():
        if r["reliable"]:
            assert r["dir_acc"] >= 80.0
            assert r["edge"] > 0.0


def test_const_model_zero_edge_like():
    panel = _panel()
    res = evaluate_per_symbol(panel, _folds(panel), _ConstUp)
    # hep ayni yon -> dir_acc ~ up orani ~ base_rate -> edge ~ 0
    for _, r in res.per_symbol.iterrows():
        if r["reliable"] and np.isfinite(r["edge"]):
            assert r["edge"] <= 1e-6


def test_predictions_no_cross_fold_train_leak_dates():
    """OOS tahmin Date'leri test pencerelerinden; train tarihleriyle cakismaz."""
    panel = _panel()
    folds = _folds(panel)
    res = evaluate_per_symbol(panel, folds, _LinModel)
    by_fold = {f.index: f for f in folds}
    dates = pd.to_datetime(panel["Date"]).to_numpy()
    for fold_idx, g in res.predictions.groupby("fold"):
        f = by_fold[fold_idx]
        train_dates = set(dates[f.train_mask])
        assert not (set(pd.to_datetime(g["Date"]).to_numpy()) & train_dates)


def test_deterministic_results():
    panel = _panel()
    folds = _folds(panel)
    a = evaluate_per_symbol(panel, folds, _LinModel).per_symbol
    b = evaluate_per_symbol(panel, folds, _LinModel).per_symbol
    pd.testing.assert_frame_equal(a, b)


def test_daily_ic_perfect_and_inverse():
    """Perfect rank uyumu IC=+1; ters siralama IC=-1."""
    dates = pd.bdate_range("2021-01-01", periods=4)
    rows = []
    for d in dates:
        for s in range(8):
            rows.append({"symbol": f"S{s}", "Date": d,
                         "y_true": float(s), "y_pred": float(s)})
    perf = daily_cross_sectional_ic(pd.DataFrame(rows), min_names=8)
    assert abs(perf["ic_mean"] - 1.0) < 1e-9
    assert perf["pct_positive"] == 1.0 and perf["n_days"] == 4

    inv = pd.DataFrame(rows).copy()
    inv["y_pred"] = -inv["y_pred"]
    r = daily_cross_sectional_ic(inv, min_names=8)
    assert abs(r["ic_mean"] + 1.0) < 1e-9


def test_sample_gap_subsamples_ic_series():
    """sample_gap_days ardisik gunleri seyreltir -> daha az gun, ortusmeyen."""
    dates = pd.bdate_range("2021-01-01", periods=60)
    rows = []
    for d in dates:
        for s in range(10):
            rows.append({"symbol": f"S{s}", "Date": d,
                         "y_true": float(s), "y_pred": float(s)})
    preds = pd.DataFrame(rows)
    full = daily_cross_sectional_ic(preds, min_names=8)
    gap = daily_cross_sectional_ic(preds, min_names=8, sample_gap_days=21)
    assert full["n_days"] == 60
    assert gap["n_days"] < full["n_days"]   # seyreltildi
    # perfect IC -> ortalama her iki sekilde de ~1
    assert abs(gap["ic_mean"] - 1.0) < 1e-9


def test_thin_dates_excluded_from_ic():
    dates = pd.bdate_range("2021-01-01", periods=2)
    rows = []
    for s in range(4):  # min_names=8 altinda
        rows.append({"symbol": f"S{s}", "Date": dates[0],
                     "y_true": float(s), "y_pred": float(s)})
    res = daily_cross_sectional_ic(pd.DataFrame(rows), min_names=8)
    assert res["n_days"] == 0


def test_result_exposes_ic_summary():
    panel = _panel()
    res = evaluate_per_symbol(panel, _folds(panel), _LinModel)
    assert set(res.ic) >= {"ic_mean", "ic_std", "icir", "pct_positive", "n_days"}
