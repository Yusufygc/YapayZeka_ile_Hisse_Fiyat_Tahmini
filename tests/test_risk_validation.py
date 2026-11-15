# -*- coding: utf-8 -*-
"""Risk dogrulama: purge, kalibrasyonun yalniz onceki bloklardan gelmesi, kapi/fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_synth import make_panel
from src.risk import validation as V
from src.risk.features import build_risk_frame, labeled_rows
from src.risk.model import METHOD_MODEL, METHOD_NAIVE, bias_factor, conformal_z
from src.risk.serving import RISK_COLUMNS, compute_risk_serving, score_risk_table
from src.risk.validation import (
    RiskValidationConfig, calibrate_from_prior, gate_passed, newey_west_t,
    run_risk_validation)

CFG = RiskValidationConfig(min_train_days=300, window_len=63, conformal_window_days=120)


@pytest.fixture(scope="module")
def frame():
    return build_risk_frame(make_panel(n_sym=20, n_days=1000, seed=3))


@pytest.fixture(scope="module")
def res(frame):
    return run_risk_validation(frame, CFG)


def test_folds_are_purged_and_embargoed(frame):
    lab = labeled_rows(frame).sort_values(["Date", "symbol"]).reset_index(drop=True)
    folds = V._make_folds(lab, CFG)
    assert len(folds) >= 5
    dates = np.sort(lab["Date"].unique())
    pos = {d: i for i, d in enumerate(dates)}
    for f in folds:
        tr, te = lab[f.train_mask], lab[f.test_mask]
        start = te["Date"].min()
        assert tr["target_date"].max() < start                   # hedefi test'e uzanan train yok
        assert tr["Date"].max() < start
        gap = pos[start] - pos[tr["Date"].max()]
        assert gap >= CFG.horizon + CFG.embargo_buffer           # embargo h+5
        assert not (set(tr.index) & set(te.index))


def test_oos_blocks_chronological_and_each_row_once(res):
    oos = res.oos
    assert not oos.duplicated(["symbol", "Date"]).any()
    starts = oos.groupby("fold")["Date"].min().sort_index()
    assert starts.is_monotonic_increasing
    assert res.metrics["n_oos_rows"] == len(oos)


def test_first_fold_uncalibrated_and_later_calibrated(res):
    first = res.oos[res.oos["fold"] == res.oos["fold"].min()]
    assert not first["calibrated"].any() and first["sigma_h"].isna().all()
    assert res.oos[res.oos["fold"] != res.oos["fold"].min()]["calibrated"].all()


def test_bias_and_z_use_only_prior_folds(res):
    """Blok k'nin c ve z'si == yalniz (fold<k, target_date<test_basi) satirlardan yeniden hesap."""
    oos = res.oos
    for fi in res.folds[1:]:
        k = fi["fold"]
        start = pd.Timestamp(fi["test_start"])
        prior = oos[(oos["fold"] < k) & (oos["target_date"] < start)]
        cal = calibrate_from_prior(prior, start, CFG)
        assert cal["c"] == pytest.approx(fi["c"]) and cal["z"] == pytest.approx(fi["z"])
        # ayni blogun kendi satirlari / sonraki bloklar sonucu degistirmez
        polluted = pd.concat([prior, oos[oos["fold"] >= k].assign(y=9.0, fwd_ret=5.0)])
        cal2 = calibrate_from_prior(polluted, start, CFG)
        assert cal2["c"] == pytest.approx(cal["c"]) and cal2["z"] == pytest.approx(cal["z"])


def test_calibration_excludes_rows_whose_target_not_yet_realised():
    d = pd.date_range("2020-01-01", periods=100, freq="B")
    cutoff = d[90]
    prior = pd.DataFrame({"Date": d, "target_date": d + pd.offsets.BDay(5), "y": -4.0,
                          "pred": -4.0, "rv_21": -4.0, "fwd_ret": 0.01})
    base = calibrate_from_prior(prior, cutoff, CFG)
    leak = prior.copy()
    late = leak["target_date"] >= cutoff
    assert late.any()
    leak.loc[late, ["y", "fwd_ret"]] = [10.0, 5.0]
    after = calibrate_from_prior(leak, cutoff, CFG)
    assert after["c"] == pytest.approx(base["c"]) and after["z"] == pytest.approx(base["z"])
    assert np.isnan(calibrate_from_prior(prior.iloc[0:0], cutoff, CFG)["c"])


def test_bias_factor_and_conformal_z_pooled_quantile():
    assert bias_factor(np.zeros(10)) == pytest.approx(1.0)
    e = np.array([-0.5, 0.5])
    assert bias_factor(e) == pytest.approx(np.mean(np.exp(2 * e)))
    assert bias_factor(np.array([])) == 1.0
    ret = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    sig = np.full(5, 0.1)
    assert conformal_z(ret, sig, 0.8) == pytest.approx(np.quantile(np.abs(ret) / 0.1, 0.8))
    assert np.isnan(conformal_z(np.array([]), np.array([])))


def test_metrics_shape_and_quintiles(res):
    m = res.metrics
    for k in ("mse_ratio", "dm_t", "qlike_model", "qlike_naive", "coverage", "n_oos_rows",
              "coverage_by_year", "mse_by_liq_quintile", "gate_passed", "method"):
        assert k in m
    assert sorted(m["mse_by_liq_quintile"]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert 0.5 < m["coverage"] < 0.95 and m["qlike_model"] > 0
    assert m["mse_ratio"] == pytest.approx(m["mse_model"] / m["mse_naive"])
    assert res.method == (METHOD_MODEL if res.passed else METHOD_NAIVE)


def test_newey_west_t():
    rng = np.random.default_rng(0)
    assert abs(newey_west_t(rng.normal(size=2000), 5)) < 3.5
    assert newey_west_t(rng.normal(0.5, 1.0, 2000), 5) > 5
    assert np.isnan(newey_west_t(np.array([1.0, 2.0]), 5))


def test_gate_thresholds():
    ok = dict(mse_ratio=0.9, dm_t=3.0, coverage=0.8)
    assert gate_passed(ok)
    assert not gate_passed({**ok, "mse_ratio": 0.96})
    assert not gate_passed({**ok, "dm_t": 1.9})
    assert not gate_passed({**ok, "coverage": 0.74}) and not gate_passed({**ok, "coverage": 0.86})
    assert gate_passed({**ok, "mse_ratio": 0.95, "dm_t": 2.0, "coverage": 0.75})
    assert not gate_passed({**ok, "dm_t": float("nan")})
    assert not gate_passed({})


def test_gate_fail_serves_naive_with_own_conformal_z(frame, monkeypatch):
    monkeypatch.setattr(V, "GATE_MAX_MSE_RATIO", 0.01)       # kapi gecilemez
    as_of = frame["Date"].max()
    table, summary, result = compute_risk_serving(frame_to_panel(frame), as_of, CFG)
    assert result.method == METHOD_NAIVE and summary["gate_passed"] is False
    day = frame[frame["Date"] == as_of].set_index("symbol")
    t = table.set_index("symbol")
    assert (t["risk_method"] == "naive_fallback").all()
    exp = np.sqrt(5) * np.exp(day.loc[t.index, "rv_21"])
    assert np.allclose(t["risk_sigma_5d"], exp) and np.allclose(t["risk_sigma_naive_5d"], exp)
    assert t["risk_z"].iloc[0] == pytest.approx(result.z_naive_final)


def test_gate_pass_serves_model(frame, monkeypatch):
    monkeypatch.setattr(V, "GATE_MAX_MSE_RATIO", 10.0)
    monkeypatch.setattr(V, "GATE_MIN_DM_T", -1e9)
    monkeypatch.setattr(V, "GATE_COVERAGE_MIN", 0.0)
    monkeypatch.setattr(V, "GATE_COVERAGE_MAX", 1.0)
    as_of = frame["Date"].max()
    table, summary, result = compute_risk_serving(frame_to_panel(frame), as_of, CFG)
    assert result.method == METHOD_MODEL and summary["gate_passed"] is True
    assert list(table.columns) == ["symbol", *RISK_COLUMNS]
    assert (table["risk_method"] == "har_x_ridge").all()
    assert (table["risk_horizon_days"] == 5).all() and (table["risk_band_level"] == 0.8).all()
    assert table["risk_level"].isin(["düşük", "orta", "yüksek"]).all()
    assert table["risk_level"].value_counts().min() >= len(table) // 3 - 1   # tertile
    assert table["risk_vol_percentile"].between(0, 100).all()
    assert (table["risk_sigma_5d"] > 0).all() and table["risk_z"].iloc[0] == pytest.approx(result.z_final)
    assert not np.allclose(table["risk_sigma_5d"], table["risk_sigma_naive_5d"])


def test_score_table_nan_features_gives_none(frame, res):
    day = frame[frame["Date"] == frame["Date"].max()].copy()
    day.loc[day.index[0], "rv_63"] = np.nan
    model = None
    from src.risk.model import PooledVolModel, VolModelConfig
    lab = labeled_rows(frame)
    model = PooledVolModel(VolModelConfig()).fit(lab)
    model.set_calibration(res.c_final, res.z_final, res.z_naive_final)
    forced = V.RiskValidationResult(res.oos, res.metrics, res.folds, METHOD_MODEL, True,
                                    res.c_final, res.z_final, res.z_naive_final, CFG)
    t = score_risk_table(day, day["Date"].iloc[0], forced, model)
    bad = t[t["symbol"] == day["symbol"].iloc[0]].iloc[0]
    assert np.isnan(bad["risk_sigma_5d"]) and bad["risk_method"] is None
    assert t["risk_sigma_5d"].notna().sum() == len(t) - 1


def frame_to_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Dogrulama testleri icin ayni sentetik panel (frame ayni tohumdan)."""
    return make_panel(n_sym=20, n_days=1000, seed=3)
