# -*- coding: utf-8 -*-
"""P1 Kol-B istatistik duzeltmeleri: NW t-stat, ortusmeyen ornekleme, trend
etiket/quintile tutarliligi, guven n_eff kapisi, vol-olcekli band, symbol_id bayragi."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from src.serving.confidence import peer_confidence
from src.serving.trend_tendency import TrendCalibration, trend_from_peer
from src.validation.pooled_oos import daily_cross_sectional_ic, split_ic_summary
from src.validation.segment_ic import segment_cross_sectional_ic


def _preds_from_ic_series(ic_target, dates, n_names=60, seed=0):
    """Her tarihte Spearman IC ~ ic_target[t] olan tahmin tablosu."""
    rng = np.random.default_rng(seed)
    rows = []
    for d, r in zip(dates, ic_target):
        z = rng.standard_normal(n_names)
        e = rng.standard_normal(n_names)
        pred = r * z + math.sqrt(max(1.0 - r * r, 1e-9)) * e
        rows.append(pd.DataFrame({
            "symbol": [f"S{i}" for i in range(n_names)], "Date": d,
            "y_true": z, "y_pred": pred}))
    return pd.concat(rows, ignore_index=True)


def _ar1(n, rho, seed=1, mu=0.03, sd=0.05):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.standard_normal(n) * sd * math.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return np.clip(mu + x, -0.95, 0.95)


# ---------------------------------------------------------------- 1. IC istatistikleri
def test_newey_west_t_smaller_than_naive_for_autocorrelated_ic():
    dates = pd.bdate_range("2020-01-01", periods=400)
    preds = _preds_from_ic_series(_ar1(400, 0.8), dates)
    res = daily_cross_sectional_ic(preds, min_names=8)
    naive_t = res["ic_mean"] / (res["ic_std"] / math.sqrt(res["n_days"]))
    assert "ic_t_nw" in res
    assert np.isfinite(res["ic_t_nw"])
    assert res["ic_t_nw"] < naive_t * 0.95   # otokorelasyon -> kucuk t (lag=4 AR(0.8)i kismen yakalar)


def test_newey_west_helper_matches_iid_limit():
    from src.validation.pooled_oos import newey_west_t
    rng = np.random.default_rng(3)
    x = rng.standard_normal(5000) + 0.1
    t_nw = newey_west_t(x, lag=4)
    naive = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    assert abs(t_nw - naive) / abs(naive) < 0.25   # iid: NW ~ naive
    assert math.isnan(newey_west_t(np.array([0.1]), lag=4))


def _holiday_dates(n_trading=60):
    """Hafta ici gunler, arada tatil bosluklari (takvim araligi duzensiz)."""
    d = pd.bdate_range("2021-01-04", periods=n_trading + 6)
    drop = {7, 8, 23, 24, 40, 41}   # 'tatil' pozisyonlari
    keep = [x for i, x in enumerate(d) if i not in drop]
    return pd.DatetimeIndex(keep[:n_trading])


def _marked_preds(dates, every=5):
    """Pozisyon %every==0 gunlerinde IC=+1, digerlerinde IC=-1."""
    rows = []
    for i, d in enumerate(dates):
        sign = 1.0 if i % every == 0 else -1.0
        for s in range(10):
            rows.append({"symbol": f"S{s}", "Date": d,
                         "y_true": float(s), "y_pred": sign * float(s)})
    return pd.DataFrame(rows)


def test_nonoverlap_takes_every_hth_trading_date_across_holidays():
    dates = _holiday_dates(60)
    res = daily_cross_sectional_ic(_marked_preds(dates), min_names=8, horizon=5)
    assert res["n_nonoverlap"] == 12                    # 60 / 5
    assert abs(res["ic_nonoverlap"] - 1.0) < 1e-9       # tam olarak 0,5,10,... tarihleri
    assert res["ic_overlap"] < 0.0                      # tum seri: 12*(+1) + 48*(-1)


def test_legacy_sample_gap_days_is_positional_not_calendar():
    dates = _holiday_dates(60)
    res = daily_cross_sectional_ic(_marked_preds(dates), min_names=8, sample_gap_days=5)
    assert res["n_days"] == 12
    assert abs(res["ic_mean"] - 1.0) < 1e-9


def test_headline_fields_and_annualization():
    dates = pd.bdate_range("2020-01-01", periods=300)
    preds = _preds_from_ic_series(_ar1(300, 0.5, seed=4), dates, seed=5)
    res = daily_cross_sectional_ic(preds, min_names=8, horizon=5)
    for k in ("ic_overlap", "icir_overlap", "ic_nonoverlap", "icir_nonoverlap",
              "icir_annualized", "ic_t_nw", "n_nonoverlap"):
        assert k in res, k
    assert res["icir"] == res["icir_overlap"]           # eski anahtar korunur
    assert res["icir_annualized"] == pytest.approx(
        res["icir_nonoverlap"] * math.sqrt(252 / 5))
    ic = np.array([x for x in preds.groupby("Date").apply(
        lambda g: pd.Series(g["y_pred"]).rank().corr(pd.Series(g["y_true"]).rank()))])
    assert res["ic_std"] == pytest.approx(ic.std(ddof=1), rel=1e-6)   # ddof=1


def test_split_ic_summary_nonoverlap_uses_positional_sampling():
    dates = _holiday_dates(60)
    p = _marked_preds(dates)
    p["is_holdout"] = False
    out = split_ic_summary(p, gap_days=5, min_names=8)
    assert out["selection_nonoverlap"]["n_days"] == 12


def test_segment_ic_default_min_names_is_20_and_has_nonoverlap_cols():
    assert inspect.signature(segment_cross_sectional_ic).parameters["min_names"].default == 20
    dates = pd.bdate_range("2021-01-01", periods=40)
    rows = []
    rng = np.random.default_rng(0)
    for d in dates:
        for s in range(25):
            y = rng.standard_normal()
            rows.append({"symbol": f"S{s}", "Date": d, "seg": "A",
                         "y_true": y, "y_pred": y + rng.standard_normal()})
    out = segment_cross_sectional_ic(pd.DataFrame(rows), "seg", horizon=5)
    assert {"icir_nonoverlap", "n_nonoverlap", "ic_t_nw"} <= set(out.columns)
    assert int(out.loc[0, "n_nonoverlap"]) == 8


# ---------------------------------------------------------------- 2. trend
def test_label_consistent_with_quintile_for_pct_25():
    t = trend_from_peer(25.0, universe_size=100)    # Q2 (20-40)
    assert t.label == "yatay"                       # eskiden "asagi" (pct<=30)
    assert trend_from_peer(15.0, 100).label == "aşağı"   # Q1
    assert trend_from_peer(65.0, 100).label == "yatay"   # Q4
    assert trend_from_peer(85.0, 100).label == "yukarı"  # Q5


def test_probability_source_and_wording():
    t = trend_from_peer(90.0, 100)
    assert t.probability_source == "oos_quintile_frequency"
    assert "kalibre" not in t.basis.lower()
    assert "OOS frekans tablosu" in t.basis
    assert trend_from_peer(None, None).probability_source is None


def test_return_std_scales_with_symbol_vol_ratio():
    base = trend_from_peer(90.0, 100).return_std
    assert trend_from_peer(90.0, 100, symbol_vol_ratio=1.0).return_std == pytest.approx(base)
    assert trend_from_peer(90.0, 100, symbol_vol_ratio=2.0).return_std == pytest.approx(2 * base)
    # gecersiz oran -> 1.0'a duser
    assert trend_from_peer(90.0, 100, symbol_vol_ratio=float("nan")).return_std == pytest.approx(base)
    assert trend_from_peer(90.0, 100, symbol_vol_ratio=-1.0).return_std == pytest.approx(base)


# ---------------------------------------------------------------- 3. confidence
def test_confidence_not_high_when_n_eff_below_50():
    c = peer_confidence(1.5, "outperform", n_eff=30)
    assert c.label != "high"
    assert any("n_eff" in w for w in c.warnings)
    assert peer_confidence(1.5, "outperform", n_eff=50).label == "high"
    assert peer_confidence(1.5, "outperform", n_eff=200).label == "high"


def test_confidence_legacy_call_without_n_eff_unchanged():
    assert peer_confidence(1.5, "outperform").label == "high"


# ---------------------------------------------------------------- 5. symbol_id bayragi
def test_global_pooled_model_use_symbol_id_flag():
    lgb = pytest.importorskip("lightgbm")  # noqa: F841
    from src.models.global_pooled_model import GlobalPooledConfig, GlobalPooledModel
    assert inspect.signature(GlobalPooledModel.__init__).parameters["use_symbol_id"].default is True
    rng = np.random.default_rng(0)
    n = 600
    sid = rng.integers(0, 6, n).astype(float)
    x = rng.standard_normal(n)
    y = sid * 0.5 + 0.1 * x                      # sinyal neredeyse sadece symbol_id
    X = np.column_stack([x, sid])
    cfg = GlobalPooledConfig(num_boost_round=40, min_data_in_leaf=5,
                             cat_indices=(1,), symbol_id_index=1)
    on = GlobalPooledModel(cfg).fit(X, y)
    off = GlobalPooledModel(cfg, use_symbol_id=False).fit(X, y)
    assert on.booster.num_feature() == 2
    assert off.booster.num_feature() == 1        # symbol_id dusuruldu
    assert off.predict(X).shape == (n,)
    r_on = float(np.corrcoef(on.predict(X), y)[0, 1])
    r_off = float(np.corrcoef(off.predict(X), y)[0, 1])
    assert r_on > r_off + 0.3                    # symbol_id ezberi kapaninca dusus
