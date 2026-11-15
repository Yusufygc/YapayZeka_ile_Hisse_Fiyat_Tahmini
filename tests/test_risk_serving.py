# -*- coding: utf-8 -*-
"""Risk model serilestirme, PeerStore migrasyonu, fiyat bandi, API esleme (sentetik)."""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from risk_synth import make_panel
from src.api.schemas.analysis import AnalysisResponse
from src.api.services.peer_service import PeerEnrichmentService
from src.risk.features import RISK_FEATURE_COLS, build_risk_frame, labeled_rows
from src.risk.model import PooledVolModel, VolModelConfig
from src.risk.serving import RISK_COLUMNS, risk_levels_from_sigma
from src.serving.nightly_scoring import attach_risk_and_band
from src.serving.peer_store import GlobalRunMeta, PeerStore


# ------------------------------------------------------------------ model
def test_model_fit_predict_roundtrip_with_metadata(tmp_path):
    lab = labeled_rows(build_risk_frame(make_panel(n_sym=12, n_days=300, seed=5)))
    m = PooledVolModel(VolModelConfig(alpha=1.0)).fit(lab)
    m.set_calibration(c=1.7, z=1.15, z_naive=1.3)
    pred = m.predict_logvol(lab)
    assert np.corrcoef(pred, lab["y"])[0, 1] > 0.5
    # sigma_h = sqrt(h) * exp(pred) * sqrt(c)
    assert np.allclose(m.sigma_h(lab), np.sqrt(5) * np.exp(pred) * np.sqrt(1.7))
    path = os.path.join(str(tmp_path), "m", "risk.joblib")
    m.save(path)
    m2 = PooledVolModel.load(path)
    assert np.allclose(m2.predict_logvol(lab), pred)
    md = m2.metadata()
    assert md["feature_cols"] == list(RISK_FEATURE_COLS) and md["horizon"] == 5
    assert md["c"] == 1.7 and md["z"] == 1.15 and md["level"] == 0.8
    assert md["fit_date_min"] <= md["fit_date_max"] and md["n_train"] == len(lab)
    assert md["fitted_at"].endswith("+00:00")                      # tz-aware


def test_model_rejects_nan_and_unfitted():
    lab = labeled_rows(build_risk_frame(make_panel(n_sym=12, n_days=200, seed=5)))
    with pytest.raises(RuntimeError):
        PooledVolModel().predict_logvol(lab)
    bad = lab.copy()
    bad.loc[bad.index[0], "rv_5"] = np.nan
    with pytest.raises(ValueError):
        PooledVolModel().fit(bad)


def test_risk_levels_are_tertiles():
    lv = risk_levels_from_sigma(pd.Series(np.arange(1.0, 10.0)))
    assert lv.value_counts().to_dict() == {"düşük": 3, "orta": 3, "yüksek": 3}
    assert lv.iloc[0] == "düşük" and lv.iloc[-1] == "yüksek"
    nan = risk_levels_from_sigma(pd.Series([1.0, np.nan, 2.0, 3.0]))
    assert nan.iloc[1] is None


# ------------------------------------------------------------- PeerStore
_OLD_SCHEMA = """
CREATE TABLE global_model_runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
  as_of_date TEXT, model_name TEXT, data_snapshot_hash TEXT, n_symbols INTEGER, n_rows INTEGER,
  horizon INTEGER, ic_mean REAL, icir REAL, pct_ic_positive REAL, config_json TEXT);
CREATE TABLE peer_scores (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
  symbol TEXT NOT NULL, as_of_date TEXT, peer_score REAL, peer_percentile REAL, peer_label TEXT,
  raw_pred REAL, universe_size INTEGER, segment_liq TEXT, segment_vol TEXT, segment_sector TEXT,
  segment_icir REAL, confidence_label TEXT, confidence_reasons TEXT, confidence_warnings TEXT,
  UNIQUE(run_id, symbol));
INSERT INTO global_model_runs (created_at, as_of_date, model_name) VALUES ('2026-01-01', 'd', 'old');
INSERT INTO peer_scores (run_id, symbol, as_of_date, peer_score) VALUES (1, 'OLD1', 'd', 0.5);
"""


def test_peer_store_migrates_old_schema_additively(tmp_path):
    db = os.path.join(str(tmp_path), "old.db")
    con = sqlite3.connect(db)
    con.executescript(_OLD_SCHEMA)
    con.commit()
    con.close()
    store = PeerStore(db)                                          # migrasyon
    store = PeerStore(db)                                          # idempotent
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(peer_scores)")}
    assert set(RISK_COLUMNS) <= cols
    old = store.get_peer_score("OLD1", run_id=1)
    assert old["peer_score"] == 0.5 and old["risk_sigma_5d"] is None   # eski satir korunur
    rid = store.insert_run(GlobalRunMeta(model_name="m", as_of_date="d2"))
    store.insert_peer_scores(rid, pd.DataFrame({
        "symbol": ["NEW1"], "risk_sigma_5d": [0.05], "risk_sigma_naive_5d": [0.06],
        "risk_vol_percentile": [70.0], "risk_level": ["yüksek"], "risk_z": [1.1],
        "risk_band_level": [0.8], "risk_horizon_days": [5], "risk_method": ["har_x_ridge"]}))
    row = store.get_peer_score("NEW1", run_id=rid)
    assert row["risk_level"] == "yüksek" and row["risk_horizon_days"] == 5
    assert row["risk_method"] == "har_x_ridge" and row["risk_z"] == pytest.approx(1.1)


# ------------------------------------------------------------------ band
def _merged(**risk):
    base = {"symbol": ["A", "B", "C"], "trend_expected_return": [0.01, 0.0, np.nan]}
    return pd.DataFrame(base), pd.DataFrame({
        "symbol": ["A", "B", "C"], "risk_sigma_5d": risk.get("sigma", [0.05, 0.03, 0.04]),
        "risk_sigma_naive_5d": 0.04, "risk_vol_percentile": 50.0, "risk_level": "orta",
        "risk_z": risk.get("z", [1.2, 1.2, 1.2]), "risk_band_level": 0.8,
        "risk_horizon_days": 5, "risk_method": "har_x_ridge"})


def test_price_band_formula():
    merged, risk = _merged()
    out = attach_risk_and_band(merged, risk, {"A": 100.0, "B": 50.0, "C": 10.0}).set_index("symbol")
    a = out.loc["A"]
    assert a["kolb_price_p50"] == pytest.approx(100 * np.exp(0.01))
    assert a["kolb_price_low"] == pytest.approx(100 * np.exp(0.01 - 1.2 * 0.05))
    assert a["kolb_price_high"] == pytest.approx(100 * np.exp(0.01 + 1.2 * 0.05))
    assert a["kolb_horizon_days"] == 5 and a["kolb_band_level"] == 0.8
    b = out.loc["B"]                                  # exp_ret=0 -> merkez Close
    assert b["kolb_price_p50"] == pytest.approx(50.0)
    assert b["kolb_price_high"] / 50.0 == pytest.approx(50.0 / b["kolb_price_low"])
    c = out.loc["C"]                                  # exp_ret NaN -> bant yok
    assert np.isnan(c["kolb_price_p50"]) and np.isnan(c["kolb_price_low"])


def test_price_band_missing_risk_or_close_is_empty():
    merged, risk = _merged(sigma=[np.nan, 0.03, 0.04])
    out = attach_risk_and_band(merged, risk, {"A": 100.0, "B": np.nan, "C": 10.0}).set_index("symbol")
    assert np.isnan(out.loc["A", "kolb_price_low"]) and not np.isnan(out.loc["A", "kolb_price_p50"])
    assert np.isnan(out.loc["B", "kolb_price_p50"]) and np.isnan(out.loc["B", "kolb_price_low"])
    none = attach_risk_and_band(merged, None, {"A": 100.0}).set_index("symbol")
    assert none["kolb_price_low"].isna().all() and none["risk_sigma_5d"].isna().all()


# ------------------------------------------------------------ API mapping
def test_peer_service_maps_risk_fields_and_validation(tmp_path):
    db = os.path.join(str(tmp_path), "svc.db")
    store = PeerStore(db)
    summary = {"mse_ratio": 0.8, "dm_t": 17.0, "coverage": 0.797, "gate_passed": True,
               "method": "har_x_ridge", "n_oos_rows": 1000}
    rid = store.insert_run(GlobalRunMeta(
        model_name="m", as_of_date="2026-06-01", icir=1.0, config={"risk": summary}))
    store.insert_peer_scores(rid, pd.DataFrame({
        "symbol": ["TUPRS"], "as_of_date": ["2026-06-01"], "peer_score": [0.1],
        "peer_percentile": [60.0], "peer_label": ["inline"], "raw_pred": [0.0],
        "universe_size": [10], "confidence_label": ["low"],
        "confidence_reasons": [[]], "confidence_warnings": [[]],
        "risk_sigma_5d": [0.052], "risk_sigma_naive_5d": [0.061], "risk_vol_percentile": [82.0],
        "risk_level": ["yüksek"], "risk_z": [1.12], "risk_band_level": [0.8],
        "risk_horizon_days": [5], "risk_method": ["har_x_ridge"]}))
    out = PeerEnrichmentService(db).enrich(
        AnalysisResponse(symbol="TUPRS", generated_at="2026-06-01T00:00:00"))
    p = out.peer
    assert p.risk_sigma_5d == pytest.approx(0.052) and p.risk_level == "yüksek"
    assert p.risk_z == pytest.approx(1.12) and p.risk_horizon_days == 5
    assert p.risk_method == "har_x_ridge" and p.risk_vol_percentile == 82.0
    assert p.risk_validation == json.loads(json.dumps(summary))


def test_peer_service_old_run_without_risk(tmp_path):
    db = os.path.join(str(tmp_path), "norisk.db")
    store = PeerStore(db)
    rid = store.insert_run(GlobalRunMeta(model_name="m", as_of_date="d"))
    store.insert_peer_scores(rid, pd.DataFrame({
        "symbol": ["AAA"], "as_of_date": ["d"], "peer_score": [0.1], "peer_percentile": [5.0],
        "peer_label": ["inline"], "raw_pred": [0.0], "universe_size": [10],
        "confidence_label": ["low"], "confidence_reasons": [[]], "confidence_warnings": [[]]}))
    p = PeerEnrichmentService(db).get_peer_block("AAA")
    assert p.available and p.risk_sigma_5d is None and p.risk_validation is None
