# -*- coding: utf-8 -*-
"""PeerStore (isolated serving DB) tests — E2 Faz 5."""

import os

import pandas as pd

from src.serving.peer_store import GlobalRunMeta, PeerStore


def _store(tmp_path) -> PeerStore:
    return PeerStore(os.path.join(str(tmp_path), "serving.db"))


def _scored():
    return pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"],
        "as_of_date": ["2026-06-03"] * 3,
        "peer_score": [0.8, 0.0, -0.8],
        "peer_percentile": [90.0, 50.0, 10.0],
        "peer_label": ["outperform", "inline", "underperform"],
        "raw_pred": [0.02, 0.0, -0.02],
        "universe_size": [3, 3, 3],
        "segment_liq": ["Q1", "Q3", "Q5"],
        "segment_vol": ["Q5", "Q3", "Q1"],
        "segment_sector": ["Industrials", "Financials", "Technology"],
        "segment_icir": [1.35, 0.66, 0.30],
        "confidence_label": ["high", "medium", "low"],
        "confidence_reasons": [["guclu"], ["orta"], ["zayif"]],
        "confidence_warnings": [[], [], []],
        "trend_label": ["yukarı", "yatay", "aşağı"],
        "trend_prob_up": [0.541, 0.509, 0.435],
        "trend_expected_return": [0.0090, 0.0050, -0.0066],
        "xai_top_features": [
            {"method": "shap_tree", "approximate": False, "caveat": "",
             "top_positive": [{"feature_name": "RSI_14_csr", "contribution": 0.3}],
             "top_negative": []},
            None,
            {"method": "shap_tree", "approximate": False, "caveat": "x",
             "top_positive": [], "top_negative": [
                 {"feature_name": "vol", "contribution": -0.2}]},
        ],
    })


def test_insert_run_and_latest(tmp_path):
    s = _store(tmp_path)
    rid = s.insert_run(GlobalRunMeta(
        model_name="GlobalPooledModel", as_of_date="2026-06-03",
        n_symbols=589, n_rows=1228434, horizon=5,
        ic_mean=0.0988, icir=1.55, pct_ic_positive=0.934,
        config={"boost": 400}))
    assert rid >= 1
    run = s.latest_run()
    assert run["model_name"] == "GlobalPooledModel"
    assert run["n_symbols"] == 589
    assert abs(run["icir"] - 1.55) < 1e-9


def test_insert_and_read_peer_scores(tmp_path):
    s = _store(tmp_path)
    rid = s.insert_run(GlobalRunMeta(model_name="m", as_of_date="2026-06-03"))
    n = s.insert_peer_scores(rid, _scored())
    assert n == 3
    aaa = s.get_peer_score("AAA")
    assert aaa["peer_label"] == "outperform"
    assert aaa["confidence_label"] == "high"
    assert aaa["confidence_reasons"] == '["guclu"]'  # JSON serialized
    assert s.get_peer_score("CCC", rid)["segment_liq"] == "Q5"


def test_trend_columns_roundtrip(tmp_path):
    s = _store(tmp_path)
    rid = s.insert_run(GlobalRunMeta(model_name="m", as_of_date="d"))
    s.insert_peer_scores(rid, _scored())
    aaa = s.get_peer_score("AAA")
    assert aaa["trend_label"] == "yukarı"
    assert abs(aaa["trend_prob_up"] - 0.541) < 1e-9
    assert abs(aaa["trend_expected_return"] - 0.0090) < 1e-9


def test_xai_top_features_roundtrip(tmp_path):
    """xai_top_features dict -> JSON yazilir, string olarak geri okunur."""
    import json
    s = _store(tmp_path)
    rid = s.insert_run(GlobalRunMeta(model_name="m", as_of_date="d"))
    s.insert_peer_scores(rid, _scored())
    aaa = json.loads(s.get_peer_score("AAA")["xai_top_features"])
    assert aaa["method"] == "shap_tree"
    assert aaa["top_positive"][0]["feature_name"] == "RSI_14_csr"
    # None -> NULL kalir
    assert s.get_peer_score("BBB")["xai_top_features"] is None


def test_migration_adds_trend_cols_to_old_db(tmp_path):
    """Eski sema (trend kolonsuz) -> PeerStore acilista ALTER ile ekler."""
    import sqlite3
    db = os.path.join(str(tmp_path), "old.db")
    old_schema = """
    CREATE TABLE peer_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, symbol TEXT,
        as_of_date TEXT, peer_score REAL, peer_percentile REAL, peer_label TEXT,
        raw_pred REAL, universe_size INTEGER, segment_liq TEXT, segment_vol TEXT,
        segment_sector TEXT, segment_icir REAL, confidence_label TEXT,
        confidence_reasons TEXT, confidence_warnings TEXT, UNIQUE(run_id, symbol));
    """
    with sqlite3.connect(db) as conn:
        conn.executescript(old_schema)
    cols = {r["name"] for r in PeerStore(db)._connect().execute(
        "PRAGMA table_info(peer_scores)")}
    assert {"trend_label", "trend_prob_up", "trend_expected_return",
            "xai_top_features"} <= cols


def test_get_run_peer_scores_sorted(tmp_path):
    s = _store(tmp_path)
    rid = s.insert_run(GlobalRunMeta(model_name="m", as_of_date="d"))
    s.insert_peer_scores(rid, _scored())
    df = s.get_run_peer_scores(rid)
    assert list(df["symbol"]) == ["AAA", "BBB", "CCC"]  # percentile desc


def test_upsert_replaces_same_run_symbol(tmp_path):
    s = _store(tmp_path)
    rid = s.insert_run(GlobalRunMeta(model_name="m", as_of_date="d"))
    s.insert_peer_scores(rid, _scored())
    upd = _scored()
    upd.loc[upd.symbol == "AAA", "peer_label"] = "inline"
    s.insert_peer_scores(rid, upd)
    assert s.get_peer_score("AAA", rid)["peer_label"] == "inline"
    # hala 3 satir (replace, duplicate degil)
    assert len(s.get_run_peer_scores(rid)) == 3


def test_missing_symbol_returns_none(tmp_path):
    s = _store(tmp_path)
    rid = s.insert_run(GlobalRunMeta(model_name="m", as_of_date="d"))
    s.insert_peer_scores(rid, _scored())
    assert s.get_peer_score("ZZZ") is None


def test_empty_store_safe(tmp_path):
    s = _store(tmp_path)
    assert s.latest_run() is None
    assert s.get_peer_score("AAA") is None
