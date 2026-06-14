# -*- coding: utf-8 -*-
"""Peer enrichment service tests — E2 Faz 5 API integration."""

import os

import pandas as pd

from src.api.schemas.analysis import AnalysisResponse
from src.api.services.peer_service import PeerEnrichmentService
from src.serving.peer_store import GlobalRunMeta, PeerStore


def _seed(tmp_path) -> str:
    db = os.path.join(str(tmp_path), "serving.db")
    store = PeerStore(db)
    rid = store.insert_run(GlobalRunMeta(
        model_name="GlobalPooledModel", as_of_date="2026-06-03",
        icir=1.55, ic_mean=0.099, pct_ic_positive=0.93))
    store.insert_peer_scores(rid, pd.DataFrame({
        "symbol": ["TUPRS"], "as_of_date": ["2026-06-03"],
        "peer_score": [0.62], "peer_percentile": [81.0],
        "peer_label": ["outperform"], "raw_pred": [0.01], "universe_size": [589],
        "segment_liq": ["Q1"], "segment_vol": ["Q5"], "segment_sector": ["Industrials"],
        "segment_icir": [1.35], "confidence_label": ["high"],
        "confidence_reasons": [["Segment sinyali guclu"]], "confidence_warnings": [[]],
        "trend_label": ["yukarı"], "trend_prob_up": [0.541],
        "trend_expected_return": [0.0090],
        "kolb_price_p50": [100.90], "kolb_price_low": [98.10], "kolb_price_high": [103.78],
        "kolb_horizon_days": [5], "kolb_band_level": [0.8],
        "xai_top_features": [{
            "method": "shap_tree", "approximate": False,
            "caveat": "Bu açıklama yalnızca LightGBM bacağı temellidir.",
            "top_positive": [{
                "feature_name": "RSI_14_csr", "human_label": "RSI ... akran",
                "importance": 0.3, "direction": "yukarı", "feature_group": "cross_sectional",
                "reason": "...", "method": "shap_tree", "contribution": 0.3,
                "approximate": False}],
            "top_negative": [{
                "feature_name": "vol", "human_label": "oynaklık",
                "importance": 0.2, "direction": "aşağı", "feature_group": "volatility",
                "reason": "...", "method": "shap_tree", "contribution": -0.2,
                "approximate": False}],
            "group_summaries": [{
                "feature_group": "cross_sectional",
                "group_label": "Akran goreli sinyaller",
                "total_importance": 0.3,
                "net_contribution": 0.3,
                "direction": "yukari",
                "top_features": ["RSI_14_csr"],
                "reason": "Akran goreli sinyaller akran siralamasini yukari yonde etkileyen faktorler arasinda.",
                "approximate_ratio": 0.0,
            }],
        }],
    }))
    return db


def _seed_no_xai(tmp_path) -> str:
    db = os.path.join(str(tmp_path), "noxai.db")
    store = PeerStore(db)
    rid = store.insert_run(GlobalRunMeta(model_name="m", as_of_date="d", icir=1.5))
    store.insert_peer_scores(rid, pd.DataFrame({
        "symbol": ["TUPRS"], "as_of_date": ["d"], "peer_score": [0.1],
        "peer_percentile": [55.0], "peer_label": ["inline"], "raw_pred": [0.0],
        "universe_size": [10], "confidence_label": ["low"],
        "confidence_reasons": [[]], "confidence_warnings": [[]],
    }))
    return db


def _resp(symbol="TUPRS") -> AnalysisResponse:
    return AnalysisResponse(symbol=symbol, generated_at="2026-06-03T00:00:00")


def test_enrich_attaches_peer_block(tmp_path):
    svc = PeerEnrichmentService(_seed(tmp_path))
    out = svc.enrich(_resp("TUPRS"))
    assert out.peer is not None and out.peer.available
    assert out.peer.peer_label == "outperform"
    assert out.peer.confidence_label == "high"
    assert out.peer.segment_icir == 1.35
    assert out.peer.icir_overall == 1.55
    assert out.peer.confidence_reasons == ["Segment sinyali guclu"]
    assert out.peer.trend_label == "yukarı"
    assert abs(out.peer.trend_prob_up - 0.541) < 1e-9
    assert abs(out.peer.trend_expected_return - 0.0090) < 1e-9
    assert abs(out.peer.kolb_price_p50 - 100.90) < 1e-9
    assert abs(out.peer.kolb_price_low - 98.10) < 1e-9
    assert abs(out.peer.kolb_price_high - 103.78) < 1e-9
    assert out.peer.kolb_horizon_days == 5
    assert abs(out.peer.kolb_band_level - 0.8) < 1e-9


def test_enrich_attaches_peer_xai(tmp_path):
    svc = PeerEnrichmentService(_seed(tmp_path))
    peer = svc.enrich(_resp("TUPRS")).peer
    assert peer.xai_available
    assert peer.xai_method == "shap_tree"
    assert "LightGBM" in peer.xai_caveat
    assert len(peer.xai_top_positive) == 1
    assert peer.xai_top_positive[0].feature_name == "RSI_14_csr"
    assert peer.xai_top_positive[0].direction == "yukarı"
    assert len(peer.xai_top_negative) == 1
    assert peer.xai_top_negative[0].contribution == -0.2
    assert len(peer.xai_group_summaries) == 1
    assert peer.xai_group_summaries[0].feature_group == "cross_sectional"
    assert "akran siralamasini yukari" in peer.xai_group_summaries[0].reason


def test_enrich_no_xai_graceful(tmp_path):
    """xai_top_features NULL -> xai_available False, blok yine doner."""
    svc = PeerEnrichmentService(_seed_no_xai(tmp_path))
    peer = svc.enrich(_resp("TUPRS")).peer
    assert peer is not None and peer.available
    assert peer.xai_available is False
    assert peer.xai_top_positive == [] and peer.xai_top_negative == []
    assert peer.xai_group_summaries == []


def test_enrich_case_insensitive(tmp_path):
    svc = PeerEnrichmentService(_seed(tmp_path))
    assert svc.enrich(_resp("tuprs")).peer.available


def test_unknown_symbol_no_block(tmp_path):
    svc = PeerEnrichmentService(_seed(tmp_path))
    out = svc.enrich(_resp("ZZZZ"))
    assert out.peer is None


def test_missing_db_is_noop(tmp_path):
    svc = PeerEnrichmentService(os.path.join(str(tmp_path), "nope.db"))
    out = svc.enrich(_resp("TUPRS"))
    assert out.peer is None  # mevcut yanit bozulmaz


def test_existing_response_untouched(tmp_path):
    """peer enrichment mevcut alanlari degistirmez (additive)."""
    svc = PeerEnrichmentService(_seed(tmp_path))
    r = _resp("TUPRS")
    r.analysis_status = "ok"
    out = svc.enrich(r)
    assert out.analysis_status == "ok"
    assert out.confidence.label == "low"  # mevcut confidence dokunulmaz
