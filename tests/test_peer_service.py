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
