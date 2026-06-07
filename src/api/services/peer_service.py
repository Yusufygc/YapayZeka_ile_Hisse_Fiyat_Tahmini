# -*- coding: utf-8 -*-
"""E2 Faz 5 — peer (cross-sectional) enrichment for /analysis/{symbol}.

Nightly batch'in yazdigi PeerStore'dan sembolun peer skorunu okur ve
AnalysisResponse'a additive `peer` blogu ekler. PeerStore yok / sembol yok /
herhangi bir hata -> sessizce no-op (mevcut yanit korunur, asla bozulmaz).
"""
from __future__ import annotations

import json
import os
from typing import Optional

from src.api.schemas.analysis import AnalysisResponse, PeerBlock, XaiFactorItem


def _default_db_path() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "data", "serving_pool.db")


def _json_list(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    try:
        parsed = json.loads(v)
        return [str(x) for x in parsed] if isinstance(parsed, list) else [str(parsed)]
    except (ValueError, TypeError):
        return [str(v)]


def _xai_factor(raw: dict) -> XaiFactorItem:
    """XAI JSON sozlugunu XaiFactorItem'a tasir (geriye uyumlu)."""
    return XaiFactorItem(
        feature_name=str(raw.get("feature_name", "")),
        human_label=str(raw.get("human_label", "")),
        importance=float(raw.get("importance", 0) or 0),
        direction=str(raw.get("direction", "")),
        feature_group=raw.get("feature_group"),
        reason=raw.get("reason"),
        method=raw.get("method"),
        contribution=raw.get("contribution"),
        approximate=raw.get("approximate"),
    )


def _parse_peer_xai(v) -> Optional[dict]:
    """peer_scores.xai_top_features JSON'unu parse et. Yok/bozuk -> None."""
    if not v:
        return None
    obj = v
    if isinstance(v, str):
        try:
            obj = json.loads(v)
        except (ValueError, TypeError):
            return None
    if not isinstance(obj, dict):
        return None
    pos = [_xai_factor(f) for f in obj.get("top_positive", []) if isinstance(f, dict)]
    neg = [_xai_factor(f) for f in obj.get("top_negative", []) if isinstance(f, dict)]
    if not pos and not neg:
        return None
    return {
        "method": str(obj.get("method", "")),
        "caveat": str(obj.get("caveat", "")),
        "top_positive": pos,
        "top_negative": neg,
    }


class PeerEnrichmentService:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or _default_db_path()

    def get_peer_block(self, symbol: str) -> Optional[PeerBlock]:
        if not os.path.exists(self._db_path):
            return None
        try:
            from src.serving.peer_store import PeerStore

            store = PeerStore(self._db_path)
            row = store.get_peer_score(str(symbol).upper().strip())
            if row is None:
                return None
            run = store.latest_run() or {}
            xai = _parse_peer_xai(row.get("xai_top_features"))
            return PeerBlock(
                available=True,
                as_of_date=row.get("as_of_date"),
                peer_score=row.get("peer_score"),
                peer_percentile=row.get("peer_percentile"),
                peer_label=row.get("peer_label"),
                universe_size=row.get("universe_size"),
                segment_liq=row.get("segment_liq"),
                segment_vol=row.get("segment_vol"),
                segment_sector=row.get("segment_sector"),
                segment_icir=row.get("segment_icir"),
                confidence_label=row.get("confidence_label"),
                confidence_reasons=_json_list(row.get("confidence_reasons")),
                confidence_warnings=_json_list(row.get("confidence_warnings")),
                model_run_id=row.get("run_id"),
                icir_overall=run.get("icir"),
                trend_label=row.get("trend_label"),
                trend_prob_up=row.get("trend_prob_up"),
                trend_expected_return=row.get("trend_expected_return"),
                xai_available=xai is not None,
                xai_method="" if xai is None else xai["method"],
                xai_caveat="" if xai is None else xai["caveat"],
                xai_top_positive=[] if xai is None else xai["top_positive"],
                xai_top_negative=[] if xai is None else xai["top_negative"],
            )
        except Exception:  # serving katmani API'yi asla bozmasin
            return None

    def enrich(self, response: AnalysisResponse) -> AnalysisResponse:
        block = self.get_peer_block(response.symbol)
        if block is not None:
            response.peer = block
        return response
