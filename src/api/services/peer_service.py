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

from src.api.schemas.analysis import AnalysisResponse, PeerBlock


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
            )
        except Exception:  # serving katmani API'yi asla bozmasin
            return None

    def enrich(self, response: AnalysisResponse) -> AnalysisResponse:
        block = self.get_peer_block(response.symbol)
        if block is not None:
            response.peer = block
        return response
