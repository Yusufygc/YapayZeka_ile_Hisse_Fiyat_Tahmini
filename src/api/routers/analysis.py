# -*- coding: utf-8 -*-
"""GET /analysis/{symbol} router (API v2: yalniz gecelik havuz ciktisi)."""
from __future__ import annotations

import logging
import os

try:
    from fastapi import APIRouter, HTTPException
except ImportError:
    raise ImportError("FastAPI yüklü değil: pip install fastapi")

from src.api.observability import log_event
from src.api.schemas.analysis import AnalysisResponse
from src.api.services.analysis_service import AnalysisService
from src.api.services.response_cache import compute_freshness_fingerprint, get_default_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analiz"])

_service = AnalysisService()
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_cache = get_default_cache()


def _freshness_fingerprint(symbol: str):
    """Cache anahtarina giren tazelik parmak izi (serving run id + piyasa verisi)."""
    return compute_freshness_fingerprint(
        symbol, serving_db_path=_service.serving_db_path, data_dir=_service.data_dir
    )


@router.get("/v1/analysis/{symbol}", response_model=AnalysisResponse)
def get_analysis_v1(symbol: str) -> AnalysisResponse:
    """/analysis ile ayni sozlesme; versiyonlu alias (api_version=2)."""
    return get_analysis(symbol)


@router.get("/analysis/{symbol}", response_model=AnalysisResponse)
def get_analysis(symbol: str) -> AnalysisResponse:
    """Belirtilen hisse icin gecelik havuzdan (Kol-B + risk) analiz payload'u doner.

    - **symbol**: BIST hisse kodu (buyuk/kucuk harf duyarsiz, or. ``TUPRS``)

    ``analysis_status`` degerleri:
    - ``ok``: Sembol son kosuda var, veri guncel, guven medium/high.
    - ``stale_data``: Piyasa verisi veya gecelik kosu bayat.
    - ``low_confidence``: Sonuc var ama guven etiketi ``low``.
    - ``no_data``: Sembol son gecelik kosuda yok.

    Kirici degisiklik (api_version=2): ``model``, ``forecast``, ``performance``,
    ``xai`` ve refresh-job alanlari kaldirildi.
    """
    try:
        normalized = str(symbol).upper().strip()
        # Parmak izi build'den ONCE alinir: build sirasinda kaynak degisirse bir
        # sonraki istek miss olur (bayat yanit servis edilmez).
        fingerprint = _freshness_fingerprint(normalized)
        cached = _cache.get(normalized, fingerprint)
        if cached is not None:
            return cached

        result = _service.build(symbol)
        peer = result.peer
        log_event(
            _PROJECT_ROOT,
            "analysis_response",
            symbol=result.symbol,
            analysis_status=result.analysis_status,
            confidence=result.confidence.label,
            run_id=None if result.serving_run is None else result.serving_run.run_id,
            risk_method=None if peer is None else getattr(peer, "risk_method", None),
        )
        _cache.set(normalized, result, fingerprint)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - istemciye 500 + log
        logger.exception("analysis build failed for %s", symbol)
        raise HTTPException(status_code=500, detail=str(exc))
