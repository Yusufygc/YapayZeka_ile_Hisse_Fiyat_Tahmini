# -*- coding: utf-8 -*-
"""GET /analysis/{symbol} router."""
from __future__ import annotations

import os

try:
    from fastapi import APIRouter, HTTPException
except ImportError:
    raise ImportError("FastAPI yüklü değil: pip install fastapi")

from src.api.schemas.analysis import AnalysisResponse
from src.api.observability import log_event
from src.api.services.analysis_service import AnalysisService

router = APIRouter(tags=["Analiz"])

_service = AnalysisService()
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@router.get("/v1/analysis/{symbol}", response_model=AnalysisResponse)
def get_analysis_v1(symbol: str) -> AnalysisResponse:
    """Sprint 8 A8.6 — /analysis ile ayni sozlesme; gelecek breaking change
    icin versiyonlu alias. Davranis birebir ayni."""
    return get_analysis(symbol)


@router.get("/analysis/{symbol}", response_model=AnalysisResponse)
def get_analysis(symbol: str) -> AnalysisResponse:
    """Belirtilen hisse için tek seferlik analiz payload'u döner.

    Yanıt şeması: yeniTasarim/04 API sözleşmesiyle uyumludur.

    - **symbol**: BIST hisse kodu (büyük/küçük harf duyarsız, ör. ``TUPRS``)

    ``analysis_status`` değerleri:
    - ``ok``: Analiz geçerli ve güncel.
    - ``stale_data``: Veri eski; tahmin dikkatli yorumlanmalı.
    - ``no_model``: Hisse için kayıtlı eligible model yok.
    - ``no_forecast``: Model var ama forecast üretilmemiş.
    - ``low_confidence``: Sonuç var ama güven etiketi ``low``.
    - ``xai_unavailable``: Forecast var ama XAI çıktısı yok.
    - ``error``: Beklenmeyen hata.
    """
    try:
        result = _service.build(symbol)
        source = result.forecast_source
        log_event(
            _PROJECT_ROOT,
            "analysis_response",
            symbol=result.symbol,
            analysis_status=result.analysis_status,
            refresh_status=result.refresh_status,
            refresh_reason=result.refresh_reason,
            forecast_source=None
            if source is None
            else {
                "model_name": source.model_name,
                "source_experiment_id": source.source_experiment_id,
                "last_observed_date": source.last_observed_date,
            },
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
