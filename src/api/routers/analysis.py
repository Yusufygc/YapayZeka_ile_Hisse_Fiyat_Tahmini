# -*- coding: utf-8 -*-
"""GET /analysis/{symbol} router."""
from __future__ import annotations

from typing import Any, Dict

try:
    from fastapi import APIRouter, HTTPException
except ImportError:
    raise ImportError("FastAPI yüklü değil: pip install fastapi")

from src.api.schemas.analysis import AnalysisResponse
from src.api.services.analysis_service import AnalysisService

router = APIRouter(tags=["Analiz"])

_service = AnalysisService()


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
        return _service.build(symbol)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
