# -*- coding: utf-8 -*-
"""
src/api/main.py - ts_forecasting_lab FastAPI Sonuç Servisi (API v2)

Proje kökünden çalıştır:
    python -m src.api.main            (127.0.0.1:8001; AI_CORE_API_PORT ile degisir)

Interaktif docs:
    http://localhost:8001/docs       (Swagger UI)
    http://localhost:8001/redoc      (ReDoc)

Servis YALNIZ gecelik havuz çıktısını (data/serving_pool.db, PeerStore) sunar:

    GET /health                      — servis, serving havuzu ve gecelik kosu sagligi
    GET /symbols                     — son gecelik kosudaki semboller
    GET /analysis/{symbol}           — sembol analizi (API v2; /v1/analysis/{symbol} alias)

Per-sembol yon modeli (Kol-A) emekli oldu: /best-model, /experiments, /metrics,
/leaderboard, /run/* ve /refresh/* endpoint'leri kaldirildi.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict

# Proje kökünü path'e ekle (uvicorn proje kökünden çalıştırılırsa gerekli)
_API_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_API_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:
    raise ImportError(
        "FastAPI yüklü değil. Kurmak için:\n"
        "  pip install fastapi uvicorn"
    ) from exc

from src.api.routers.analysis import router as analysis_router
from src.api.runtime_config import get_cors_settings
from src.api.services.rate_limit import (
    get_default_limiter,
    rate_limit_middleware_factory,
)
from src.api.services.serving_health import nightly_run_status, serving_pool_status
from src.api.services.symbols_service import list_serving_symbols

logger = logging.getLogger(__name__)

API_VERSION = "2"

app = FastAPI(
    title="ts_forecasting_lab API",
    description=(
        "BIST gecelik havuz (pooled) model çıktısına erişim servisi (API v2).\n\n"
        "Yön/göreli sinyal Kol-B pooled modelinden, risk ve fiyat aralığı risk "
        "modelinden gelir; veri `data/serving_pool.db` içindeki son gecelik "
        "koşudan okunur. İstek sırasında model eğitilmez."
    ),
    version="2.0.0",
    contact={"name": "ts_forecasting_lab"},
)

# CORS — aynı makinedeki başka servisler (React dashboard, Merge_PortfoySim vb.)
_CORS_SETTINGS = get_cors_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_SETTINGS.allow_origins,
    allow_origin_regex=_CORS_SETTINGS.allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sprint 9 A9.3 — IP rate limit middleware (env: AI_CORE_RATE_LIMIT_PER_MINUTE).
_RATE_LIMITER = get_default_limiter()
_RATE_MW = rate_limit_middleware_factory()
if _RATE_LIMITER.enabled() and _RATE_MW is not None:
    app.add_middleware(_RATE_MW)

app.include_router(analysis_router)


@app.get("/health", tags=["Sistem"])
def health_check() -> Dict[str, Any]:
    """Servis sağlık kontrolü: serving havuzu tazeliği ve son gecelik koşu durumu."""
    pool = serving_pool_status()
    nightly = nightly_run_status()
    risk = nightly.get("risk") if isinstance(nightly.get("risk"), dict) else None
    risk_failed = risk is not None and risk.get("status") == "failed"
    healthy = (
        bool(pool.get("available"))
        and not pool.get("stale")
        and not nightly.get("interrupted")
        and not risk_failed  # risk adimi coktuyse fiyat bandi bos: servis "degraded"
    )
    return {
        "status": "ok" if healthy else "degraded",
        "api_version": API_VERSION,
        "cors": {
            "mode": _CORS_SETTINGS.mode,
            "allow_origins": _CORS_SETTINGS.allow_origins,
            "allow_origin_regex": _CORS_SETTINGS.allow_origin_regex,
        },
        "registered_symbols": int(pool.get("n_symbols", 0)) if pool.get("available") else 0,
        "serving_pool": pool,
        "nightly_run": nightly,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/symbols", tags=["Hisseler"])
def list_symbols() -> Dict[str, Any]:
    """Son gecelik havuz kosusundaki hisse kodlarini listeler."""
    try:
        symbols = list_serving_symbols()
    except (sqlite3.Error, OSError):
        logger.exception("symbols query failed (serving DB)")
        raise HTTPException(status_code=503, detail="Serving database unavailable")
    except Exception:  # noqa: BLE001 - istemciye genel 500; ayrinti yalniz sunucu logunda
        logger.exception("symbols query failed")
        raise HTTPException(status_code=500, detail="Internal server error")
    return {"count": len(symbols), "symbols": symbols}


def _bind_address() -> tuple[str, int]:
    """Varsayilan YALNIZ loopback (127.0.0.1); disariya acmak icin AI_CORE_API_HOST/PORT."""
    host = os.getenv("AI_CORE_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.getenv("AI_CORE_API_PORT", "8001"))
    except ValueError:
        logger.warning("AI_CORE_API_PORT gecersiz; 8001 kullaniliyor")
        port = 8001
    return host, port


if __name__ == "__main__":
    try:
        import uvicorn
        _host, _port = _bind_address()
        uvicorn.run(
            "src.api.main:app",
            host=_host,
            port=_port,
            reload=True,
            reload_dirs=[_PROJECT_ROOT],
        )
    except ImportError:
        print("uvicorn yüklü değil: pip install uvicorn")
