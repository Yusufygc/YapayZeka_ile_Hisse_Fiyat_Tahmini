# -*- coding: utf-8 -*-
"""
src/api/main.py - ts_forecasting_lab FastAPI Sonuç Servisi (Faz 5.4)

Proje kökünden çalıştır:
    uvicorn src.api.main:app --reload --port 8000

Interaktif docs:
    http://localhost:8000/docs       (Swagger UI)
    http://localhost:8000/redoc      (ReDoc)

Endpoints:
    GET /health                      — servis sağlık kontrolü
    GET /best-model/{symbol}         — hisse için en iyi model
    GET /experiments/{symbol}        — deney geçmişi
    GET /metrics/{symbol}            — model karşılaştırma tablosu
    GET /leaderboard                 — tüm hisseler lider tablosu
    GET /symbols                     — kayıtlı tüm hisse kodları
    POST /run/{symbol}               — tek hisse pipeline'ını tetikle (arka planda)
    GET /run/status/{job_id}         — tetiklenen işin durumu
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Proje kökünü path'e ekle (uvicorn proje kökünden çalıştırılırsa gerekli)
_API_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_API_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:
    raise ImportError(
        "FastAPI yüklü değil. Kurmak için:\n"
        "  pip install fastapi uvicorn\n"
        "ya da dl_env ortamında:\n"
        "  conda activate dl_env && pip install fastapi uvicorn"
    ) from exc

from src.database.stock_model_db import StockModelDB
from src.api.routers.analysis import router as analysis_router
from src.api.runtime_config import get_cors_settings
from src.api.services.pipeline_jobs import (
    RunRequest,
    get_job,
    known_job_ids,
    start_pipeline_job,
)
from src.api.services.rate_limit import (
    get_default_limiter,
    rate_limit_middleware_factory,
)

# ─────────────────────────────────────────────────────────────────────────────
# App ve DB başlatma
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ts_forecasting_lab API",
    description=(
        "BIST hisse tahmin pipeline'ının sonuçlarına erişim servisi.\n\n"
        "SQLite veritabanındaki model deneyleri, metrikler ve lider tablosunu "
        "HTTP üzerinden sunar. Merge_PortfoySim gibi dış uygulamalar bu API "
        "aracılığıyla en iyi model seçimini ve metriklerini sorgulayabilir."
    ),
    version="1.0.0",
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

_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "stock_models.db")

def _get_db() -> StockModelDB:
    return StockModelDB(_DB_PATH)


def _parse_payload_json(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job:
        return None
    raw = job.get("payload_json")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return None
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint'ler  (RunRequest şeması + job tracker: src/api/services/pipeline_jobs.py)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistem"])
def health_check() -> Dict[str, Any]:
    """
    Servis sağlık kontrolü.
    DB erişimi ve kayıtlı hisse sayısını döner.
    """
    try:
        db = _get_db()
        schema = db.get_schema_status()
        latest_refresh = db.get_latest_refresh_job()
        db_ok = bool(schema.get("ok"))
        total_symbols = len(db.get_leaderboard(top_n=9999))
    except Exception as exc:
        db_ok = False
        total_symbols = 0
        schema = {"ok": False, "missing_tables": [], "table_counts": {}, "error": str(exc)}
        latest_refresh = None

    return {
        "status": "ok" if db_ok else "degraded",
        "db_path": _DB_PATH,
        "db_accessible": db_ok,
        "schema": schema,
        "latest_refresh_job": None
        if latest_refresh is None
        else {
            **latest_refresh,
            "payload": _parse_payload_json(latest_refresh),
        },
        "cors": {
            "mode": _CORS_SETTINGS.mode,
            "allow_origins": _CORS_SETTINGS.allow_origins,
            "allow_origin_regex": _CORS_SETTINGS.allow_origin_regex,
        },
        "runtime": {
            "project_root": _PROJECT_ROOT,
            "python": sys.executable,
        },
        "registered_symbols": total_symbols,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/symbols", tags=["Hisseler"])
def list_symbols() -> Dict[str, Any]:
    """
    Veritabanında kayıtlı tüm hisse kodlarını listeler.
    """
    try:
        db = _get_db()
        leaders = db.get_leaderboard(top_n=9999)
        symbols = [r["stock_symbol"] for r in leaders]
        return {"count": len(symbols), "symbols": symbols}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/best-model/{symbol}", tags=["Modeller"])
def get_best_model(symbol: str) -> Dict[str, Any]:
    """
    Belirtilen hisse için en iyi modelin tüm bilgilerini döner.

    - **symbol**: Hisse kodu (büyük/küçük harf duyarsız, ör. `TUPRS`)

    Dönen model; composite_score'a göre tüm denemeler içinden seçilir.
    """
    symbol = symbol.upper()
    try:
        db = _get_db()
        result = db.get_best_model(symbol)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} için kayıtlı model bulunamadı. "
                   "Önce pipeline'ı çalıştırın: python -m src.cli.interactive",
        )
    return result


@app.get("/experiments/{symbol}", tags=["Deneyler"])
def get_experiments(
    symbol: str,
    model_name: Optional[str] = Query(None, description="Model adına göre filtrele"),
    limit: int = Query(20, ge=1, le=500, description="Maksimum sonuç sayısı"),
) -> Dict[str, Any]:
    """
    Belirtilen hisse için deney geçmişini döner.

    - **symbol**: Hisse kodu
    - **model_name**: Opsiyonel — belirli bir modeli filtrele (ör. `XGBoost`)
    - **limit**: Döndürülecek maksimum satır sayısı (varsayılan: 20)
    """
    symbol = symbol.upper()
    try:
        db = _get_db()
        rows = db.get_experiments(stock_symbol=symbol, model_name=model_name, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "symbol": symbol,
        "count": len(rows),
        "model_filter": model_name,
        "experiments": rows,
    }


@app.get("/metrics/{symbol}", tags=["Modeller"])
def get_model_comparison(symbol: str) -> Dict[str, Any]:
    """
    Belirtilen hisse için tüm modellerin ortalama metrik karşılaştırmasını döner.

    Composite score, directional accuracy, Sharpe, MAE, RMSE gibi metrikleri
    model bazında gruplayarak özetler.
    """
    symbol = symbol.upper()
    try:
        db = _get_db()
        rows = db.get_model_comparison(symbol)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} için metrik verisi bulunamadı.",
        )
    return {
        "symbol": symbol,
        "model_count": len(rows),
        "comparison": rows,
    }


@app.get("/leaderboard", tags=["Hisseler"])
def get_leaderboard(
    top_n: int = Query(20, ge=1, le=500, description="Kaç hisse döndürülsün"),
) -> Dict[str, Any]:
    """
    Tüm hisseler arasında composite_score'a göre lider tablosunu döner.

    Her satır bir hissenin en iyi modelini temsil eder.
    Portföy seçimi ve önceliklendirme için kullanılabilir.
    """
    try:
        db = _get_db()
        rows = db.get_leaderboard(top_n=top_n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "top_n": top_n,
        "count": len(rows),
        "leaderboard": rows,
    }


@app.post("/run/{symbol}", tags=["Pipeline"])
def trigger_pipeline(
    symbol: str,
    request: RunRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Belirtilen hisse için pipeline'ı arka planda başlatır.

    **Not:** Bu endpoint pipeline'ı tetikler ve hemen bir `job_id` döner.
    İşin durumunu `GET /run/status/{job_id}` ile takip edebilirsiniz.

    Paralel çalıştırmak için `python -m src.cli.batch` kullanın.
    """
    symbol = symbol.upper()
    job_id = start_pipeline_job(symbol, request, _PROJECT_ROOT, background_tasks)
    return {
        "job_id": job_id,
        "symbol": symbol,
        "status": "queued",
        "message": f"Pipeline kuyruğa alındı. Durum: GET /run/status/{job_id}",
    }


@app.get("/run/status/{job_id}", tags=["Pipeline"])
def get_run_status(job_id: str) -> Dict[str, Any]:
    """
    Tetiklenen bir pipeline işinin durumunu döner.

    - **job_id**: `/run/{symbol}` endpoint'inden dönen job_id
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' bulunamadı. Geçerli job listesi: {known_job_ids()}",
        )
    return job


@app.get("/refresh/status/{job_id}", tags=["Analiz"])
def get_refresh_status(job_id: str) -> Dict[str, Any]:
    """Return analysis auto-refresh job status."""
    try:
        db = _get_db()
        job = db.get_refresh_job(job_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Refresh job not found: {job_id}")
    return {
        **job,
        "payload": _parse_payload_json(job),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Doğrudan çalıştırma (geliştirme için)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[_PROJECT_ROOT],
        )
    except ImportError:
        print("uvicorn yüklü değil: pip install uvicorn")

