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
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Proje kökünü path'e ekle (uvicorn proje kökünden çalıştırılırsa gerekli)
_API_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_API_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as PydanticModel
except ImportError as exc:
    raise ImportError(
        "FastAPI yüklü değil. Kurmak için:\n"
        "  pip install fastapi uvicorn\n"
        "ya da dl_env ortamında:\n"
        "  conda activate dl_env && pip install fastapi uvicorn"
    ) from exc

from src.database.stock_model_db import StockModelDB

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "stock_models.db")

def _get_db() -> StockModelDB:
    return StockModelDB(_DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic şemaları
# ─────────────────────────────────────────────────────────────────────────────

class RunRequest(PydanticModel):
    mode: str = "walk_forward"
    models: Optional[List[str]] = None
    data_dir: str = "data"


class RunStatus(PydanticModel):
    job_id: str
    symbol: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# In-memory job tracker (production'da Redis kullanılır)
# ─────────────────────────────────────────────────────────────────────────────

_jobs: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint'ler
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistem"])
def health_check() -> Dict[str, Any]:
    """
    Servis sağlık kontrolü.
    DB erişimi ve kayıtlı hisse sayısını döner.
    """
    try:
        db = _get_db()
        leaders = db.get_leaderboard(top_n=1)
        db_ok = True
        total_symbols = len(db.get_leaderboard(top_n=9999))
    except Exception as exc:
        db_ok = False
        total_symbols = 0

    return {
        "status": "ok" if db_ok else "degraded",
        "db_path": _DB_PATH,
        "db_accessible": db_ok,
        "registered_symbols": total_symbols,
        "timestamp": datetime.now().isoformat(),
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
    job_id = str(uuid.uuid4())[:8]

    job: Dict[str, Any] = {
        "job_id": job_id,
        "symbol": symbol,
        "status": "queued",
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
    }
    _jobs[job_id] = job

    def _bg_run() -> None:
        _jobs[job_id]["status"] = "running"
        try:
            data_file = os.path.join(_PROJECT_ROOT, request.data_dir, f"{symbol}.csv")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_file}")

            from src.pipeline.orchestrator import ForecastingPipeline
            pipeline = ForecastingPipeline(
                data_file=data_file,
                validation_mode=request.mode,
                selected_models=request.models,
            )
            pipeline.run_all()
            _jobs[job_id]["status"] = "completed"
        except Exception as exc:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)
        finally:
            _jobs[job_id]["finished_at"] = datetime.now().isoformat()

    background_tasks.add_task(_bg_run)

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
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' bulunamadı. Geçerli job listesi: {list(_jobs.keys())}",
        )
    return job


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

