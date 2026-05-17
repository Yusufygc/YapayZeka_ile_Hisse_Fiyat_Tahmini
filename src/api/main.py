# -*- coding: utf-8 -*-
"""
src/api/main.py - ts_forecasting_lab FastAPI SonuÃ§ Servisi (Faz 5.4)

Proje kÃ¶kÃ¼nden Ã§alÄ±ÅŸtÄ±r:
    uvicorn src.api.main:app --reload --port 8000

Interaktif docs:
    http://localhost:8000/docs       (Swagger UI)
    http://localhost:8000/redoc      (ReDoc)

Endpoints:
    GET /health                      â€” servis saÄŸlÄ±k kontrolÃ¼
    GET /best-model/{symbol}         â€” hisse iÃ§in en iyi model
    GET /experiments/{symbol}        â€” deney geÃ§miÅŸi
    GET /metrics/{symbol}            â€” model karÅŸÄ±laÅŸtÄ±rma tablosu
    GET /leaderboard                 â€” tÃ¼m hisseler lider tablosu
    GET /symbols                     â€” kayÄ±tlÄ± tÃ¼m hisse kodlarÄ±
    POST /run/{symbol}               â€” tek hisse pipeline'Ä±nÄ± tetikle (arka planda)
    GET /run/status/{job_id}         â€” tetiklenen iÅŸin durumu
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Proje kÃ¶kÃ¼nÃ¼ path'e ekle (uvicorn proje kÃ¶kÃ¼nden Ã§alÄ±ÅŸtÄ±rÄ±lÄ±rsa gerekli)
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
        "FastAPI yÃ¼klÃ¼ deÄŸil. Kurmak iÃ§in:\n"
        "  pip install fastapi uvicorn\n"
        "ya da dl_env ortamÄ±nda:\n"
        "  conda activate dl_env && pip install fastapi uvicorn"
    ) from exc

from src.database.stock_model_db import StockModelDB

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# App ve DB baÅŸlatma
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app = FastAPI(
    title="ts_forecasting_lab API",
    description=(
        "BIST hisse tahmin pipeline'Ä±nÄ±n sonuÃ§larÄ±na eriÅŸim servisi.\n\n"
        "SQLite veritabanÄ±ndaki model deneyleri, metrikler ve lider tablosunu "
        "HTTP Ã¼zerinden sunar. Merge_PortfoySim gibi dÄ±ÅŸ uygulamalar bu API "
        "aracÄ±lÄ±ÄŸÄ±yla en iyi model seÃ§imini ve metriklerini sorgulayabilir."
    ),
    version="1.0.0",
    contact={"name": "ts_forecasting_lab"},
)

# CORS â€” aynÄ± makinedeki baÅŸka servisler (React dashboard, Merge_PortfoySim vb.)
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pydantic ÅŸemalarÄ±
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# In-memory job tracker (production'da Redis kullanÄ±lÄ±r)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_jobs: Dict[str, Dict[str, Any]] = {}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Endpoint'ler
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/health", tags=["Sistem"])
def health_check() -> Dict[str, Any]:
    """
    Servis saÄŸlÄ±k kontrolÃ¼.
    DB eriÅŸimi ve kayÄ±tlÄ± hisse sayÄ±sÄ±nÄ± dÃ¶ner.
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
    VeritabanÄ±nda kayÄ±tlÄ± tÃ¼m hisse kodlarÄ±nÄ± listeler.
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
    Belirtilen hisse iÃ§in en iyi modelin tÃ¼m bilgilerini dÃ¶ner.

    - **symbol**: Hisse kodu (bÃ¼yÃ¼k/kÃ¼Ã§Ã¼k harf duyarsÄ±z, Ã¶r. `TUPRS`)

    DÃ¶nen model; composite_score'a gÃ¶re tÃ¼m denemeler iÃ§inden seÃ§ilir.
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
            detail=f"{symbol} iÃ§in kayÄ±tlÄ± model bulunamadÄ±. "
                   "Ã–nce pipeline'Ä± Ã§alÄ±ÅŸtÄ±rÄ±n: python -m src.cli.interactive",
        )
    return result


@app.get("/experiments/{symbol}", tags=["Deneyler"])
def get_experiments(
    symbol: str,
    model_name: Optional[str] = Query(None, description="Model adÄ±na gÃ¶re filtrele"),
    limit: int = Query(20, ge=1, le=500, description="Maksimum sonuÃ§ sayÄ±sÄ±"),
) -> Dict[str, Any]:
    """
    Belirtilen hisse iÃ§in deney geÃ§miÅŸini dÃ¶ner.

    - **symbol**: Hisse kodu
    - **model_name**: Opsiyonel â€” belirli bir modeli filtrele (Ã¶r. `XGBoost`)
    - **limit**: DÃ¶ndÃ¼rÃ¼lecek maksimum satÄ±r sayÄ±sÄ± (varsayÄ±lan: 20)
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
    Belirtilen hisse iÃ§in tÃ¼m modellerin ortalama metrik karÅŸÄ±laÅŸtÄ±rmasÄ±nÄ± dÃ¶ner.

    Composite score, directional accuracy, Sharpe, MAE, RMSE gibi metrikleri
    model bazÄ±nda gruplayarak Ã¶zetler.
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
            detail=f"{symbol} iÃ§in metrik verisi bulunamadÄ±.",
        )
    return {
        "symbol": symbol,
        "model_count": len(rows),
        "comparison": rows,
    }


@app.get("/leaderboard", tags=["Hisseler"])
def get_leaderboard(
    top_n: int = Query(20, ge=1, le=500, description="KaÃ§ hisse dÃ¶ndÃ¼rÃ¼lsÃ¼n"),
) -> Dict[str, Any]:
    """
    TÃ¼m hisseler arasÄ±nda composite_score'a gÃ¶re lider tablosunu dÃ¶ner.

    Her satÄ±r bir hissenin en iyi modelini temsil eder.
    PortfÃ¶y seÃ§imi ve Ã¶nceliklendirme iÃ§in kullanÄ±labilir.
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
    Belirtilen hisse iÃ§in pipeline'Ä± arka planda baÅŸlatÄ±r.

    **Not:** Bu endpoint pipeline'Ä± tetikler ve hemen bir `job_id` dÃ¶ner.
    Ä°ÅŸin durumunu `GET /run/status/{job_id}` ile takip edebilirsiniz.

    Paralel Ã§alÄ±ÅŸtÄ±rmak iÃ§in `python -m src.cli.batch` kullanÄ±n.
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
                raise FileNotFoundError(f"Veri dosyasÄ± bulunamadÄ±: {data_file}")

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
        "message": f"Pipeline kuyruÄŸa alÄ±ndÄ±. Durum: GET /run/status/{job_id}",
    }


@app.get("/run/status/{job_id}", tags=["Pipeline"])
def get_run_status(job_id: str) -> Dict[str, Any]:
    """
    Tetiklenen bir pipeline iÅŸinin durumunu dÃ¶ner.

    - **job_id**: `/run/{symbol}` endpoint'inden dÃ¶nen job_id
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' bulunamadÄ±. GeÃ§erli job listesi: {list(_jobs.keys())}",
        )
    return job


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DoÄŸrudan Ã§alÄ±ÅŸtÄ±rma (geliÅŸtirme iÃ§in)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        print("uvicorn yÃ¼klÃ¼ deÄŸil: pip install uvicorn")

