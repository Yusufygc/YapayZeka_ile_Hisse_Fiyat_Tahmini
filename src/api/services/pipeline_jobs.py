# -*- coding: utf-8 -*-
"""POST /run/{symbol} için bellek-içi pipeline job takibi + arka plan koşucusu.

Sorumluluklar:
  - `RunRequest` / `RunStatus` istek-yanıt şemaları.
  - Tek-process bellek-içi job kaydı (`_jobs`). Üretimde Redis vb. tercih edilir.
  - `start_pipeline_job()`: ForecastingPipeline'ı BackgroundTasks ile kuyruğa alır.

FastAPI route katmanı (`src/api/main.py`) yalnız bu servise delege eder.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from fastapi import BackgroundTasks
    from pydantic import BaseModel as PydanticModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FastAPI yüklü değil. Kurmak için: pip install fastapi uvicorn"
    ) from exc


class RunRequest(PydanticModel):
    mode: str = "walk_forward"
    models: Optional[List[str]] = None
    data_dir: str = "data"
    auto_update_data: bool = True


class RunStatus(PydanticModel):
    job_id: str
    symbol: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None


# Üretimde Redis kullanılır; tek-process serve için bellek-içi yeterli.
_jobs: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Job kaydını döner; yoksa None."""
    return _jobs.get(job_id)


def known_job_ids() -> List[str]:
    """Bilinen tüm job id'leri (404 mesajı için)."""
    return list(_jobs.keys())


def start_pipeline_job(
    symbol: str,
    request: RunRequest,
    project_root: str,
    background_tasks: BackgroundTasks,
) -> str:
    """Pipeline'ı arka planda kuyruğa alır ve job_id döner.

    İş durumu `_jobs[job_id]` içinde queued → running → completed/error olarak
    güncellenir. Veri dosyası yoksa job 'error' statüsüne düşer.
    """
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "job_id": job_id,
        "symbol": symbol,
        "status": "queued",
        "started_at": _now_iso(),
        "finished_at": None,
        "error": None,
    }

    def _bg_run() -> None:
        _jobs[job_id]["status"] = "running"
        try:
            data_file = os.path.join(project_root, request.data_dir, f"{symbol}.csv")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_file}")

            from src.pipeline.config import (
                DataConfig,
                ExecutionConfig,
                ModelConfig,
                PipelineConfig,
                ValidationConfig,
            )
            from src.pipeline.orchestrator import ForecastingPipeline

            pipeline = ForecastingPipeline(cfg=PipelineConfig(
                data=DataConfig(
                    data_file=data_file,
                    auto_update_data=bool(request.auto_update_data),
                    auto_update_interactive=False,
                ),
                validation=ValidationConfig(validation_mode=request.mode),
                models=ModelConfig(selected_models=request.models),
                execution=ExecutionConfig(),
            ))
            pipeline.run_all()
            _jobs[job_id]["status"] = "completed"
        except Exception as exc:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)
        finally:
            _jobs[job_id]["finished_at"] = _now_iso()

    background_tasks.add_task(_bg_run)
    return job_id
