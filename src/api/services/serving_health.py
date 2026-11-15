# -*- coding: utf-8 -*-
"""`/health` icin Kol-B serving havuzu tazelik ozeti (K6).

PeerStore'u ACMAZ/OLUSTURMAZ (salt-okunur sqlite); dosya yok / bozuk ise
`available=False` doner ve istisna metnini sizdirmaz.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.database.connection import open_readonly
from src.serving.paths import default_serving_db_path

logger = logging.getLogger(__name__)

_ISTANBUL = timezone(timedelta(hours=3))  # Turkiye sabit UTC+3 (DST yok)
_DEFAULT_STALE_DAYS = 7


def serving_pool_status(
    db_path: Optional[str] = None,
    today: Optional[date] = None,
    stale_days: int = _DEFAULT_STALE_DAYS,
) -> Dict[str, Any]:
    """Son gece skorlama kosusunun `as_of_date` / yas / kapsam ozeti.

    age_days: bugun - as_of_date (takvim gunu). stale: age_days > stale_days.
    """
    path = db_path or default_serving_db_path()
    if not os.path.exists(path):
        return {"available": False, "reason": "no_db"}
    try:
        conn = open_readonly(path)
        try:
            row = conn.execute(
                "SELECT run_id, model_name, as_of_date, n_symbols, n_rows, created_at "
                "FROM global_model_runs ORDER BY run_id DESC LIMIT 1").fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:  # /health asla patlamasin, detay sizmasin
        logger.warning("serving havuzu okunamadi (%s): %s", path, exc)
        return {"available": False, "reason": "unreadable"}
    if row is None:
        return {"available": False, "reason": "no_runs"}
    run_id, model_name, as_of_raw, n_symbols, n_rows, created_at = row
    try:
        as_of = datetime.fromisoformat(str(as_of_raw)[:10]).date()
    except ValueError:
        return {"available": False, "reason": "bad_as_of_date"}
    age = ((today or datetime.now(_ISTANBUL).date()) - as_of).days
    return {
        "available": True,
        "run_id": int(run_id),
        "model_name": model_name,
        "as_of_date": as_of.isoformat(),
        "age_days": int(age),
        "stale": bool(age > int(stale_days)),
        "n_symbols": int(n_symbols or 0),
        "n_rows": int(n_rows or 0),
        "created_at": created_at,
    }


# Tamamlanmamis bir kosu bu sureden uzun suredir "calisiyor" gorunuyorsa oldurulmus sayilir
# (Task Scheduler sinirinin biraz ustu; gece isi ~2 sa icinde bitmeli).
_RUNNING_MAX_HOURS = 3.0


def nightly_run_status(
    db_path: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Son gecelik kosunun durum dosyasi (`nightly_status.json`, serving DB'nin yaninda) ozeti.

    `interrupted=True`: kosu bitmedi (completed=false) ve ya cokme kaydi var ya da
    ``_RUNNING_MAX_HOURS``'tan eski -> surec oldurulmus/donmus. `healthy`: tamamlandi, cikis 0. `risk`: risk adiminin durumu
    (`{status: ok|fallback|failed, method, error}`; eski durum dosyalarinda None).
    Dosya yok/bozuk ise `available=False`; istisna metni sizdirilmaz.
    """
    # Yalniz stdlib modul (agir bagimlilik yuklemez); durum dosyasi serving DB'nin yaninda.
    from src.serving.run_status import read_status, status_path_for_db

    try:
        data = read_status(status_path_for_db(db_path or default_serving_db_path()))
    except ValueError:
        return {"available": False, "reason": "unreadable"}
    if data is None:
        return {"available": False, "reason": "no_status_file"}
    completed = bool(data.get("completed"))
    exit_code = data.get("exit_code")
    interrupted = False
    if not completed:
        try:
            started = datetime.fromisoformat(str(data.get("started_at")))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            age_h = ((now or datetime.now(timezone.utc)) - started).total_seconds() / 3600.0
            interrupted = data.get("phase") == "crashed" or age_h > _RUNNING_MAX_HOURS
        except ValueError:
            interrupted = True
    return {
        "available": True,
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "phase": data.get("phase"),
        "completed": completed,
        "interrupted": interrupted,
        "exit_code": exit_code,
        "healthy": bool(completed and exit_code == 0),
        "scoring": data.get("scoring"),
        "stale_symbols_after_refresh": data.get("stale_symbols_after_refresh"),
        "stale_fraction": data.get("stale_fraction"),
        "refresh": data.get("refresh"),
        "risk": data.get("risk"),
        "error": data.get("error"),
    }
