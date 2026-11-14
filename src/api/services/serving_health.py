# -*- coding: utf-8 -*-
"""`/health` icin Kol-B serving havuzu tazelik ozeti (K6).

PeerStore'u ACMAZ/OLUSTURMAZ (salt-okunur sqlite); dosya yok / bozuk ise
`available=False` doner ve istisna metnini sizdirmaz.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, Optional

_DEFAULT_STALE_DAYS = 7


def default_serving_db_path() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "data", "serving_pool.db")


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
        uri = "file:" + path.replace("\\", "/") + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2)
        try:
            row = conn.execute(
                "SELECT run_id, model_name, as_of_date, n_symbols, n_rows, created_at "
                "FROM global_model_runs ORDER BY run_id DESC LIMIT 1").fetchone()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - /health asla patlamasin, detay sizmasin
        return {"available": False, "reason": "unreadable"}
    if row is None:
        return {"available": False, "reason": "no_runs"}
    run_id, model_name, as_of_raw, n_symbols, n_rows, created_at = row
    try:
        as_of = datetime.fromisoformat(str(as_of_raw)[:10]).date()
    except ValueError:
        return {"available": False, "reason": "bad_as_of_date"}
    age = ((today or date.today()) - as_of).days
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
