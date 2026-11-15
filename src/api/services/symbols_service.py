# -*- coding: utf-8 -*-
"""`/symbols` icin: son gecelik havuz kosusundaki semboller (PeerStore, salt-okunur kullanim)."""
from __future__ import annotations

import os
from typing import Optional

from src.api.services.serving_health import default_serving_db_path


def list_serving_symbols(db_path: Optional[str] = None) -> list[str]:
    """PeerStore'un SON kosusundaki sembolleri alfabetik dondurur.

    DB yok / kosu yok -> bos liste. PeerStore dosya yoksa OLUSTURULMAZ (once varlik kontrolu).
    """
    path = db_path or default_serving_db_path()
    if not os.path.exists(path):
        return []
    from src.serving.peer_store import PeerStore

    store = PeerStore(path)
    run = store.latest_run()
    if not run:
        return []
    df = store.get_run_peer_scores(int(run["run_id"]))
    if df.empty:
        return []
    return sorted({str(s).upper() for s in df["symbol"].dropna()})
