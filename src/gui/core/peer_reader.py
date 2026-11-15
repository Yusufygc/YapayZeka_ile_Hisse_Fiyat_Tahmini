# -*- coding: utf-8 -*-
"""
src.gui.core.peer_reader - Gecelik havuz (PeerStore) icin salt-okunur okuyucu.

`data/serving_pool.db` son kosusunu ve `data/nightly_status.json` durumunu okur.
PeerStore SALT-OKUNUR acilir (`open_readonly`): dosya olusturulmaz, migrasyon/journal_mode
degismez; dosya yoksa bos goruntu doner. Risk kolonlari (risk_*)
henuz yazilmamis eski DB'lerde `None` gelir.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.serving.paths import default_serving_db_path
from src.serving.run_status import read_status, status_path_for_db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerSnapshot:
    run: Optional[Dict[str, Any]] = None
    rows: List[Dict[str, Any]] = field(default_factory=list)
    nightly: Optional[Dict[str, Any]] = None
    nightly_error: Optional[str] = None
    error: Optional[str] = None


def _clean(value: Any) -> Any:
    """NaN -> None; numpy skalerleri -> Python."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


class PeerReader:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or default_serving_db_path()

    def load(self) -> PeerSnapshot:
        nightly, nightly_error = self._read_nightly()
        if not os.path.exists(self.db_path):
            return PeerSnapshot(nightly=nightly, nightly_error=nightly_error,
                                error="Serving veritabani yok (gece isi henuz calismadi).")
        try:
            from src.serving.peer_store import PeerStore

            store = PeerStore.open_readonly(self.db_path)
            run = store.latest_run()
            if run is None:
                return PeerSnapshot(nightly=nightly, nightly_error=nightly_error,
                                    error="Serving veritabaninda kosu yok.")
            df = store.get_run_peer_scores(int(run["run_id"]))
        except (sqlite3.Error, OSError) as exc:  # GUI acilisini bozma
            logger.warning("PeerStore okunamadi (%s): %s", self.db_path, exc)
            return PeerSnapshot(nightly=nightly, nightly_error=nightly_error,
                                error=f"Serving veritabani okunamadi: {type(exc).__name__}")
        rows = [{k: _clean(v) for k, v in rec.items()} for rec in df.to_dict("records")]
        return PeerSnapshot(run=run, rows=rows, nightly=nightly, nightly_error=nightly_error)

    def _read_nightly(self) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            return read_status(status_path_for_db(self.db_path)), None
        except ValueError as exc:
            return None, str(exc)
