# -*- coding: utf-8 -*-
"""2026-11 persistence audit fixes: PeerStore pragmalari/atomiklik."""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from src.serving.peer_store import GlobalRunMeta, PeerStore


def _path(tmp_path, name="serving_pool.db") -> str:
    return os.path.join(str(tmp_path), name)


# ----------------------------------------------------------- WAL + busy_timeout
def test_peer_store_connection_has_wal_and_busy_timeout(tmp_path):
    store = PeerStore(_path(tmp_path))
    with store._connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000


def test_peer_store_write_run_is_one_transaction(tmp_path):
    store = PeerStore(_path(tmp_path))
    meta = GlobalRunMeta(model_name="m", as_of_date="2026-06-03")
    good = pd.DataFrame({"symbol": ["AAA", "BBB"], "peer_score": [1.0, 2.0]})
    rid, n = store.write_run(meta, good)
    assert n == 2 and store.latest_run()["run_id"] == rid

    # Skor yazımı sırasında hata: NOT NULL symbol ihlali -> run başlığı da geri alınmalı.
    bad2 = pd.DataFrame({"symbol": [None], "peer_score": [1.0]})
    with pytest.raises(sqlite3.IntegrityError):
        store.write_run(meta, bad2)
    assert store.latest_run()["run_id"] == rid  # yetim run yok
