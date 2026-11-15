# -*- coding: utf-8 -*-
"""2026-11 persistence audit fixes: PeerStore pragmalari/atomiklik, tasinabilir yollar."""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from src.database import paths as db_paths
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


# --------------------------------------------------------------- portable paths
def test_legacy_absolute_path_is_remapped_to_local_outputs(tmp_path):
    model = tmp_path / "outputs" / "X" / "runs" / "r" / "models" / "m.pt"
    model.parent.mkdir(parents=True)
    model.write_text("w")
    legacy = r"D:\old_machine\ts_lab\outputs\X\runs\r\models\m.pt"

    resolved = db_paths.resolve_model_path(legacy, project_root=str(tmp_path))

    assert os.path.normpath(resolved) == os.path.normpath(str(model))
    # Yerelde karşılığı yoksa orijinal döner (sessizce yanlış dosya uydurulmaz).
    missing = r"D:\old_machine\outputs\Y\m.pt"
    assert db_paths.resolve_model_path(missing, project_root=str(tmp_path)) == missing
    # Göreceli yol proje köküne çözülür; outputs_root verilirse onun altına.
    rel = db_paths.resolve_model_path("outputs/X/runs/r/models/m.pt", project_root=str(tmp_path))
    assert os.path.normpath(rel) == os.path.normpath(str(model))
    other = tmp_path / "elsewhere"
    via_root = db_paths.resolve_model_path(
        "outputs/X/m.pt", project_root=str(tmp_path), outputs_root=str(other)
    )
    assert os.path.normpath(via_root) == os.path.normpath(str(other / "X" / "m.pt"))


def test_to_storable_path_is_idempotent_and_keeps_foreign_paths(tmp_path):
    inside = os.path.join(db_paths.PROJECT_ROOT, "outputs", "A", "m.pt")
    once = db_paths.to_storable_path(inside)
    assert once == "outputs/A/m.pt"
    assert db_paths.to_storable_path(once) == once
    assert db_paths.to_storable_path("") == ""
    # pytest tmp dizini repo icinde (.codex_tmp/pytest/...) olabilir; "dis" yol icin proje
    # koku acikca tmp altindaki ayri bir dizine verilir -> yol gercekten kok DISINDA.
    fake_root = str(tmp_path / "repo")
    outside = str(tmp_path / "other" / "m.pt")
    assert db_paths.to_storable_path(outside, project_root=fake_root) == outside
    inside_fake = os.path.join(fake_root, "outputs", "A", "m.pt")
    assert db_paths.to_storable_path(inside_fake, project_root=fake_root) == "outputs/A/m.pt"
