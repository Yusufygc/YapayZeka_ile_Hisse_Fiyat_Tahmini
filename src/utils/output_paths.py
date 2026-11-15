# -*- coding: utf-8 -*-
"""Çıktı dizini yardımcıları: `latest/` senkronu, işaretçi ve run_id çözümü.

`outputs/{SYMBOL}/latest/` artık run dizininin tam kopyası DEĞİLDİR. Yalnızca
küçük raporlar (json/csv/png/md/...) kopyalanır ve `LATEST_RUN.json` işaretçisi
yazılır. Büyük model artefaktları (`.pkl`, ...) işaretçinin gösterdiği
`runs/{RUN_ID}/` altında kalır; okuyucular `resolve_latest_artifact` ile çözer.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

LATEST_POINTER_NAME = "LATEST_RUN.json"
LATEST_MAX_COPY_BYTES = 5 * 1024 * 1024
LATEST_COPY_SUFFIXES = (".json", ".csv", ".png", ".md", ".txt", ".html")

_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_")


def _inside(root: str, path: str) -> bool:
    root_abs = os.path.abspath(root)
    path_abs = os.path.abspath(path)
    return path_abs == root_abs or path_abs.startswith(root_abs + os.sep)


def run_id_from_path(path: str) -> Optional[str]:
    """`.../runs/{RUN_ID}/...` yolundan run_id çıkarır (Windows/POSIX ayırıcıları)."""
    parts = [p for p in re.split(r"[\\/]+", str(path or "")) if p]
    for idx, part in enumerate(parts[:-1]):
        if part == "runs" and _RUN_ID_RE.match(parts[idx + 1]):
            return parts[idx + 1]
    return None


def write_latest_pointer(symbol_root: str, run_id: str, directory: Optional[str] = None) -> str:
    """`LATEST_RUN.json` yazar: {run_id, path (symbol_root'a göreli), created_at}."""
    target_dir = directory or os.path.join(symbol_root, "latest")
    os.makedirs(target_dir, exist_ok=True)
    payload = {
        "run_id": run_id,
        "path": f"runs/{run_id}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    pointer = os.path.join(target_dir, LATEST_POINTER_NAME)
    tmp = pointer + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, pointer)
    return pointer


def read_latest_pointer(symbol_root: str) -> Optional[Dict[str, Any]]:
    pointer = os.path.join(symbol_root, "latest", LATEST_POINTER_NAME)
    if not os.path.isfile(pointer):
        return None
    try:
        with open(pointer, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("LATEST_RUN.json okunamadi (%s): %s", pointer, exc)
        return None
    return data if isinstance(data, dict) and data.get("run_id") else None


def latest_run_dir(symbol_root: str) -> Optional[str]:
    """İşaretçinin gösterdiği run dizini (symbol_root dışına çıkamaz)."""
    pointer = read_latest_pointer(symbol_root)
    if not pointer:
        return None
    rel = str(pointer.get("path") or f"runs/{pointer['run_id']}")
    run_dir = os.path.normpath(os.path.join(symbol_root, rel))
    if not _inside(symbol_root, run_dir) or not os.path.isdir(run_dir):
        return None
    return run_dir


def resolve_latest_artifact(symbol_root: str, name: str) -> Optional[str]:
    """`latest/{name}` dosyasını döndürür; yoksa işaretçinin run dizininde arar.

    `name` symbol kökü içinde göreli olmalıdır (ör. `models/xgboost_model.pkl`).
    Bulunamazsa `None`.
    """
    rel = str(name or "").replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/") or os.path.isabs(name):
        return None
    local = os.path.join(symbol_root, "latest", rel)
    if os.path.isfile(local):
        return local
    run_dir = latest_run_dir(symbol_root)
    if run_dir is None:
        return None
    candidate = os.path.join(run_dir, rel)
    return candidate if os.path.isfile(candidate) else None


def copy_small_files(
    run_dir: str,
    dest_dir: str,
    *,
    max_bytes: int = LATEST_MAX_COPY_BYTES,
    suffixes: tuple = LATEST_COPY_SUFFIXES,
) -> int:
    """Yalnızca küçük rapor dosyalarını kopyalar; kopyalanan dosya sayısını döndürür."""
    copied = 0
    for dirpath, _dirs, files in os.walk(run_dir):
        rel_dir = os.path.relpath(dirpath, run_dir)
        for fname in files:
            if not fname.lower().endswith(suffixes):
                continue
            src = os.path.join(dirpath, fname)
            if os.path.getsize(src) > max_bytes:
                continue
            out_dir = dest_dir if rel_dir == "." else os.path.join(dest_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(out_dir, fname))
            copied += 1
    return copied


def acquire_sync_lock(
    lock_path: str,
    *,
    timeout_seconds: float = 60.0,
    stale_seconds: float = 900.0,
) -> int:
    start = time.monotonic()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
            return fd
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_seconds:
                    os.unlink(lock_path)
                    continue
            except OSError:
                continue
            if time.monotonic() - start >= timeout_seconds:
                raise TimeoutError(f"Latest output lock alinamadi: {lock_path}")
            time.sleep(0.25)


def sync_latest_output(
    output_root: str,
    run_dir: str,
    latest_dir: str,
    run_id: str,
    *,
    max_bytes: int = LATEST_MAX_COPY_BYTES,
) -> str:
    """`latest/` dizinini hafif biçimde günceller (küçük dosyalar + işaretçi)."""
    root = os.path.abspath(output_root)
    tmp_dir = os.path.abspath(os.path.join(root, f"latest.__tmp__{run_id}"))
    lock_path = os.path.abspath(os.path.join(root, ".latest_sync.lock"))
    if not all(_inside(root, p) and os.path.abspath(p) != root for p in (run_dir, latest_dir, tmp_dir, lock_path)):
        raise RuntimeError("Output latest senkronizasyon hedefi output root disinda.")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    try:
        copy_small_files(run_dir, tmp_dir, max_bytes=max_bytes)
        write_latest_pointer(root, run_id, directory=tmp_dir)
        lock_fd = acquire_sync_lock(lock_path)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    try:
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        os.replace(tmp_dir, latest_dir)
    finally:
        try:
            os.close(lock_fd)
        finally:
            if os.path.exists(lock_path):
                os.unlink(lock_path)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
    return os.path.abspath(latest_dir)
