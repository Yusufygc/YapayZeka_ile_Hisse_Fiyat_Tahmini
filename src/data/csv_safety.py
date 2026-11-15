# -*- coding: utf-8 -*-
"""csv_safety.py - Sembol CSV'leri icin kilit, atomik yazim ve yedek rotasyonu."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

DEFAULT_LOCK_TIMEOUT_S = 60.0
MAX_BACKUPS = 3


class CsvLockTimeout(RuntimeError):
    """CSV kilidi zaman asimi suresinde alinamadi."""


@contextmanager
def csv_lock(csv_path: str, timeout: float = DEFAULT_LOCK_TIMEOUT_S) -> Iterator[None]:
    """Sembol basina surecler-arasi kilit (``<csv>.lock``); olen surecin kilidi OS tarafindan birakilir."""
    lock = FileLock(f"{csv_path}.lock", timeout=timeout)
    try:
        lock.acquire()
    except Timeout as exc:
        raise CsvLockTimeout(f"{os.path.basename(csv_path)} kilidi {timeout:.0f}s icinde alinamadi") from exc
    try:
        yield
    finally:
        lock.release()


def atomic_write_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Ayni klasorde gecici dosyaya yaz, sonra os.replace ile yerine koy."""
    directory = os.path.dirname(os.path.abspath(csv_path))
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(csv_path) + ".", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        os.replace(tmp_path, csv_path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as rm_exc:
                logger.warning("gecici dosya silinemedi (%s): %s", tmp_path, rm_exc)
        raise


def backup_before_rewrite(
    csv_path: str, *, today: datetime | pd.Timestamp, keep: int = MAX_BACKUPS
) -> str:
    """CSV'yi ``<csv_dir>/yedek/<DOSYA>.bak_<YYYYMMDD>`` olarak kopyalar; en fazla ``keep`` yedek tutar."""
    backup_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "yedek")
    os.makedirs(backup_dir, exist_ok=True)
    base = os.path.basename(csv_path)
    target = os.path.join(backup_dir, f"{base}.bak_{pd.Timestamp(today).strftime('%Y%m%d')}")
    shutil.copy2(csv_path, target)
    prefix = f"{base}.bak_"
    backups = sorted(f for f in os.listdir(backup_dir) if f.startswith(prefix))
    for old in backups[:-keep] if keep > 0 else backups:
        try:
            os.remove(os.path.join(backup_dir, old))
        except OSError as exc:
            logger.warning("eski yedek silinemedi (%s): %s", old, exc)
    return target


def atomic_append_text(csv_path: str, text: str) -> None:
    """Mevcut dosya baytlarini birebir koruyup sonuna ``text`` ekler; gecici dosya + os.replace.

    Okuyucular hicbir zaman yarim yazilmis dosya gormez; eski satirlarin bicimi degismez.
    """
    with open(csv_path, "rb") as fh:
        original = fh.read()
    eol = b"\r\n" if b"\r\n" in original else b"\n"
    payload = original
    if payload and not payload.endswith(b"\n"):
        payload += eol
    payload += text.replace("\r\n", "\n").replace("\n", eol.decode()).encode("utf-8")
    directory = os.path.dirname(os.path.abspath(csv_path))
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(csv_path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(payload)
        os.replace(tmp_path, csv_path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as rm_exc:
                logger.warning("gecici dosya silinemedi (%s): %s", tmp_path, rm_exc)
        raise
