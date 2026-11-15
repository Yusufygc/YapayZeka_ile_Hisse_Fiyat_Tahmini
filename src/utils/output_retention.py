# -*- coding: utf-8 -*-
"""`outputs/` saklama (retention) mantığı: eski run silme, latest inceltme, pkl sıkıştırma.

`tools/prune_outputs.py` bu modülün ince CLI sarmalayıcısıdır. Varsayılan davranış
salt-okunurdur (dry-run); silme/yazma yalnızca `apply=True` ile yapılır.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from src.database.connection import connect
from src.utils.output_paths import (
    LATEST_COPY_SUFFIXES,
    LATEST_MAX_COPY_BYTES,
    LATEST_POINTER_NAME,
    read_latest_pointer,
    run_id_from_path,
    write_latest_pointer,
)

logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,10}$")
_PICKLE_MAGIC = b"\x80"  # sıkıştırılmamış (düz pickle/joblib) dosyalar bununla başlar


@dataclass
class RunEntry:
    symbol: str
    run_id: str
    path: str
    size: int
    keep: bool
    reason: str


@dataclass
class PruneReport:
    runs: List[RunEntry] = field(default_factory=list)
    deleted: List[RunEntry] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)

    @property
    def to_delete(self) -> List[RunEntry]:
        return [r for r in self.runs if not r.keep]

    @property
    def bytes_to_free(self) -> int:
        return sum(r.size for r in self.to_delete)


def dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
    return total


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def symbol_dirs(outputs_dir: str, symbols: Optional[Iterable[str]] = None) -> List[str]:
    wanted = {s.upper() for s in symbols} if symbols else None
    found = []
    for name in sorted(os.listdir(outputs_dir)):
        if not _SYMBOL_RE.match(name):
            continue
        if wanted is not None and name not in wanted:
            continue
        if os.path.isdir(os.path.join(outputs_dir, name, "runs")):
            found.append(name)
    return found


def db_referenced_run_ids(db_path: str) -> Set[str]:
    """best_models/experiments.model_path içindeki run_id segmentleri (salt-okunur)."""
    if not db_path or not os.path.isfile(db_path):
        return set()
    uri = Path(os.path.abspath(db_path)).as_uri() + "?mode=ro"
    run_ids: Set[str] = set()
    try:
        with connect(uri, uri=True) as conn:
            for table in ("best_models", "experiments"):
                for (model_path,) in conn.execute(
                    f"SELECT DISTINCT model_path FROM {table} WHERE model_path IS NOT NULL AND model_path != ''"
                ):
                    run_id = run_id_from_path(model_path)
                    if run_id:
                        run_ids.add(run_id)
    except sqlite3.Error as exc:
        logger.warning("DB okunamadi (%s): %s", db_path, exc)
        raise
    return run_ids


def _manifest_status(run_dir: str) -> Optional[str]:
    try:
        with open(os.path.join(run_dir, "run_manifest.json"), "r", encoding="utf-8") as fh:
            return json.load(fh).get("status")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _legacy_latest_run_id(symbol_root: str) -> Optional[str]:
    try:
        with open(os.path.join(symbol_root, "latest", "run_manifest.json"), "r", encoding="utf-8") as fh:
            value = json.load(fh).get("run_id")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return str(value) if value else None


def plan_prune(
    outputs_dir: str,
    *,
    keep: int = 5,
    db_path: Optional[str] = None,
    symbols: Optional[Iterable[str]] = None,
    running_grace_hours: float = 24.0,
    now: Optional[float] = None,
) -> PruneReport:
    import time

    now = time.time() if now is None else now
    referenced = db_referenced_run_ids(db_path) if db_path else set()
    report = PruneReport()
    for symbol in symbol_dirs(outputs_dir, symbols):
        symbol_root = os.path.join(outputs_dir, symbol)
        runs_root = os.path.join(symbol_root, "runs")
        run_ids = sorted(
            (n for n in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, n))),
            reverse=True,
        )
        pointer = read_latest_pointer(symbol_root)
        latest_ids = {str(pointer["run_id"])} if pointer else set()
        legacy = _legacy_latest_run_id(symbol_root)
        if legacy:
            latest_ids.add(legacy)
        for idx, run_id in enumerate(run_ids):
            path = os.path.join(runs_root, run_id)
            reason, keep_it = "", False
            if idx < keep:
                reason, keep_it = f"son_{keep}", True
            elif run_id in referenced:
                reason, keep_it = "db_referansli", True
            elif run_id in latest_ids:
                reason, keep_it = "latest", True
            elif (
                _manifest_status(path) == "running"
                and (now - os.path.getmtime(path)) < running_grace_hours * 3600
            ):
                reason, keep_it = "calisiyor", True
            report.runs.append(
                RunEntry(symbol, run_id, path, dir_size(path), keep_it, reason or "silinecek")
            )
    return report


def apply_prune(report: PruneReport, outputs_dir: str) -> PruneReport:
    outputs_abs = os.path.abspath(outputs_dir)
    for entry in report.to_delete:
        target = os.path.abspath(entry.path)
        expected_parent = os.path.join(outputs_abs, entry.symbol, "runs")
        if os.path.dirname(target) != expected_parent or os.path.islink(target):
            report.failed.append(f"{entry.path}: beklenmeyen yol, atlandi")
            continue
        try:
            shutil.rmtree(target)
            report.deleted.append(entry)
        except OSError as exc:
            logger.warning("Run silinemedi %s: %s", target, exc)
            report.failed.append(f"{entry.path}: {exc}")
    return report


def slim_latest(
    outputs_dir: str,
    *,
    apply: bool,
    symbols: Optional[Iterable[str]] = None,
    max_bytes: int = LATEST_MAX_COPY_BYTES,
) -> Dict[str, int]:
    """Eski tam-kopya `latest/` dizinlerindeki büyük dosyaları kaldırıp işaretçi yazar."""
    freed = 0
    slimmed = 0
    for symbol in symbol_dirs(outputs_dir, symbols):
        symbol_root = os.path.join(outputs_dir, symbol)
        latest = os.path.join(symbol_root, "latest")
        if not os.path.isdir(latest) or os.path.isfile(os.path.join(latest, LATEST_POINTER_NAME)):
            continue
        run_id = _legacy_latest_run_id(symbol_root)
        if not run_id or not os.path.isdir(os.path.join(symbol_root, "runs", run_id)):
            logger.warning("%s: latest'in kaynak run'i bulunamadi, atlandi", symbol)
            continue
        victims = []
        for dirpath, _dirs, files in os.walk(latest):
            for name in files:
                full = os.path.join(dirpath, name)
                if not name.lower().endswith(LATEST_COPY_SUFFIXES) or os.path.getsize(full) > max_bytes:
                    victims.append(full)
        freed += sum(os.path.getsize(v) for v in victims)
        slimmed += 1
        if apply:
            for victim in victims:
                os.unlink(victim)
            write_latest_pointer(symbol_root, run_id)
    return {"symbols": slimmed, "bytes": freed}


def _is_uncompressed(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(1) == _PICKLE_MAGIC


def find_compress_candidates(outputs_dir: str, *, min_bytes: int, symbols=None) -> List[str]:
    found = []
    for symbol in symbol_dirs(outputs_dir, symbols):
        for dirpath, _dirs, files in os.walk(os.path.join(outputs_dir, symbol)):
            for name in files:
                full = os.path.join(dirpath, name)
                if not name.lower().endswith(".pkl"):
                    continue
                try:
                    if os.path.getsize(full) >= min_bytes and _is_uncompressed(full):
                        found.append(full)
                except OSError as exc:
                    logger.warning("Dosya okunamadi %s: %s", full, exc)
    return found


def estimate_compressed_size(paths: List[str], *, sample_files: int = 5, sample_bytes: int = 4 << 20) -> float:
    """İlk dosyaların başından zlib(3) oranı ölçüp toplam sıkıştırılmış boyutu tahmin eder."""
    ratios = []
    for path in sorted(paths, key=os.path.getsize, reverse=True)[:sample_files]:
        with open(path, "rb") as fh:
            chunk = fh.read(sample_bytes)
        if chunk:
            ratios.append(len(zlib.compress(chunk, 3)) / len(chunk))
    ratio = sum(ratios) / len(ratios) if ratios else 1.0
    return ratio * sum(os.path.getsize(p) for p in paths)


def _same_tree_ensemble(a: object, b: object) -> bool:
    """sklearn ağaç topluluklarını ağaç dizileri üzerinden karşılaştırır.

    `joblib.hash` RandomForest gibi nesnelerde aynı dosyanın iki yüklemesinde bile
    farklı çıkabiliyor (pickle iç durumu deterministik değil). Ağaç yapısı
    (children/feature/threshold/value) birebir aynıysa model aynıdır.
    """
    import numpy as np

    ests_a, ests_b = getattr(a, "estimators_", None), getattr(b, "estimators_", None)
    if ests_a is None or ests_b is None or type(a) is not type(b) or len(ests_a) != len(ests_b):
        return False
    fields = ("children_left", "children_right", "feature", "threshold", "value")
    for ea, eb in zip(np.ravel(ests_a), np.ravel(ests_b)):
        ta, tb = getattr(ea, "tree_", None), getattr(eb, "tree_", None)
        if ta is None or tb is None:
            return False
        if not all(np.array_equal(getattr(ta, f), getattr(tb, f)) for f in fields):
            return False
    return True


def _same_object(a: object, b: object) -> bool:
    import joblib

    return joblib.hash(a) == joblib.hash(b) or _same_tree_ensemble(a, b)


def compress_file(path: str) -> int:
    """Tek bir .pkl'yi joblib compress=3 ile yeniden yazar; kazanılan bayt sayısını döndürür.

    Atomik değiştirme: geçici dosyaya yaz, yeniden yükleyip `joblib.hash` eşitliğini
    doğrula, sonra `os.replace`. Doğrulama başarısızsa orijinal dokunulmaz.
    """
    import joblib

    original = joblib.load(path)
    tmp = path + ".compress.tmp"
    try:
        joblib.dump(original, tmp, compress=3)
        if not _same_object(joblib.load(tmp), original):
            raise ValueError("sikistirilmis dosya orijinalle esit degil")
        saved = os.path.getsize(path) - os.path.getsize(tmp)
        if saved <= 0:
            os.unlink(tmp)
            return 0
        os.replace(tmp, path)
        return saved
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def compress_existing(paths: List[str]) -> Dict[str, object]:
    saved, done, failed = 0, 0, []
    for path in paths:
        try:
            gained = compress_file(path)
        except Exception as exc:  # pkl yuklenemez/dogrulama hatasi: dosyayi atla, raporla
            logger.warning("Sikistirma atlandi %s: %s", path, exc)
            failed.append(f"{path}: {exc}")
            continue
        saved += gained
        done += 1 if gained else 0
    return {"files": done, "bytes_saved": saved, "failed": failed}


def clean_tmp(project_root: str, *, apply: bool) -> int:
    target = os.path.join(project_root, ".codex_tmp")
    if not os.path.isdir(target) or os.path.islink(target):
        return 0
    size = dir_size(target)
    if apply:
        shutil.rmtree(target, ignore_errors=False)
    return size
