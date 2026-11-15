# -*- coding: utf-8 -*-
"""Pooled panel icin sembol-paralel hesap (PooledLoaderConfig.n_jobs != 1).

`PooledPanelLoader.load()` sembol basina FeaturePipeline calistirir (tek cekirdek, ~50 sn).
Bu modul sembolleri parcalara (chunk) boler, parcalari joblib/loky SUREC havuzunda
isler (thread degil: pandas/ta GIL'e takilir) ve sonuclari parca sirasiyla dondurur.

Birebirlik: iscinin yaptigi is `PooledPanelLoader._engineer_symbol_safe` ile AYNI koddur
(ayri bir "worker kopyasi" yok); isci yalniz sembole ait yan bilgiyi (report durumu,
rejected_rows, quarantined_days, ozellik onbellegi sayaclari, uyari loglari) toplayip
dondurur, ana surec bunlari sirali sembol dongusunde sekansiyel yolun yapacagi gibi birlestirir.

Hafif import: bu modulun ust seviyesi yalniz stdlib + pandas; `pooled_loader` (FeaturePipeline,
ta, pyarrow) isci fonksiyonu icinde tembel yuklenir. tensorflow/torch/sklearn YUKLENMEZ.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # dongusel import onlemi (pooled_loader bu modulu ice aktarir)
    from src.data.pooled_loader import PooledLoaderConfig

MAX_AUTO_WORKERS = 8      # olcum: bkz. docs/wiki/data-pipeline.md (daha fazla isci kazanc getirmedi)
AUTO_CHUNK_MAX = 20
AUTO_CHUNK_MIN = 4


@dataclass(frozen=True)
class SymbolTask:
    """Bir sembolun iscide hesaplanmasi icin gereken girdi (picklable)."""
    symbol: str
    sector: str
    delisted_on: pd.Timestamp | None
    path: str
    clean_frame: pd.DataFrame | None = None     # temiz depodan gelen cerceve (yoksa CSV'den)


@dataclass
class SymbolResult:
    symbol: str
    frame: pd.DataFrame | None
    status: str | None            # iscideki report[symbol] (hata/atlama nedeni) ya da None
    rejected_rows: int | None
    quarantined_days: int | None


@dataclass
class ChunkResult:
    results: list[SymbolResult]
    cache: dict[str, int]                                   # hits / misses / errors
    logs: list[tuple[str, int, str]] = field(default_factory=list)   # (logger, seviye, mesaj)


def resolve_workers(n_jobs: int, n_items: int) -> int:
    """n_jobs: 1 sekansiyel, >1 tam bu kadar, -1 otomatik. Is sayisini asmaz."""
    if n_jobs == -1:
        workers = max(1, min((os.cpu_count() or 2) - 1, MAX_AUTO_WORKERS))
    else:
        workers = int(n_jobs)
    return max(1, min(workers, n_items))


def resolve_chunk_size(chunk_size: int, n_items: int, workers: int) -> int:
    """0 = otomatik: isci basina ~4 parca (yuk dengesi), [4, 20] araliginda."""
    if chunk_size > 0:
        return chunk_size
    per = -(-n_items // max(1, workers * 4))
    return max(AUTO_CHUNK_MIN, min(AUTO_CHUNK_MAX, per))


def split_chunks(tasks: Sequence[SymbolTask], size: int) -> list[list[SymbolTask]]:
    return [list(tasks[i:i + size]) for i in range(0, len(tasks), size)]


class _CaptureHandler(logging.Handler):
    """Isci surecteki WARNING+ kayitlarini toplar (ana surec yeniden yayinlar)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[tuple[str, int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.name, record.levelno, record.getMessage()))


def run_chunk(
    tasks: list[SymbolTask], cfg: "PooledLoaderConfig", fingerprint: str | None,
) -> ChunkResult:
    """ISCI: bir parcadaki sembolleri sirayla hesapla. Modul seviyesinde, picklable."""
    from src.data.pooled_loader import PooledPanelLoader      # tembel: hafif ust-import
    from src.features.pooled_feature_cache import PooledFeatureCache

    capture = _CaptureHandler()
    root = logging.getLogger()
    root.addHandler(capture)
    try:
        # n_jobs=1 + depo/onbellek ana surecte; isci yalniz hesap yapar.
        loader = PooledPanelLoader(
            replace(cfg, n_jobs=1, clean_store_dir=None, feature_cache_dir=None))
        loader._paths = {t.symbol: t.path for t in tasks}
        loader._clean_frames = {t.symbol: t.clean_frame for t in tasks
                                if t.clean_frame is not None}
        if cfg.feature_cache_dir:
            loader._fcache = PooledFeatureCache(cfg.feature_cache_dir, fingerprint=fingerprint)
        results = []
        for t in tasks:
            frame = loader._engineer_symbol_safe(t.symbol, t.sector, t.delisted_on)
            results.append(SymbolResult(
                symbol=t.symbol, frame=frame, status=loader.report.get(t.symbol),
                rejected_rows=loader.rejected_rows.get(t.symbol),
                quarantined_days=loader.quarantined_days.get(t.symbol)))
        cache = loader._fcache.stats() if loader._fcache else {"hits": 0, "misses": 0, "errors": 0}
        return ChunkResult(results=results, cache=cache, logs=capture.records)
    finally:
        root.removeHandler(capture)


def run_parallel(
    tasks: Sequence[SymbolTask], cfg: "PooledLoaderConfig", fingerprint: str | None,
    workers: int, chunk_size: int,
) -> list[ChunkResult]:
    """Parcalari loky surec havuzunda calistir; sonuclar parca (=sembol) sirasinda."""
    from joblib import Parallel, delayed

    chunks = split_chunks(tasks, resolve_chunk_size(chunk_size, len(tasks), workers))
    return Parallel(n_jobs=min(workers, len(chunks)), backend="loky", max_nbytes=None)(
        delayed(run_chunk)(c, cfg, fingerprint) for c in chunks)


def replay_logs(logs: Sequence[tuple[str, int, str]]) -> None:
    """Isci uyarilarini ana surecin logger'larindan yeniden yayinla (sessiz kalmasin)."""
    for name, level, message in logs:
        logging.getLogger(name).log(level, "%s", message)
