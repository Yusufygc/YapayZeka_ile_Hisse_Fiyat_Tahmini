# -*- coding: utf-8 -*-
"""Kol-B (pooled panel) icin sembol basi Parquet ozellik onbellegi.

`PooledPanelLoader` her sembol icin `FeaturePipeline.engineer_features` calistirir
(~0.12 sn/sembol; ~585 sembolde ~65 sn). Girdi (temizlenmis OHLCV) ve kod degismediyse
sonuc deterministiktir -> diskten okunur.

Kapsam: yalniz `macro_df=None` cagrilari (pooled yukleyici). Sonuc `symbol`'e bagli
degildir ama anahtara yine de girer (dosya basina tek sembol).

Anahtar (sha256):
  _SCHEMA_VERSION (biçim degisikligi icin elle) + kod parmak izi (`code_fingerprint`:
  feature_pipeline ve transitif src bagimliliklari + pandas/numpy/ta surumu) +
  girdi cercevesinin tam icerigi (kolon adi, dtype, deger baytlari) + feature_mode +
  enable_calendar_features (+ takvim aciksa FOMC csv icerigi) + sembol.

Depolama: `{cache_dir}/{SEMBOL}.parquet`, sembol basina TEK girdi (uzerine yazilir ->
dizin sembol sayisiyla sinirli, TTL gerekmez). Anahtar Parquet sema metadata'sinda.
Yazma atomik (gecici dosya + os.replace). Bozuk dosya -> uyari, miss, yeniden yazilir.
Artimli (son K satir) hesap bilincli olarak YOK: bkz. docs/wiki/data-pipeline.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.code_fingerprint import POOLED_ROOTS, code_fingerprint

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1"            # onbellek BICIMI degisirse elle artir (kod degisimi otomatik)
_META_KEY = b"tsfl.feature_cache_key"
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")   # yol gecisi (path traversal) onlemi
_FOMC_PATH = os.path.join("data", "meta", "fomc_calendar.csv")  # FeaturePipeline varsayilani
_READ_ERRORS = (OSError, ValueError, KeyError, pa.ArrowException)


def frame_content_hash(df: pd.DataFrame) -> str:
    """Kolon adi + dtype + deger baytlari (index HARIC) uzerinden sha256 hex.

    Sayisal/tarih kolonlari ham baytlardan (tam esitlik; NaN/-0.0 ayrimi korunur),
    diger dtype'lar `hash_pandas_object` ile hashlenir.
    """
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    for name in df.columns:
        col = df[name]
        h.update(b"\0" + str(name).encode("utf-8") + b"\0" + str(col.dtype).encode())
        if isinstance(col.dtype, np.dtype) and col.dtype.kind in "fiubMm":
            h.update(np.ascontiguousarray(col.to_numpy()).tobytes())
        else:
            h.update(pd.util.hash_pandas_object(col, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _fomc_digest() -> str:
    try:
        with open(_FOMC_PATH, "rb") as fh:
            return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return "missing"


class PooledFeatureCache:
    """Sembol basi Parquet onbellegi. `cache_dir` yoksa olusturulur."""

    def __init__(self, cache_dir: str, fingerprint: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.fingerprint = fingerprint if fingerprint is not None else code_fingerprint(POOLED_ROOTS)
        self.hits = 0
        self.misses = 0
        self.errors = 0
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ key
    def make_key(
        self, frame: pd.DataFrame, *, symbol: str, feature_mode: str,
        enable_calendar_features: bool,
    ) -> str:
        payload = {
            "schema": _SCHEMA_VERSION,
            "code": self.fingerprint,
            "symbol": symbol,
            "feature_mode": feature_mode,
            "calendar": bool(enable_calendar_features),
            "fomc": _fomc_digest() if enable_calendar_features else None,
            "content": frame_content_hash(frame),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _path(self, symbol: str) -> str | None:
        if not _SYMBOL_RE.match(symbol):
            logger.warning("feature_cache: gecersiz sembol adi, onbellek atlandi: %r", symbol)
            return None
        return os.path.join(self.cache_dir, f"{symbol}.parquet")

    # ------------------------------------------------------------------- io
    def get(self, symbol: str, key: str) -> pd.DataFrame | None:
        """Anahtar eslesirse cerceve, aksi halde None. Sayaclari ARTIRMAZ."""
        path = self._path(symbol)
        if path is None or not os.path.exists(path):
            return None
        try:
            meta = pq.read_schema(path).metadata or {}
            if meta.get(_META_KEY) != key.encode():
                return None
            return pq.read_table(path).to_pandas()
        except _READ_ERRORS as exc:
            self.errors += 1
            logger.warning("feature_cache: bozuk dosya %s (%s); yeniden uretilecek", path, exc)
            return None

    def put(self, symbol: str, key: str, frame: pd.DataFrame) -> bool:
        """Atomik yaz (temp + os.replace). Basarisizlik uyari + errors, hesap sonucu etkilenmez."""
        path = self._path(symbol)
        if path is None:
            return False
        tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            table = pa.Table.from_pandas(frame, preserve_index=None)
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), _META_KEY: key.encode()})
            pq.write_table(table, tmp, compression="zstd", compression_level=3)
            os.replace(tmp, path)
            return True
        except (OSError, ValueError, TypeError, pa.ArrowException) as exc:
            self.errors += 1
            logger.warning("feature_cache: %s yazilamadi (%s)", path, exc)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError as rm_exc:
                    logger.warning("feature_cache: gecici dosya silinemedi %s (%s)", tmp, rm_exc)
            return False

    def get_or_compute(
        self, symbol: str, raw: pd.DataFrame, *, feature_mode: str,
        enable_calendar_features: bool, compute: Callable[[], pd.DataFrame | None],
    ) -> pd.DataFrame | None:
        """`raw` (engineer_features girdisi) icin onbellek; miss'te `compute()` + yaz."""
        key = self.make_key(raw, symbol=symbol, feature_mode=feature_mode,
                            enable_calendar_features=enable_calendar_features)
        hit = self.get(symbol, key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        feat = compute()
        if feat is not None and not feat.empty:
            self.put(symbol, key, feat)
        return feat

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "errors": self.errors}
