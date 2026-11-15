# -*- coding: utf-8 -*-
"""Temizlenmis OHLCV icin turetilmis Parquet deposu (CSV'ler KAYNAK olarak kalir).

Sorun: panel kurulumu her kosuda ~585 CSV'yi okuyup `clean_ohlcv_csv` ile temizler
(~14 sn). Girdi degismediyse sonuc deterministiktir. Bu modul temizlenmis veriyi
artimli olarak tek bir Parquet'te tutar; CSV'ler (DataUpdater/GUI/araclar) yine
tek dogruluk kaynagidir, depo yalniz turevdir ve silinirse kendiliginden yeniden kurulur.

Duzen (`root`, varsayilan `data/clean`):
  panel.parquet     uzun format: symbol, Date, Open, High, Low, Close, Volume
                    (sembol-sirali, sembol ici orijinal satir sirasi; clean_ohlcv_csv
                    ciktisinin `drop_zero_volume=True` hali — panel yolunun kullandigi).
  manifest.parquet  sembol basina: source_path, source_sha256, source_size,
                    source_mtime_ns, n_rows, first_date, last_date, rejected_rows,
                    cleaned_at, schema (kolon:dtype listesi).
  panel.parquet.lock  surecler-arasi kilit (`csv_safety.csv_lock`).

Artimli yenileme (`refresh`): once boyut+mtime_ns (hizli yol); uyusmazsa sha256;
yalniz sha degisen/yeni semboller yeniden temizlenir, kalkanlar dusurulur. Panel ve
manifest gecici dosya + `os.replace` ile yazilir (manifest SON = commit noktasi);
ikisini bagdastiran `panel_id` Parquet metadata'sindadir. Temizleyici kod
parmak izi (`ohlcv_clean.py` + `dates.py` kapanisi) veya depo bicim surumu degisirse
ya da dosyalar bozuksa depo UYARIYLA bastan kurulur.

Birebirlik: `load()` sembol cercevesi, `clean_ohlcv_csv(path)` ciktisinin standart
OHLCV kolonlarina izdusumuyle (`assert_frame_equal(check_exact=True)`) aynidir: ayni
RangeIndex, kolon sirasi, dtype (manifest `schema`), `attrs["rejected_rows"]`
(manifestten). Standart disi kolonlar (Dividends vb.) depolanmaz; panel yolu zaten
yalniz standart OHLCV'yi kullanir.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.csv_safety import csv_lock
from src.data.ohlcv_clean import clean_ohlcv_csv
from src.features.code_fingerprint import code_fingerprint

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"                      # depo BICIMI degisirse elle artir (kod degisimi otomatik)
STD_COLS = ("Date", "Open", "High", "Low", "Close", "Volume")
_CLEANER_ROOTS = ("src/data/ohlcv_clean.py",)     # + transitif src.* (dates.py)
_CLEANER_LIBS = ("pandas", "numpy")
_ID_KEY = b"tsfl.panel_id"
_VER_KEY = b"tsfl.schema_version"
_FP_KEY = b"tsfl.cleaner_fingerprint"
_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, pa.ArrowException)
_MAX_EXACT_INT = 2 ** 53                  # float64'te tam temsil edilen tam sayi siniri
_HASH_CHUNK = 1 << 20

PANEL_SCHEMA = pa.schema([
    ("symbol", pa.string()), ("Date", pa.timestamp("ns")),
    ("Open", pa.float64()), ("High", pa.float64()), ("Low", pa.float64()),
    ("Close", pa.float64()), ("Volume", pa.float64())])
MANIFEST_SCHEMA = pa.schema([
    ("symbol", pa.string()), ("source_path", pa.string()), ("source_sha256", pa.string()),
    ("source_size", pa.int64()), ("source_mtime_ns", pa.int64()), ("n_rows", pa.int64()),
    ("first_date", pa.timestamp("ns")), ("last_date", pa.timestamp("ns")),
    ("rejected_rows", pa.int64()), ("cleaned_at", pa.string()), ("schema", pa.string())])


class CleanStoreError(RuntimeError):
    """Depo okunamiyor/tutarsiz (cagiran once `refresh` etmeli)."""


@dataclass
class RefreshReport:
    unchanged: int = 0
    changed: int = 0          # kaynak sha'si degisti -> yeniden temizlendi
    new: int = 0
    removed: int = 0
    failed: list[str] = field(default_factory=list)    # temizlenemeyen (depoda YOK) semboller
    hashed: int = 0           # sha256 hesaplanan dosya (hizli yol kacirildi)
    rebuilt: str | None = None  # tam yeniden kurulum nedeni (None = artimli)
    seconds: float = 0.0

    def summary(self) -> str:
        base = (f"unchanged={self.unchanged} changed={self.changed} new={self.new} "
                f"removed={self.removed} failed={len(self.failed)} hashed={self.hashed} "
                f"in {self.seconds:.2f}s")
        return f"{base} (tam kurulum: {self.rebuilt})" if self.rebuilt else base


def cleaner_fingerprint() -> str:
    """Temizleyici kodun parmak izi (kaynak kapanisi + pandas/numpy surumleri)."""
    return code_fingerprint(_CLEANER_ROOTS, _CLEANER_LIBS)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _dtype_str(df: pd.DataFrame, cols: Sequence[str]) -> str:
    return ";".join(f"{c}:{df[c].dtype}" for c in cols)


def _parse_schema(spec: str) -> list[tuple[str, str]]:
    return [tuple(item.split(":", 1)) for item in spec.split(";")] if spec else []  # type: ignore[misc]


def _frame_to_table(symbol: str, df: pd.DataFrame, cols: Sequence[str]) -> pa.Table:
    """Temiz cerceve -> PANEL_SCHEMA tablosu. Tam temsil edilemeyen dtype/deger -> ValueError."""
    n = len(df)
    arrays: list[pa.Array] = [pa.array([symbol] * n, type=pa.string())]
    for name in STD_COLS:
        if name not in cols:
            arrays.append(pa.nulls(n, PANEL_SCHEMA.field(name).type))
            continue
        col = df[name]
        if name == "Date":
            if str(col.dtype) != "datetime64[ns]":
                raise ValueError(f"{symbol}: Date dtype {col.dtype} desteklenmiyor")
            arrays.append(pa.array(col.to_numpy(), type=pa.timestamp("ns"), from_pandas=True))
            continue
        if col.dtype.kind == "i" and n and int(np.abs(col.to_numpy()).max()) > _MAX_EXACT_INT:
            raise ValueError(f"{symbol}: {name} tam sayisi float64'te tam temsil edilemez")
        if col.dtype.kind not in "if":
            raise ValueError(f"{symbol}: {name} dtype {col.dtype} desteklenmiyor")
        arrays.append(pa.array(col.to_numpy(dtype="float64"), type=pa.float64(), from_pandas=True))
    return pa.Table.from_arrays(arrays, schema=PANEL_SCHEMA)


def _table_to_frame(tbl: pa.Table, entry: Mapping) -> pd.DataFrame:
    """Sembol dilimi -> clean_ohlcv_csv ciktisiyla ozdes cerceve (kolon/dtype manifestten)."""
    spec = _parse_schema(str(entry["schema"]))
    if not spec:
        return pd.DataFrame()
    pdf = tbl.select([name for name, _ in spec]).to_pandas().copy()
    for name, dtype in spec:
        if str(pdf[name].dtype) != dtype:
            pdf[name] = pdf[name].astype(dtype)
    pdf.attrs["rejected_rows"] = int(entry["rejected_rows"])
    return pdf


class CleanStore:
    """Temizlenmis OHLCV Parquet deposu. Bkz. modul dokumani."""

    def __init__(self, root: str = os.path.join("data", "clean")) -> None:
        self.root = root
        self.panel_path = os.path.join(root, "panel.parquet")
        self.manifest_path = os.path.join(root, "manifest.parquet")

    # ------------------------------------------------------------------ public
    def refresh(self, paths: Mapping[str, str]) -> RefreshReport:
        """Depoyu `paths` ({sembol: csv}) ile esitle; yalniz degisenleri yeniden temizle."""
        t0 = time.perf_counter()
        rep = RefreshReport()
        os.makedirs(self.root, exist_ok=True)
        with csv_lock(self.panel_path):
            entries, panel_id, rep.rebuilt = self._read_manifest()
            plan = self._classify(paths, entries, rep)
            dirty = bool(plan["clean"] or plan["removed"])
            slices: dict[str, pa.Table] = {}
            if entries and dirty:
                try:
                    slices = self._read_slices(entries, set(plan["keep"]))
                except _READ_ERRORS as exc:
                    logger.warning("clean_store: panel okunamadi (%s); depo bastan kuruluyor", exc)
                    entries, rep.rebuilt = {}, f"panel bozuk: {exc}"
                    rep.unchanged = rep.hashed = 0
                    plan = self._classify(paths, entries, rep)
            fresh, tables = self._clean_pending(plan["clean"], rep)
            rep.removed = len(plan["removed"])
            new_entries = {**plan["keep"], **fresh}
            tables.update(slices)
            if dirty or rep.rebuilt:
                panel_id = uuid.uuid4().hex
                self._write_panel(tables, new_entries, panel_id)
                self._write_manifest(new_entries, panel_id)
            elif plan["touched"]:
                self._write_manifest(new_entries, panel_id)
        rep.seconds = time.perf_counter() - t0
        logger.info("clean_store refresh: %s", rep.summary())
        return rep

    def load(self, symbols: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
        """{sembol: cerceve}; depoda olmayan semboller yok sayilir. Once `refresh` gerekir."""
        with csv_lock(self.panel_path):
            entries, _, reason = self._read_manifest()
            if reason:
                raise CleanStoreError(f"depo kullanilamaz ({reason or 'bos'}): {self.root}")
            want = set(entries) if symbols is None else set(symbols) & set(entries)
            try:
                slices = self._read_slices(entries, want)
            except _READ_ERRORS as exc:
                raise CleanStoreError(f"panel okunamadi: {exc}") from exc
        return {s: _table_to_frame(slices[s], entries[s]) for s in sorted(slices)}

    def content_hash(self, symbols: Sequence[str] | None = None) -> str:
        """sha256: (sembol, kaynak sha256) siralisi + bicim surumu + temizleyici parmak izi."""
        with csv_lock(self.panel_path):
            entries, _, reason = self._read_manifest()
        if reason:
            raise CleanStoreError(f"depo kullanilamaz ({reason}): {self.root}")
        keys = sorted(entries if symbols is None else set(symbols) & set(entries))
        h = hashlib.sha256()
        h.update(f"v{SCHEMA_VERSION}|{cleaner_fingerprint()}".encode())
        for s in keys:
            h.update(f"\n{s}\0{entries[s]['source_sha256']}".encode())
        return h.hexdigest()

    # ---------------------------------------------------------------- internals
    def _read_manifest(self) -> tuple[dict[str, dict], str, str | None]:
        """(entries, panel_id, rebuild_reason). Gecersiz/uyumsuz depo -> ({}, '', neden)."""
        if not os.path.exists(self.manifest_path) and not os.path.exists(self.panel_path):
            return {}, "", "yeni depo"
        try:
            meta = pq.read_schema(self.manifest_path).metadata or {}
            if meta.get(_VER_KEY) != SCHEMA_VERSION.encode():
                return {}, "", "depo bicim surumu degisti"
            if meta.get(_FP_KEY) != cleaner_fingerprint().encode():
                return {}, "", "temizleyici kod parmak izi degisti"
            rows = pq.read_table(self.manifest_path).to_pylist()
            entries = {r["symbol"]: r for r in rows}
            pmeta = pq.read_schema(self.panel_path).metadata or {}
            if pmeta.get(_ID_KEY) != meta.get(_ID_KEY):
                raise ValueError("panel_id uyusmuyor (yarim kalmis yazim?)")
            if pq.read_metadata(self.panel_path).num_rows != sum(r["n_rows"] for r in rows):
                raise ValueError("panel satir sayisi manifestle uyusmuyor")
            return entries, (meta[_ID_KEY] or b"").decode(), None
        except _READ_ERRORS as exc:
            logger.warning("clean_store: depo bozuk/okunamadi (%s); bastan kurulacak: %s",
                           exc, self.root)
            return {}, "", f"bozuk depo: {exc}"

    def _classify(self, paths: Mapping[str, str], entries: dict[str, dict],
                  rep: RefreshReport) -> dict:
        """Hizli yol (boyut+mtime_ns) -> sha256 -> {keep, clean, removed, touched}."""
        keep: dict[str, dict] = {}
        clean: dict[str, tuple[str, dict, bool]] = {}   # sym -> (path, stat/sha bilgisi, yeni_mi)
        touched = False
        for sym, path in paths.items():
            st = os.stat(path)
            old = entries.get(sym)
            src = _norm(path)
            if (old and old["source_path"] == src and old["source_size"] == st.st_size
                    and old["source_mtime_ns"] == st.st_mtime_ns):
                keep[sym] = old
                rep.unchanged += 1
                continue
            sha = file_sha256(path)
            rep.hashed += 1
            info = {"source_path": src, "source_sha256": sha, "source_size": st.st_size,
                    "source_mtime_ns": st.st_mtime_ns}
            if old and old["source_sha256"] == sha:      # icerik ayni: yalniz stat/yol tazele
                keep[sym] = {**old, **info}
                rep.unchanged += 1
                touched = True
            else:
                clean[sym] = (path, info, old is None)
        removed = sorted(set(entries) - set(paths))
        return {"keep": keep, "clean": clean, "removed": removed, "touched": touched}

    def _clean_pending(self, pending: Mapping[str, tuple[str, dict, bool]],
                       rep: RefreshReport) -> tuple[dict[str, dict], dict[str, pa.Table]]:
        entries: dict[str, dict] = {}
        tables: dict[str, pa.Table] = {}
        for sym, (path, info, is_new) in pending.items():
            try:
                df = clean_ohlcv_csv(path, label=sym)
                cols = [c for c in df.columns if c in STD_COLS]
                df = df[cols]
                table = _frame_to_table(sym, df, cols)
            except _READ_ERRORS + (pd.errors.ParserError, UnicodeError) as exc:
                # depoya girmez; cagiran (loader) dogrudan CSV'ye duser ve hatayi kendisi raporlar
                logger.warning("clean_store: %s temizlenemedi (%s); depoya alinmadi", sym, exc)
                rep.failed.append(sym)
                continue
            has_rows = bool(len(df)) and "Date" in df.columns
            entries[sym] = {
                **info, "symbol": sym, "n_rows": int(len(df)),
                "first_date": df["Date"].iloc[0] if has_rows else None,
                "last_date": df["Date"].iloc[-1] if has_rows else None,
                "rejected_rows": int(df.attrs.get("rejected_rows", 0)),
                "cleaned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "schema": _dtype_str(df, cols)}
            tables[sym] = table
            if is_new:
                rep.new += 1
            else:
                rep.changed += 1
        return entries, tables

    def _read_slices(self, entries: Mapping[str, dict], want) -> dict[str, pa.Table]:
        """Panel dosyasindan istenen sembollerin dilimleri; sinir tutarliligi dogrulanir."""
        table = pq.read_table(self.panel_path)
        order = sorted(entries)
        offs = np.concatenate([[0], np.cumsum([entries[s]["n_rows"] for s in order])])
        if int(offs[-1]) != table.num_rows:
            raise ValueError("panel satir sayisi manifestle uyusmuyor")
        out: dict[str, pa.Table] = {}
        sym_col = table.column("symbol")
        for i, sym in enumerate(order):
            a, b = int(offs[i]), int(offs[i + 1])
            if sym not in want or a == b:
                if sym in want:
                    out[sym] = table.slice(a, 0)
                continue
            if sym_col[a].as_py() != sym or sym_col[b - 1].as_py() != sym:
                raise ValueError(f"panel sembol siniri {sym} icin tutarsiz")
            out[sym] = table.slice(a, b - a)
        return out

    def _write_panel(self, tables: Mapping[str, pa.Table], entries: Mapping[str, dict],
                     panel_id: str) -> None:
        parts = [tables[s].replace_schema_metadata(None) for s in sorted(entries)]
        table = (pa.concat_tables(parts) if parts else PANEL_SCHEMA.empty_table())
        table = table.replace_schema_metadata({_ID_KEY: panel_id.encode()})
        self._atomic_write(table, self.panel_path)

    def _write_manifest(self, entries: Mapping[str, dict], panel_id: str) -> None:
        rows = [{c: entries[s].get(c) for c in MANIFEST_SCHEMA.names} for s in sorted(entries)]
        table = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA).replace_schema_metadata({
            _ID_KEY: panel_id.encode(), _VER_KEY: SCHEMA_VERSION.encode(),
            _FP_KEY: cleaner_fingerprint().encode()})
        self._atomic_write(table, self.manifest_path)

    @staticmethod
    def _atomic_write(table: pa.Table, dest: str) -> None:
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(dest) + ".", suffix=".tmp",
                                   dir=os.path.dirname(os.path.abspath(dest)))
        os.close(fd)
        try:
            pq.write_table(table, tmp, compression="zstd")
            os.replace(tmp, dest)
        except BaseException:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError as rm_exc:
                    logger.warning("gecici dosya silinemedi (%s): %s", tmp, rm_exc)
            raise


def _norm(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def panel_content_hash(df: pd.DataFrame, cols: Sequence[str] = ("symbol", "Date", "Close", "target")) -> str:
    """Depo yokken gercek veri icerigi parmak izi: secili kolonlarin satir-hash'leri uzerinden sha256."""
    use = [c for c in cols if c in df.columns]
    h = hashlib.sha256(("|".join(use) + f"|{len(df)}").encode())
    h.update(pd.util.hash_pandas_object(df[use], index=False).to_numpy().tobytes())
    return h.hexdigest()
