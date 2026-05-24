# -*- coding: utf-8 -*-
"""
universe_sync.py — bist_universe.csv otomatik senkronizasyon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data/ klasorundeki hisse CSV dosyalarini tarayip eksik sembolleri
bist_universe.csv'ye ekler. Idempotent: mevcut satirlari korur, yalnizca
eksik sembolleri append eder. Survivorship bias kontrolu (data_services.py)
yanlis pozitif uyari uretmesin diye DataManager init sirasinda calistirilir.

Listed_Date = ilgili hissenin CSV'sindeki min(Tarih), Delisted_Date bos
(Active varsayim). Cross-process race icin filelock kullanilir; yoksa
basit retry-loop fallback devreye girer.
"""

from __future__ import annotations

import glob
import os
import random
import time
from typing import Iterable

import pandas as pd

try:
    from filelock import FileLock, Timeout as FileLockTimeout

    _HAS_FILELOCK = True
except ImportError:
    _HAS_FILELOCK = False
    FileLockTimeout = Exception


REQUIRED_COLUMNS = ("Symbol", "Listed_Date", "Delisted_Date", "Status", "Source", "Note")
OPTIONAL_COLUMNS = ("Sector", "Sector_Index")
DEFAULT_EXCLUDE_DIRS = ("macro", "meta", "feature_cache", "optuna")
DEFAULT_EXCLUDE_FILES = ("bist_universe.csv",)
_LOCK_TIMEOUT_SEC = 10
_RETRY_ATTEMPTS = 3
_RETRY_BASE_SLEEP_SEC = 0.05


def _read_csv_first_date(csv_path: str) -> str | None:
    try:
        df = pd.read_csv(csv_path, usecols=["Tarih"])
    except (ValueError, KeyError):
        try:
            df = pd.read_csv(csv_path, usecols=["Date"])
            df.rename(columns={"Date": "Tarih"}, inplace=True)
        except Exception:
            return None
    except Exception:
        return None

    if df.empty or "Tarih" not in df.columns:
        return None

    try:
        dates = pd.to_datetime(df["Tarih"], format="mixed", dayfirst=True, errors="coerce")
    except Exception:
        dates = pd.to_datetime(df["Tarih"], dayfirst=True, errors="coerce")

    dates = dates.dropna()
    if dates.empty:
        return None
    return dates.min().strftime("%Y-%m-%d")


def _enumerate_candidates(
    data_dir: str,
    exclude_dirs: Iterable[str],
    exclude_files: Iterable[str],
) -> list[tuple[str, str]]:
    excl_dirs_norm = {d.lower() for d in exclude_dirs}
    excl_files_norm = {f.lower() for f in exclude_files}
    out: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        fname = os.path.basename(path)
        if fname.lower() in excl_files_norm:
            continue
        parent = os.path.basename(os.path.dirname(path)).lower()
        if parent in excl_dirs_norm:
            continue
        symbol = os.path.splitext(fname)[0].upper()
        if not symbol:
            continue
        out.append((symbol, path))
    return out


def _load_universe_df(universe_path: str) -> tuple[pd.DataFrame | None, str | None]:
    if not os.path.exists(universe_path) or os.path.getsize(universe_path) == 0:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS[:-1], *OPTIONAL_COLUMNS, "Note"]), None
    try:
        df = pd.read_csv(universe_path)
    except Exception as exc:
        return None, f"read_failed: {exc}"
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, f"invalid_schema_missing: {','.join(missing)}"
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df, None


def _atomic_write(df: pd.DataFrame, universe_path: str) -> None:
    tmp_path = f"{universe_path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, universe_path)


def _write_with_lock(df: pd.DataFrame, universe_path: str) -> None:
    if _HAS_FILELOCK:
        lock_path = f"{universe_path}.lock"
        try:
            with FileLock(lock_path, timeout=_LOCK_TIMEOUT_SEC):
                _atomic_write(df, universe_path)
            return
        except FileLockTimeout:
            pass
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            _atomic_write(df, universe_path)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(_RETRY_BASE_SLEEP_SEC + random.random() * 0.15)
    if last_exc is not None:
        raise last_exc


def sync_universe(
    data_dir: str,
    universe_path: str,
    *,
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS,
    exclude_files: tuple[str, ...] = DEFAULT_EXCLUDE_FILES,
    dry_run: bool = False,
) -> dict:
    result: dict = {
        "path": universe_path,
        "added": [],
        "skipped": [],
        "errors": [],
        "dry_run": dry_run,
    }

    if not os.path.isdir(data_dir):
        result["errors"].append({"reason": "data_dir_missing", "path": data_dir})
        return result

    universe_df, schema_err = _load_universe_df(universe_path)
    if universe_df is None:
        result["errors"].append({"reason": "universe_schema", "detail": schema_err})
        return result

    existing = {str(s).strip().upper() for s in universe_df["Symbol"].dropna().tolist()}
    candidates = _enumerate_candidates(data_dir, exclude_dirs, exclude_files)

    new_rows: list[dict] = []
    for symbol, csv_path in candidates:
        if symbol in existing:
            result["skipped"].append(symbol)
            continue
        listed_date = _read_csv_first_date(csv_path)
        if listed_date is None:
            result["errors"].append({"symbol": symbol, "reason": "no_valid_date", "path": csv_path})
            continue
        rel_path = os.path.relpath(csv_path, start=os.path.dirname(universe_path)).replace(
            "\\", "/"
        )
        new_rows.append(
            {
                "Symbol": symbol,
                "Listed_Date": listed_date,
                "Delisted_Date": "",
                "Status": "Active",
                "Source": "auto-discovered",
                "Sector": "",
                "Sector_Index": "",
                "Note": f"Auto-populated from {rel_path} first trading date",
            }
        )
        result["added"].append(symbol)

    if not new_rows:
        print(
            f"  [UNIVERSE] up to date ({len(existing)} symbols, {len(result['skipped'])} matched)"
        )
        return result

    if dry_run:
        print(f"  [UNIVERSE] dry-run: {len(new_rows)} symbol eklenecek: {result['added']}")
        return result

    column_order = [*REQUIRED_COLUMNS[:-1], *OPTIONAL_COLUMNS, "Note"]
    merged = pd.concat(
        [universe_df, pd.DataFrame(new_rows, columns=column_order)],
        ignore_index=True,
    )
    merged = merged[[column for column in column_order if column in merged.columns]]

    try:
        _write_with_lock(merged, universe_path)
    except Exception as exc:
        result["errors"].append({"reason": "write_failed", "detail": str(exc)})
        return result

    print(f"  [UNIVERSE] auto-added {len(new_rows)} symbol: {result['added']}")
    return result
