# -*- coding: utf-8 -*-
"""DuckDB analiz katmanı için kaynak okuyucular (salt-okunur).

- `scan_runs`: outputs/{SYM}/runs/{RUN_ID}/ dizinlerini Python ile tarar
  (manifestler heterojen olduğundan JSON'u Python okur; DuckDB `read_json` değil).
- `read_sqlite_tables`: SQLite dosyasını `mode=ro` URI ile açar (yazma imkansız),
  tabloları pandas'a okur. DuckDB `sqlite_scanner` eklentisi KULLANILMAZ (ağdan
  indirme gerektirir); yalnız gömülü parquet/json yetenekleri kullanılır.
- `csv_view_sql`: rapor CSV'lerini `all_varchar` + dinamik `TRY_CAST` ile okuyan
  view SQL'i üretir (dosyalar arası tip tutarsızlığına dayanıklı).
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.database.connection import connect

logger = logging.getLogger(__name__)

SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,10}$")
REQUIRED_RUN_FILES = (
    "run_manifest.json",
    "csv/backtest_report_wf.csv",
    "csv/backtest_report_final_holdout.csv",
    "csv/validation_protocol_report.csv",
)
RUNNING_WINDOW_HOURS = 2.0

# Rapor CSV'lerinde sayısal OLMAYAN kolonlar (geri kalanı DOUBLE'a çevrilir).
REPORT_TEXT_COLS = frozenset({"Model", "Signal_Diagnosis", "Beats_BuyHold_NetReturn"})
EXPERIMENT_TEXT_COLS = frozenset(
    {
        "Timestamp", "Model", "Validation_Params", "Dataset_Hash", "Target_Mode",
        "Feature_Mode", "Scaling_Mode", "Date_Range", "Validation_Mode", "Run_ID",
    }
)

RUN_COLUMNS = [
    "symbol", "run_id", "run_dir", "status", "has_manifest", "manifest_valid",
    "has_wf_report", "has_final_report", "has_validation_report", "has_experiment_log",
    "generated_at", "started_at", "finished_at", "mode", "data_hash",
    "feature_pipeline_version", "model_config_hash", "signal_config_hash",
    "random_seed", "git_commit", "research_policy", "research_phase",
    "uses_final_holdout_for_selection", "final_holdout_status", "model_list", "n_models",
]


@dataclass
class RunScan:
    """`scan_runs` sonucu: run tablosu satırları + okunacak CSV dosyaları."""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    report_files: List[str] = field(default_factory=list)
    experiment_files: List[str] = field(default_factory=list)


def posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(text: str) -> str:
    """SQL string literal (tek tırnak kaçışlı). Yalnız kendi globladığımız yollar için."""
    return "'" + str(text).replace("'", "''") + "'"


def sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def scan_runs(outputs_dir: Path, running_window_hours: float = RUNNING_WINDOW_HOURS) -> RunScan:
    """outputs/{SYM}/runs/{RUN_ID} dizinlerini tarar; kaynak yoksa boş sonuç."""
    scan = RunScan()
    if not outputs_dir.is_dir():
        return scan
    now = datetime.now(timezone.utc)
    for sym_dir in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        runs_dir = sym_dir / "runs"
        if not runs_dir.is_dir():
            continue
        if not SYMBOL_RE.match(sym_dir.name.upper()):
            logger.warning("analytics: geçersiz sembol dizini atlandı: %s", sym_dir.name)
            continue
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            row = _run_row(sym_dir.name, run_dir, now, running_window_hours)
            scan.rows.append(row)
            for rel, flag, bucket in (
                ("csv/backtest_report_wf.csv", "has_wf_report", scan.report_files),
                ("csv/backtest_report_final_holdout.csv", "has_final_report", scan.report_files),
                ("experiments/experiment_log.csv", "has_experiment_log", scan.experiment_files),
            ):
                if row[flag]:
                    bucket.append(posix(run_dir / rel))
    return scan


def _run_row(symbol: str, run_dir: Path, now: datetime, window_hours: float) -> Dict[str, Any]:
    manifest, valid = _read_manifest(run_dir / "run_manifest.json")
    has_manifest = (run_dir / "run_manifest.json").exists()
    missing = [rel for rel in REQUIRED_RUN_FILES if not (run_dir / rel).exists()]
    models = [str(m) for m in (manifest.get("model_list") or []) if str(m).strip()]
    holdout = manifest.get("final_holdout_status")
    if isinstance(holdout, dict):
        holdout = holdout.get("status")
    row: Dict[str, Any] = {
        "symbol": symbol,
        "run_id": run_dir.name,
        "run_dir": posix(run_dir),
        "status": _derive_status(manifest, has_manifest, valid, missing, run_dir, now, window_hours),
        "has_manifest": has_manifest,
        "manifest_valid": valid,
        "has_wf_report": (run_dir / "csv/backtest_report_wf.csv").exists(),
        "has_final_report": (run_dir / "csv/backtest_report_final_holdout.csv").exists(),
        "has_validation_report": (run_dir / "csv/validation_protocol_report.csv").exists(),
        "has_experiment_log": (run_dir / "experiments/experiment_log.csv").exists(),
        "generated_at": _text(manifest.get("generated_at")),
        "started_at": _text(manifest.get("started_at")),
        "finished_at": _text(manifest.get("finished_at")),
        "mode": _text(manifest.get("validation_protocol")),
        "final_holdout_status": _text(holdout),
        "model_list": models,
        "n_models": len(models),
    }
    for key in (
        "data_hash", "feature_pipeline_version", "model_config_hash",
        "signal_config_hash", "git_commit", "research_policy", "research_phase",
    ):
        row[key] = _text(manifest.get(key))
    row["random_seed"] = manifest.get("random_seed") if isinstance(manifest.get("random_seed"), int) else None
    flag = manifest.get("uses_final_holdout_for_selection")
    row["uses_final_holdout_for_selection"] = flag if isinstance(flag, bool) else None
    return row


def _derive_status(
    manifest: Dict[str, Any],
    has_manifest: bool,
    valid: bool,
    missing: Sequence[str],
    run_dir: Path,
    now: datetime,
    window_hours: float,
) -> str:
    explicit = manifest.get("status")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    if not has_manifest:
        # Manifest yazılmadan kalan dizin: yakın zamanda değiştiyse hâlâ koşuyor olabilir.
        age = now - datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
        return "running" if age <= timedelta(hours=window_hours) else "failed"
    if not valid:
        return "failed"
    return "incomplete" if missing else "completed"


def _read_manifest(path: Path) -> tuple[Dict[str, Any], bool]:
    if not path.exists():
        return {}, False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("analytics: manifest okunamadı %s: %s", path, exc)
        return {}, False
    if not isinstance(data, dict):
        logger.warning("analytics: manifest nesne değil: %s", path)
        return {}, False
    return data, True


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def runs_frame(scan: RunScan) -> pd.DataFrame:
    return pd.DataFrame(scan.rows, columns=RUN_COLUMNS)


# --------------------------------------------------------------------------- CSV views
def _sniff_delimiter(path: str) -> str:
    with open(path, encoding="utf-8-sig") as handle:
        header = handle.readline()
    return ";" if header.count(";") > header.count(",") else ","


def _raw_select(files: Sequence[str]) -> str:
    groups: Dict[str, List[str]] = {}
    for path in files:
        groups.setdefault(_sniff_delimiter(path), []).append(path)
    parts = []
    for delim, paths in sorted(groups.items()):
        listing = "[" + ", ".join(sql_quote(p) for p in paths) + "]"
        parts.append(
            f"SELECT * FROM read_csv({listing}, delim={sql_quote(delim)}, header=true, "
            "all_varchar=true, union_by_name=true, filename=true, sample_size=-1)"
        )
    return " UNION ALL BY NAME ".join(parts)


def create_csv_view(con: Any, name: str, files: Sequence[str], kind: str) -> None:
    """`kind`: 'report' (symbol/run_id/split/model) veya 'experiment' (symbol/run_id)."""
    raw = f"_raw_{name}"
    con.execute(f"CREATE OR REPLACE VIEW {raw} AS {_raw_select(files)}")
    columns = [row[0] for row in con.execute(f"DESCRIBE {raw}").fetchall()]
    text_cols = REPORT_TEXT_COLS if kind == "report" else EXPERIMENT_TEXT_COLS
    casts = [
        f"NULLIF(TRY_CAST(replace({sql_ident(c)}, ',', '.') AS DOUBLE), 'NaN'::DOUBLE) AS {sql_ident(c)}"
        for c in columns
        if c != "filename" and c not in text_cols
    ]
    replace = f" REPLACE ({', '.join(casts)})" if casts else ""
    if kind == "report":
        dirs = r"([^/\\]+)[/\\]runs[/\\]([^/\\]+)[/\\]csv[/\\][^/\\]+$"
        head = (
            f"regexp_extract(filename, {sql_quote(dirs)}, 1) AS symbol, "
            f"regexp_extract(filename, {sql_quote(dirs)}, 2) AS run_id, "
            "regexp_extract(filename, 'backtest_report_(wf|final_holdout)[.]csv$', 1) AS split, "
            '"Model" AS model'
        )
    else:
        dirs = r"([^/\\]+)[/\\]runs[/\\]([^/\\]+)[/\\]experiments[/\\][^/\\]+$"
        head = (
            f"regexp_extract(filename, {sql_quote(dirs)}, 1) AS symbol, "
            f"regexp_extract(filename, {sql_quote(dirs)}, 2) AS run_id"
        )
    # Küçük (birkaç bin satır) anlık görüntü: her sorguda yüzlerce CSV'yi yeniden okumamak için
    # tabloya materyalize edilir (AnalyticsDB açıldığı andaki durumu yansıtır).
    con.execute(
        f"CREATE OR REPLACE TABLE {name} AS SELECT {head}, * EXCLUDE (filename){replace} FROM {raw}"
    )
    con.execute(f"DROP VIEW {raw}")


# --------------------------------------------------------------------------- SQLite bridge
def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """SQLite'ı salt-okunur açar (`file:...?mode=ro`); yazma girişimi OperationalError verir."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return connect(uri, uri=True)


def read_sqlite_tables(db_path: Path, tables: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """`tables`: {sqlite_tablo: view_adı}. Olmayan tablolar atlanır; DB yoksa boş sözlük."""
    if not db_path.is_file():
        return {}
    result: Dict[str, pd.DataFrame] = {}
    conn = open_readonly(db_path)
    try:
        present = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, view in tables.items():
            if table not in present:
                continue
            frame = pd.read_sql_query(f"SELECT * FROM {table}", conn)  # noqa: S608 (sabit allowlist)
            result[view] = _coerce_declared_types(frame, conn, table)
    finally:
        conn.close()
    return result


def _coerce_declared_types(frame: pd.DataFrame, conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Hepsi NULL kolonlar DuckDB'de tür belirsizliği yaratır; SQLite bildirimli türüne çevir."""
    for _, name, decl, *_ in conn.execute(f"PRAGMA table_info({table})").fetchall():
        if name not in frame.columns:
            continue
        kind = str(decl or "").upper()
        if "INT" in kind:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("Int64")
        elif any(tag in kind for tag in ("REAL", "FLOA", "DOUB")):
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
        else:
            frame[name] = frame[name].astype("string")
    return frame


__all__ = [
    "RunScan", "scan_runs", "runs_frame", "create_csv_view", "open_readonly",
    "read_sqlite_tables", "posix", "sql_quote",
]
