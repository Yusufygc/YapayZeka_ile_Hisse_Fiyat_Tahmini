# -*- coding: utf-8 -*-
"""SQLite maintenance commands for AI_Core."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.database.stock_model_db import StockModelDB


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "stock_models.db"
DEFAULT_OUTPUTS_BASE = PROJECT_ROOT / "outputs"
_BACKTEST_SUFFIX_TO_VALIDATION = {
    "final_holdout": "final_holdout",
    "wf": "walk_forward",
    "latest": "single_split",
}
_BACKTEST_FIELD_MAP = {
    "RMSE_vs_benchmark": "rmse_vs_benchmark",
    "Net_Return": "net_return",
    "BuyHold_Return": "buyhold_return",
    "Max_Drawdown": "max_drawdown",
    "Trade_Count": "trade_count",
    "Signal_Diagnosis": "signal_diagnosis",
}


def _summary(db_path: Path) -> Dict[str, Any]:
    db = StockModelDB(str(db_path))
    schema = db.get_schema_status()
    return {
        "db_path": str(db_path),
        "schema_ok": schema["ok"],
        "missing_tables": schema["missing_tables"],
        "table_counts": schema["table_counts"],
    }


def backup_reset(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    db_path = Path(db_path)
    backup_path = None
    if db_path.exists():
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{db_path.stem}_{stamp}{db_path.suffix}"
        shutil.move(str(db_path), str(backup_path))
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + suffix)
            if sidecar.exists():
                shutil.move(str(sidecar), str(backup_path) + suffix)

    db = StockModelDB(str(db_path))
    schema = db.get_schema_status()
    return {
        "action": "backup-reset",
        "db_path": str(db_path),
        "backup_path": None if backup_path is None else str(backup_path),
        "schema_ok": schema["ok"],
        "missing_tables": schema["missing_tables"],
        "table_counts": schema["table_counts"],
    }


def _read_backtest_report(path: Path) -> pd.DataFrame:
    attempts = (
        {"sep": None, "engine": "python"},
        {"sep": ";"},
        {},
    )
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            frame = pd.read_csv(path, **kwargs)
            if len(frame.columns) == 1 and ";" in str(frame.columns[0]):
                continue
            frame = frame.copy()
            frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
            return frame
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def _validation_mode_from_report(path: Path) -> Optional[str]:
    stem = path.stem
    prefix = "backtest_report_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix):]
    return _BACKTEST_SUFFIX_TO_VALIDATION.get(suffix)


def _symbol_from_report(path: Path, outputs_base: Path) -> Optional[str]:
    try:
        relative = path.resolve().relative_to(outputs_base.resolve())
    except ValueError:
        return None
    return relative.parts[0].upper() if relative.parts else None


def _run_id_from_report(path: Path) -> Optional[str]:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "runs" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _clean_metric_value(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def backfill_run_metrics(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    outputs_base: Path = DEFAULT_OUTPUTS_BASE,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    db_path = Path(db_path)
    outputs_base = Path(outputs_base)
    db = StockModelDB(str(db_path))
    symbols = [symbol.upper()] if symbol else [
        path.name.upper()
        for path in outputs_base.iterdir()
        if path.is_dir() and (path / "runs").is_dir()
    ] if outputs_base.exists() else []

    scanned_reports = 0
    updated_rows = 0
    unmatched_rows = []
    failed_reports = []

    with db._connect() as conn:
        for current_symbol in symbols:
            for report_path in sorted((outputs_base / current_symbol / "runs").glob("*/csv/backtest_report_*.csv")):
                validation_mode = _validation_mode_from_report(report_path)
                run_id = _run_id_from_report(report_path)
                report_symbol = _symbol_from_report(report_path, outputs_base) or current_symbol
                if validation_mode is None or run_id is None:
                    continue
                scanned_reports += 1
                try:
                    frame = _read_backtest_report(report_path)
                except Exception as exc:
                    failed_reports.append({"path": str(report_path), "error": type(exc).__name__})
                    continue
                if "Model" not in frame.columns:
                    failed_reports.append({"path": str(report_path), "error": "missing Model column"})
                    continue

                for _, row in frame.iterrows():
                    model_name = str(row.get("Model", "")).strip()
                    if not model_name:
                        continue
                    values = {
                        db_col: _clean_metric_value(row.get(report_col))
                        for report_col, db_col in _BACKTEST_FIELD_MAP.items()
                    }
                    cursor = conn.execute(
                        """
                        UPDATE experiments
                        SET rmse_vs_benchmark = COALESCE(?, rmse_vs_benchmark),
                            net_return        = ?,
                            buyhold_return    = ?,
                            max_drawdown      = ?,
                            trade_count       = ?,
                            signal_diagnosis  = ?
                        WHERE stock_symbol = ?
                          AND run_id = ?
                          AND model_name = ?
                          AND validation_mode = ?
                        """,
                        (
                            values["rmse_vs_benchmark"],
                            values["net_return"],
                            values["buyhold_return"],
                            values["max_drawdown"],
                            values["trade_count"],
                            values["signal_diagnosis"],
                            report_symbol,
                            run_id,
                            model_name,
                            validation_mode,
                        ),
                    )
                    if cursor.rowcount:
                        updated_rows += int(cursor.rowcount)
                    else:
                        unmatched_rows.append({
                            "symbol": report_symbol,
                            "run_id": run_id,
                            "model_name": model_name,
                            "validation_mode": validation_mode,
                        })

        db.schema_repository.refresh_best_models_from_production_experiments(conn)

    summary = _summary(db_path)
    return {
        "action": "backfill-run-metrics",
        "db_path": str(db_path),
        "outputs_base": str(outputs_base),
        "symbols": symbols,
        "scanned_reports": scanned_reports,
        "updated_rows": updated_rows,
        "unmatched_rows": len(unmatched_rows),
        "unmatched_examples": unmatched_rows[:10],
        "failed_reports": failed_reports,
        "schema_ok": summary["schema_ok"],
        "table_counts": summary["table_counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI_Core SQLite maintenance")
    sub = parser.add_subparsers(dest="command", required=True)
    backup_parser = sub.add_parser("backup-reset", help="Backup current DB and create a fresh schema")
    backup_parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    summary_parser = sub.add_parser("summary", help="Print DB schema and table counts")
    summary_parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    backfill_parser = sub.add_parser("backfill-run-metrics", help="Backfill DB trade metrics from run backtest reports")
    backfill_parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    backfill_parser.add_argument("--outputs-base", default=str(DEFAULT_OUTPUTS_BASE))
    backfill_parser.add_argument("--symbol", default=None)
    args = parser.parse_args(argv)

    if args.command == "backup-reset":
        result = backup_reset(Path(args.db_path))
    elif args.command == "summary":
        result = _summary(Path(args.db_path))
    elif args.command == "backfill-run-metrics":
        result = backfill_run_metrics(
            db_path=Path(args.db_path),
            outputs_base=Path(args.outputs_base),
            symbol=args.symbol,
        )
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
