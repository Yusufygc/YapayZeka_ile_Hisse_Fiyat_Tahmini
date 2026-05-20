# -*- coding: utf-8 -*-
"""SQLite maintenance commands for AI_Core."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.database.stock_model_db import StockModelDB


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "stock_models.db"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI_Core SQLite maintenance")
    sub = parser.add_subparsers(dest="command", required=True)
    backup_parser = sub.add_parser("backup-reset", help="Backup current DB and create a fresh schema")
    backup_parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    summary_parser = sub.add_parser("summary", help="Print DB schema and table counts")
    summary_parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args(argv)

    if args.command == "backup-reset":
        result = backup_reset(Path(args.db_path))
    elif args.command == "summary":
        result = _summary(Path(args.db_path))
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
