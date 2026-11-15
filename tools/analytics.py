# -*- coding: utf-8 -*-
"""DuckDB analiz katmanı CLI'si (salt-okunur; bkz. docs/wiki/analytics-layer.md).

Kullanım:
    python tools/analytics.py list
    python tools/analytics.py views
    python tools/analytics.py query leaderboard --format md
    python tools/analytics.py query run_status --symbol THYAO
    python tools/analytics.py query forecast_accuracy --format csv --out out/acc.csv
    python tools/analytics.py sql "SELECT symbol, count(*) n FROM runs GROUP BY 1"
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis.analytics_db import AnalyticsDB, AnalyticsError  # noqa: E402
from src.analysis.analytics_queries import QUERIES, resolve_sql  # noqa: E402


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty and len(frame.columns) == 0:
        return ""
    cols = [str(c) for c in frame.columns]
    rows = [[("" if pd.isna(v) else str(v)).replace("|", "\\|") for v in rec] for rec in frame.itertuples(index=False)]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render(frame: pd.DataFrame, fmt: str) -> str:
    if fmt == "csv":
        return frame.to_csv(index=False)
    if fmt == "md":
        return _to_markdown(frame)
    return frame.to_string(index=False) if not frame.empty else "(sonuç yok)"


def _emit(frame: pd.DataFrame, fmt: str, out: Optional[str]) -> None:
    text = render(frame, fmt)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")
        print(f"{len(frame)} satır -> {out}")
    else:
        print(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Salt-okunur DuckDB analiz katmanı.")
    parser.add_argument("--project-root", default=None, help="Proje kökü (varsayılan: bu repo).")
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--main-db", default=None)
    parser.add_argument("--serving-db", default=None)
    parser.add_argument("--clean-store-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Adlandırılmış sorguları ve kullanılabilirliğini listele.")
    sub.add_parser("views", help="Kurulan view'ları ve satır sayılarını listele.")
    query = sub.add_parser("query", help="Adlandırılmış sorgu çalıştır.")
    query.add_argument("name", choices=sorted(QUERIES))
    query.add_argument("--symbol", default=None)
    query.add_argument("--format", choices=["table", "csv", "md"], default="table")
    query.add_argument("--out", default=None)
    raw = sub.add_parser("sql", help="Tek bir SELECT/WITH sorgusu çalıştır.")
    raw.add_argument("statement")
    raw.add_argument("--format", choices=["table", "csv", "md"], default="table")
    raw.add_argument("--out", default=None)
    return parser


def run(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    db = AnalyticsDB(
        project_root=args.project_root,
        outputs_dir=args.outputs_dir,
        main_db=args.main_db,
        serving_db=args.serving_db,
        clean_store_dir=args.clean_store_dir,
    )
    try:
        if args.command == "list":
            views = set(db.views())
            rows = [
                {
                    "name": name,
                    "available": resolve_sql(spec, views) is not None,
                    "symbol_filter": spec.has_symbol,
                    "description": spec.description,
                }
                for name, spec in QUERIES.items()
            ]
            _emit(pd.DataFrame(rows), "table", None)
        elif args.command == "views":
            rows = [
                {"view": v, "rows": int(db.sql(f"SELECT count(*) AS n FROM {v}").iloc[0, 0])}
                for v in db.views()
            ]
            _emit(pd.DataFrame(rows, columns=["view", "rows"]), "table", None)
        elif args.command == "query":
            _emit(db.query(args.name, symbol=args.symbol), args.format, args.out)
        else:
            _emit(db.sql(args.statement), args.format, args.out)
    except AnalyticsError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 2
    finally:
        db.close()
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    return run()


if __name__ == "__main__":
    sys.exit(main())
