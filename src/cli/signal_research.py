#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dry-run CLI for Plan 1 signal research tables."""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis.signal_research import (
    DEFAULT_LONG_HISTORY_YEARS,
    DEFAULT_SHORT_HISTORY_YEARS,
    build_decision_report,
    check_universe,
    ensure_data_for_symbols,
    parse_csv_list,
    plan_runs,
    run_research_matrix,
    summarize_completed_runs,
)
from src.utils.reporting_utils import write_csv_and_aligned_view


def _emit_frame(frame, *, output_format: str, out: str | None) -> None:
    if out:
        paths = write_csv_and_aligned_view(frame, out)
        print(json.dumps({"rows": len(frame), "csv": paths["csv"]}, ensure_ascii=False, indent=2))
        return
    if output_format == "csv":
        print(frame.to_csv(sep=";", index=False), end="")
        return
    safe = frame.astype(object).where(frame.notna(), None)
    print(json.dumps(safe.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan 1 sinyal araştırması dry-run araçları.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check-universe")
    check_parser.add_argument("--symbols", required=True)
    check_parser.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"))
    check_parser.add_argument(
        "--universe", default=os.path.join(_PROJECT_ROOT, "data", "bist_universe.csv")
    )
    check_parser.add_argument(
        "--min-history-years",
        type=float,
        default=None,
        help="Backward-compatible alias for --long-history-years; not an exclusion filter.",
    )
    check_parser.add_argument(
        "--long-history-years", type=float, default=DEFAULT_LONG_HISTORY_YEARS
    )
    check_parser.add_argument(
        "--short-history-years", type=float, default=DEFAULT_SHORT_HISTORY_YEARS
    )
    check_parser.add_argument("--out", default=None)
    check_parser.add_argument("--format", choices=["json", "csv"], default="json")

    ensure_parser = subparsers.add_parser("ensure-data")
    ensure_parser.add_argument("--symbols", required=True)
    ensure_parser.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"))
    ensure_parser.add_argument("--old-data-dir", default=os.path.join(_PROJECT_ROOT, "data", "old"))
    ensure_parser.add_argument("--no-restore", action="store_true")
    ensure_parser.add_argument("--out", default=None)
    ensure_parser.add_argument("--format", choices=["json", "csv"], default="json")

    plan_parser = subparsers.add_parser("plan-runs")
    plan_parser.add_argument("--symbols", required=True)
    plan_parser.add_argument("--models", default=None)
    plan_parser.add_argument("--policies", default=None)
    plan_parser.add_argument("--out", default=None)
    plan_parser.add_argument("--format", choices=["json", "csv"], default="json")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--symbols", required=True)
    run_parser.add_argument("--models", default="all")
    run_parser.add_argument("--policies", default="V0,V1,V2,V3,V4")
    run_parser.add_argument("--mode", default="walk_forward")
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--phase", default="plan1")
    run_parser.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"))
    run_parser.add_argument("--old-data-dir", default=os.path.join(_PROJECT_ROOT, "data", "old"))
    run_parser.add_argument("--outputs-base", default=os.path.join(_PROJECT_ROOT, "outputs"))
    run_parser.add_argument(
        "--universe", default=os.path.join(_PROJECT_ROOT, "data", "bist_universe.csv")
    )
    run_parser.add_argument("--out", default=None)
    run_parser.add_argument("--format", choices=["json", "csv"], default="json")

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--symbols", required=True)
    summarize_parser.add_argument("--outputs-base", default=os.path.join(_PROJECT_ROOT, "outputs"))
    summarize_parser.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"))
    summarize_parser.add_argument(
        "--sector-file", default=os.path.join(_PROJECT_ROOT, "data", "bist_universe.csv")
    )
    summarize_parser.add_argument(
        "--min-history-years",
        type=float,
        default=None,
        help="Backward-compatible alias for --long-history-years; not an exclusion filter.",
    )
    summarize_parser.add_argument(
        "--long-history-years", type=float, default=DEFAULT_LONG_HISTORY_YEARS
    )
    summarize_parser.add_argument(
        "--short-history-years", type=float, default=DEFAULT_SHORT_HISTORY_YEARS
    )
    summarize_parser.add_argument("--out", default=None)
    summarize_parser.add_argument("--sector-summary-out", default=None)
    summarize_parser.add_argument("--history-summary-out", default=None)
    summarize_parser.add_argument("--decision-report-out", default=None)
    summarize_parser.add_argument("--format", choices=["json", "csv"], default="json")

    args = parser.parse_args()

    if args.command == "check-universe":
        frame = check_universe(
            symbols=parse_csv_list(args.symbols),
            data_dir=args.data_dir,
            universe_file=args.universe,
            min_history_years=args.min_history_years,
            long_history_years=args.long_history_years,
            short_history_years=args.short_history_years,
        )
        _emit_frame(frame, output_format=args.format, out=args.out)
        return 0

    if args.command == "ensure-data":
        frame = ensure_data_for_symbols(
            symbols=parse_csv_list(args.symbols),
            data_dir=args.data_dir,
            old_data_dir=args.old_data_dir,
            restore_from_old=not args.no_restore,
        )
        _emit_frame(frame, output_format=args.format, out=args.out)
        return 0

    if args.command == "plan-runs":
        frame = plan_runs(
            symbols=parse_csv_list(args.symbols),
            models=parse_csv_list(args.models) or None,
            policies=parse_csv_list(args.policies) or None,
        )
        _emit_frame(frame, output_format=args.format, out=args.out)
        return 0

    if args.command == "run":
        frame = run_research_matrix(
            symbols=parse_csv_list(args.symbols),
            data_dir=args.data_dir,
            outputs_base=args.outputs_base,
            universe_file=args.universe,
            models=parse_csv_list(args.models) or None,
            policies=parse_csv_list(args.policies) or None,
            mode=args.mode,
            workers=args.workers,
            resume=args.resume,
            dry_run=args.dry_run,
            old_data_dir=args.old_data_dir,
            phase=args.phase,
        )
        _emit_frame(frame, output_format=args.format, out=args.out)
        return 0

    leaderboard, sector_summary, history_summary = summarize_completed_runs(
        outputs_base=args.outputs_base,
        symbols=parse_csv_list(args.symbols),
        data_dir=args.data_dir,
        sector_file=args.sector_file,
        min_history_years=args.min_history_years,
        long_history_years=args.long_history_years,
        short_history_years=args.short_history_years,
    )
    if args.out:
        paths = write_csv_and_aligned_view(leaderboard, args.out)
        summary_out = args.sector_summary_out
        if summary_out is None:
            root, ext = os.path.splitext(args.out)
            summary_out = f"{root}_sector_summary{ext or '.csv'}"
        summary_paths = write_csv_and_aligned_view(sector_summary, summary_out)
        history_out = args.history_summary_out
        if history_out is None:
            root, ext = os.path.splitext(args.out)
            history_out = f"{root}_history_effect_summary{ext or '.csv'}"
        history_paths = write_csv_and_aligned_view(history_summary, history_out)
        report_out = args.decision_report_out
        if report_out is None:
            report_out = os.path.join(
                os.path.dirname(paths["csv"]), "plan1_final_decision_report.md"
            )
        _write_text_report(
            report_out,
            build_decision_report(
                leaderboard=leaderboard,
                sector_summary=sector_summary,
                history_summary=history_summary,
            ),
        )
        print(
            json.dumps(
                {
                    "rows": len(leaderboard),
                    "csv": paths["csv"],
                    "sector_summary_csv": summary_paths["csv"],
                    "history_effect_summary_csv": history_paths["csv"],
                    "decision_report_md": report_out,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.history_summary_out:
        write_csv_and_aligned_view(history_summary, args.history_summary_out)
    if args.decision_report_out:
        _write_text_report(
            args.decision_report_out,
            build_decision_report(
                leaderboard=leaderboard,
                sector_summary=sector_summary,
                history_summary=history_summary,
            ),
        )
    _emit_frame(leaderboard, output_format=args.format, out=None)
    return 0


def _write_text_report(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
