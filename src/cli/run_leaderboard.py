#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build run-level leaderboard diagnostics from outputs/{SYMBOL}/runs."""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis.run_leaderboard import (
    DEFAULT_LONG_HISTORY_YEARS,
    DEFAULT_SHORT_HISTORY_YEARS,
    build_history_effect_summary,
    build_multi_symbol_leaderboard,
    build_run_leaderboard,
    build_sector_summary,
    leaderboard_to_records,
    list_symbols_with_runs,
)
from src.utils.reporting_utils import write_csv_and_aligned_view


def _parse_symbol_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run-level final holdout dayanıklılık leaderboard'u üretir."
    )
    parser.add_argument("--symbol", default=None, help="Tek sembol: ARDYZ")
    parser.add_argument(
        "--symbols", default=None, help="Virgülle ayrılmış semboller: ARDYZ,ASELS,LOGO"
    )
    parser.add_argument(
        "--all", action="store_true", help="outputs-base altındaki tüm runs klasörlerini tara."
    )
    parser.add_argument("--outputs-base", default=os.path.join(_PROJECT_ROOT, "outputs"))
    parser.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"))
    parser.add_argument(
        "--sector-file", default=os.path.join(_PROJECT_ROOT, "data", "bist_universe.csv")
    )
    parser.add_argument(
        "--min-history-years",
        type=float,
        default=None,
        help="Backward-compatible alias for --long-history-years; not an exclusion filter.",
    )
    parser.add_argument("--long-history-years", type=float, default=DEFAULT_LONG_HISTORY_YEARS)
    parser.add_argument("--short-history-years", type=float, default=DEFAULT_SHORT_HISTORY_YEARS)
    parser.add_argument("--out", default=None, help="Opsiyonel CSV çıktı yolu.")
    parser.add_argument(
        "--sector-summary-out", default=None, help="Opsiyonel sektör özet CSV çıktı yolu."
    )
    parser.add_argument(
        "--history-summary-out",
        default=None,
        help="Opsiyonel veri gecmisi etkisi ozet CSV cikti yolu.",
    )
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    args = parser.parse_args()

    symbols = _parse_symbol_list(args.symbols)
    if args.symbol:
        symbols.insert(0, args.symbol.upper())
    if args.all:
        symbols.extend(list_symbols_with_runs(args.outputs_base))
    symbols = sorted(set(symbols))
    if not symbols:
        parser.error("--symbol, --symbols veya --all seçeneklerinden biri gerekli.")

    build_kwargs = {
        "outputs_base": args.outputs_base,
        "data_dir": args.data_dir,
        "sector_file": args.sector_file,
        "min_history_years": args.min_history_years,
        "long_history_years": args.long_history_years,
        "short_history_years": args.short_history_years,
    }
    if len(symbols) == 1:
        frame = build_run_leaderboard(symbol=symbols[0], **build_kwargs)
    else:
        frame = build_multi_symbol_leaderboard(symbols=symbols, **build_kwargs)
    sector_summary = build_sector_summary(frame)
    history_summary = build_history_effect_summary(frame)

    if args.out:
        paths = write_csv_and_aligned_view(frame, args.out)
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
        print(
            json.dumps(
                {
                    "symbols": symbols,
                    "rows": len(frame),
                    "csv": paths["csv"],
                    "sector_summary_csv": summary_paths["csv"],
                    "history_effect_summary_csv": history_paths["csv"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.sector_summary_out:
        write_csv_and_aligned_view(sector_summary, args.sector_summary_out)
    if args.history_summary_out:
        write_csv_and_aligned_view(history_summary, args.history_summary_out)

    if args.format == "csv":
        print(frame.to_csv(sep=";", index=False), end="")
        return 0

    print(json.dumps(leaderboard_to_records(frame), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
