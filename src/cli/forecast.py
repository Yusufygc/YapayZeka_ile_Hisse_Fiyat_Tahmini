#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src.cli.forecast - BIST-compliant forward forecast command.

Example:
    python -m src.cli.forecast --stocks TUPRS,ASELS --horizon-days 5
    python -m src.cli.forecast --stocks TUPRS --model Ridge\ Return
    python -m src.cli.forecast --stocks TUPRS --use-macro
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from typing import List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.forecasting.runner import ForecastRunner


def _parse_stocks(raw: str) -> List[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="BIST uyumlu ileri tahminleri SQLite DB'ye yazar.")
    parser.add_argument("--stocks", required=True, help="Virgulle ayrilmis semboller: TUPRS,ASELS")
    parser.add_argument("--horizon-days", type=int, default=5, help="Kac BIST islem gunu tahmin edilecek.")
    parser.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"), help="OHLCV CSV dizini.")
    parser.add_argument("--db-path", default=os.path.join(_PROJECT_ROOT, "data", "stock_models.db"), help="SQLite DB yolu.")
    parser.add_argument("--calendar-path", default=os.path.join(_PROJECT_ROOT, "data", "meta", "bist_calendar.csv"), help="BIST takvim CSV yolu.")
    parser.add_argument("--model", default=None, help="DB best_model yerine zorunlu model adi.")
    parser.add_argument("--use-macro", action="store_true", help="Forecast egitiminde makro ozellikleri opt-in ac.")
    parser.add_argument("--no-macro", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-update", action="store_true", help="Forecast oncesi hisse CSV guncellemesini kapat.")
    parser.add_argument("--verbose", action="store_true", help="Detayli pipeline/model loglarini goster.")
    parser.add_argument("--resolve", action="store_true", help="Mevcut OHLCV CSV kapanislariyla eski forecast'leri cozumle.")
    args = parser.parse_args()

    stocks = _parse_stocks(args.stocks)
    if not stocks:
        raise SystemExit("--stocks en az bir sembol icermeli.")

    runner = ForecastRunner(
        project_root=_PROJECT_ROOT,
        db_path=args.db_path,
        calendar_path=args.calendar_path,
    )

    failures = 0
    for symbol in stocks:
        data_file = os.path.join(args.data_dir, f"{symbol}.csv")
        if not os.path.exists(data_file):
            failures += 1
            print(f"[ERROR] {symbol}: veri dosyasi yok -> {data_file}")
            continue
        try:
            if args.resolve:
                resolved = runner.db.resolve_forecasts_from_csv(symbol, data_file)
                print(f"[OK] {symbol}: {resolved} forecast point cozumlendi.")
            if args.verbose:
                result = runner.run_symbol(
                    symbol=symbol,
                    data_file=data_file,
                    horizon_days=args.horizon_days,
                    force_model_name=args.model,
                    use_macro=bool(args.use_macro and not args.no_macro),
                    auto_update_data=not args.no_update,
                    auto_update_interactive=False,
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = runner.run_symbol(
                        symbol=symbol,
                        data_file=data_file,
                        horizon_days=args.horizon_days,
                        force_model_name=args.model,
                        use_macro=bool(args.use_macro and not args.no_macro),
                        auto_update_data=not args.no_update,
                        auto_update_interactive=False,
                    )
            print(
                f"[OK] {symbol}: run_id={result.run_id} model={result.model_name} "
                f"trend={result.trend_label} weekly_return={result.weekly_expected_return:.4%}"
            )
            for point in result.points:
                print(
                    "  "
                    f"h{point['horizon_index']} {point['target_date']} "
                    f"close={point['bounded_predicted_close']:.4f} "
                    f"ret={point['predicted_return']:.4%}"
                )
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {symbol}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
