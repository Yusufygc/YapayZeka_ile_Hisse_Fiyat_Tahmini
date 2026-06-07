# -*- coding: utf-8 -*-
"""Forward interval coverage karşılaştırma raporu (B2 vs conformal).

Çözümlenmiş forecast'ların ampirik kapsama oranı + ortalama band genişliğini
üreteç yöntemine (interval_method) göre tablolar. Tez "kalibrasyon kanıtı"
tablosunun (naive band %X vs conformal %Y, hedef %90) veri kaynağıdır.

Kullanım:
    python tools/interval_coverage_report.py --db data/stock_models.db
    python tools/interval_coverage_report.py --db data/stock_models.db --symbol GARAN
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Optional


_QUERY = """
SELECT method,
       AVG(coverage) AS avg_coverage,
       AVG(width)    AS avg_width,
       COUNT(*)      AS resolved_runs
FROM (
    SELECT fas.run_id,
           (SELECT fp.interval_method
              FROM forecast_points fp
             WHERE fp.run_id = fas.run_id
               AND fp.interval_method IS NOT NULL
             LIMIT 1)                AS method,
           fas.interval_coverage     AS coverage,
           fas.interval_avg_width    AS width
    FROM forecast_accuracy_summary fas
    JOIN forecast_runs fr ON fr.id = fas.run_id
    WHERE fas.interval_coverage IS NOT NULL
      {symbol_filter}
)
WHERE method IS NOT NULL
GROUP BY method
ORDER BY method
"""


def build_report(db_path: str, symbol: Optional[str] = None) -> list[dict]:
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"DB bulunamadı: {db_path}")
    symbol_filter = "AND fr.stock_symbol = ?" if symbol else ""
    sql = _QUERY.format(symbol_filter=symbol_filter)
    params = (symbol.upper(),) if symbol else ()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("Çözümlenmiş interval forecast'ı yok (önce forecast üret + resolve et).")
        return
    print(f"{'method':<16}{'avg_coverage(%)':>18}{'avg_width':>14}{'runs':>8}")
    print("-" * 56)
    for r in rows:
        cov = r["avg_coverage"]
        wid = r["avg_width"]
        print(
            f"{str(r['method']):<16}"
            f"{(f'{cov:.2f}' if cov is not None else '-'):>18}"
            f"{(f'{wid:.4f}' if wid is not None else '-'):>14}"
            f"{int(r['resolved_runs']):>8}"
        )
    print("\nNot: hedef nominal kapsama tipik %80 (residual_b2) / %90 (conformal).")
    print("Tez tablosu: coverage hedefe ne kadar yakın + band genişliği (dar = bilgili).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward interval coverage raporu (B2 vs conformal).")
    parser.add_argument("--db", default=os.path.join("data", "stock_models.db"), help="SQLite DB yolu.")
    parser.add_argument("--symbol", default=None, help="Tek sembolle sınırla (opsiyonel).")
    args = parser.parse_args()
    rows = build_report(args.db, args.symbol)
    _print_table(rows)


if __name__ == "__main__":
    main()
