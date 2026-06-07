# -*- coding: utf-8 -*-
"""backfill_sectors.py - bist_universe.csv Sector kolonunu yfinance ile doldurur.

E2 Faz b: pooled global model'in kosullandirma (conditioning) feature'i icin her
hisseye tutarli bir sektor etiketi gerek. Faz 0 audit'i ~557 hissenin Sector'unun
bos oldugunu gosterdi (yalniz orijinal 28 kataloglu, hepsi farkli kisa etiketle).

Bu script her sembol icin yfinance `Ticker("{SYM}.IS").info`'dan GICS sektorunu
(ve industry'sini) ceker ve `data/bist_universe.csv`'nin `Sector` kolonunu TEK TIP
GICS sozcuk dagarciigiyla doldurur. `Sector_Index` (macro sektor-getiri feature'ini
besleyen 7-endeks alani) DOKUNULMAZ — sadece descriptive `Sector` guncellenir.

Cozemedigi sembol -> Sector="Unknown".

Kullanim:
    python tools/backfill_sectors.py                 # tum universe, uniform doldur
    python tools/backfill_sectors.py --only-blank    # yalniz bos Sector'lari doldur
    python tools/backfill_sectors.py --limit 5 --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yfinance as yf

_UNIVERSE = os.path.join(_PROJECT_ROOT, "data", "bist_universe.csv")


def _fetch_sector(symbol: str) -> tuple[str, str]:
    """(sector, industry) doner; bulunamazsa ('Unknown', '')."""
    try:
        info = yf.Ticker(f"{symbol}.IS").info or {}
    except Exception:
        return "Unknown", ""
    sector = str(info.get("sector") or "").strip() or "Unknown"
    industry = str(info.get("industry") or "").strip()
    return sector, industry


def _is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="bist_universe Sector backfill (yfinance)")
    p.add_argument("--universe", type=str, default=_UNIVERSE)
    p.add_argument("--only-blank", action="store_true", help="Yalniz bos Sector'lari doldur (varsayilan: tumunu uniform doldur)")
    p.add_argument("--sleep", type=float, default=0.4, help="Istekler arasi bekleme (rate-limit)")
    p.add_argument("--limit", type=int, default=0, help="Ilk N sembol (test)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.path.exists(args.universe):
        print(f"[BACKFILL] universe yok: {args.universe}")
        sys.exit(1)

    uni = pd.read_csv(args.universe, encoding="utf-8-sig")
    if "Sector" not in uni.columns:
        uni["Sector"] = ""

    targets = uni.copy()
    if args.only_blank:
        targets = targets[targets["Sector"].map(_is_blank)]
    if args.limit:
        targets = targets.head(args.limit)
    symbols = [str(s).strip().upper() for s in targets["Symbol"]]
    print(f"[BACKFILL] {len(symbols)}/{len(uni)} sembol islenecek | only_blank={args.only_blank} dry_run={args.dry_run}")

    resolved, unknown = 0, 0
    for i, sym in enumerate(symbols, 1):
        sector, industry = _fetch_sector(sym)
        if sector == "Unknown":
            unknown += 1
        else:
            resolved += 1
        if not args.dry_run:
            mask = uni["Symbol"].astype(str).str.upper() == sym
            uni.loc[mask, "Sector"] = sector
            # Industry yfinance'den geliyor ama universe semasinda kolon yok;
            # schema churn olmasin diye yalniz konsola yansitiliyor (Faz 3 feature).
        if i <= 10 or i % 50 == 0 or args.dry_run:
            print(f"  [{i}/{len(symbols)}] {sym:10s} -> {sector} / {industry}")
        time.sleep(args.sleep)

    print(f"[BACKFILL] resolved={resolved} unknown={unknown}")
    if args.dry_run:
        print("[BACKFILL] dry-run: universe YAZILMADI")
        return

    tmp = args.universe + ".tmp"
    uni.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, args.universe)
    print(f"[BACKFILL] universe guncellendi -> {args.universe}")
    print("[BACKFILL] sektor dagilimi:")
    print(uni["Sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
