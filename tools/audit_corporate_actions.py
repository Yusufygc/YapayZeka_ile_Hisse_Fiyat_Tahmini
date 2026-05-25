# -*- coding: utf-8 -*-
"""
audit_corporate_actions.py - Corporate Action Anomali Tarayicisi.

Sprint 2 (2026-05-25) Plan A2.2:
  data/*.csv altindaki tum hisse CSV'lerini tarar; her hisse icin
  log_return hesaplar ve |log_return| > 0.30 olan gunleri raporlar.

Boyle gunler:
  - Bolunmus (split) hisse: 1:2 split -> log_return ≈ -0.69
  - Buyuk temettu dagitimi: ex-day fiyat sicrayisi
  - Veri hatasi: yfinance edge case, manuel duzeltme gerekli

Cikti:
  outputs/_audits/corporate_action_audit_{timestamp}.csv kolonlari:
    Symbol, Date, prev_close, close, log_return, abs_log_return,
    auto_adjust_active (always True after Sprint 2 A2.1),
    severity (high/extreme), notes

Kullanim:
    python tools/audit_corporate_actions.py
    python tools/audit_corporate_actions.py --universe data/bist_universe.csv
    python tools/audit_corporate_actions.py --threshold 0.20
    python tools/audit_corporate_actions.py --data-dir data --out outputs/_audits
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd


_DEFAULT_THRESHOLD = 0.30
_EXTREME_THRESHOLD = 0.50
_DATE_CANDIDATES = ("Date", "Tarih", "date")
_CLOSE_CANDIDATES = ("Close", "Kapanis", "Kapanış", "close", "Adj_Close")


def _find_column(columns, candidates):
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    for cand in candidates:
        hit = lookup.get(cand.strip().lower())
        if hit is not None:
            return hit
    return None


def _load_close_series(csv_path: str) -> Optional[pd.DataFrame]:
    """CSV'yi yukler, Date + Close kolonlarini standardize eder."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  [WARN] {os.path.basename(csv_path)} okunamadi: {exc}")
        return None
    if df.empty:
        return None

    date_col = _find_column(df.columns, _DATE_CANDIDATES)
    close_col = _find_column(df.columns, _CLOSE_CANDIDATES)
    if date_col is None or close_col is None:
        return None

    # Format "mixed" hem ISO (2024-01-15) hem TR (15/01/2024) parser eder.
    # dayfirst=True ISO datelerini bozar (2024-01-02 -> 2024-02-01).
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    except Exception:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        except Exception:
            return None

    df = df[[date_col, close_col]].copy()
    df.columns = ["Date", "Close"]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Date", "Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _symbol_from_path(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    stem, _ = os.path.splitext(base)
    return stem.upper()


def _filter_universe(csv_paths: List[str], universe_path: Optional[str]) -> List[str]:
    if universe_path is None or not os.path.exists(universe_path):
        return csv_paths
    try:
        uni = pd.read_csv(universe_path)
    except Exception as exc:
        print(f"  [WARN] universe okunamadi ({universe_path}): {exc}")
        return csv_paths
    sym_col = _find_column(uni.columns, ["symbol", "Symbol", "ticker", "Ticker"])
    if sym_col is None:
        return csv_paths
    allowed = {str(s).strip().upper() for s in uni[sym_col].dropna()}
    return [p for p in csv_paths if _symbol_from_path(p) in allowed]


def scan_corporate_actions(
    data_dir: str = "data",
    threshold: float = _DEFAULT_THRESHOLD,
    universe: Optional[str] = None,
) -> pd.DataFrame:
    """data_dir altindaki tum *.csv dosyalarini tarar, anomali satirlarini doner."""
    pattern = os.path.join(data_dir, "*.csv")
    csv_paths = sorted(glob.glob(pattern))
    csv_paths = _filter_universe(csv_paths, universe)
    if not csv_paths:
        print(f"  [WARN] {data_dir} altinda CSV yok (universe={universe}).")
        return pd.DataFrame()

    rows: List[dict] = []
    scanned = 0
    for path in csv_paths:
        df = _load_close_series(path)
        if df is None or len(df) < 2:
            continue
        scanned += 1
        symbol = _symbol_from_path(path)
        prev_close = df["Close"].shift(1)
        log_ret = np.log(df["Close"] / prev_close)
        abs_ret = log_ret.abs()
        mask = abs_ret >= threshold

        if not mask.any():
            continue

        flagged = df.loc[mask].copy()
        flagged["prev_close"] = prev_close.loc[mask].values
        flagged["log_return"] = log_ret.loc[mask].values
        flagged["abs_log_return"] = abs_ret.loc[mask].values
        flagged["Symbol"] = symbol
        flagged["auto_adjust_active"] = True  # Sprint 2 A2.1 sonrasi True
        flagged["severity"] = np.where(
            flagged["abs_log_return"] >= _EXTREME_THRESHOLD, "extreme", "high"
        )
        flagged["notes"] = np.where(
            flagged["log_return"] < 0,
            "potansiyel split veya buyuk temettu (-)",
            "potansiyel split tersine veya veri hatasi (+)",
        )
        rows.extend(flagged.to_dict("records"))

    if not rows:
        print(f"  [INFO] {scanned} hisse tarandi; threshold={threshold} ile anomali yok.")
        return pd.DataFrame()

    report = pd.DataFrame(rows)
    report = report[
        ["Symbol", "Date", "prev_close", "Close", "log_return", "abs_log_return",
         "auto_adjust_active", "severity", "notes"]
    ]
    report.rename(columns={"Close": "close"}, inplace=True)
    report.sort_values(["abs_log_return", "Symbol", "Date"], ascending=[False, True, True], inplace=True)
    print(f"  [OK] {scanned} hisse tarandi; {len(report)} anomali bulundu.")
    return report


def write_report(report: pd.DataFrame, out_dir: str) -> Optional[str]:
    if report.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"corporate_action_audit_{ts}.csv")
    report.to_csv(out_path, index=False, encoding="utf-8")
    latest_path = os.path.join(out_dir, "corporate_action_audit_latest.csv")
    report.to_csv(latest_path, index=False, encoding="utf-8")
    print(f"  [OK] Audit raporu: {out_path}")
    print(f"  [OK] Latest snapshot: {latest_path}")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Corporate action audit (split/temettu anomali tarayicisi)."
    )
    parser.add_argument("--data-dir", default="data", help="OHLCV CSV dizini")
    parser.add_argument(
        "--threshold", type=float, default=_DEFAULT_THRESHOLD,
        help=f"|log_return| esigi (default {_DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--universe", default=None, help="Hisse listesi CSV (opsiyonel)")
    parser.add_argument("--out", default="outputs/_audits", help="Cikti dizini")
    args = parser.parse_args(argv)

    report = scan_corporate_actions(
        data_dir=args.data_dir,
        threshold=float(args.threshold),
        universe=args.universe,
    )
    write_report(report, out_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
