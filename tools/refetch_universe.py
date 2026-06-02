# -*- coding: utf-8 -*-
"""refetch_universe.py - Tum BIST universe'unu yfinance'den yeniden ceker.

E2 Faz 0.5: pooled global model oncesi veri tabanini tek tip + guncel hale getirir.
Faz 0 audit'i uc sorunu ortaya cikardi (docs/wiki/e2-pooled-global-model-epic.md):
  1. Freshness: 569 hisse 2025-10-24 snapshot'inda donmus.
  2. Karisik tarih formati (ISO / dd-mm / dd.mm).
  3. Universe metadata yalniz 28 sembol kataloglu.

Bu script her ticker icin yfinance'den TAM gecmisi cekip `data/{TICKER}.csv`'yi
TEK TIP ISO (%Y-%m-%d) tarih formatiyla SIFIRDAN yazar (append degil, overwrite)
ve her cekilen hisse icin `data/bist_universe.csv` satirini gunceller.

CSV semasi (mevcut pipeline ile ayni):
    Tarih,Açılış,Yüksek,Düşük,Kapanış,Düzeltilmiş_Kapanış,Hacim

auto_adjust=True ZORUNLU (YFinanceProvider invariant'i: split/temettu sizintisini
onler). Bu nedenle OHLC tamami split/temettu-duzeltilmis gelir; Kapanis ve
Duzeltilmis_Kapanis ayni adjusted close olur.

Kullanim:
    # Tum universe (data/*.csv + bist_universe birlesimini) yeniden cek:
    python tools/refetch_universe.py

    # Sadece belirli hisseler:
    python tools/refetch_universe.py --symbols EREGL,AKBNK,TUPRS

    # Once kucuk dene (ilk 5), gercek data'yi bozmadan gecici dizine:
    python tools/refetch_universe.py --limit 5 --data-dir _refetch_smoke

    # Ne cekilecegini goster, indirme yapma:
    python tools/refetch_universe.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.data_updater import YFinanceProvider  # auto_adjust=True invariant

CSV_COLUMNS = ["Tarih", "Açılış", "Yüksek", "Düşük", "Kapanış", "Düzeltilmiş_Kapanış", "Hacim"]
UNIVERSE_COLUMNS = [
    "Symbol", "Listed_Date", "Delisted_Date", "Status",
    "Source", "Sector", "Sector_Index", "Note",
]
_DEFAULT_START = "2015-01-01"
_EXCLUDE = {"bist_universe", "advisory_history"}
_FRESH_DAYS = 10  # son islem bu kadar gun once ise Active, degilse Inactive (halt/delist suphesi)


# --------------------------------------------------------------------------- #
#  Sembol listesi
# --------------------------------------------------------------------------- #

def _symbols_from_data(data_dir: str) -> list[str]:
    out = []
    for p in glob.glob(os.path.join(data_dir, "*.csv")):
        stem = os.path.basename(p)[:-4]
        if stem not in _EXCLUDE:
            out.append(stem)
    return out


def _symbols_from_universe(universe_path: str) -> list[str]:
    if not os.path.exists(universe_path):
        return []
    df = pd.read_csv(universe_path, encoding="utf-8-sig")
    return [str(s).strip().upper() for s in df.get("Symbol", [])]


def _resolve_symbols(args: argparse.Namespace, real_data_dir: str) -> list[str]:
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        syms = set(_symbols_from_data(real_data_dir)) | set(_symbols_from_universe(args.universe))
        syms = sorted(syms)
    if args.limit:
        syms = syms[: args.limit]
    return syms


# --------------------------------------------------------------------------- #
#  Tek hisse cekme + sema
# --------------------------------------------------------------------------- #

def _to_schema(frame: pd.DataFrame) -> pd.DataFrame | None:
    """yfinance (auto_adjust=True) frame'ini TR semaya, ISO tarihle cevirir."""
    if frame is None or frame.empty:
        return None
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    data.columns = [str(c).replace("Datetime", "Date") for c in data.columns]
    lower = {str(c).strip().lower(): c for c in data.columns}

    def col(name: str):
        return data[lower[name]] if name in lower else None

    date = pd.to_datetime(col("date"), errors="coerce")
    close = col("close")
    if date is None or close is None:
        return None
    out = pd.DataFrame({
        "Tarih": date.dt.strftime("%Y-%m-%d"),
        "Açılış": col("open"),
        "Yüksek": col("high"),
        "Düşük": col("low"),
        "Kapanış": close,
        "Düzeltilmiş_Kapanış": close,  # auto_adjust=True -> close zaten adjusted
        "Hacim": col("volume"),
    })
    out = out.dropna(subset=["Tarih", "Kapanış"]).drop_duplicates(subset=["Tarih"], keep="last")
    out = out.sort_values("Tarih").reset_index(drop=True)
    return out if not out.empty else None


def _fetch_one(provider: YFinanceProvider, symbol: str, start: str, end: str) -> pd.DataFrame | None:
    ticker = f"{symbol}.IS"
    frame = provider.download(ticker, start=start, end=end)
    return _to_schema(frame)


# --------------------------------------------------------------------------- #
#  Universe upsert
# --------------------------------------------------------------------------- #

def _load_universe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        for c in UNIVERSE_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df[UNIVERSE_COLUMNS].copy()
    return pd.DataFrame(columns=UNIVERSE_COLUMNS)


def _upsert_universe_row(uni: pd.DataFrame, symbol: str, df_csv: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    first_date = df_csv["Tarih"].iloc[0]
    last_date = pd.Timestamp(df_csv["Tarih"].iloc[-1])
    fresh = (today - last_date).days <= _FRESH_DAYS
    status = "Active" if fresh else "Inactive"
    note = "" if fresh else f"Son islem {last_date.date()}; guncel degil (halt/delist suphesi)."
    source = f"yfinance_refetch_{today.date()}"

    mask = uni["Symbol"].astype(str).str.upper() == symbol
    if mask.any():
        idx = uni.index[mask][0]
        uni.at[idx, "Listed_Date"] = first_date
        uni.at[idx, "Status"] = status
        uni.at[idx, "Source"] = source
        uni.at[idx, "Note"] = note or uni.at[idx, "Note"]
        # Sector / Sector_Index / Delisted_Date mevcutsa KORUNUR (elle girilmis olabilir)
    else:
        uni.loc[len(uni)] = {
            "Symbol": symbol, "Listed_Date": first_date, "Delisted_Date": "",
            "Status": status, "Source": source, "Sector": "", "Sector_Index": "",
            "Note": note,
        }
    return uni


def _write_universe(uni: pd.DataFrame, path: str) -> None:
    uni = uni.sort_values("Symbol").reset_index(drop=True)
    tmp = path + ".tmp"
    uni[UNIVERSE_COLUMNS].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIST universe yfinance yeniden cekme")
    p.add_argument("--symbols", type=str, default=None, help="Virgullu liste; bos ise data/ + universe birlesimi")
    p.add_argument("--data-dir", type=str, default="data", help="CSV cikti dizini (gecici test icin degistir)")
    p.add_argument("--universe", type=str, default=os.path.join(_PROJECT_ROOT, "data", "bist_universe.csv"))
    p.add_argument("--start", type=str, default=_DEFAULT_START)
    p.add_argument("--end", type=str, default=None, help="Varsayilan: yarin (bugun dahil)")
    p.add_argument("--min-rows", type=int, default=20, help="Bu satirdan az gelen hisse atlanir")
    p.add_argument("--sleep", type=float, default=0.3, help="Istekler arasi bekleme (rate-limit)")
    p.add_argument("--limit", type=int, default=0, help="Ilk N sembol (test)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    real_data_dir = os.path.join(_PROJECT_ROOT, "data")
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(_PROJECT_ROOT, args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    today = pd.Timestamp(datetime.today().date())
    end = args.end or (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    symbols = _resolve_symbols(args, real_data_dir)

    print(f"[REFETCH] {len(symbols)} sembol | {args.start} -> {end} | data_dir={data_dir}")
    if args.dry_run:
        print("[DRY-RUN] cekilecek semboller:", ", ".join(symbols))
        return

    provider = YFinanceProvider()
    uni = _load_universe(args.universe)
    ok, no_data, failed = [], [], []

    for i, sym in enumerate(symbols, 1):
        try:
            df_csv = _fetch_one(provider, sym, args.start, end)
        except Exception as exc:
            failed.append((sym, str(exc)[:120]))
            print(f"  [{i}/{len(symbols)}] {sym:10s} FAIL: {str(exc)[:80]}")
            time.sleep(args.sleep)
            continue
        if df_csv is None or len(df_csv) < args.min_rows:
            no_data.append(sym)
            print(f"  [{i}/{len(symbols)}] {sym:10s} no-data ({0 if df_csv is None else len(df_csv)} satir)")
            time.sleep(args.sleep)
            continue
        out_path = os.path.join(data_dir, f"{sym}.csv")
        df_csv.to_csv(out_path, index=False, encoding="utf-8-sig")
        uni = _upsert_universe_row(uni, sym, df_csv, today)
        ok.append(sym)
        print(f"  [{i}/{len(symbols)}] {sym:10s} OK {len(df_csv)} satir ({df_csv['Tarih'].iloc[0]} -> {df_csv['Tarih'].iloc[-1]})")
        time.sleep(args.sleep)

    # universe yalniz gercek data dizinine yazarken guncellensin (smoke temp dizinini kirletme)
    if os.path.abspath(data_dir) == os.path.abspath(real_data_dir):
        _write_universe(uni, args.universe)
        print(f"[REFETCH] universe guncellendi -> {args.universe}")
    else:
        print(f"[REFETCH] smoke modu (data_dir != data/): universe YAZILMADI")

    print(f"[REFETCH] ok={len(ok)} no_data={len(no_data)} failed={len(failed)}")
    if no_data:
        print(f"[REFETCH] no-data semboller (delist/gecersiz olabilir): {no_data[:30]}{' ...' if len(no_data)>30 else ''}")
    if failed:
        print(f"[REFETCH] failed: {[s for s,_ in failed][:30]}")


if __name__ == "__main__":
    main()
