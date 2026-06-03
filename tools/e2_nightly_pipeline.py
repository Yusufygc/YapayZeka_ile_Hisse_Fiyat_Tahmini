# -*- coding: utf-8 -*-
"""E2 Faz 8 — gecelik serving pipeline orkestratoru.

Tek giris noktasi: (a) BIST islem-gunu kapisi -> (b) evren-geneli artimli veri
tazeleme (yfinance) -> (c) global model skorlama batch'i (PeerStore guncelle).
Windows Task Scheduler her gun 03:00'te scripts/nightly_serving.ps1 ile cagirir.

Tasarim:
- Veri tazeleme mevcut `DataUpdater.check_and_update` (graceful, per-symbol)
  uzerine ince evren dongusu. Hata = devam; ozet sayim loglanir.
- Skorlama kanitli `tools/e2_faz5_nightly_scoring.py` aracini DEGISTIRMEDEN
  subprocess ile ayni yorumlayiciyla cagirir.
- Islem-gunu kapisi pandas-market-calendars (XIST). Fail-open: takvim hatasi ->
  yine de calis (bayat veriden iyidir).

Run (dl_env):
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1
    python tools/e2_nightly_pipeline.py --db data/serving_pool.db --boost 400

Bayraklar: --skip-data, --skip-trading-gate, --limit N (test/manuel).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta

# BIST seans kapanisi ~18:00. Bundan sonraki kosular AYNI gunun seansini hedefler;
# daha erken (sabah) kosular bir ONCEKI seansi (dun) hedefler.
_MARKET_CLOSE_HOUR = 19

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_updater import DataUpdater  # noqa: E402

# Loader ile ayni: panel disi yardimci CSV'ler sembol degil.
_EXCLUDE_STEMS = {"bist_universe", "advisory_history"}
_SCORING_TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "e2_faz5_nightly_scoring.py")


def _log(m: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------- trading gate
def is_trading_day(d: date, calendar: str = "XIST") -> bool:
    """d BIST (XIST) islem gunu mu?

    Birincil: pandas-market-calendars XIST takvimi (tatilleri de bilir).
    Fallback (lib/takvim hatasi): hafta-ici kontrolu (Pzt-Cum=True). Hafta
    sonlarini yine de keser; tatilleri kacirir ama bayat-veri/bos-kosudan iyidir.
    """
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar(calendar)
        days = cal.valid_days(start_date=(d - timedelta(days=7)).isoformat(),
                              end_date=d.isoformat())
        valid = {ts.date() for ts in days}
        return d in valid
    except Exception as exc:  # lib yok / takvim hatasi -> hafta-ici fallback
        _log(f"[WARN] XIST takvimi kullanilamadi ({exc}); hafta-ici fallback")
        return d.weekday() < 5


def gate_target_date(now: datetime | None = None) -> date:
    """Kapinin kontrol edecegi seans tarihi (saate gore).

    Aksam kosusu (>= _MARKET_CLOSE_HOUR, or. 21:00): bugunun seansi kapandi ve
    verisi hazir -> bugun. Sabah kosusu (< close): en son seans dun -> dun.
    """
    now = now or datetime.now()
    return now.date() if now.hour >= _MARKET_CLOSE_HOUR else (now.date() - timedelta(days=1))


# --------------------------------------------------------------- data refresh
def _universe_symbols(data_dir: str) -> list[str]:
    out = []
    for p in glob.glob(os.path.join(data_dir, "*.csv")):
        stem = os.path.basename(p)[:-4]
        if stem not in _EXCLUDE_STEMS:
            out.append(stem)
    return sorted(out)


def refresh_universe(data_dir: str, *, sleep_s: float = 0.2,
                     limit: int = 0, progress_every: int = 50) -> dict:
    """Evrendeki her sembolu artimli tazele. Returns durum sayim sozlugu."""
    syms = _universe_symbols(data_dir)
    if limit > 0:
        syms = syms[:limit]
    counts = {"updated": 0, "up_to_date": 0, "skipped": 0, "failed": 0}
    rows_added = 0
    _log(f"veri tazeleme: {len(syms)} sembol")
    for i, sym in enumerate(syms, 1):
        path = os.path.join(data_dir, f"{sym}.csv")
        try:
            res = DataUpdater.check_and_update(path, sym, interactive=False)
            counts[res.status] = counts.get(res.status, 0) + 1
            rows_added += int(res.rows_added or 0)
        except Exception as exc:  # tek sembol patlamasi pipeline'i durdurmasin
            counts["failed"] += 1
            _log(f"[WARN] {sym} tazeleme hatasi: {str(exc)[:80]}")
        if sleep_s:
            time.sleep(sleep_s)
        if progress_every and i % progress_every == 0:
            _log(f"  ... {i}/{len(syms)} (updated={counts['updated']} "
                 f"up_to_date={counts['up_to_date']} failed={counts['failed']})")
    counts["rows_added"] = rows_added
    _log(f"veri tazeleme bitti: {counts}")
    return counts


# ------------------------------------------------------------------- scoring
def run_scoring(db: str, boost: int, data_dir: str, universe: str,
                limit: int = 0) -> int:
    """e2_faz5 skorlama aracini subprocess ile cagir. Returns exit kodu."""
    cmd = [sys.executable, _SCORING_TOOL, "--db", db, "--boost", str(boost),
           "--data-dir", data_dir, "--universe", universe]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    _log(f"skorlama baslatiliyor: {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd)
    _log(f"skorlama exit={proc.returncode}")
    return proc.returncode


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--universe", default="data/bist_universe.csv")
    ap.add_argument("--db", default="data/serving_pool.db")
    ap.add_argument("--boost", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep-s", type=float, default=0.2)
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-trading-gate", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    _log("=== gecelik serving pipeline basladi ===")

    # (a) islem-gunu kapisi: hedef seans saate gore (aksam=bugun, sabah=dun).
    if not args.skip_trading_gate:
        target = gate_target_date()
        if not is_trading_day(target):
            _log(f"skip: {target} BIST islem gunu degil (hafta sonu/tatil). "
                 f"cikis 0.")
            return 0
        _log(f"islem gunu: {target} -> devam")

    # (b) veri tazeleme
    if args.skip_data:
        _log("veri tazeleme atlandi (--skip-data)")
    else:
        refresh_universe(args.data_dir, sleep_s=args.sleep_s, limit=args.limit)

    # (c) skorlama
    rc = run_scoring(args.db, args.boost, args.data_dir, args.universe,
                     limit=args.limit)

    _log(f"=== pipeline bitti: exit={rc}, sure={time.time()-t0:.0f}s ===")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
