# -*- coding: utf-8 -*-
"""repair_symbol_csv.py - bozuk sembol OHLCV CSV satirlarini onarir.

Sorun: bazi dosyalarda (orn. ASELS, EREGL) satirlar basliktan BIR alan eksik
(6 alan, `Duzeltilmis_Kapanis`/`Adj Close` yok). Naif okuyucu Hacmi Adj Close
kolonuna kaydirir (Close = Hacim kontaminasyonu).

Onarim:
  * alan sayisi = baslik-1 olan satirlara Adj Close = Close eklenir,
  * tarih bicimi dosyanin cogunluk bicimine (iso / dd/mm/yyyy / dd.mm.yyyy)
    normalize edilir (parse_dates; takassiz),
  * degismeyen satirlar BAYT-BAYT korunur (BOM + satir sonu dahil),
  * degisen dosyanin orijinali `<backup-dir>/<SYM>.csv.bak_<YYYYMMDD>` olarak
    yedeklenir (mevcut yedegin ustune YAZILMAZ).
Onarilamayan (baska alan sayisi) satirlar dokunulmadan raporlanir.

Kullanim:
    python tools/repair_symbol_csv.py --dry-run            # yalniz tara + raporla
    python tools/repair_symbol_csv.py                      # data/*.csv onar
    python tools/repair_symbol_csv.py --symbols ASELS,EREGL --backup-suffix bak_20261115
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.dates import parse_dates  # noqa: E402

_DATE_NAMES = ("tarih", "date")
_CLOSE_NAMES = ("kapanış", "kapanis", "close")
_ADJ_NAMES = ("düzeltilmiş_kapanış", "duzeltilmis_kapanis", "adj close", "adj_close")
_FMT_PATTERNS = (
    ("iso", re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"), "%Y-%m-%d"),
    ("dmy/", re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"), "%d/%m/%Y"),
    ("dmy.", re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$"), "%d.%m.%Y"),
    ("dmy-", re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"), "%d-%m-%Y"),
)
_SKIP_STEMS = {"bist_universe", "advisory_history"}
_ISTANBUL = timezone(timedelta(hours=3))


@dataclass
class FileReport:
    path: str
    status: str = "ok"            # ok | repaired | would_repair | skipped | unfixable_rows
    wrong_rows: int = 0
    fixed_rows: int = 0
    date_normalized: int = 0
    unfixable_rows: list[int] = field(default_factory=list)   # 1-tabanli satir no
    majority_format: str = ""
    note: str = ""


def _find(header: list[str], names: tuple[str, ...]) -> int | None:
    low = [h.strip().lower() for h in header]
    for name in names:
        if name in low:
            return low.index(name)
    return None


def _classify(value: str) -> tuple[str, str] | None:
    v = value.strip()
    for name, pattern, fmt in _FMT_PATTERNS:
        if pattern.match(v):
            return name, fmt
    return None


def analyze_text(text: str, label: str = "") -> tuple[str, FileReport]:
    """Metni analiz et, onarilmis metni don (degisiklik yoksa ayni metin)."""
    rep = FileReport(path=label)
    lines = text.splitlines(keepends=True)
    if not lines:
        rep.status, rep.note = "skipped", "bos dosya"
        return text, rep
    header = next(csv.reader([lines[0]]))
    n = len(header)
    d_idx, c_idx, a_idx = (_find(header, _DATE_NAMES), _find(header, _CLOSE_NAMES),
                           _find(header, _ADJ_NAMES))
    if d_idx is None or c_idx is None:
        rep.status, rep.note = "skipped", "OHLCV olmayan dosya (Date/Close kolonu yok)"
        return text, rep

    parsed = [(i, next(csv.reader([ln])) if ln.strip() else None) for i, ln in enumerate(lines)]
    fmt_count: Counter = Counter()
    for i, row in parsed[1:]:
        if row is not None and len(row) >= 1:
            cls = _classify(row[d_idx]) if d_idx < len(row) else None
            if cls:
                fmt_count[cls] += 1
    majority = fmt_count.most_common(1)[0][0] if fmt_count else None
    rep.majority_format = majority[0] if majority else ""

    out = list(lines)
    for i, row in parsed[1:]:
        if row is None:
            continue
        changed = False
        if len(row) != n:
            rep.wrong_rows += 1
            if len(row) == n - 1 and a_idx is not None and c_idx < len(row):
                row = row[:a_idx] + [row[c_idx]] + row[a_idx:]
                rep.fixed_rows += 1
                changed = True
            else:
                rep.unfixable_rows.append(i + 1)
                continue
        if majority and d_idx < len(row):
            cls = _classify(row[d_idx])
            if cls and cls != majority:
                new = parse_dates([row[d_idx].strip()]).iloc[0]
                if new == new:  # NaT degil
                    row[d_idx] = new.strftime(majority[1])
                    rep.date_normalized += 1
                    changed = True
        if changed:
            eol = lines[i][len(lines[i].rstrip("\r\n")):]
            buf = io.StringIO()
            csv.writer(buf, lineterminator="").writerow(row)
            out[i] = buf.getvalue() + eol
    if rep.unfixable_rows:
        rep.status = "unfixable_rows"
    elif rep.fixed_rows or rep.date_normalized:
        rep.status = "would_repair"
    return "".join(out), rep


def process_file(path: str, backup_dir: str, suffix: str, dry_run: bool) -> FileReport:
    with open(path, "rb") as fh:
        raw = fh.read()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    new_text, rep = analyze_text(text, label=path)
    rep.path = path
    if new_text == text or rep.status == "skipped":
        return rep
    if dry_run:
        return rep
    stem = os.path.basename(path)
    os.makedirs(backup_dir, exist_ok=True)
    backup = os.path.join(backup_dir, f"{stem}.{suffix}")
    if os.path.exists(backup):
        rep.note = f"yedek zaten var, ustune yazilmadi: {backup}"
    else:
        shutil.copy2(path, backup)
        rep.note = f"yedek: {backup}"
    tmp = path + ".tmp_repair"
    with open(tmp, "wb") as fh:
        fh.write((b"\xef\xbb\xbf" if has_bom else b"") + new_text.encode("utf-8"))
    os.replace(tmp, path)
    if rep.status == "would_repair":
        rep.status = "repaired"
    return rep


def run(data_dir: str, symbols: list[str] | None, backup_dir: str, suffix: str,
        dry_run: bool) -> list[FileReport]:
    paths = sorted(p for p in (os.path.join(data_dir, f) for f in os.listdir(data_dir))
                   if p.endswith(".csv") and os.path.isfile(p))
    reports = []
    for p in paths:
        stem = os.path.basename(p)[:-4]
        if stem in _SKIP_STEMS or (symbols and stem.upper() not in symbols):
            continue
        reports.append(process_file(p, backup_dir, suffix, dry_run))
    return reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sembol CSV bozuk satir onarici")
    ap.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"))
    ap.add_argument("--backup-dir", default=None, help="varsayilan: <data-dir>/yedek")
    ap.add_argument("--backup-suffix", default=None,
                    help="varsayilan: bak_<bugun YYYYMMDD, Istanbul>")
    ap.add_argument("--symbols", default="", help="virgullu liste (bos = hepsi)")
    ap.add_argument("--dry-run", action="store_true", help="dosya yazma/yedekleme yapma")
    args = ap.parse_args(argv)
    suffix = args.backup_suffix or f"bak_{datetime.now(_ISTANBUL):%Y%m%d}"
    backup_dir = args.backup_dir or os.path.join(args.data_dir, "yedek")
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    reports = run(args.data_dir, syms, backup_dir, suffix, args.dry_run)
    flagged = [r for r in reports if r.status not in ("ok", "skipped")]
    for r in flagged:
        print(f"{os.path.basename(r.path)}: {r.status} yanlis_alan={r.wrong_rows} "
              f"duzeltilen={r.fixed_rows} tarih_normalize={r.date_normalized} "
              f"cogunluk_bicim={r.majority_format} onarilamayan={r.unfixable_rows[:5]} {r.note}")
    print(f"taranan={len(reports)} sorunlu={len(flagged)} dry_run={args.dry_run}")
    return 1 if any(r.status == "unfixable_rows" for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
