# -*- coding: utf-8 -*-
"""prune_outputs.py - `outputs/` saklama (retention) araci.

Varsayilan DRY-RUN'dir: hicbir sey silinmez/yazilmaz, yalniz rapor basilir.
`--apply` ile gercekten uygulanir.

Korunanlar (her sembol icin):
  * en yeni N run (`--keep`, varsayilan 5),
  * SQLite `best_models` / `experiments` model_path'inde run_id'si gecen run'lar,
  * `latest/LATEST_RUN.json` isaretcisinin gosterdigi run,
  * manifest durumu "running" olan ve son 24 saatte degisen run'lar.

Opsiyonel:
  --slim-latest        eski tam-kopya latest/ dizinlerini inceltir + isaretci yazar
  --compress-existing  kalan sikistirilmamis .pkl dosyalarini joblib compress=3 ile
                       yeniden yazar (yuklenip dogrulanir, atomik degistirilir)
  --clean-tmp          proje kokundeki `.codex_tmp` dizinini siler

Kullanim:
    python tools/prune_outputs.py                       # dry-run, 5 run sakla
    python tools/prune_outputs.py --keep 3 --slim-latest --compress-existing
    python tools/prune_outputs.py --apply --keep 5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import output_retention as ret  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", default=os.path.join(_PROJECT_ROOT, "outputs"))
    p.add_argument("--db", default=os.path.join(_PROJECT_ROOT, "data", "stock_models.db"))
    p.add_argument("--keep", type=int, default=5, help="Sembol basina saklanacak en yeni run sayisi")
    p.add_argument("--symbols", default=None, help="Virgulle ayrilmis sembol listesi (varsayilan: hepsi)")
    p.add_argument("--apply", action="store_true", help="Gercekten uygula (varsayilan dry-run)")
    p.add_argument("--dry-run", action="store_true", help="Varsayilan; yalniz rapor (--apply ile birlikte verilemez)")
    p.add_argument("--slim-latest", action="store_true")
    p.add_argument("--compress-existing", action="store_true")
    p.add_argument("--min-compress-mb", type=float, default=1.0)
    p.add_argument("--clean-tmp", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.apply and args.dry_run:
        print("[HATA] --apply ve --dry-run birlikte kullanilamaz.")
        return 2
    if args.keep < 1:
        print("[HATA] --keep en az 1 olmali.")
        return 2
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    apply = bool(args.apply)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    outputs_dir = os.path.abspath(args.outputs_dir)
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] outputs={outputs_dir} keep={args.keep}")
    if not os.path.isdir(outputs_dir):
        print("[HATA] outputs dizini bulunamadi.")
        return 1

    before = ret.dir_size(outputs_dir)
    report = ret.plan_prune(outputs_dir, keep=args.keep, db_path=args.db, symbols=symbols)
    kept = [r for r in report.runs if r.keep]
    print(
        f"  Run toplam={len(report.runs)} sakla={len(kept)} sil={len(report.to_delete)} "
        f"-> bosalacak {ret.format_bytes(report.bytes_to_free)}"
    )
    by_reason: dict = {}
    for r in kept:
        by_reason[r.reason] = by_reason.get(r.reason, 0) + 1
    print("  Saklama nedenleri:", ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())) or "-")
    for entry in report.to_delete:
        print(f"    - sil {entry.symbol}/{entry.run_id} ({ret.format_bytes(entry.size)})")

    exit_code = 0
    if apply:
        ret.apply_prune(report, outputs_dir)
        print(f"  Silinen run={len(report.deleted)} basarisiz={len(report.failed)}")
        for msg in report.failed:
            print(f"    [HATA] {msg}")
        exit_code = 1 if report.failed else 0

    if args.slim_latest:
        slim = ret.slim_latest(outputs_dir, apply=apply, symbols=symbols)
        print(f"  latest inceltme: {slim['symbols']} sembol, {ret.format_bytes(slim['bytes'])}")

    if args.compress_existing:
        cands = ret.find_compress_candidates(
            outputs_dir, min_bytes=int(args.min_compress_mb * 1024 * 1024), symbols=symbols
        )
        if not apply:  # silinecek run'lardaki dosyalar sayilmaz
            doomed = tuple(os.path.abspath(e.path) + os.sep for e in report.to_delete)
            cands = [c for c in cands if not os.path.abspath(c).startswith(doomed)]
        total = sum(os.path.getsize(p) for p in cands)
        if apply:
            res = ret.compress_existing(cands)
            print(
                f"  Sikistirma: {res['files']}/{len(cands)} dosya, "
                f"kazanc {ret.format_bytes(res['bytes_saved'])}"
            )
            for msg in res["failed"]:
                print(f"    [UYARI] {msg}")
        else:
            est = ret.estimate_compressed_size(cands) if cands else 0
            print(
                f"  Sikistirma adayi: {len(cands)} dosya, {ret.format_bytes(total)} "
                f"(tahmini kazanc ~{ret.format_bytes(max(total - est, 0))})"
            )

    if args.clean_tmp:
        size = ret.clean_tmp(_PROJECT_ROOT, apply=apply)
        print(f"  .codex_tmp: {ret.format_bytes(size)} {'silindi' if apply else 'silinecek'}")

    if apply:
        after = ret.dir_size(outputs_dir)
        print(f"  outputs boyutu: {ret.format_bytes(before)} -> {ret.format_bytes(after)}")
    else:
        print("  (dry-run: hicbir sey degistirilmedi; uygulamak icin --apply)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
