# -*- coding: utf-8 -*-
"""Kol-B gecerlilik raporu (denetim K2 + K3).

Baslik IC/ICIR'i dort acidan olcer:
  1. secim penceresi vs FINAL HOLDOUT (K3)
  2. ortusmeli vs ortusmesiz (`--gap-days`) ornekleme (K3)
  3. tum evren vs TRADABLE alt kume (medyan TL ciro >= `--liq-floor-tl`) (K2)
  4. model vs kural-tabanli 5 gunluk ters-momentum kiyasi, likidite kantili kirilimiyla (K2)

AGIR ISLEM: tum evrende LightGBM x (6 fold + holdout). Hizli deneme icin
    python tools/e2_kolb_validity_report.py --limit 60 --boost 50
Tam koşu:
    python tools/e2_kolb_validity_report.py --boost 400
Cikti: outputs/_audits/kolb_validity_<tarih>.json ve .md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.pooled_dataset import build_pooled_dataset
from src.data.pooled_loader import PooledLoaderConfig
from src.models.global_pooled_model import GlobalPooledConfig, make_global_model_factory
from src.validation.baselines import ic_report, reversal_factor_predictions
from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward
from src.validation.pooled_oos import (
    PerSymbolOOSConfig,
    evaluate_per_symbol,
    split_ic_summary,
)


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _clean(obj):
    """NaN/inf -> None, numpy -> python (JSON uyumlu)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    try:
        f = float(obj)
    except (TypeError, ValueError):
        return obj if isinstance(obj, (str, bool)) or obj is None else str(obj)
    if isinstance(obj, bool):
        return obj
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f) if float(f).is_integer() and not isinstance(obj, float) else f


def _row(name: str, ic: dict | None) -> str:
    if not ic or not ic.get("n_days"):
        return f"| {name} | - | - | - | 0 |"
    return (f"| {name} | {ic['ic_mean']:+.4f} | {ic['icir']:.2f} | "
            f"{100 * ic['pct_positive']:.0f}% | {ic['n_days']} |")


def _markdown(rep: dict) -> str:
    L = ["# Kol-B gecerlilik raporu", "",
         f"- Tarih: {rep['generated_at']}  |  evren: {rep['n_symbols']} sembol, "
         f"{rep['n_rows']} etiketli satir  |  boost={rep['boost']}  |  tradable esigi: "
         f"{rep['liq_floor_tl']:,.0f} TL/gun",
         "", "## Secim penceresi vs final holdout (K3)", "",
         "| Bolum | IC | ICIR | %IC>0 | gun |", "|---|---|---|---|---|"]
    s = rep["split"]
    for k, lbl in [("selection", "Secim (ortusmeli)"),
                   ("selection_nonoverlap", "Secim (ortusmesiz)"),
                   ("holdout", "Holdout (ortusmeli)"),
                   ("holdout_nonoverlap", "Holdout (ortusmesiz)")]:
        L.append(_row(lbl, s.get(k)))
    for title, key in [("Model", "model"), ("Ters-momentum kiyasi (-past5)", "baseline")]:
        r = rep[key]
        L += ["", f"## {title} (secim penceresi) — K2", "",
              "| Kume | IC | ICIR | %IC>0 | gun |", "|---|---|---|---|---|",
              _row("Tum evren", r.get("all")),
              _row("Tum evren (ortusmesiz)", r.get("all_nonoverlap")),
              _row("Tradable", r.get("tradable")),
              _row("Tradable (ortusmesiz)", r.get("tradable_nonoverlap")), "",
              "| Likidite kantili (Q1=en az likit) | IC | ICIR | %IC>0 | gun |",
              "|---|---|---|---|---|"]
        for q in r.get("liquidity_quantiles", []):
            L.append(_row(q["segment"], {
                "ic_mean": q["ic_mean"], "icir": q["icir"],
                "pct_positive": q["pct_positive"], "n_days": q["n_days"]}))
    L += ["", "> Yorum: model IC'si tradable alt kumede ve ortusmesiz orneklemde belirgin dusuyorsa "
          "baslik rakam likit olmayan hisselerin mikro-yapisindan geliyordur; ters-momentum "
          "kiyasi ayni kantil desenini veriyorsa sinyalin bu kismi model-bagimsizdir.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--universe", default="data/bist_universe.csv")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--min-names", type=int, default=15)
    ap.add_argument("--boost", type=int, default=400)
    ap.add_argument("--liq-floor-tl", type=float, default=3_000_000.0)
    ap.add_argument("--gap-days", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fallback-dir", action="append", default=[],
                    help="birincil data-dir'de olmayan semboller icin yedek klasor (tekrarlanabilir), "
                         "ornek: --fallback-dir data/yedek (K7)")
    ap.add_argument("--quarantine-jumps", action=argparse.BooleanOptionalAction, default=True,
                    help="|1g log-getiri|>0.30 olcek kiriklari etrafindaki satirlari at (K8)")
    ap.add_argument("--feature-cache-dir", default=os.path.join("data", "feature_cache", "pooled"),
                    help="sembol basi Parquet ozellik onbellegi")
    ap.add_argument("--no-feature-cache", action="store_true", help="ozellik onbellegini kapat")
    ap.add_argument("--clean-store-dir", default=os.path.join("data", "clean"),
                    help="temizlenmis OHLCV Parquet deposu (CSV kaynak kalir)")
    ap.add_argument("--no-clean-store", action="store_true", help="Parquet temiz veri deposunu kapat")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="sembol-paralel ozellik hesabi isci sayisi: -1 otomatik (min(cpu-1, 8)), 1 sekansiyel; "
                         "sonuc n_jobs'tan bagimsiz birebirdir")
    ap.add_argument("--out-dir", default=os.path.join("outputs", "_audits"))
    args = ap.parse_args()

    t0 = time.time()
    ds = build_pooled_dataset(
        PooledLoaderConfig(data_dir=args.data_dir, universe_file=args.universe,
                           target_horizon=args.horizon, min_rows=args.min_rows,
                           fallback_dirs=tuple(args.fallback_dir),
                           quarantine_jumps=args.quarantine_jumps,
                           feature_cache_dir=None if args.no_feature_cache else args.feature_cache_dir,
                           clean_store_dir=None if args.no_clean_store else args.clean_store_dir,
                           n_jobs=args.n_jobs),
        min_names=args.min_names, limit=args.limit)
    aug = ds.aug
    _log(f"panel: {len(aug)} etiketli satir, {aug['symbol'].nunique()} sembol, "
         f"son etiketli gun {aug['Date'].max().date()}")

    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=63, n_windows=6,
        min_train_days=504)).split(aug)
    _log(f"{len(folds)} fold (holdout dahil) kosuluyor ...")
    res = evaluate_per_symbol(
        aug, folds,
        make_global_model_factory(ds.cat_idx, GlobalPooledConfig(num_boost_round=args.boost)),
        PerSymbolOOSConfig(feature_cols=ds.feats, target_col="target_cs", include_holdout=True))

    preds = res.predictions
    sel = preds[~preds["is_holdout"]]
    split = split_ic_summary(preds, gap_days=args.gap_days, min_names=10)
    model_rep = ic_report(sel, panel=aug, min_turnover_tl=args.liq_floor_tl,
                          gap_days=args.gap_days, min_names=10)
    base = reversal_factor_predictions(aug, window=5)
    keys = sel[["symbol", "Date"]].drop_duplicates()
    base = base.merge(keys, on=["symbol", "Date"], how="inner")
    base_rep = ic_report(base, panel=aug, min_turnover_tl=args.liq_floor_tl,
                         gap_days=args.gap_days, min_names=10)

    rep = _clean({
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_symbols": int(aug["symbol"].nunique()), "n_rows": int(len(aug)),
        "boost": args.boost, "liq_floor_tl": args.liq_floor_tl, "gap_days": args.gap_days,
        "split": split, "model": model_rep, "baseline": base_rep,
    })
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"kolb_validity_{datetime.now():%Y%m%d_%H%M}")
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write(_markdown(rep))
    _log(f"yazildi: {stem}.json / .md  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
