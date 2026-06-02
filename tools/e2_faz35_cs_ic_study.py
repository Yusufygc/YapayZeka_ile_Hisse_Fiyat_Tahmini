# -*- coding: utf-8 -*-
"""E2 Faz 3.5 — full-universe cross-sectional IC study.

Decisive test: does the pooled global model's cross-sectional IC (+0.09 on 39
symbols) hold on the FULL BIST universe? Wider cross-section usually stabilizes
IC. Compares absolute vs cross-sectional target with the same GlobalPooledModel
through the leakage-safe pooled CV + OOS harness.

Run (dl_env), background-friendly:
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1
    python tools/e2_faz35_cs_ic_study.py --min-rows 300 --boost 400

Outputs:
    outputs/e2_faz35_cs_ic_study.md         (summary)
    outputs/e2_faz35_per_symbol.csv         (per-symbol OOS, cross-sectional)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

# repo-root sys.path (python tools/x.py -> sys.path[0]=tools/, src bulunmaz)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cross_sectional import (
    add_cross_sectional_features,
    add_cross_sectional_target,
)
from src.data.pooled_loader import PooledLoaderConfig, PooledPanelLoader
from src.models.global_pooled_model import (
    GlobalPooledConfig,
    build_pooled_features,
    make_global_model_factory,
)
from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward
from src.validation.pooled_oos import PerSymbolOOSConfig, evaluate_per_symbol


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--universe", default="data/bist_universe.csv")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--min-names", type=int, default=15, help="cross-section gunluk min sembol")
    ap.add_argument("--window-len", type=int, default=63)
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--min-train-days", type=int, default=504)
    ap.add_argument("--boost", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="0=tum evren; >0 ilk N sembol")
    args = ap.parse_args()

    t0 = time.time()
    _log(f"loader: min_rows={args.min_rows} horizon={args.horizon}")
    loader = PooledPanelLoader(PooledLoaderConfig(
        data_dir=args.data_dir, universe_file=args.universe,
        target_horizon=args.horizon, min_rows=args.min_rows,
    ))
    if args.limit > 0:
        syms = loader._symbols()[: args.limit]
        loader._symbols = lambda: syms
    panel = loader.load()
    _log(f"panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols "
         f"(load {time.time()-t0:.0f}s)")

    # cross-sectional hedef
    panel = add_cross_sectional_target(
        panel, raw_target="target", out_col="target_cs",
        method="rank", min_names=args.min_names)
    _log(f"after cross-sectional filter: {len(panel)} rows, "
         f"{panel['symbol'].nunique()} symbols, {panel['Date'].nunique()} dates")

    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=args.window_len,
        n_windows=args.n_windows, min_train_days=args.min_train_days,
    )).split(panel)
    sel = [f for f in folds if not f.is_final_holdout]
    _log(f"folds: {len(folds)} ({len(sel)} selectable + holdout)")

    # Faz 3.6: mevcut causal ozelliklerin cross-sectional goreli versiyonu.
    aug0, feats0, _ = build_pooled_features(panel)
    cs_base = [c for c in feats0 if c not in ("symbol_id", "sector_code")]
    panel, cs_new = add_cross_sectional_features(panel, cs_base)
    _log(f"cross-sectional features added: {len(cs_new)} (base {len(cs_base)})")

    aug, feats, cat_idx = build_pooled_features(panel)
    assert not any(c.startswith("target") for c in feats), "target leak!"
    feats_nocs = [c for c in feats if not (c.endswith("_csr") or c.endswith("_csz"))]
    _log(f"features: {len(feats)} total, {len(feats_nocs)} without cs-feats "
         f"(cat at {cat_idx})")
    factory = make_global_model_factory(cat_idx, GlobalPooledConfig(num_boost_round=args.boost))

    variants = [
        ("ABSOLUTE", "target", feats_nocs),
        ("CROSS-SECTIONAL", "target_cs", feats_nocs),
        ("CS+CSFEAT", "target_cs", feats),
    ]
    results = {}
    for label, tcol, fcols in variants:
        _log(f"fitting GlobalPooledModel — {label} (target={tcol}, {len(fcols)} feats) ...")
        ts = time.time()
        res = evaluate_per_symbol(
            aug, folds, factory, PerSymbolOOSConfig(feature_cols=fcols, target_col=tcol))
        ic = res.ic
        rel = res.per_symbol[res.per_symbol["reliable"]]
        results[label] = (res, ic, rel)
        _log(f"  {label}: IC {ic['ic_mean']:+.4f} ICIR {ic['icir']:+.3f} "
             f"%IC>0 {100*ic['pct_positive']:.1f} n_days {ic['n_days']} "
             f"reliable_syms {len(rel)} ({time.time()-ts:.0f}s)")

    # --- rapor ---
    os.makedirs("outputs", exist_ok=True)
    lines = ["# E2 Faz 3.5 — Full-Universe Cross-Sectional IC Study", ""]
    lines.append(f"- panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
                 f"{panel['Date'].nunique()} dates, h={args.horizon}")
    lines.append(f"- folds: {len(sel)} selectable + holdout, min_names={args.min_names}, "
                 f"boost={args.boost}")
    lines.append("")
    lines.append("| target | daily IC | ICIR | %IC>0 | n_days | reliable syms |")
    lines.append("|---|---|---|---|---|---|")
    for label, (_, ic, rel) in results.items():
        lines.append(f"| {label} | {ic['ic_mean']:+.4f} | {ic['icir']:+.3f} | "
                     f"{100*ic['pct_positive']:.1f} | {ic['n_days']} | {len(rel)} |")
    lines.append("")
    best_label = "CS+CSFEAT" if "CS+CSFEAT" in results else "CROSS-SECTIONAL"
    rcs, ics, relcs = results[best_label]
    lines.append(f"- {best_label} reliable per-symbol dir_acc mean: "
                 f"{relcs['dir_acc'].mean():.2f}, edge mean: {relcs['edge'].mean():+.2f}")
    lines.append(f"- total elapsed: {time.time()-t0:.0f}s")
    with open("outputs/e2_faz35_cs_ic_study.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    rcs.per_symbol.to_csv("outputs/e2_faz35_per_symbol.csv", index=False, encoding="utf-8-sig")
    _log("wrote outputs/e2_faz35_cs_ic_study.md + per_symbol.csv")
    _log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
