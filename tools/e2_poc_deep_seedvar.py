# -*- coding: utf-8 -*-
"""E2 PoC adim A — DEEP-MLP seed varyansi vs LightGBM.

Soru: PoC'ta MLP LGB'yi +%5 ICIR ile gecti (ICIR 1.635 vs 1.553). Bu GERCEK mi,
yoksa torch'un run-arasi nondeterminizminin gurultusu mu? LGB deterministik (tek
deger). MLP'yi N farkli seed ile kosup ICIR dagilimina bak:
  - LGB ICIR, MLP dagiliminin ALTINDA ise -> kazanc saglam.
  - LGB, MLP min..max araliginda ise -> fark gurultu, kazanc supheli.

AYNI panel snapshot + AYNI fold + AYNI metrik (e2_poc_deep_ic ile birebir).

Run (dl_env, background):
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1 & set TF_CPP_MIN_LOG_LEVEL=3
    python tools/e2_poc_deep_seedvar.py --seeds 42,7,123,2024,99 --epochs 15

Outputs:
    outputs/e2_poc_deep_seedvar.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

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

# PoC araci ayni klasorde; TorchMLPModel + factory'yi tekrar kullan.
from tools.e2_poc_deep_ic import make_mlp_factory


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--universe", default="data/bist_universe.csv")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--min-names", type=int, default=15)
    ap.add_argument("--window-len", type=int, default=63)
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--min-train-days", type=int, default=504)
    ap.add_argument("--boost", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", default="42,7,123,2024,99")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", default="256,128,64")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

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
    _log(f"panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols (load {time.time()-t0:.0f}s)")

    panel = add_cross_sectional_target(
        panel, raw_target="target", out_col="target_cs", method="rank", min_names=args.min_names)
    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=args.window_len,
        n_windows=args.n_windows, min_train_days=args.min_train_days,
    )).split(panel)
    sel = [f for f in folds if not f.is_final_holdout]

    aug0, feats0, _ = build_pooled_features(panel)
    cs_base = [c for c in feats0 if c not in ("symbol_id", "sector_code")]
    panel, _ = add_cross_sectional_features(panel, cs_base)
    aug, feats, cat_idx = build_pooled_features(panel)
    assert not any(c.startswith("target") for c in feats), "target leak!"
    cardinalities = [int(aug[feats[i]].max()) + 1 for i in cat_idx]
    cfg_oos = PerSymbolOOSConfig(feature_cols=feats, target_col="target_cs")
    _log(f"panel ready: {len(panel)} rows, {panel['Date'].nunique()} dates, "
         f"{len(feats)} feats, folds={len(sel)}+holdout")

    # --- LGB deterministik baseline (tek kosu) ---
    _log(f"LightGBM baseline (boost={args.boost}) ...")
    ts = time.time()
    lgb_ic = evaluate_per_symbol(
        aug, folds, make_global_model_factory(cat_idx, GlobalPooledConfig(num_boost_round=args.boost)),
        cfg_oos).ic
    _log(f"  LGB: ICIR {lgb_ic['icir']:+.3f} IC {lgb_ic['ic_mean']:+.4f} ({time.time()-ts:.0f}s)")

    # --- MLP, seed dagilimi ---
    mlp_base = {
        "hidden": [int(x) for x in args.hidden.split(",")],
        "dropout": args.dropout, "lr": args.lr, "wd": args.wd,
        "epochs": args.epochs, "batch": args.batch,
    }
    mlp_runs = []  # (seed, icir, ic_mean, pct_pos)
    for sd in seeds:
        ts = time.time()
        cfg = {**mlp_base, "seed": sd}
        ic = evaluate_per_symbol(aug, folds, make_mlp_factory(cat_idx, cardinalities, cfg), cfg_oos).ic
        mlp_runs.append((sd, ic["icir"], ic["ic_mean"], ic["pct_positive"]))
        _log(f"  MLP seed={sd}: ICIR {ic['icir']:+.3f} IC {ic['ic_mean']:+.4f} "
             f"%IC>0 {100*ic['pct_positive']:.1f} ({time.time()-ts:.0f}s)")

    icirs = np.array([r[1] for r in mlp_runs], dtype=float)
    m, s = float(icirs.mean()), float(icirs.std())
    lo, hi = float(icirs.min()), float(icirs.max())
    lgb_icir = lgb_ic["icir"]
    lgb_in_range = lo <= lgb_icir <= hi
    if lgb_in_range:
        verdict = "GURULTU SUPHESI: LGB ICIR MLP min..max icinde -> deep kazanci saglam degil"
    elif lgb_icir < lo:
        verdict = "SAGLAM: LGB ICIR MLP dagiliminin ALTINDA -> deep kazanci gercek"
    else:  # lgb_icir > hi
        verdict = "DEEP KAYBEDIYOR: LGB ICIR MLP dagiliminin USTUNDE -> LGB daha iyi"

    # --- rapor ---
    os.makedirs("outputs", exist_ok=True)
    lines = ["# E2 PoC A — DEEP-MLP seed varyansi vs LightGBM", ""]
    lines.append(f"- panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
                 f"{panel['Date'].nunique()} dates, h={args.horizon}")
    lines.append(f"- config: CS+CSFEAT (target_cs), {len(feats)} feats, folds={len(sel)}+holdout")
    lines.append(f"- MLP: hidden={mlp_base['hidden']} dropout={args.dropout} lr={args.lr} "
                 f"wd={args.wd} epochs={args.epochs} batch={args.batch}")
    lines.append(f"- seeds: {seeds}")
    lines.append("")
    lines.append(f"**LightGBM (deterministik): ICIR {lgb_ic['icir']:+.3f}**")
    lines.append("")
    lines.append("| seed | ICIR | IC | %IC>0 |")
    lines.append("|---|---|---|---|")
    for sd, icir, icm, pp in mlp_runs:
        lines.append(f"| {sd} | {icir:+.3f} | {icm:+.4f} | {100*pp:.1f} |")
    lines.append(f"| **mean** | **{m:+.3f}** | | |")
    lines.append("")
    lines.append(f"- MLP ICIR: mean {m:+.3f}, std {s:.3f}, min {lo:+.3f}, max {hi:+.3f}")
    lines.append(f"- LGB ICIR {lgb_ic['icir']:+.3f} MLP araliginda mi? {lgb_in_range}")
    lines.append(f"- **VERDICT: {verdict}**")
    lines.append(f"- elapsed: {time.time()-t0:.0f}s")
    with open("outputs/e2_poc_deep_seedvar.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    _log(f"VERDICT: {verdict}")
    _log("wrote outputs/e2_poc_deep_seedvar.md")
    _log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
