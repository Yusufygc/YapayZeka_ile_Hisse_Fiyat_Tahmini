# -*- coding: utf-8 -*-
"""E2 PoC adim B — LightGBM + DEEP-MLP ensemble (cross-sectional IC).

Adim A bulgusu: deep kazanci gercek ama marjinal (+%4 ICIR). Asil deger: deep'in
hatalari LGB'den farkli -> ENSEMBLE ikisini birlestirince ikisini de gecebilir.

Blend = tarih-ici RANK ortalamasi (IC rank-bazli oldugu icin z-score degil rank):
  her tarih d icin: rank(pred_lgb) ve rank(pred_mlp) [pct] -> agirlikli ortalama.
MLP nondeterminizmini sondurmek icin MLP = COK-SEED y_pred ortalamasi.

AYNI panel snapshot + AYNI fold + AYNI metrik (e2_poc_deep_ic/seedvar ile birebir).

Kiyas: LGB tek / MLP-avg tek / ENSEMBLE (birkac agirlik).

Run (dl_env, background):
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1 & set TF_CPP_MIN_LOG_LEVEL=3
    python tools/e2_poc_deep_ensemble.py --seeds 42,7,123 --epochs 15

Outputs:
    outputs/e2_poc_deep_ensemble.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

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
from src.validation.pooled_oos import (
    PerSymbolOOSConfig,
    daily_cross_sectional_ic,
    evaluate_per_symbol,
)
from tools.e2_poc_deep_ic import make_mlp_factory


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ic_from(preds: pd.DataFrame, pred_col: str) -> dict:
    """daily_cross_sectional_ic istenen pred kolonunu 'y_pred' bekler."""
    df = preds[["Date", "symbol", "y_true", pred_col]].rename(columns={pred_col: "y_pred"})
    return daily_cross_sectional_ic(df)


def _pct_rank_within_date(preds: pd.DataFrame, col: str) -> pd.Series:
    return preds.groupby("Date")[col].rank(method="average", pct=True)


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
    ap.add_argument("--seeds", default="42,7,123")
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

    KEY = ["symbol", "Date"]

    # --- LGB ---
    _log(f"LightGBM (boost={args.boost}) ...")
    ts = time.time()
    res_lgb = evaluate_per_symbol(
        aug, folds, make_global_model_factory(cat_idx, GlobalPooledConfig(num_boost_round=args.boost)),
        cfg_oos)
    base = res_lgb.predictions[KEY + ["y_true"]].copy()
    base["pred_lgb"] = res_lgb.predictions["y_pred"].to_numpy()
    _log(f"  LGB: ICIR {res_lgb.ic['icir']:+.3f} ({time.time()-ts:.0f}s)")

    # --- MLP cok-seed; y_pred ortalamasi ---
    mlp_base = {
        "hidden": [int(x) for x in args.hidden.split(",")],
        "dropout": args.dropout, "lr": args.lr, "wd": args.wd,
        "epochs": args.epochs, "batch": args.batch,
    }
    mlp_pred_cols = []
    for sd in seeds:
        ts = time.time()
        r = evaluate_per_symbol(aug, folds, make_mlp_factory(cat_idx, cardinalities, {**mlp_base, "seed": sd}), cfg_oos)
        col = f"pred_mlp_{sd}"
        p = r.predictions[KEY].copy()
        p[col] = r.predictions["y_pred"].to_numpy()
        base = base.merge(p, on=KEY, how="left")
        mlp_pred_cols.append(col)
        _log(f"  MLP seed={sd}: ICIR {r.ic['icir']:+.3f} ({time.time()-ts:.0f}s)")

    base["pred_mlp"] = base[mlp_pred_cols].mean(axis=1)

    # --- IC: bilesenler ---
    ic_lgb = _ic_from(base, "pred_lgb")
    ic_mlp = _ic_from(base, "pred_mlp")
    _log(f"LGB ICIR {ic_lgb['icir']:+.3f} | MLP-avg ICIR {ic_mlp['icir']:+.3f}")

    # --- ensemble: tarih-ici pct-rank agirlikli blend ---
    base["rank_lgb"] = _pct_rank_within_date(base, "pred_lgb")
    base["rank_mlp"] = _pct_rank_within_date(base, "pred_mlp")
    weights = [("50/50", 0.5), ("30LGB/70MLP", 0.3), ("70LGB/30MLP", 0.7)]
    ens_results = {}
    for label, w_lgb in weights:
        base["pred_ens"] = w_lgb * base["rank_lgb"] + (1 - w_lgb) * base["rank_mlp"]
        ens_results[label] = _ic_from(base, "pred_ens")
        _log(f"ensemble {label}: ICIR {ens_results[label]['icir']:+.3f} IC {ens_results[label]['ic_mean']:+.4f}")

    # --- rapor ---
    os.makedirs("outputs", exist_ok=True)
    lines = ["# E2 PoC B — LightGBM + DEEP-MLP Ensemble (cross-sectional IC)", ""]
    lines.append(f"- panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
                 f"{panel['Date'].nunique()} dates, h={args.horizon}")
    lines.append(f"- config: CS+CSFEAT (target_cs), {len(feats)} feats, folds={len(sel)}+holdout")
    lines.append(f"- MLP: hidden={mlp_base['hidden']} epochs={args.epochs} seeds={seeds} (y_pred avg)")
    lines.append(f"- blend: tarih-ici pct-rank agirlikli ortalama")
    lines.append("")
    lines.append("| model | daily IC | ICIR | %IC>0 | n_days |")
    lines.append("|---|---|---|---|---|")
    rows = [("LightGBM", ic_lgb), ("DEEP-MLP (avg)", ic_mlp)]
    for label, ic in ens_results.items():
        rows.append((f"ENSEMBLE {label}", ic))
    for label, ic in rows:
        lines.append(f"| {label} | {ic['ic_mean']:+.4f} | {ic['icir']:+.3f} | "
                     f"{100*ic['pct_positive']:.1f} | {ic['n_days']} |")
    lines.append("")
    best_lbl, best_ic = max(rows, key=lambda kv: (kv[1]["icir"] if np.isfinite(kv[1]["icir"]) else -9))
    lines.append(f"- BEST: {best_lbl} (ICIR {best_ic['icir']:+.3f})")
    lines.append(f"- elapsed: {time.time()-t0:.0f}s")
    with open("outputs/e2_poc_deep_ensemble.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    _log(f"BEST: {best_lbl} ICIR {best_ic['icir']:+.3f}")
    _log("wrote outputs/e2_poc_deep_ensemble.md")
    _log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
