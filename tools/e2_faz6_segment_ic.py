# -*- coding: utf-8 -*-
"""E2 Faz 6 — stratified per-segment cross-sectional IC study.

Best config (cross-sectional target + cs-features) ile OOS tahminleri uretir,
segmentlere (likidite kovasi / volatilite kovasi / sektor) ayirip her segment
icin gunluk cross-sectional IC dagilimini cikarir. Serving guven skorunun
segment-tabanini besler (Faz 5).

Run (dl_env):
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1
    python tools/e2_faz6_segment_ic.py --min-rows 300 --boost 400

Outputs:
    outputs/e2_faz6_segment_ic.md
    outputs/e2_faz6_segment_<liq|vol|sector>.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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
from src.validation.pooled_oos import PerSymbolOOSConfig, evaluate_per_symbol
from src.validation.segment_ic import (
    attach_segments,
    segment_cross_sectional_ic,
    symbol_segments,
)


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--universe", default="data/bist_universe.csv")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--min-names", type=int, default=15)
    ap.add_argument("--seg-min-names", type=int, default=10)
    ap.add_argument("--n-buckets", type=int, default=5)
    ap.add_argument("--boost", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    loader = PooledPanelLoader(PooledLoaderConfig(
        data_dir=args.data_dir, universe_file=args.universe,
        target_horizon=args.horizon, min_rows=args.min_rows))
    if args.limit > 0:
        syms = loader._symbols()[: args.limit]
        loader._symbols = lambda: syms
    panel = loader.load()
    panel = add_cross_sectional_target(panel, min_names=args.min_names)
    _log(f"panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
         f"{panel['Date'].nunique()} dates")

    seg_table = symbol_segments(panel, n_buckets=args.n_buckets)

    aug0, f0, _ = build_pooled_features(panel)
    panel, _ = add_cross_sectional_features(
        panel, [c for c in f0 if c not in ("symbol_id", "sector_code")])
    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=63, n_windows=6,
        min_train_days=504)).split(panel)
    aug, feats, ci = build_pooled_features(panel)
    assert not any(c.startswith("target") for c in feats), "target leak!"
    _log(f"fitting best config (CS+CSFEAT, {len(feats)} feats) ...")
    res = evaluate_per_symbol(
        aug, folds, make_global_model_factory(ci, GlobalPooledConfig(num_boost_round=args.boost)),
        PerSymbolOOSConfig(feature_cols=feats, target_col="target_cs"))
    _log(f"overall IC {res.ic['ic_mean']:+.4f} ICIR {res.ic['icir']:+.3f} "
         f"%IC>0 {100*res.ic['pct_positive']:.1f} ({time.time()-t0:.0f}s)")

    preds = attach_segments(res.predictions, seg_table)
    os.makedirs("outputs", exist_ok=True)
    lines = ["# E2 Faz 6 — Stratified Segment IC", "",
             f"- panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
             f"{panel['Date'].nunique()} dates, h={args.horizon}, boost={args.boost}",
             f"- overall IC {res.ic['ic_mean']:+.4f} / ICIR {res.ic['icir']:+.3f} / "
             f"%IC>0 {100*res.ic['pct_positive']:.1f}", ""]
    for col, key in [("liq_bucket", "liq"), ("vol_bucket", "vol"), ("sector", "sector")]:
        tbl = segment_cross_sectional_ic(preds, group_col=col, min_names=args.seg_min_names)
        tbl.to_csv(f"outputs/e2_faz6_segment_{key}.csv", index=False, encoding="utf-8-sig")
        lines.append(f"## by {col}")
        lines.append("| segment | IC | ICIR | %IC>0 | n_days | n_syms |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in tbl.iterrows():
            lines.append(f"| {r['segment']} | {r['ic_mean']:+.4f} | {r['icir']:+.3f} | "
                         f"{100*r['pct_positive']:.1f} | {int(r['n_days'])} | {int(r['n_symbols'])} |")
        lines.append("")
        _log(f"segment by {col}: {len(tbl)} groups")
    lines.append(f"- total elapsed: {time.time()-t0:.0f}s")
    with open("outputs/e2_faz6_segment_ic.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    _log(f"wrote outputs/e2_faz6_segment_ic.md  DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
