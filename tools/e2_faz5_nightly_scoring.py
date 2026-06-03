# -*- coding: utf-8 -*-
"""E2 Faz 5 — nightly universe scoring batch (real).

Tum hisseleri skorlar + siralar + segment/confidence ile PeerStore'a yazar.
API (GET /analysis/{symbol}) bu snapshot'i okur. Per-query egitim yok.

Akis:
  load panel -> target_cs + cs-features -> (OOS ile segment ICIR referansi)
  -> tum gecmise GlobalPooledModel egit -> en guncel evreni skorla
  -> segment + confidence -> PeerStore.

Run (dl_env):
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1
    python tools/e2_faz5_nightly_scoring.py --db data/serving_pool.db --boost 400
"""
from __future__ import annotations

import argparse
import hashlib
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
    GlobalPooledModel,
    build_pooled_features,
    make_global_model_factory,
)
from src.serving.nightly_scoring import (
    assemble_peer_table,
    liqlog_floor_from_turnover,
    segment_icir_from_table,
)
from src.serving.peer_store import GlobalRunMeta, PeerStore
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
    ap.add_argument("--db", default="data/serving_pool.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--min-names", type=int, default=15)
    ap.add_argument("--boost", type=int, default=400)
    ap.add_argument("--stale-days", type=int, default=10)
    ap.add_argument("--liq-floor-tl", type=float, default=3_000_000.0,
                    help="tradable alt esigi: medyan gunluk TL ciro (default 3M=P20, Q1'i kapatir)")
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
    seg_table = symbol_segments(panel, n_buckets=5)
    # tradable/stale referans: per-symbol son tarih + medyan likidite
    latest_global = panel["Date"].max()
    sym_last = panel.groupby("symbol")["Date"].max()
    sym_liq = panel.groupby("symbol")["liq_log"].median()
    _log(f"panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, latest {latest_global.date()}")

    aug0, f0, _ = build_pooled_features(panel)
    panel, _ = add_cross_sectional_features(
        panel, [c for c in f0 if c not in ("symbol_id", "sector_code")])
    aug, feats, ci = build_pooled_features(panel)
    assert not any(c.startswith("target") for c in feats), "target leak!"

    # --- segment ICIR referansi (OOS, likidite kovasi birincil) ---
    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=63, n_windows=6,
        min_train_days=504)).split(aug)
    _log("OOS segment IC (likidite) hesaplaniyor ...")
    res = evaluate_per_symbol(
        aug, folds, make_global_model_factory(ci, GlobalPooledConfig(num_boost_round=args.boost)),
        PerSymbolOOSConfig(feature_cols=feats, target_col="target_cs"))
    preds_seg = attach_segments(res.predictions, seg_table)
    icir_maps = {
        "liq": segment_icir_from_table(
            segment_cross_sectional_ic(preds_seg, group_col="liq_bucket", min_names=10)),
        "vol": segment_icir_from_table(
            segment_cross_sectional_ic(preds_seg, group_col="vol_bucket", min_names=10)),
        "sector": segment_icir_from_table(
            segment_cross_sectional_ic(preds_seg, group_col="sector", min_names=10)),
    }
    _log(f"overall ICIR {res.ic['icir']:+.3f}; liq {icir_maps['liq']}")
    _log(f"vol {icir_maps['vol']}")

    # --- final model: TUM gecmise egit ---
    _log("final GlobalPooledModel tum panele egitiliyor ...")
    model = GlobalPooledModel(GlobalPooledConfig(num_boost_round=args.boost, cat_indices=tuple(ci)))
    model.fit(aug[feats].to_numpy(dtype=float), aug["target_cs"].to_numpy(dtype=float))

    # --- en guncel evreni skorla ---
    latest_rows = aug[aug["Date"] == aug["Date"].max()]
    liq_floor = liqlog_floor_from_turnover(args.liq_floor_tl)
    def tradable_for(s: str) -> bool:
        return float(sym_liq.get(s, -1e9)) >= liq_floor
    def stale_for(s: str) -> bool:
        last = sym_last.get(s)
        return last is None or (latest_global - last).days > args.stale_days

    table = assemble_peer_table(
        model, latest_rows, feats, seg_table, icir_maps=icir_maps,
        tradable_for=tradable_for, stale_for=stale_for)
    _log(f"scored {len(table)} symbols at {table['as_of_date'].iloc[0] if len(table) else '-'}")

    # --- persist ---
    snap = hashlib.sha1(
        f"{len(aug)}|{aug['Date'].max()}|{sorted(aug['symbol'].unique())[:5]}".encode()
    ).hexdigest()[:12]
    store = PeerStore(args.db)
    rid = store.insert_run(GlobalRunMeta(
        model_name="GlobalPooledModel(CS+CSFEAT)",
        as_of_date=str(aug["Date"].max().date()),
        data_snapshot_hash=snap, n_symbols=int(panel["symbol"].nunique()),
        n_rows=int(len(aug)), horizon=args.horizon,
        ic_mean=res.ic["ic_mean"], icir=res.ic["icir"],
        pct_ic_positive=res.ic["pct_positive"],
        config={"boost": args.boost, "icir_maps": icir_maps,
                "liq_floor_tl": args.liq_floor_tl, "stale_days": args.stale_days}))
    n = store.insert_peer_scores(rid, table)
    dist = table["confidence_label"].value_counts().to_dict()
    _log(f"PeerStore run_id={rid}: {n} peer_scores yazildi -> {args.db}")
    _log(f"confidence dagilimi: {dist}")
    if "trend_label" in table.columns:
        _log(f"trend dagilimi: {table['trend_label'].value_counts().to_dict()}")
    _log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
