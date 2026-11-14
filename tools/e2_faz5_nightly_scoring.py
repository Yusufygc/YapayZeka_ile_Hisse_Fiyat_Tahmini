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

from src.data.pooled_dataset import build_pooled_dataset
from src.data.pooled_loader import PooledLoaderConfig
from src.models.ensemble_pooled_model import (
    EnsemblePooledConfig,
    EnsemblePooledModel,
)
from src.models.global_pooled_model import (
    GlobalPooledConfig,
    GlobalPooledModel,
    make_global_model_factory,
)
from src.models.torch_mlp_model import TorchMLPConfig
from src.serving.nightly_scoring import (
    assemble_peer_table,
    coverage_warnings,
    latest_scoring_rows,
    liqlog_floor_from_turnover,
    segment_icir_from_table,
)
from src.serving.peer_store import GlobalRunMeta, PeerStore
from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward
from src.validation.pooled_oos import PerSymbolOOSConfig, evaluate_per_symbol
from src.validation.segment_ic import (
    attach_date_segments,
    latest_date_segments,
    segment_cross_sectional_ic,
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
    ap.add_argument("--model", choices=["lgb", "ensemble"], default="lgb",
                    help="final skorlama modeli: lgb (varsayilan) | ensemble (LGB+3-seed MLP, Faz 9)")
    ap.add_argument("--mlp-epochs", type=int, default=15, help="ensemble MLP epoch")
    ap.add_argument("--stale-days", type=int, default=10)
    ap.add_argument("--liq-floor-tl", type=float, default=3_000_000.0,
                    help="tradable alt esigi: medyan gunluk TL ciro (default 3M=P20, Q1'i kapatir)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fallback-dir", action="append", default=[],
                    help="birincil data-dir'de olmayan semboller icin yedek klasor (tekrarlanabilir), "
                         "ornek: --fallback-dir data/yedek (K7)")
    ap.add_argument("--quarantine-jumps", action=argparse.BooleanOptionalAction, default=True,
                    help="|1g log-getiri|>0.30 olcek kiriklari etrafindaki satirlari at (K8)")
    args = ap.parse_args()

    t0 = time.time()
    # K4: hedefi gerceklesmemis son h satir da yuklenir (keep_unlabeled) -> skorlama
    # en guncel PIYASA gunune yapilir; egitim/OOS yalniz etiketli alt kumeyi kullanir.
    ds = build_pooled_dataset(
        PooledLoaderConfig(
            data_dir=args.data_dir, universe_file=args.universe,
            target_horizon=args.horizon, min_rows=args.min_rows,
            fallback_dirs=tuple(args.fallback_dir), quarantine_jumps=args.quarantine_jumps),
        min_names=args.min_names, limit=args.limit)
    full_aug, aug, panel, feats, ci = ds.full, ds.aug, ds.labeled, ds.feats, ds.cat_idx
    # K5: segment kovalari nedensel (tarih-bazli); serving tablosu skorlama gunu satirlarindan.
    latest_rows, as_of = latest_scoring_rows(full_aug)
    seg_table = latest_date_segments(latest_rows, n_buckets=5)
    # tradable/stale referans: per-symbol son PIYASA tarihi (etiketsiz satirlar dahil)
    latest_global = ds.latest_market_day
    sym_last = full_aug.groupby("symbol")["Date"].max()
    sym_liq = panel.groupby("symbol")["liq_log"].median()
    _log(f"panel: {len(panel)} labeled rows, {panel['symbol'].nunique()} symbols, "
         f"latest market day {latest_global.date()} "
         f"(latest labeled {panel['Date'].max().date()})")

    # K6/K7: kapsama ve veri yasi uyarilari (sessiz gecmez; run config'ine de yazilir)
    n_universe = 0
    if os.path.exists(args.universe):
        n_universe = int(pd.read_csv(args.universe, encoding="utf-8-sig")["Symbol"].nunique())
    warns = coverage_warnings(
        int(full_aug["symbol"].nunique()), n_universe, latest_global,
        pd.Timestamp.today(), stale_days=args.stale_days)
    for w in warns:
        _log(f"UYARI {w}")
    meta = ds.loader_meta
    if meta["survivorship_status"] != "delisted_handled":
        _log("UYARI survivorship: evrende delisted sembol yok (hayatta kalma yanliligi riski)")
    if meta["fallback_symbols"]:
        _log(f"fallback klasorden yuklenen sembol: {len(meta['fallback_symbols'])}")
    if meta["quarantined_days"]:
        _log(f"karantina: {len(meta['quarantined_days'])} sembolde "
             f"{sum(meta['quarantined_days'].values())} satir atildi")

    # --- segment ICIR referansi (OOS, likidite kovasi birincil) ---
    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=63, n_windows=6,
        min_train_days=504)).split(aug)
    _log("OOS segment IC (likidite) hesaplaniyor ...")
    res = evaluate_per_symbol(
        aug, folds, make_global_model_factory(ci, GlobalPooledConfig(num_boost_round=args.boost)),
        PerSymbolOOSConfig(feature_cols=feats, target_col="target_cs"))
    preds_seg = attach_date_segments(res.predictions, aug, n_buckets=5)
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
    X_full = aug[feats].to_numpy(dtype=float)
    y_full = aug["target_cs"].to_numpy(dtype=float)
    if args.model == "ensemble":
        cardinalities = [int(aug[feats[i]].max()) + 1 for i in ci]
        _log(f"final ENSEMBLE (LGB + {len([42,7,123])}-seed MLP) tum panele egitiliyor ...")
        model = EnsemblePooledModel(EnsemblePooledConfig(
            lgb=GlobalPooledConfig(num_boost_round=args.boost),
            mlp=TorchMLPConfig(epochs=args.mlp_epochs),
            cat_indices=tuple(ci), cat_cardinalities=tuple(cardinalities)))
        model_name = "EnsemblePooled(LGB+MLP, CS+CSFEAT)"
    else:
        _log("final GlobalPooledModel tum panele egitiliyor ...")
        model = GlobalPooledModel(GlobalPooledConfig(num_boost_round=args.boost, cat_indices=tuple(ci)))
        model_name = "GlobalPooledModel(CS+CSFEAT)"
    model.fit(X_full, y_full)

    # --- en guncel evreni skorla (latest_rows/as_of yukarida) ---
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
        f"{len(aug)}|{as_of}|{sorted(aug['symbol'].unique())[:5]}".encode()
    ).hexdigest()[:12]
    store = PeerStore(args.db)
    rid = store.insert_run(GlobalRunMeta(
        model_name=model_name,
        as_of_date=str(as_of.date()),
        data_snapshot_hash=snap, n_symbols=int(panel["symbol"].nunique()),
        n_rows=int(len(aug)), horizon=args.horizon,
        ic_mean=res.ic["ic_mean"], icir=res.ic["icir"],
        pct_ic_positive=res.ic["pct_positive"],
        config={"boost": args.boost, "model": args.model, "icir_maps": icir_maps,
                "liq_floor_tl": args.liq_floor_tl, "stale_days": args.stale_days,
                "warnings": warns,
                "survivorship": meta["survivorship_status"],
                "fallback_symbols": len(meta["fallback_symbols"]),
                "quarantined_symbols": len(meta["quarantined_days"])}))
    n = store.insert_peer_scores(rid, table)
    dist = table["confidence_label"].value_counts().to_dict()
    _log(f"PeerStore run_id={rid}: {n} peer_scores yazildi -> {args.db}")
    _log(f"confidence dagilimi: {dist}")
    if "trend_label" in table.columns:
        _log(f"trend dagilimi: {table['trend_label'].value_counts().to_dict()}")
    _log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
