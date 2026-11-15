# -*- coding: utf-8 -*-
"""Gecelik serving pipeline — E2 Faz 8 (tek modul, surec-ici).

`tools/e2_nightly_pipeline.py` (orkestrator) + `tools/e2_faz5_nightly_scoring.py`
(skorlama) burada birlestirildi; araclar artik bu modulu cagiran ince sarmalayicilar.

Akis: (a) BIST islem-gunu kapisi -> (b) evren-geneli artimli veri tazeleme (yfinance)
-> (c) global model skorlama -> PeerStore. Skorlama alt-surec DEGIL, ayni surecte.

Calistirma:
    python -m src.serving.nightly --db data/serving_pool.db --boost 400 \
        --model ensemble --fallback-dir data/yedek
    python -m src.serving.nightly --help

`score_main()` yalniz skorlama (kapi/tazeleme yok) = eski e2_faz5 davranisi.
Islem-gunu kapisi pandas-market-calendars (XIST); fail-open (takvim hatasi ->
hafta-ici kontrolu).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import time
from datetime import date, datetime, timedelta
from typing import Optional, Sequence

import pandas as pd

from src.data.data_updater import DataUpdater
from src.data.freshness import check_freshness, warn_if_stale
from src.serving.nightly_scoring import (
    assemble_peer_table,
    coverage_warnings,
    latest_scoring_rows,
    liqlog_floor_from_turnover,
    segment_icir_from_table,
    segment_neff_from_table,
    symbol_vol_ratios,
)

# BIST seans kapanisi ~18:00. Bundan sonraki kosular AYNI gunun seansini hedefler;
# daha erken (sabah) kosular bir ONCEKI seansi (dun) hedefler.
_MARKET_CLOSE_HOUR = 19

# Loader ile ayni: panel disi yardimci CSV'ler sembol degil.
_EXCLUDE_STEMS = {"bist_universe", "advisory_history"}


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
        return d in {ts.date() for ts in days}
    except Exception as exc:  # lib yok / takvim hatasi -> hafta-ici fallback
        _log(f"[WARN] XIST takvimi kullanilamadi ({exc}); hafta-ici fallback")
        return d.weekday() < 5


def gate_target_date(now: Optional[datetime] = None) -> date:
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


def refresh_universe(data_dir: str, *, sleep_s: float = 0.0,
                     limit: int = 0, progress_every: int = 50) -> dict:
    """Evrendeki her sembolu artimli tazele. Returns durum sayim sozlugu.

    Toplu yol: ``DataUpdater.update_many`` (yf.download chunk + retry + sembol kilidi).
    Toplu cagri tumden patlarsa sembol basina tekli yola duser. ``sleep_s`` yalniz
    bu dusus yolunda semboller arasi beklemedir.
    """
    syms = _universe_symbols(data_dir)
    if limit > 0:
        syms = syms[:limit]
    counts = {"updated": 0, "up_to_date": 0, "skipped": 0, "failed": 0}
    rows_added = 0
    _log(f"veri tazeleme: {len(syms)} sembol (toplu)")
    targets = [(os.path.join(data_dir, f"{s}.csv"), s) for s in syms]
    try:
        results = DataUpdater.update_many(targets)
    except Exception as exc:  # toplu yol patlarsa tekli yola dus
        _log(f"[WARN] toplu tazeleme hatasi ({str(exc)[:80]}); tekli yola dusuluyor")
        results = {}
        for i, (path, sym) in enumerate(targets, 1):
            try:
                results[sym] = DataUpdater.check_and_update(path, sym, interactive=False)
            except Exception as one_exc:  # tek sembol patlamasi pipeline'i durdurmasin
                results[sym] = None
                _log(f"[WARN] {sym} tazeleme hatasi: {str(one_exc)[:80]}")
            if sleep_s:
                time.sleep(sleep_s)
            if progress_every and i % progress_every == 0:
                _log(f"  ... {i}/{len(targets)}")
    for sym in syms:
        res = results.get(sym)
        if res is None:
            counts["failed"] += 1
            continue
        counts[res.status] = counts.get(res.status, 0) + 1
        rows_added += int(res.rows_added or 0)
    counts["rows_added"] = rows_added
    _log(f"veri tazeleme bitti: {counts}")
    return counts


# ------------------------------------------------------------------- scoring
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m src.serving.nightly",
        description="Gecelik serving: islem-gunu kapisi -> veri tazeleme -> skorlama -> PeerStore")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--universe", default="data/bist_universe.csv")
    ap.add_argument("--db", default="data/serving_pool.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--min-names", type=int, default=15)
    ap.add_argument("--boost", type=int, default=400)
    ap.add_argument("--model", choices=["lgb", "ensemble"], default="lgb",
                    help="skorlama modeli: lgb (varsayilan) | ensemble (LGB+3-seed MLP, Faz 9; ~4x sure)")
    ap.add_argument("--mlp-epochs", type=int, default=15, help="ensemble MLP epoch")
    ap.add_argument("--stale-days", type=int, default=10)
    ap.add_argument("--stale-business-days", type=int, default=5,
                    help="bugune gore is gunu esigi: sembolun son tarihi bundan fazla geride ise "
                         "BAYAT isaretlenir (confidence dusurulur; kosu durmaz)")
    ap.add_argument("--liq-floor-tl", type=float, default=3_000_000.0,
                    help="tradable alt esigi: medyan gunluk TL ciro (default 3M=P20, Q1'i kapatir)")
    ap.add_argument("--limit", type=int, default=0, help="ilk N sembol (tazeleme + skorlama; test/manuel)")
    ap.add_argument("--fallback-dir", action="append", default=[],
                    help="birincil data-dir'de olmayan semboller icin yedek klasor (tekrarlanabilir), "
                         "ornek: --fallback-dir data/yedek (K7)")
    ap.add_argument("--quarantine-jumps", action=argparse.BooleanOptionalAction, default=True,
                    help="|1g log-getiri|>0.30 olcek kiriklari etrafindaki satirlari at (K8)")
    # Gece isinde veri her gun degisir -> her sembol iska + ~11 sn yazma maliyeti.
    # Bu yuzden varsayilan KAPALI; ayni gun tekrar kosularinda --feature-cache ile ac.
    ap.add_argument("--feature-cache", action="store_true",
                    help="sembol basi Parquet ozellik onbellegini ac (ayni veride tekrar kosular icin)")
    ap.add_argument("--feature-cache-dir", default=os.path.join("data", "feature_cache", "pooled"),
                    help="ozellik onbellegi dizini (--feature-cache ile)")
    ap.add_argument("--sleep-s", type=float, default=0.0, help="yalniz tekli dusus yolunda semboller arasi bekleme")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-trading-gate", action="store_true")
    return ap


def assemble_serving_table(model, full_aug: pd.DataFrame, latest_rows: pd.DataFrame,
                           as_of: pd.Timestamp, feats: list[str], seg_table: pd.DataFrame,
                           icir_maps: dict, neff_maps: dict, **kwargs) -> pd.DataFrame:
    """Kol-B alanlari dahil serving tablosu: segment n_eff + nedensel sembol vol orani.

    vol orani: sembol trailing 20g vol / o gunku evren medyani, yalniz Date<=as_of
    satirlarindan (`symbol_vol_ratios`). n_eff: ortusmeyen tarih sayisi (confidence kapisi).
    """
    return assemble_peer_table(
        model, latest_rows, feats, seg_table, icir_maps=icir_maps,
        neff_maps=neff_maps, symbol_vol_ratio=symbol_vol_ratios(full_aug, as_of), **kwargs)


def run_scoring(args: argparse.Namespace) -> int:
    """Global modeli tum gecmise egit, en guncel evreni skorla, PeerStore'a yaz (surec-ici)."""
    # Agir bagimliliklar (lightgbm/torch) yalniz skorlamada yuklenir (--help / kapi hizli kalir).
    from src.data.pooled_dataset import build_pooled_dataset
    from src.data.pooled_loader import PooledLoaderConfig
    from src.models.ensemble_pooled_model import EnsemblePooledConfig, EnsemblePooledModel
    from src.models.global_pooled_model import (
        GlobalPooledConfig, GlobalPooledModel, make_global_model_factory)
    from src.models.torch_mlp_model import TorchMLPConfig
    from src.serving.peer_store import GlobalRunMeta, PeerStore
    from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward
    from src.validation.pooled_oos import PerSymbolOOSConfig, evaluate_per_symbol
    from src.validation.segment_ic import (
        attach_date_segments, latest_date_segments, segment_cross_sectional_ic)

    t0 = time.time()
    # K4: hedefi gerceklesmemis son h satir da yuklenir (keep_unlabeled) -> skorlama
    # en guncel PIYASA gunune yapilir; egitim/OOS yalniz etiketli alt kumeyi kullanir.
    ds = build_pooled_dataset(
        PooledLoaderConfig(
            data_dir=args.data_dir, universe_file=args.universe,
            target_horizon=args.horizon, min_rows=args.min_rows,
            fallback_dirs=tuple(args.fallback_dir), quarantine_jumps=args.quarantine_jumps,
            feature_cache_dir=args.feature_cache_dir if args.feature_cache else None),
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
    fc = meta.get("feature_cache", {})
    if fc.get("enabled"):
        _log(f"ozellik onbellegi: {fc['hits']} hit, {fc['misses']} miss, {fc['errors']} hata")
    if meta["survivorship_status"] != "delisted_handled":
        _log("UYARI survivorship: evrende delisted sembol yok (hayatta kalma yanliligi riski)")
    if meta["fallback_symbols"]:
        _log(f"fallback klasorden yuklenen sembol: {len(meta['fallback_symbols'])}")
    if meta.get("rejected_rows"):
        _log(f"UYARI bozuk CSV satiri: {len(meta['rejected_rows'])} sembolde "
             f"{sum(meta['rejected_rows'].values())} satir karantinaya alindi")
    if meta["quarantined_days"]:
        _log(f"karantina: {len(meta['quarantined_days'])} sembolde "
             f"{sum(meta['quarantined_days'].values())} satir atildi")

    # --- segment ICIR referansi (OOS; ortusmeyen ICIR + n_eff tercih edilir) ---
    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=63, n_windows=6,
        min_train_days=504)).split(aug)
    _log("OOS segment IC (likidite) hesaplaniyor ...")
    res = evaluate_per_symbol(
        aug, folds, make_global_model_factory(ci, GlobalPooledConfig(num_boost_round=args.boost)),
        PerSymbolOOSConfig(feature_cols=feats, target_col="target_cs"))
    preds_seg = attach_date_segments(res.predictions, aug, n_buckets=5)
    seg_tables = {
        axis: segment_cross_sectional_ic(preds_seg, group_col=col, min_names=10,
                                         horizon=args.horizon)
        for axis, col in (("liq", "liq_bucket"), ("vol", "vol_bucket"), ("sector", "sector"))}
    icir_maps = {a: segment_icir_from_table(t) for a, t in seg_tables.items()}
    neff_maps = {a: segment_neff_from_table(t) for a, t in seg_tables.items()}
    _log(f"overall ICIR {res.ic['icir']:+.3f}; liq {icir_maps['liq']}")
    _log(f"vol {icir_maps['vol']}")

    # --- final model: TUM gecmise egit ---
    X_full = aug[feats].to_numpy(dtype=float)
    y_full = aug["target_cs"].to_numpy(dtype=float)
    if args.model == "ensemble":
        cardinalities = [int(aug[feats[i]].max()) + 1 for i in ci]
        _log("final ENSEMBLE (LGB + 3-seed MLP) tum panele egitiliyor ...")
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

    # Bugune gore tazelik kapisi: panel-ici goreli esik (stale_days) tek basina, tum veri
    # aylarca bayatsa hicbir sembolu isaretlemez. Bayat semboller uyari + confidence'ta isaret.
    stale_now = warn_if_stale(
        sym_last.to_dict(), "Gecelik serving tazelik kapisi",
        max_business_days=args.stale_business_days)
    glob_fresh = check_freshness(
        latest_global, max_business_days=args.stale_business_days, label="panel en guncel piyasa gunu")
    _log(("UYARI " if glob_fresh.stale else "") + glob_fresh.message)
    if stale_now:
        _log(f"UYARI bayat sembol (is gunu > {args.stale_business_days}): "
             f"{len(stale_now)}/{len(sym_last)} -> confidence'ta stale isaretlenir")

    def stale_for(s: str) -> bool:
        last = sym_last.get(s)
        return (last is None or s in stale_now
                or (latest_global - last).days > args.stale_days)

    table = assemble_serving_table(
        model, full_aug, latest_rows, as_of, feats, seg_table, icir_maps, neff_maps,
        tradable_for=tradable_for, stale_for=stale_for)
    _log(f"scored {len(table)} symbols at {table['as_of_date'].iloc[0] if len(table) else '-'}")

    # --- persist ---
    snap = hashlib.sha1(
        f"{len(aug)}|{as_of}|{sorted(aug['symbol'].unique())[:5]}".encode()
    ).hexdigest()[:12]
    store = PeerStore(args.db)
    # probability_source PeerStore kolonu degil (peer_store.py disinda) -> run config'inde.
    prob_src = (sorted(table["probability_source"].dropna().unique().tolist())
                if "probability_source" in table.columns else [])
    rid, n = store.write_run(GlobalRunMeta(
        model_name=model_name,
        as_of_date=str(as_of.date()),
        data_snapshot_hash=snap, n_symbols=int(panel["symbol"].nunique()),
        n_rows=int(len(aug)), horizon=args.horizon,
        ic_mean=res.ic["ic_mean"], icir=res.ic["icir"],
        pct_ic_positive=res.ic["pct_positive"],
        config={"boost": args.boost, "model": args.model, "icir_maps": icir_maps,
                "neff_maps": neff_maps, "trend_probability_source": prob_src,
                "liq_floor_tl": args.liq_floor_tl, "stale_days": args.stale_days,
                "warnings": warns,
                "stale_business_days": args.stale_business_days,
                "stale_symbols_now": len(stale_now),
                "panel_freshness": glob_fresh.message,
                "rejected_rows_symbols": len(meta.get("rejected_rows", {})),
                "survivorship": meta["survivorship_status"],
                "fallback_symbols": len(meta["fallback_symbols"]),
                "quarantined_symbols": len(meta["quarantined_days"])}), table)
    _log(f"PeerStore run_id={rid}: {n} peer_scores yazildi -> {args.db}")
    _log(f"confidence dagilimi: {table['confidence_label'].value_counts().to_dict()}")
    if "trend_label" in table.columns:
        _log(f"trend dagilimi: {table['trend_label'].value_counts().to_dict()}")
    _log(f"DONE in {time.time()-t0:.0f}s")
    return 0


# ---------------------------------------------------------------------- main
def score_main(argv: Optional[Sequence[str]] = None) -> int:
    """Yalniz skorlama (kapi/tazeleme yok) — eski tools/e2_faz5_nightly_scoring.py davranisi."""
    return run_scoring(build_parser().parse_args(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    t0 = time.time()
    _log("=== gecelik serving pipeline basladi ===")

    # (a) islem-gunu kapisi: hedef seans saate gore (aksam=bugun, sabah=dun).
    if not args.skip_trading_gate:
        target = gate_target_date()
        if not is_trading_day(target):
            _log(f"skip: {target} BIST islem gunu degil (hafta sonu/tatil). cikis 0.")
            return 0
        _log(f"islem gunu: {target} -> devam")

    # (b) veri tazeleme
    if args.skip_data:
        _log("veri tazeleme atlandi (--skip-data)")
    else:
        refresh_universe(args.data_dir, sleep_s=args.sleep_s, limit=args.limit)

    # (c) skorlama (surec-ici)
    _log(f"skorlama baslatiliyor: model={args.model} boost={args.boost} db={args.db}")
    rc = run_scoring(args)

    _log(f"=== pipeline bitti: exit={rc}, sure={time.time()-t0:.0f}s ===")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
