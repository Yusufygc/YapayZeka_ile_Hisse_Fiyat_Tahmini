# -*- coding: utf-8 -*-
"""E2 Faz 7 — confidence-stratified ABSOLUTE directional accuracy study.

Soru: "guvendigimizde daha mi isabetliyiz?" Yani serving confidence (segment
composite ICIR x tradability kapilari) YUKSEK olan isimlerde, peer-rank'in MUTLAK
yon (yukari/asagi) isabeti, dusuk-confidence isimlere gore belirgin daha yuksek
mi? Eger evetse urun hikayesi guclu: guven etiketi gercek isabetle ortusur.

Akis:
  1. Faz 6 ile ayni OOS uretimi (CS hedef + CS-features, h=5).
  2. y_true=target_cs ile per-eksen segment ICIR (liq/vol/sector) -> icir_maps.
  3. her satira composite_icir + tradability kapisi -> confidence (high/med/low).
  4. mutlak hedef (`target`, log-getiri) (symbol,Date) ile geri eklenir -> y_abs.
  5. her (fold,Date) ici y_pred quintile (Q1..Q5).
  6. confidence x quintile kir: P(up), mean_ret, ekstrem-yon isabeti
     (Q5->yukari, Q1->asagi tahmininin dogrulugu).

Run (dl_env):
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1
    python tools/e2_faz7_confidence_diracc.py --min-rows 300 --boost 400 --liq-floor-tl 3000000

Outputs:
    outputs/e2_faz7_confidence_diracc.md
    outputs/e2_faz7_confidence_diracc.csv
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
from src.serving.confidence import ConfidenceThresholds, peer_confidence
from src.serving.nightly_scoring import composite_icir, liqlog_floor_from_turnover
from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward
from src.validation.pooled_oos import PerSymbolOOSConfig, evaluate_per_symbol
from src.validation.segment_ic import (
    attach_segments,
    segment_cross_sectional_ic,
    symbol_segments,
)


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _icir_map(preds: pd.DataFrame, col: str, min_names: int) -> dict[str, float]:
    tbl = segment_cross_sectional_ic(preds, group_col=col, min_names=min_names)
    out = {}
    for _, r in tbl.iterrows():
        v = r.get("icir")
        out[str(r["segment"])] = float(v) if v is not None and np.isfinite(v) else float("nan")
    return out


def _quintile(s: pd.Series) -> pd.Series:
    """y_pred -> Q1..Q5 (Q5 en yuksek tahmin). Az isimde sabit-kova guvenli."""
    try:
        return pd.qcut(s.rank(method="first"), 5,
                       labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str)
    except ValueError:
        return pd.Series(["Q3"] * len(s), index=s.index)


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
    ap.add_argument("--liq-floor-tl", type=float, default=3_000_000.0)
    ap.add_argument("--quintile-min-names", type=int, default=15)
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
    # per-symbol medyan liq_log (tradability kapisi icin).
    sym_liq = panel.groupby("symbol")["liq_log"].median().to_dict()
    floor = liqlog_floor_from_turnover(args.liq_floor_tl)

    aug0, f0, _ = build_pooled_features(panel)
    panel, _ = add_cross_sectional_features(
        panel, [c for c in f0 if c not in ("symbol_id", "sector_code")])
    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=63, n_windows=6,
        min_train_days=504)).split(panel)
    aug, feats, ci = build_pooled_features(panel)
    assert not any(c.startswith("target") for c in feats), "target leak!"
    _log(f"fitting CS+CSFEAT ({len(feats)} feats) ...")
    res = evaluate_per_symbol(
        aug, folds, make_global_model_factory(ci, GlobalPooledConfig(num_boost_round=args.boost)),
        PerSymbolOOSConfig(feature_cols=feats, target_col="target_cs"))
    _log(f"overall IC {res.ic['ic_mean']:+.4f} ICIR {res.ic['icir']:+.3f} "
         f"%IC>0 {100*res.ic['pct_positive']:.1f} ({time.time()-t0:.0f}s)")

    preds = attach_segments(res.predictions, seg_table)

    # --- per-eksen segment ICIR (y_true=target_cs ile, serving ile tutarli) ---
    icir_maps = {
        "liq": _icir_map(preds, "liq_bucket", args.seg_min_names),
        "vol": _icir_map(preds, "vol_bucket", args.seg_min_names),
        "sector": _icir_map(preds, "sector", args.seg_min_names),
    }

    # --- mutlak hedef geri ekle (yon olcumu icin) ---
    abs_tbl = aug[["symbol", "Date", "target"]].rename(columns={"target": "y_abs"})
    preds = preds.merge(abs_tbl, on=["symbol", "Date"], how="left")
    preds = preds.dropna(subset=["y_abs"]).reset_index(drop=True)

    # --- per-satir confidence ---
    cic = [composite_icir(r["liq_bucket"], r["vol_bucket"], r["sector"], icir_maps)
           for _, r in preds.iterrows()]
    preds["segment_icir"] = cic
    thr = ConfidenceThresholds()
    labels = []
    for _, r in preds.iterrows():
        tradable = float(sym_liq.get(r["symbol"], float("nan"))) >= floor
        # peer_label burada onemsiz (yon olcumu quintile'dan); 'inline' ver -> kapi yok.
        c = peer_confidence(r["segment_icir"], peer_label="inline",
                            tradable=tradable, stale=False, universe_ok=True, thr=thr)
        labels.append(c.label)
    preds["confidence"] = labels

    # --- per (fold,Date) quintile ---
    q = []
    for _, g in preds.groupby(["fold", "Date"]):
        if g["symbol"].nunique() < args.quintile_min_names:
            q.append(pd.Series(["NA"] * len(g), index=g.index))
        else:
            q.append(_quintile(g["y_pred"]))
    preds["quintile"] = pd.concat(q).sort_index()
    use = preds[preds["quintile"] != "NA"].copy()
    use["up"] = (use["y_abs"] > 0).astype(float)

    base_up = float(use["up"].mean())
    _log(f"measured rows {len(use)}  base P(up)={base_up:.4f}")

    # --- confidence x quintile kir ---
    rows = []
    for conf in ["high", "medium", "low"]:
        sub = use[use["confidence"] == conf]
        if sub.empty:
            continue
        for qq in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            s = sub[sub["quintile"] == qq]
            if s.empty:
                continue
            rows.append({"confidence": conf, "quintile": qq, "n": len(s),
                         "pct_up": float(s["up"].mean()),
                         "mean_ret": float(s["y_abs"].mean())})
    grid = pd.DataFrame(rows)

    # --- ekstrem-yon isabeti per confidence (Q5->up, Q1->down) ---
    summ = []
    for conf in ["high", "medium", "low", "ALL"]:
        sub = use if conf == "ALL" else use[use["confidence"] == conf]
        if sub.empty:
            continue
        q5 = sub[sub["quintile"] == "Q5"]
        q1 = sub[sub["quintile"] == "Q1"]
        n_ext = len(q5) + len(q1)
        if n_ext == 0:
            continue
        # Q5 -> 'up' tahmin dogru = up==1; Q1 -> 'down' tahmin dogru = up==0
        correct = float(q5["up"].sum() + (len(q1) - q1["up"].sum()))
        dir_acc = correct / n_ext
        summ.append({
            "confidence": conf, "n": len(sub),
            "base_up": float(sub["up"].mean()),
            "q5_pct_up": float(q5["up"].mean()) if len(q5) else float("nan"),
            "q1_pct_up": float(q1["up"].mean()) if len(q1) else float("nan"),
            "extreme_dir_acc": dir_acc,
            "ret_spread_q5_q1": (float(q5["y_abs"].mean()) - float(q1["y_abs"].mean()))
            if len(q5) and len(q1) else float("nan"),
        })
    summary = pd.DataFrame(summ)

    os.makedirs("outputs", exist_ok=True)
    grid.to_csv("outputs/e2_faz7_confidence_diracc.csv", index=False, encoding="utf-8-sig")

    lines = ["# E2 Faz 7 — Confidence-stratified absolute directional accuracy", "",
             f"- panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
             f"{panel['Date'].nunique()} dates, h={args.horizon}, boost={args.boost}",
             f"- overall IC {res.ic['ic_mean']:+.4f} / ICIR {res.ic['icir']:+.3f} / "
             f"%IC>0 {100*res.ic['pct_positive']:.1f}",
             f"- liq floor: {args.liq_floor_tl:,.0f} TL/gun -> liq_log>={floor:.3f}",
             f"- measured rows {len(use)}  base P(up)={base_up:.4f}", "",
             "## Confidence dagilimi (satir bazli)"]
    vc = use["confidence"].value_counts()
    for k in ["high", "medium", "low"]:
        lines.append(f"- {k}: {int(vc.get(k, 0))}")
    lines += ["",
              "## Ozet: ekstrem-yon isabeti (Q5=yukari, Q1=asagi tahmin)",
              "| confidence | n | base P(up) | Q5 %up | Q1 %up | dir_acc(Q5+Q1) | ret_spread Q5-Q1 |",
              "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['confidence']} | {int(r['n'])} | {r['base_up']:.4f} | "
            f"{r['q5_pct_up']:.4f} | {r['q1_pct_up']:.4f} | "
            f"{r['extreme_dir_acc']:.4f} | {r['ret_spread_q5_q1']:+.4f} |")
    lines += ["", "## Detay: confidence x quintile",
              "| confidence | quintile | n | %up | mean_ret |", "|---|---|---|---|---|"]
    for _, r in grid.iterrows():
        lines.append(f"| {r['confidence']} | {r['quintile']} | {int(r['n'])} | "
                     f"{r['pct_up']:.4f} | {r['mean_ret']:+.4f} |")
    lines.append(f"\n- total elapsed: {time.time()-t0:.0f}s")
    with open("outputs/e2_faz7_confidence_diracc.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    _log("wrote outputs/e2_faz7_confidence_diracc.md")

    # konsol ozet
    _log("SUMMARY extreme-dir-acc by confidence:")
    for _, r in summary.iterrows():
        _log(f"  {r['confidence']:>6}: n={int(r['n']):>6} base_up={r['base_up']:.3f} "
             f"Q5up={r['q5_pct_up']:.3f} Q1up={r['q1_pct_up']:.3f} "
             f"dir_acc={r['extreme_dir_acc']:.3f} spread={r['ret_spread_q5_q1']:+.4f}")
    _log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
