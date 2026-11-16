# -*- coding: utf-8 -*-
"""Kalibrasyon raporu: urunun soyledigi guven/olasilik gercekte tutuyor mu?

Iki olcum, ikisi de walk-forward OOS tahminleri uzerinde:

  A. Trend tablosu (`TrendCalibration`): peer quintile -> P(5g getiri > 0) ve ort. getiri.
     Sabit tablo, fold bazinda ve tarih-kumelenmis (ortusmeyen tarih) guven araligiyla
     yeniden olculur; Brier skoru taban orana (0.5 ve o gunun evren orani) gore verilir.
  B. Guven etiketi (low/medium/high): her fold icin segment ICIR haritalari YALNIZ onceki
     fold'larin OOS tahminlerinden kurulur (ileriye donuk, sizintisiz), `peer_confidence`
     ile etiketlenir; etiket bazinda gerceklesen kesitsel IC ve Q5-Q1 getiri farki olculur.
     Dogru kalibre bir etiket monoton olmali: high > medium > low.

Model: gece kosusunun OOS adimiyla ayni (LightGBM, target_cs). Risk bandi kapsamasi zaten
gece kosusunda olculuyor (`risk_validation.coverage`), burada tekrar edilmez.

    python tools/calibration_report.py --n-windows 12 --boost 400
    python tools/calibration_report.py --limit 80 --boost 50 --n-windows 6   # hizli deneme
Cikti: outputs/_audits/calibration_<tarih>.json ve .md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.pooled_dataset import build_pooled_dataset
from src.data.pooled_loader import PooledLoaderConfig
from src.models.global_pooled_model import GlobalPooledConfig, make_global_model_factory
from src.serving.confidence import ConfidenceThresholds, peer_confidence, relative_thresholds
from src.serving.nightly import build_parser as nightly_parser
from src.serving.nightly_scoring import (
    composite_icir, composite_n_eff, liqlog_floor_from_turnover,
    segment_icir_from_table, segment_neff_from_table)
from src.serving.trend_tendency import TrendCalibration
from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward
from src.validation.pooled_oos import PerSymbolOOSConfig, evaluate_per_symbol
from src.validation.segment_ic import attach_date_segments, segment_cross_sectional_ic

_IST = timezone(timedelta(hours=3))
_H = 5
_AXES = (("liq", "liq_bucket"), ("vol", "vol_bucket"), ("sector", "sector"))


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _nonoverlap_dates(dates: pd.Series, h: int = _H) -> set:
    """Her h'inci tarih: ortusen 5g hedeflerin otokorelasyonunu kirar."""
    u = np.sort(pd.unique(dates))
    return set(u[::h])


def _mean_ci(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else None, "ci95": None, "n": int(len(x))}
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    m = float(x.mean())
    return {"mean": m, "ci95": [m - 1.96 * se, m + 1.96 * se], "n": int(len(x))}


def add_quintiles(preds: pd.DataFrame) -> pd.DataFrame:
    """Tarih ici peer percentile (urunle ayni formul) ve quintile 0..4."""
    out = preds.copy()
    g = out.groupby("Date")["y_pred"]
    pct = 100.0 * (g.rank(method="average") - 0.5) / g.transform("size")
    out["pct"] = pct
    out["q"] = np.clip((pct // 20.0).astype(int), 0, 4)
    # akranlarini gecme: getirisi o gunun evren medyaninin ustunde
    out["beat"] = (out["ret"] > out.groupby("Date")["ret"].transform("median")).astype(float)
    return out


def trend_calibration(df: pd.DataFrame, cfg: TrendCalibration) -> dict:
    """A: quintile -> P(up), ort. getiri; tarih-kumelenmis CI; Brier ve fold kararliligi."""
    keep = _nonoverlap_dates(df["Date"])
    dfn = df[df["Date"].isin(keep)]
    rows = []
    for q in range(5):
        dq = dfn[dfn["q"] == q]
        per_date_up = dq.groupby("Date")["ret"].apply(lambda r: float((r > 0).mean()))
        per_date_ret = dq.groupby("Date")["ret"].mean()
        per_date_beat = dq.groupby("Date")["beat"].mean()
        rows.append({
            "quintile": f"Q{q + 1}",
            "table_prob_up": cfg.quintile_prob_up[q],
            "table_exp_ret": cfg.quintile_expected_return[q],
            "oos_prob_up": _mean_ci(per_date_up.to_numpy()),
            "oos_exp_ret": _mean_ci(per_date_ret.to_numpy()),
            "oos_prob_outperform": _mean_ci(per_date_beat.to_numpy()),
        })
    per_fold = {}
    for f, g in sorted(df.groupby("fold"), key=lambda kv: kv[1]["Date"].min()):
        per_fold[int(f)] = {
            "start": str(pd.Timestamp(g["Date"].min()).date()),
            "end": str(pd.Timestamp(g["Date"].max()).date()),
            "prob_up_by_q": [float((g.loc[g["q"] == q, "ret"] > 0).mean()) for q in range(5)],
            "base_up": float((g["ret"] > 0).mean()),
        }
    up = (df["ret"] > 0).astype(float).to_numpy()
    p_table = np.asarray(cfg.quintile_prob_up)[df["q"].to_numpy()]
    p_date = df.groupby("Date")["ret"].transform(lambda r: float((r > 0).mean())).to_numpy()
    beat = df["beat"].to_numpy()
    p_beat = df.groupby("q")["beat"].transform("mean").to_numpy()
    brier = {
        "outperform_in_sample_quintile": float(np.mean((p_beat - beat) ** 2)),
        "outperform_coin_0.5": float(np.mean((0.5 - beat) ** 2)),
        "table": float(np.mean((p_table - up) ** 2)),
        "coin_0.5": float(np.mean((0.5 - up) ** 2)),
        # "kahin" taban: o gunun evren yukselis orani (gelecek bilgisi; ust sinir referansi)
        "same_day_base_rate_oracle": float(np.mean((p_date - up) ** 2)),
    }
    spread = df[df["Date"].isin(keep)].groupby("Date")[["q", "ret"]].apply(
        lambda g: g.loc[g["q"] == 4, "ret"].mean() - g.loc[g["q"] == 0, "ret"].mean())
    day_up = df[df["Date"].isin(keep)].groupby("Date")["ret"].apply(lambda r: float((r > 0).mean()))
    return {"quintiles": rows, "per_fold": per_fold, "brier": brier,
            "q5_minus_q1_ret": _mean_ci(spread.to_numpy()),
            "cross_date_std_of_base_up": float(day_up.std(ddof=1)) if len(day_up) > 1 else None}


def _daily_ic(g: pd.DataFrame, min_names: int) -> float:
    if len(g) < min_names:
        return float("nan")
    return float(g["y_pred"].rank().corr(g["ret"].rank()))


def forward_confidence_labels(df: pd.DataFrame, liq_floor: float) -> pd.DataFrame:
    """B: fold k satirlarini yalniz fold < k OOS tahminlerinden kurulan ICIR ile etiketle."""
    # fold indeksi kronolojik DEGIL (0 = en yeni pencere); test baslangicina gore sirala
    folds = list(df.groupby("fold")["Date"].min().sort_values().index)
    out = []
    for i, k in enumerate(folds):
        if i < 2:   # en az iki onceki fold (n_eff ~25) olmadan ICIR anlamsiz
            continue
        prior = df[df["fold"].isin(folds[:i])]
        tables = {a: segment_cross_sectional_ic(prior, group_col=c, min_names=10, horizon=_H)
                  for a, c in _AXES}
        icir_maps = {a: segment_icir_from_table(t) for a, t in tables.items()}
        neff_maps = {a: segment_neff_from_table(t) for a, t in tables.items()}
        cur = df[df["fold"] == k].copy()
        cur["seg_icir_prior"] = [
            composite_icir(r.liq_bucket, r.vol_bucket, r.sector, icir_maps)
            for r in cur[["liq_bucket", "vol_bucket", "sector"]].itertuples(index=False)]
        cur["neff_prior"] = [
            composite_n_eff(r.liq_bucket, r.vol_bucket, r.sector, neff_maps)
            for r in cur[["liq_bucket", "vol_bucket", "sector"]].itertuples(index=False)]
        cur["tradable"] = np.isfinite(cur["liq_log"]) & (cur["liq_log"] >= liq_floor)
        old_thr = ConfidenceThresholds()
        new_l, old_l = {}, {}
        for _d, g in cur.groupby("Date"):
            thr = relative_thresholds(g["seg_icir_prior"])     # urun: skorlama gunu evreni
            for idx, icir, neff, trd in zip(g.index, g["seg_icir_prior"], g["neff_prior"], g["tradable"]):
                new_l[idx] = peer_confidence(icir, "x", tradable=bool(trd), thr=thr, n_eff=neff).label
                old_l[idx] = (peer_confidence(icir, "x", thr=old_thr, n_eff=neff).label
                              if trd else "low")              # eski: likidite kapisi -> low
        cur["conf"] = pd.Series(new_l)
        cur["conf_old"] = pd.Series(old_l)
        out.append(cur)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def confidence_by_label(lab: pd.DataFrame, col: str, min_names: int = 20) -> dict:
    """Etiket bazinda: gerceklesen kesitsel IC (ortusmeyen), Q5-Q1 farki, satir payi."""
    keep = _nonoverlap_dates(lab["Date"])
    res = {}
    for name, g in lab.groupby(col):
        gn = g[g["Date"].isin(keep)]
        ics = gn.groupby("Date")[["y_pred", "ret"]].apply(lambda d: _daily_ic(d, min_names)).to_numpy()
        ic = _mean_ci(ics)
        finite = ics[np.isfinite(ics)]
        icir = float(finite.mean() / finite.std(ddof=1)) if len(finite) > 2 and finite.std() > 0 else None
        # quintile'lar TUM evrende (urundeki gibi); etiket icinde Q5-Q1 getiri farki
        spread = gn.groupby("Date")[["q", "ret"]].apply(
            lambda d: d.loc[d["q"] == 4, "ret"].mean() - d.loc[d["q"] == 0, "ret"].mean())
        res[str(name)] = {
            "share_rows": float(len(g) / len(lab)),
            "ic": ic, "icir_nonoverlap": icir,
            "q5_minus_q1_ret": _mean_ci(spread.to_numpy()),
            "q5_prob_up": float((g.loc[g["q"] == 4, "ret"] > 0).mean()) if (g["q"] == 4).any() else None,
        }
    return res


def icir_tercile_check(lab: pd.DataFrame, min_names: int = 20) -> dict:
    """Onceki ICIR tahmini (surekli) sonraki fold IC'sini siraliyor mu? Tertil bazinda IC."""
    lab = lab[np.isfinite(lab["seg_icir_prior"])].copy()
    if lab.empty:
        return {}
    lab["icir_tercile"] = pd.qcut(lab["seg_icir_prior"].rank(method="first"), 3,
                                  labels=["alt", "orta", "ust"])
    keep = _nonoverlap_dates(lab["Date"])
    out = {}
    for t, g in lab[lab["Date"].isin(keep)].groupby("icir_tercile", observed=True):
        ics = g.groupby("Date")[["y_pred", "ret"]].apply(lambda d: _daily_ic(d, min_names)).to_numpy()
        out[str(t)] = {"prior_icir_median": float(g["seg_icir_prior"].median()), "ic": _mean_ci(ics)}
    return out


def _fmt_ci(d: dict, pct: bool = False) -> str:
    if not d or d.get("mean") is None:
        return "-"
    f = (lambda v: f"{100 * v:+.2f}%") if pct else (lambda v: f"{v:.3f}")
    ci = d.get("ci95")
    return f"{f(d['mean'])} [{f(ci[0])}, {f(ci[1])}]" if ci else f(d["mean"])


def to_markdown(rep: dict) -> str:
    L = [f"# Kalibrasyon raporu ({rep['generated_at'][:10]})", "",
         f"OOS: {rep['n_rows']} satir, {rep['n_dates']} tarih, {rep['n_folds']} fold "
         f"({rep['oos_start']} - {rep['oos_end']}). Guven araliklari ortusmeyen tarihler "
         f"(her {_H}. gun) uzerinden, tarih-kumelenmis.", "",
         "## A. Trend tablosu (peer quintile -> mutlak yon)", "",
         "| Q | tablo P(up) | OOS P(up) [95% CI] | tablo ort. getiri | OOS ort. getiri [95% CI] "
         "| OOS P(akrani gecer) [95% CI] |",
         "|---|---|---|---|---|---|"]
    for r in rep["trend"]["quintiles"]:
        L.append(f"| {r['quintile']} | {r['table_prob_up']:.3f} | {_fmt_ci(r['oos_prob_up'])} | "
                 f"{100 * r['table_exp_ret']:+.2f}% | {_fmt_ci(r['oos_exp_ret'], pct=True)} | "
                 f"{_fmt_ci(r['oos_prob_outperform'])} |")
    b = rep["trend"]["brier"]
    L += ["", f"Brier: tablo {b['table']:.4f}, yazi-tura {b['coin_0.5']:.4f}, "
              f"ayni-gun evren orani (kahin) {b['same_day_base_rate_oracle']:.4f}. "
              f"Akrani gecme: quintile frekansi {b['outperform_in_sample_quintile']:.4f}, "
              f"yazi-tura {b['outperform_coin_0.5']:.4f}.",
          f"Q5-Q1 5g getiri farki: {_fmt_ci(rep['trend']['q5_minus_q1_ret'], pct=True)}. "
          f"Gunluk evren yukselis oraninin tarihler arasi std: "
          f"{rep['trend']['cross_date_std_of_base_up']:.3f}.", "",
          "Fold bazinda P(up) Q1..Q5 (taban):", ""]
    for f, v in rep["trend"]["per_fold"].items():
        L.append(f"- fold {f} ({v['start']} - {v['end']}): "
                 + " ".join(f"{p:.2f}" for p in v["prob_up_by_q"]) + f" (taban {v['base_up']:.2f})")
    for key, title in (("by_label", "yeni: evren ici tertil, likidite ayri"),
                       ("by_label_old", "eski: sabit 1.0/0.5, likidite -> low")):
        L += ["", f"## B. Guven etiketi, ileriye donuk ({title})", "",
              "| etiket | satir payi | IC [95% CI] | ICIR | Q5-Q1 [95% CI] | Q5 P(up) |",
              "|---|---|---|---|---|---|"]
        for name in ("high", "medium", "low"):
            v = rep["confidence"][key].get(name)
            if not v:
                continue
            icir = f"{v['icir_nonoverlap']:.2f}" if v["icir_nonoverlap"] is not None else "-"
            q5 = f"{v['q5_prob_up']:.3f}" if v["q5_prob_up"] is not None else "-"
            L.append(f"| {name} | {100 * v['share_rows']:.0f}% | {_fmt_ci(v['ic'])} | {icir} | "
                     f"{_fmt_ci(v['q5_minus_q1_ret'], pct=True)} | {q5} |")
    L += ["", "## C. Onceki segment ICIR tertili -> sonraki fold IC", ""]
    for t, v in rep["confidence"]["icir_terciles"].items():
        L.append(f"- {t}: onceki ICIR medyan {v['prior_icir_median']:.2f} -> IC {_fmt_ci(v['ic'])}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--boost", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/_audits")
    a = ap.parse_args(argv)
    n = nightly_parser().parse_args([])    # veri/evren/karantina ayarlari gece kosusuyla ayni

    ds = build_pooled_dataset(
        PooledLoaderConfig(
            data_dir=n.data_dir, universe_file=n.universe, target_horizon=_H,
            min_rows=n.min_rows, fallback_dirs=tuple(n.fallback_dir),
            quarantine_jumps=n.quarantine_jumps, clean_store_dir=n.clean_store_dir,
            n_jobs=n.n_jobs),
        min_names=n.min_names, limit=a.limit)
    aug, feats, ci = ds.aug, ds.feats, ds.cat_idx
    _log(f"panel {len(aug)} satir, {aug['symbol'].nunique()} sembol")
    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=_H, window_len=63, n_windows=a.n_windows, min_train_days=504)).split(aug)
    _log(f"{len(folds)} fold; OOS LightGBM ...")
    res = evaluate_per_symbol(
        aug, folds, make_global_model_factory(ci, GlobalPooledConfig(num_boost_round=a.boost)),
        PerSymbolOOSConfig(feature_cols=feats, target_col="target_cs", include_holdout=True))
    preds = attach_date_segments(res.predictions, aug, n_buckets=5)
    preds = preds.merge(aug[["symbol", "Date", "target", "liq_log"]].drop_duplicates(["symbol", "Date"]),
                        on=["symbol", "Date"], how="left").rename(columns={"target": "ret"})
    preds = add_quintiles(preds[np.isfinite(preds["ret"])])
    _log(f"OOS {len(preds)} satir; olcumler ...")

    lab = forward_confidence_labels(preds, liqlog_floor_from_turnover(n.liq_floor_tl))
    rep = {
        "generated_at": datetime.now(_IST).isoformat(timespec="seconds"),
        "n_rows": int(len(preds)), "n_dates": int(preds["Date"].nunique()),
        "n_folds": int(preds["fold"].nunique()),
        "oos_start": str(pd.Timestamp(preds["Date"].min()).date()),
        "oos_end": str(pd.Timestamp(preds["Date"].max()).date()),
        "config": {"n_windows": a.n_windows, "boost": a.boost, "limit": a.limit},
        "trend": trend_calibration(preds, TrendCalibration()),
        "confidence": {
            "n_rows_labeled": int(len(lab)),
            "by_label": confidence_by_label(lab, "conf") if len(lab) else {},
            "by_label_old": confidence_by_label(lab, "conf_old") if len(lab) else {},
            "icir_terciles": icir_tercile_check(lab) if len(lab) else {},
        },
    }
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"calibration_{rep['generated_at'][:10]}")
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=str)
    md = to_markdown(rep)
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    _log(f"yazildi: {stem}.json / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
