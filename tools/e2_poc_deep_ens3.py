# -*- coding: utf-8 -*-
"""E2 PoC (D karar kapisi) — 3-bacak ensemble (LGB + MLP + Seq-LSTM).

Soru: Seq-LSTM 3. bacak olarak eklenince mevcut 2-bacak (LGB+MLP) ensemble'in
en iyi ICIR'ini (~1.670) gecer mi? LSTM standalone LGB'yi ezmedi (full: ICIR
1.599 vs 1.557) ama pred rank corr 0.53 = decorrelated -> ensemble katkisi
olabilir. Bu arac onu OLCER; net kazanc yoksa LSTM raf.

Adil kosul: TEK manuel fold dongusu, UC bacak da AYNI gecerli-satir evreninde
(W lookback'i olan satirlar) ve AYNI fold mask'lerinde egitilir. Blend within-date
pct-rank uzerinde (serving rank_to_peer_scores ile ayni mantik). Metrik
daily_cross_sectional_ic.

Bacaklar:
  - LGB  : GlobalPooledModel (CS+CSFEAT, cat native)
  - MLP  : TorchMLPModel (embedding feedforward), N seed ortalamasi (production=3)
  - LSTM : SeqLSTMModel (20-gun sequence), M seed ortalamasi

Run (dl_env), background-friendly:
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1
    python tools/e2_poc_deep_ens3.py --boost 400 --mlp-epochs 15 --lstm-epochs 8

Outputs:
    outputs/e2_poc_deep_ens3.md
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
    GlobalPooledModel,
    build_pooled_features,
)
from src.models.torch_mlp_model import TorchMLPConfig, TorchMLPModel
from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward
from src.validation.pooled_oos import daily_cross_sectional_ic
from tools.e2_poc_deep_lstm import SeqLSTMModel, build_lookback_index


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pct_rank_within_date(df: pd.DataFrame, col: str) -> pd.Series:
    """col'u her tarih icinde [0,1] pct-rank'a cevir (NaN -> 0.5)."""
    r = df.groupby("Date")[col].rank(pct=True)
    return r.fillna(0.5)


def _ic_from(preds: pd.DataFrame, pred_col: str, min_names: int) -> dict:
    d = preds.rename(columns={pred_col: "y_pred"})[["symbol", "Date", "y_true", "y_pred"]]
    return daily_cross_sectional_ic(d, min_names=min_names)


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
    # MLP
    ap.add_argument("--mlp-epochs", type=int, default=15)
    ap.add_argument("--mlp-seeds", type=int, default=3)
    # LSTM
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--lstm-epochs", type=int, default=8)
    ap.add_argument("--lstm-seeds", type=int, default=2)
    ap.add_argument("--lstm-hidden", type=int, default=64)
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
    _log(f"panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols (load {time.time()-t0:.0f}s)")

    panel = add_cross_sectional_target(
        panel, raw_target="target", out_col="target_cs", method="rank", min_names=args.min_names)

    aug0, feats0, _ = build_pooled_features(panel)
    cs_base = [c for c in feats0 if c not in ("symbol_id", "sector_code")]
    panel, _ = add_cross_sectional_features(panel, cs_base)
    aug, feats, cat_idx = build_pooled_features(panel)
    assert not any(c.startswith("target") for c in feats), "target leak!"

    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=args.window_len,
        n_windows=args.n_windows, min_train_days=args.min_train_days,
    )).split(aug)
    sel = [f for f in folds if not f.is_final_holdout]
    _log(f"folds: {len(folds)} ({len(sel)} selectable + holdout)")

    num_cols = [feats[i] for i in range(len(feats)) if i not in cat_idx]
    cat_cols = [feats[i] for i in cat_idx]
    cardinalities = [int(aug[c].max()) + 1 for c in cat_cols]
    _log(f"features: {len(feats)} ({len(num_cols)} num + {len(cat_cols)} cat) {dict(zip(cat_cols, cardinalities))}")

    X_lgb = aug[feats].to_numpy(dtype=float)
    Xnum = aug[num_cols].to_numpy(dtype=float)
    Xcat = aug[cat_cols].to_numpy(dtype=np.int64)
    y_cs = aug["target_cs"].to_numpy(dtype=float)
    sym_all = aug["symbol"].to_numpy()
    date_all = aug["Date"].to_numpy()

    lb = build_lookback_index(aug, args.window)
    has_lb = (lb >= 0).all(axis=1)
    _log(f"lookback W={args.window}: {int(has_lb.sum())}/{len(aug)} valid rows")

    lstm_cfg = {"hidden": args.lstm_hidden, "dropout": 0.2, "lr": 1e-3,
                "wd": 1e-5, "epochs": args.lstm_epochs, "batch": 4096, "seed": 42}

    pred_rows: list[pd.DataFrame] = []
    n_used = 0
    for f in sel:
        tr = f.train_mask & has_lb
        te = f.test_mask & has_lb
        if not tr.any() or not te.any():
            continue
        tr_rows = np.where(tr)[0]
        te_rows = np.where(te)[0]
        n_used += 1
        tf = time.time()

        # --- LGB ---
        lgb = GlobalPooledModel(GlobalPooledConfig(
            num_boost_round=args.boost, cat_indices=tuple(cat_idx)))
        lgb.fit(X_lgb[tr_rows], y_cs[tr_rows])
        p_lgb = lgb.predict(X_lgb[te_rows])

        # --- MLP (seed ort) ---
        acc_mlp = np.zeros(len(te_rows), dtype=float)
        for s in range(args.mlp_seeds):
            m = TorchMLPModel(TorchMLPConfig(
                epochs=args.mlp_epochs, seed=42 + s * 101,
                cat_indices=tuple(cat_idx), cat_cardinalities=tuple(cardinalities)))
            m.fit(X_lgb[tr_rows], y_cs[tr_rows])
            acc_mlp += m.predict(X_lgb[te_rows])
        p_mlp = acc_mlp / args.mlp_seeds

        # --- LSTM (seed ort) ---
        acc_lstm = np.zeros(len(te_rows), dtype=float)
        for s in range(args.lstm_seeds):
            cfg_s = dict(lstm_cfg, seed=lstm_cfg["seed"] + s * 101)
            ls = SeqLSTMModel(Xnum, Xcat, y_cs, lb, cardinalities, cfg_s)
            ls.fit(tr_rows)
            acc_lstm += ls.predict(te_rows)
        p_lstm = acc_lstm / args.lstm_seeds

        pred_rows.append(pd.DataFrame({
            "symbol": sym_all[te_rows], "Date": date_all[te_rows],
            "y_true": y_cs[te_rows], "lgb": p_lgb, "mlp": p_mlp, "lstm": p_lstm}))
        _log(f"  fold {f.index} [{f.test_date_start.date()}..{f.test_date_end.date()}] "
             f"n_tr={len(tr_rows)} n_te={len(te_rows)} ({time.time()-tf:.0f}s)")

    preds = pd.concat(pred_rows, ignore_index=True)

    # within-date pct-rank her bacak
    preds["r_lgb"] = _pct_rank_within_date(preds, "lgb")
    preds["r_mlp"] = _pct_rank_within_date(preds, "mlp")
    preds["r_lstm"] = _pct_rank_within_date(preds, "lstm")

    mn = args.min_names
    rows = []
    # standalone (ham pred)
    rows.append(("LGB", _ic_from(preds, "lgb", mn)))
    rows.append(("MLP", _ic_from(preds, "mlp", mn)))
    rows.append(("LSTM", _ic_from(preds, "lstm", mn)))

    # 2-bacak LGB+MLP (mevcut sampiyon, 50/50 rank-blend)
    preds["b2_lgb_mlp"] = 0.5 * preds["r_lgb"] + 0.5 * preds["r_mlp"]
    rows.append(("2-leg LGB+MLP (0.5/0.5)", _ic_from(preds, "b2_lgb_mlp", mn)))

    # 3-bacak kombolar (rank-blend)
    combos = {
        "3-leg equal (1/3)": (1 / 3, 1 / 3, 1 / 3),
        "3-leg 0.4/0.4/0.2": (0.4, 0.4, 0.2),
        "3-leg 0.4/0.3/0.3": (0.4, 0.3, 0.3),
        "3-leg 0.5/0.3/0.2": (0.5, 0.3, 0.2),
    }
    for name, (wl, wm, ws) in combos.items():
        col = "blend_" + name
        preds[col] = wl * preds["r_lgb"] + wm * preds["r_mlp"] + ws * preds["r_lstm"]
        rows.append((name, _ic_from(preds, col, mn)))

    # decorrelation matrisi (gunluk pred rank corr)
    def _avg_corr(a: str, b: str) -> float:
        cs = []
        for _, g in preds.groupby("Date"):
            if len(g) >= mn:
                ra, rb = g[a].rank().to_numpy(), g[b].rank().to_numpy()
                if ra.std() > 0 and rb.std() > 0:
                    cs.append(float(np.corrcoef(ra, rb)[0, 1]))
        return float(np.mean(cs)) if cs else float("nan")

    c_lm = _avg_corr("lgb", "mlp")
    c_ll = _avg_corr("lgb", "lstm")
    c_ml = _avg_corr("mlp", "lstm")

    for label, ic in rows:
        _log(f"  {label:28s} IC {ic['ic_mean']:+.4f} ICIR {ic['icir']:+.3f} "
             f"%IC>0 {100*ic['pct_positive']:.1f} n_days {ic['n_days']}")
    _log(f"corr LGB-MLP {c_lm:+.3f}  LGB-LSTM {c_ll:+.3f}  MLP-LSTM {c_ml:+.3f}")

    os.makedirs("outputs", exist_ok=True)
    out = ["# E2 PoC (D karar kapisi) — 3-bacak ensemble (LGB+MLP+LSTM)", ""]
    out.append(f"- panel: {len(aug)} rows, {aug['symbol'].nunique()} symbols, "
               f"{aug['Date'].nunique()} dates, h={args.horizon}, folds={n_used}")
    out.append(f"- config: CS+CSFEAT (target_cs), {len(feats)} feats, min_names={mn}")
    out.append(f"- MLP: epochs={args.mlp_epochs} seeds={args.mlp_seeds} | "
               f"LSTM: W={args.window} epochs={args.lstm_epochs} seeds={args.lstm_seeds} "
               f"hidden={args.lstm_hidden} | LGB: boost={args.boost}")
    out.append("")
    out.append("| config | daily IC | ICIR | %IC>0 | n_days |")
    out.append("|---|---|---|---|---|")
    for label, ic in rows:
        out.append(f"| {label} | {ic['ic_mean']:+.4f} | {ic['icir']:+.3f} | "
                   f"{100*ic['pct_positive']:.1f} | {ic['n_days']} |")
    out.append("")
    out.append(f"- pred rank corr: LGB-MLP {c_lm:+.3f}, LGB-LSTM {c_ll:+.3f}, "
               f"MLP-LSTM {c_ml:+.3f} (dusuk = decorrelated = ensemble degeri)")
    out.append(f"- total elapsed: {time.time()-t0:.0f}s")
    with open("outputs/e2_poc_deep_ens3.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    _log("wrote outputs/e2_poc_deep_ens3.md")
    _log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
