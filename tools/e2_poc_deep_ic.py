# -*- coding: utf-8 -*-
"""E2 PoC — Pooled DEEP model vs LightGBM on cross-sectional IC.

Question (Kol B genislemesi): bir sinir agi (derin), ayni cross-sectional
gorevde LightGBM'in ICIR'ini gecebilir mi? Esit kosul:
  - AYNI panel snapshot, AYNI purged date-walk-forward fold'lar,
  - AYNI en-iyi config (CS+CSFEAT, target_cs),
  - AYNI metrik (gunluk cross-sectional IC / ICIR, pooled_oos harness).

Bu yuzden LightGBM baseline AYNI kosuda yeniden hesaplanir (tarihsel 1.55 ile
kiyas haksiz olurdu — data/ CSV'leri tazeleniyor, snapshot kayar).

DEEP = embedding'li feedforward MLP (torch):
  - kategorik symbol_id + sector_code -> ogrenilen embedding (LightGBM'in native
    categorical destegine adil karsilik; ham int magnitude DEGIL).
  - sayisal ozellikler train-only standardize (leakage yok; fit icinde).
  - Neden MLP (LSTM degil): mevcut ozellikler zaten zamansal ozet (lag/momentum/
    MA). Sequence LSTM ancak HAM seri ile deger katar = buyuk redesign. MLP "NN
    bu ozelliklerden trees'ten fazlasini cikarir mi"yi dogrudan/hizli/leak-safe
    olcer. Kazanirsa ham-sequence LSTM Faz 2.

Run (dl_env), background-friendly:
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1
    python tools/e2_poc_deep_ic.py --boost 400 --epochs 12

Outputs:
    outputs/e2_poc_deep_ic.md
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
from src.validation.pooled_oos import PerSymbolOOSConfig, evaluate_per_symbol


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── DEEP model: embedding'li MLP (torch), sklearn-vari fit/predict ────────────
class TorchMLPModel:
    """pooled_oos harness uyumlu (fit/predict, satir-bazli).

    cat_indices: feature matrisindeki kategorik kolon pozisyonlari
    cat_cardinalities: her kategorik kolonun vocab boyutu (global, CV-oncesi bilinir)
    """

    def __init__(self, cat_indices, cat_cardinalities, cfg):
        self.cat_idx = list(cat_indices)
        self.cardinalities = list(cat_cardinalities)
        self.cfg = cfg
        self.net = None
        self.mu = None
        self.sd = None
        self.num_idx = None

    def _build(self, n_numeric):
        import torch
        import torch.nn as nn

        emb_dims = [min(50, (c + 1) // 2) for c in self.cardinalities]
        embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(self.cardinalities, emb_dims)])
        in_dim = n_numeric + sum(emb_dims)
        h = self.cfg["hidden"]
        layers, prev = [], in_dim
        for hid in h:
            layers += [nn.Linear(prev, hid), nn.ReLU(), nn.BatchNorm1d(hid), nn.Dropout(self.cfg["dropout"])]
            prev = hid
        layers += [nn.Linear(prev, 1)]

        class Net(nn.Module):
            def __init__(self, embs, mlp, cat_idx, num_idx):
                super().__init__()
                self.embs = embs
                self.mlp = nn.Sequential(*mlp)
                self.cat_idx = cat_idx
                self.num_idx = num_idx

            def forward(self, x):
                num = x[:, self.num_idx]
                parts = [num]
                for j, emb in enumerate(self.embs):
                    idx = x[:, self.cat_idx[j]].long().clamp(0, emb.num_embeddings - 1)
                    parts.append(emb(idx))
                return self.mlp(torch.cat(parts, dim=1)).squeeze(-1)

        return Net(embs, layers, self.cat_idx, self.num_idx)

    def fit(self, X, y):
        import torch

        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.num_idx = [j for j in range(X.shape[1]) if j not in self.cat_idx]

        # train-only standardize (leakage yok)
        num = X[:, self.num_idx]
        self.mu = num.mean(axis=0)
        self.sd = num.std(axis=0)
        self.sd[self.sd == 0] = 1.0

        Xs = X.copy()
        Xs[:, self.num_idx] = (num - self.mu) / self.sd
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

        self.net = self._build(len(self.num_idx))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        loss_fn = torch.nn.MSELoss()
        Xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        n = len(yt)
        bs = self.cfg["batch"]
        self.net.train()
        for ep in range(self.cfg["epochs"]):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                bi = perm[i:i + bs]
                opt.zero_grad()
                out = self.net(Xt[bi])
                loss = loss_fn(out, yt[bi])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch

        X = np.asarray(X, dtype=float)
        Xs = X.copy()
        Xs[:, self.num_idx] = (X[:, self.num_idx] - self.mu) / self.sd
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(Xs, dtype=torch.float32)).numpy()
        return np.asarray(out, dtype=float).ravel()


def make_mlp_factory(cat_indices, cat_cardinalities, cfg):
    def _f():
        return TorchMLPModel(cat_indices, cat_cardinalities, cfg)
    return _f


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
    # MLP hiperparametreleri
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", default="256,128,64")
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
    _log(f"after cs filter: {len(panel)} rows, {panel['symbol'].nunique()} symbols, {panel['Date'].nunique()} dates")

    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=args.window_len,
        n_windows=args.n_windows, min_train_days=args.min_train_days,
    )).split(panel)
    sel = [f for f in folds if not f.is_final_holdout]
    _log(f"folds: {len(folds)} ({len(sel)} selectable + holdout)")

    # cs-features (Faz 3.6) — en iyi config
    aug0, feats0, _ = build_pooled_features(panel)
    cs_base = [c for c in feats0 if c not in ("symbol_id", "sector_code")]
    panel, cs_new = add_cross_sectional_features(panel, cs_base)
    aug, feats, cat_idx = build_pooled_features(panel)
    assert not any(c.startswith("target") for c in feats), "target leak!"
    _log(f"features: {len(feats)} total (cat at {cat_idx})")

    # kategorik vocab boyutlari (global, CV-oncesi -> leakage degil)
    cardinalities = [int(aug[feats[i]].max()) + 1 for i in cat_idx]
    _log(f"cat cardinalities: {dict(zip([feats[i] for i in cat_idx], cardinalities))}")

    cfg_oos = PerSymbolOOSConfig(feature_cols=feats, target_col="target_cs")

    results = {}

    # --- LightGBM baseline (ayni kosu) ---
    _log(f"fitting LightGBM baseline (boost={args.boost}) ...")
    ts = time.time()
    lgb_factory = make_global_model_factory(cat_idx, GlobalPooledConfig(num_boost_round=args.boost))
    res_lgb = evaluate_per_symbol(aug, folds, lgb_factory, cfg_oos)
    results["LightGBM"] = res_lgb.ic
    _log(f"  LightGBM: IC {res_lgb.ic['ic_mean']:+.4f} ICIR {res_lgb.ic['icir']:+.3f} "
         f"%IC>0 {100*res_lgb.ic['pct_positive']:.1f} n_days {res_lgb.ic['n_days']} ({time.time()-ts:.0f}s)")

    # --- DEEP: embedding MLP ---
    mlp_cfg = {
        "hidden": [int(x) for x in args.hidden.split(",")],
        "dropout": args.dropout, "lr": args.lr, "wd": args.wd,
        "epochs": args.epochs, "batch": args.batch, "seed": 42,
    }
    _log(f"fitting DEEP MLP {mlp_cfg['hidden']} epochs={args.epochs} batch={args.batch} ...")
    ts = time.time()
    mlp_factory = make_mlp_factory(cat_idx, cardinalities, mlp_cfg)
    res_mlp = evaluate_per_symbol(aug, folds, mlp_factory, cfg_oos)
    results["DEEP-MLP"] = res_mlp.ic
    _log(f"  DEEP-MLP: IC {res_mlp.ic['ic_mean']:+.4f} ICIR {res_mlp.ic['icir']:+.3f} "
         f"%IC>0 {100*res_mlp.ic['pct_positive']:.1f} n_days {res_mlp.ic['n_days']} ({time.time()-ts:.0f}s)")

    # --- rapor ---
    os.makedirs("outputs", exist_ok=True)
    lines = ["# E2 PoC — Pooled DEEP (MLP) vs LightGBM (cross-sectional IC)", ""]
    lines.append(f"- panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
                 f"{panel['Date'].nunique()} dates, h={args.horizon}")
    lines.append(f"- config: CS+CSFEAT (target_cs), {len(feats)} feats, folds={len(sel)}+holdout, "
                 f"min_names={args.min_names}")
    lines.append(f"- MLP: hidden={mlp_cfg['hidden']} dropout={args.dropout} lr={args.lr} "
                 f"wd={args.wd} epochs={args.epochs} batch={args.batch}")
    lines.append(f"- LGB: boost={args.boost}")
    lines.append("")
    lines.append("| model | daily IC | ICIR | %IC>0 | n_days |")
    lines.append("|---|---|---|---|---|")
    for label, ic in results.items():
        lines.append(f"| {label} | {ic['ic_mean']:+.4f} | {ic['icir']:+.3f} | "
                     f"{100*ic['pct_positive']:.1f} | {ic['n_days']} |")
    lines.append("")
    lines.append(f"- total elapsed: {time.time()-t0:.0f}s")
    with open("outputs/e2_poc_deep_ic.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    _log("wrote outputs/e2_poc_deep_ic.md")
    _log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
