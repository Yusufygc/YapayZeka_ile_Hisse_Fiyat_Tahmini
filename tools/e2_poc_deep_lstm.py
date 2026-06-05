# -*- coding: utf-8 -*-
"""E2 PoC (D) — Pooled sequence LSTM vs LightGBM on cross-sectional IC.

Faz 9'un 3. bacagi. Soru: HAM ozellik dizisini (lookback penceresi) okuyan bir
LSTM, tek-satir MLP / LightGBM'in goremedigi ZAMANSAL yapidan ek cross-sectional
sinyal cikarabilir mi? Decorrelated 3. bacak = ensemble icin deger (bireysel
olarak gecmese bile).

Adil kosul (e2_poc_deep_ic ile ayni felsefe):
  - AYNI panel snapshot, AYNI purged date-walk-forward fold'lar,
  - AYNI en-iyi config (CS+CSFEAT, target_cs),
  - AYNI metrik (gunluk cross-sectional IC / ICIR, daily_cross_sectional_ic),
  - LGB baseline AYNI kosuda, AYNI gecerli-satir evreninde yeniden hesaplanir.

Neden ayri harness (evaluate_per_symbol DEGIL): harness fit/predict'e yalniz
satir-bazli X/y gecer, symbol/Date kimligini siler. LSTM per-symbol lookback
penceresi ister -> bu PoC kendi fold dongusunu yapar ama AYNI fold mask'leri ve
AYNI IC metrigini kullanir. LGB de ayni dongude, ayni satir evreninde egitilir
(rows with >= W history) -> birebir adil IC kiyasi.

Sequence tasarimi:
  - panel [Date, symbol] sirali -> bir sembolun satirlari panel-sirasinda zaten
    date-artan. Her satir t icin lookback = ayni sembolun [t-W+1 .. t] satirlari.
  - sayisal ozellikler train-only standardize (leakage yok); lookback gecmis
    girdidir (target degil) -> kullanim serbest.
  - kategorik (symbol_id, sector_code) son adimda embedding ile head'e eklenir.
  - target = target_cs (satirin tarihindeki cross-sectional rank).

Run (dl_env), background-friendly:
    set PYTHONUNBUFFERED=1 & set TQDM_DISABLE=1
    python tools/e2_poc_deep_lstm.py --boost 400 --epochs 8 --window 20

Outputs:
    outputs/e2_poc_deep_lstm.md
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
from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward
from src.validation.pooled_oos import daily_cross_sectional_ic


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── lookback index matrisi ───────────────────────────────────────────────────
def build_lookback_index(panel: pd.DataFrame, window: int) -> np.ndarray:
    """Her panel satiri icin W uzunlugunda gecmis pencere (global pozisyonlar).

    panel [Date, symbol] sirali oldugundan bir sembolun satirlari panel-sirasinda
    date-artandir. Donen matris [N, W]; yetersiz gecmisli satir (-1 doldurulmus)
    egitim/teste alinmaz.
    """
    n = len(panel)
    lb = np.full((n, window), -1, dtype=np.int64)
    sym = panel["symbol"].to_numpy()
    order = np.argsort(sym, kind="stable")  # sembol-bazli grup; grup-ici date-artan korunur
    # her sembolun global pozisyonlarini sirayla topla
    start = 0
    sorted_sym = sym[order]
    while start < n:
        end = start
        while end < n and sorted_sym[end] == sorted_sym[start]:
            end += 1
        pos = order[start:end]  # bu sembolun global pozisyonlari (date-artan)
        for p in range(len(pos)):
            if p + 1 >= window:
                lb[pos[p]] = pos[p + 1 - window: p + 1]
        start = end
    return lb


# ─── DEEP: sequence LSTM (torch) ──────────────────────────────────────────────
def _make_lstm_class():
    import torch
    import torch.nn as nn

    class _LSTMNet(nn.Module):
        def __init__(self, n_num, cardinalities, emb_dims, hidden, dropout):
            super().__init__()
            self.lstm = nn.LSTM(n_num, hidden, batch_first=True)
            self.embs = nn.ModuleList(
                [nn.Embedding(c, d) for c, d in zip(cardinalities, emb_dims)])
            head_in = hidden + sum(emb_dims)
            self.head = nn.Sequential(
                nn.Linear(head_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1))

        def forward(self, seq, cat):
            _, (h, _) = self.lstm(seq)          # h: [1, B, hidden]
            parts = [h[-1]]
            for j, emb in enumerate(self.embs):
                idx = cat[:, j].long().clamp(0, emb.num_embeddings - 1)
                parts.append(emb(idx))
            return self.head(torch.cat(parts, dim=1)).squeeze(-1)

    return _LSTMNet


class SeqLSTMModel:
    """Pooled sequence LSTM. fit(row_idx)/predict(row_idx) — global tensorlerden
    lookback'i kendi toplar. Standardizasyon train satirlarindan (leakage yok)."""

    def __init__(self, Xnum, Xcat, y, lookback, cardinalities, cfg):
        self.Xnum = Xnum                # [N, n_num] ham (standardize ICINDE fit'te)
        self.Xcat = Xcat                # [N, n_cat] int
        self.y = y                      # [N]
        self.lb = lookback              # [N, W]
        self.cardinalities = cardinalities
        self.cfg = cfg
        self.net = None
        self.mu = None
        self.sd = None
        self.device = None

    def _prep_device(self):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, tr_rows):
        import torch

        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        self._prep_device()

        # train-only standardize (sayisal)
        tr_num = self.Xnum[tr_rows]
        self.mu = np.nanmean(tr_num, axis=0)
        self.sd = np.nanstd(tr_num, axis=0)
        self.sd[self.sd == 0] = 1.0
        Xstd = (self.Xnum - self.mu) / self.sd
        Xstd = np.nan_to_num(Xstd, nan=0.0, posinf=0.0, neginf=0.0)

        Xt = torch.tensor(Xstd, dtype=torch.float32, device=self.device)
        Ct = torch.tensor(self.Xcat, dtype=torch.long, device=self.device)
        yt = torch.tensor(self.y, dtype=torch.float32, device=self.device)
        lbt = torch.tensor(self.lb, dtype=torch.long, device=self.device)

        emb_dims = [min(50, (c + 1) // 2) for c in self.cardinalities]
        self.net = _make_lstm_class()(
            Xt.shape[1], self.cardinalities, emb_dims,
            self.cfg["hidden"], self.cfg["dropout"]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"],
                               weight_decay=self.cfg["wd"])
        loss_fn = torch.nn.MSELoss()

        rows = torch.tensor(np.asarray(tr_rows), dtype=torch.long, device=self.device)
        n = len(rows)
        bs = self.cfg["batch"]
        cat_at = Ct  # kategorik son adimda = satirin kendi degeri
        self.net.train()
        for _ in range(self.cfg["epochs"]):
            perm = rows[torch.randperm(n, device=self.device)]
            for i in range(0, n, bs):
                bi = perm[i:i + bs]
                if len(bi) < 2:  # tek-ornek atla
                    continue
                seq = Xt[lbt[bi]]            # [B, W, n_num]
                cat = cat_at[bi]            # [B, n_cat]
                opt.zero_grad()
                loss = loss_fn(self.net(seq, cat), yt[bi])
                loss.backward()
                opt.step()
        # tensorleri predict icin sakla
        self._Xt, self._Ct, self._lbt = Xt, Ct, lbt
        return self

    def predict(self, te_rows):
        import torch

        self.net.eval()
        rows = torch.tensor(np.asarray(te_rows), dtype=torch.long, device=self.device)
        bs = self.cfg["batch"]
        out = np.empty(len(rows), dtype=float)
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                bi = rows[i:i + bs]
                seq = self._Xt[self._lbt[bi]]
                cat = self._Ct[bi]
                out[i:i + len(bi)] = self.net(seq, cat).cpu().numpy()
        return out


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
    # LSTM hiperparametreleri
    ap.add_argument("--window", type=int, default=20, help="lookback gun sayisi (sequence uzunlugu)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=1, help="LSTM seed sayisi (>1 -> ortalama)")
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

    # cs-features (Faz 3.6) — en iyi config
    aug0, feats0, _ = build_pooled_features(panel)
    cs_base = [c for c in feats0 if c not in ("symbol_id", "sector_code")]
    panel, cs_new = add_cross_sectional_features(panel, cs_base)
    aug, feats, cat_idx = build_pooled_features(panel)
    assert not any(c.startswith("target") for c in feats), "target leak!"

    folds = PooledPurgedWalkForward(PooledCVConfig(
        target_horizon=args.horizon, window_len=args.window_len,
        n_windows=args.n_windows, min_train_days=args.min_train_days,
    )).split(aug)
    sel = [f for f in folds if not f.is_final_holdout]
    _log(f"folds: {len(folds)} ({len(sel)} selectable + holdout)")

    # ozellik bolumleme: sayisal (LSTM seq) vs kategorik (embedding)
    num_cols = [feats[i] for i in range(len(feats)) if i not in cat_idx]
    cat_cols = [feats[i] for i in cat_idx]
    cardinalities = [int(aug[c].max()) + 1 for c in cat_cols]
    _log(f"features: {len(feats)} ({len(num_cols)} numeric seq + {len(cat_cols)} cat), "
         f"cat={dict(zip(cat_cols, cardinalities))}")

    Xnum = aug[num_cols].to_numpy(dtype=float)
    Xcat = aug[cat_cols].to_numpy(dtype=np.int64)
    y_cs = aug["target_cs"].to_numpy(dtype=float)
    sym_all = aug["symbol"].to_numpy()
    date_all = aug["Date"].to_numpy()
    X_lgb = aug[feats].to_numpy(dtype=float)

    # lookback index (yetersiz gecmis -> -1, satir atlanir)
    ts = time.time()
    lb = build_lookback_index(aug, args.window)
    has_lb = (lb >= 0).all(axis=1)
    _log(f"lookback W={args.window}: {int(has_lb.sum())}/{len(aug)} rows valid "
         f"(build {time.time()-ts:.0f}s)")

    lstm_cfg = {
        "hidden": args.hidden, "dropout": args.dropout, "lr": args.lr,
        "wd": args.wd, "epochs": args.epochs, "batch": args.batch, "seed": 42,
    }

    lgb_pred_rows: list[pd.DataFrame] = []
    lstm_pred_rows: list[pd.DataFrame] = []
    n_used = 0
    for f in sel:
        tr = f.train_mask & has_lb
        te = f.test_mask & has_lb
        if not tr.any() or not te.any():
            continue
        tr_rows = np.where(tr)[0]
        te_rows = np.where(te)[0]
        n_used += 1

        # --- LGB (ayni satir evreni) ---
        tl = time.time()
        lgb = GlobalPooledModel(GlobalPooledConfig(
            num_boost_round=args.boost, cat_indices=tuple(cat_idx)))
        lgb.fit(X_lgb[tr_rows], y_cs[tr_rows])
        yl = lgb.predict(X_lgb[te_rows])
        lgb_pred_rows.append(pd.DataFrame({
            "symbol": sym_all[te_rows], "Date": date_all[te_rows],
            "y_true": y_cs[te_rows], "y_pred": yl}))
        t_lgb = time.time() - tl

        # --- LSTM (seed ortalamasi) ---
        ts2 = time.time()
        acc = np.zeros(len(te_rows), dtype=float)
        for s in range(args.seeds):
            cfg_s = dict(lstm_cfg, seed=lstm_cfg["seed"] + s * 101)
            m = SeqLSTMModel(Xnum, Xcat, y_cs, lb, cardinalities, cfg_s)
            m.fit(tr_rows)
            acc += m.predict(te_rows)
        ylstm = acc / args.seeds
        lstm_pred_rows.append(pd.DataFrame({
            "symbol": sym_all[te_rows], "Date": date_all[te_rows],
            "y_true": y_cs[te_rows], "y_pred": ylstm}))
        _log(f"  fold {f.index} [{f.test_date_start.date()}..{f.test_date_end.date()}] "
             f"n_tr={len(tr_rows)} n_te={len(te_rows)} "
             f"LGB {t_lgb:.0f}s LSTM {time.time()-ts2:.0f}s")

    lgb_preds = pd.concat(lgb_pred_rows, ignore_index=True)
    lstm_preds = pd.concat(lstm_pred_rows, ignore_index=True)
    ic_lgb = daily_cross_sectional_ic(lgb_preds, min_names=args.min_names)
    ic_lstm = daily_cross_sectional_ic(lstm_preds, min_names=args.min_names)

    # decorrelation: gunluk pred rank korelasyonu (dusuk -> ensemble degeri yuksek)
    merged = lgb_preds.merge(lstm_preds, on=["symbol", "Date"], suffixes=("_lgb", "_lstm"))
    corrs = []
    for _, g in merged.groupby("Date"):
        if len(g) >= args.min_names:
            ra = g["y_pred_lgb"].rank().to_numpy()
            rb = g["y_pred_lstm"].rank().to_numpy()
            if ra.std() > 0 and rb.std() > 0:
                corrs.append(float(np.corrcoef(ra, rb)[0, 1]))
    mean_corr = float(np.mean(corrs)) if corrs else float("nan")

    _log(f"LGB : IC {ic_lgb['ic_mean']:+.4f} ICIR {ic_lgb['icir']:+.3f} "
         f"%IC>0 {100*ic_lgb['pct_positive']:.1f} n_days {ic_lgb['n_days']}")
    _log(f"LSTM: IC {ic_lstm['ic_mean']:+.4f} ICIR {ic_lstm['icir']:+.3f} "
         f"%IC>0 {100*ic_lstm['pct_positive']:.1f} n_days {ic_lstm['n_days']}")
    _log(f"pred rank corr (LGB vs LSTM): {mean_corr:+.3f} (dusuk -> ensemble degeri yuksek)")

    os.makedirs("outputs", exist_ok=True)
    lines = ["# E2 PoC (D) — Pooled sequence LSTM vs LightGBM (cross-sectional IC)", ""]
    lines.append(f"- panel: {len(aug)} rows, {aug['symbol'].nunique()} symbols, "
                 f"{aug['Date'].nunique()} dates, h={args.horizon}")
    lines.append(f"- config: CS+CSFEAT (target_cs), {len(feats)} feats "
                 f"({len(num_cols)} numeric seq + {len(cat_cols)} cat), folds={n_used}, "
                 f"min_names={args.min_names}")
    lines.append(f"- lookback W={args.window}: {int(has_lb.sum())}/{len(aug)} valid rows")
    lines.append(f"- LSTM: hidden={args.hidden} dropout={args.dropout} lr={args.lr} "
                 f"wd={args.wd} epochs={args.epochs} batch={args.batch} seeds={args.seeds}")
    lines.append(f"- LGB: boost={args.boost}")
    lines.append("")
    lines.append("| model | daily IC | ICIR | %IC>0 | n_days |")
    lines.append("|---|---|---|---|---|")
    for label, ic in (("LightGBM", ic_lgb), ("Seq-LSTM", ic_lstm)):
        lines.append(f"| {label} | {ic['ic_mean']:+.4f} | {ic['icir']:+.3f} | "
                     f"{100*ic['pct_positive']:.1f} | {ic['n_days']} |")
    lines.append("")
    lines.append(f"- pred rank corr (LGB vs LSTM): {mean_corr:+.3f} "
                 f"(dusuk = decorrelated = ensemble icin degerli)")
    lines.append(f"- total elapsed: {time.time()-t0:.0f}s")
    with open("outputs/e2_poc_deep_lstm.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    _log("wrote outputs/e2_poc_deep_lstm.md")
    _log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
