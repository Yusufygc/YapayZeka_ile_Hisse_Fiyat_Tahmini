# -*- coding: utf-8 -*-
"""Per-symbol out-of-sample aggregation harness for the E2 pooled model.

E2 Faz 2 son parca. Tasarim: docs/wiki/e2-faz2-pooled-cv-design.md.

Amac: `PooledPurgedWalkForward` fold'larini bir modelle tahmin et, test
satirlarini SEMBOL bazinda grupla, her sembol icin OOS metrik DAGILIMI uret.
Bu dagilim Faz 5 serving guven skorunu besler:
  - per-symbol Dir_Acc / RMSE / edge-over-base-rate
  - fold-bazli stabilite: positive_fold_ratio (Dir_Acc >= 50 olan fold orani)

Tasarim kararlari:
  - target = log-return -> isaret dogrudan yon; prev_close/price-mode gereksiz.
  - Model protokolu sklearn-vari: `model_factory()` -> `.fit(X, y)` / `.predict(X)`.
    Her fold icin TAZE model (fold'lar arasi durum sizmasi yok).
  - Olcekleme model_factory icinde (or. Pipeline). Loader global quantile
    uygulamaz (lookahead); burada da panel-genisi fit yok -> sadece fold-train.
  - final_holdout fold'lari varsayilan DISLANIR (secim/raporlama ayrimi). Holdout
    metrikleri ayrica istenirse include_holdout=True.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.data.pooled_matrix import pooled_feature_matrix, pooled_target_array
from src.validation.pooled_cv import PooledFold

ModelFactory = Callable[[], object]


@dataclass(frozen=True)
class PerSymbolOOSConfig:
    target_col: str = "target"
    symbol_col: str = "symbol"
    feature_cols: Sequence[str] | None = None  # None -> otomatik (asagi _NON_FEAT haric)
    min_test_per_symbol: int = 20    # bundan az OOS satirli sembol guvenilmez
    neutral_eps: float = 0.0         # |pred| <= eps -> notr (yon sayilmaz)
    include_holdout: bool = False


# panel'de ozellik OLMAYAN kolonlar (loader ciktisi + CV alanlari). target*
# varyantlari (target, target_cs, ...) ASLA feature olmamali -> sizinti.
_NON_FEAT = {
    "symbol", "Date", "target_date", "sector", "symbol_id",
    "liq_log", "vol", "sector_code",
}


def _is_non_feature(col: str) -> bool:
    return col in _NON_FEAT or col == "target" or col.startswith("target_")


def _auto_feature_cols(panel: pd.DataFrame) -> list[str]:
    cols = []
    for c in panel.columns:
        if _is_non_feature(c):
            continue
        if pd.api.types.is_numeric_dtype(panel[c]):
            cols.append(c)
    return cols


def _dir_acc(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    """Yonsel dogruluk (%). Notr tahminler (|pred|<=eps) paydadan dusulur."""
    mask = np.abs(y_pred) > eps
    if not mask.any():
        return float("nan")
    correct = np.sign(y_true[mask]) == np.sign(y_pred[mask])
    return 100.0 * float(np.mean(correct))


def _base_rate(y_true: np.ndarray) -> float:
    """Cogunluk-yonu naif dogrulugu (%): hep up ya da hep down tahmininin en iyisi."""
    if len(y_true) == 0:
        return float("nan")
    up = float(np.mean(y_true > 0))
    return 100.0 * max(up, 1.0 - up)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """scipy'siz Spearman: degerleri rank'la, Pearson uygula."""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def daily_cross_sectional_ic(
    predictions: pd.DataFrame, min_names: int = 8, date_col: str = "Date",
    sample_gap_days: int = 0,
) -> dict:
    """Gunluk cross-sectional IC (semboller-arasi rank korelasyonu) ozeti.

    Pooled/cross-sectional modelin DOGRU degerlendirmesi: her tarih icin
    semboller-arasi corr(y_pred, y_true). Per-symbol dir_acc goreli siralama
    becerisini olcemez; bu olcer. ICIR = mean/std (bilgi orani).

    sample_gap_days>0 : IC serisini >= bu kadar TAKVIM gunu arali tarihlerle
    alt-ornekle (greedy). h-gunluk hedef pencereleri ortustugu icin ardisik
    gunlerin IC'si autocorrelation tasir -> ICIR'i SISIRIR (std kucuk). Ortusmeyen
    ornekleme (gap≈h) durust ICIR verir.
    """
    pairs = []  # (date, ic)
    for d, g in predictions.groupby(date_col):
        if g["symbol"].nunique() < min_names:
            continue
        ic = _spearman(g["y_pred"].to_numpy(), g["y_true"].to_numpy())
        if np.isfinite(ic):
            pairs.append((pd.Timestamp(d), float(ic)))
    pairs.sort(key=lambda p: p[0])
    if sample_gap_days and sample_gap_days > 0 and pairs:
        kept = []
        last = None
        for d, ic in pairs:
            if last is None or (d - last).days >= sample_gap_days:
                kept.append((d, ic))
                last = d
        pairs = kept
    ics = [ic for _, ic in pairs]
    if not ics:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "icir": float("nan"), "pct_positive": float("nan"), "n_days": 0}
    arr = np.array(ics, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": (mean / std) if std > 0 else float("nan"),
        "pct_positive": float((arr > 0).mean()),
        "n_days": len(arr),
    }


@dataclass
class PerSymbolOOSResult:
    per_symbol: pd.DataFrame   # bir satir/sembol: dir_acc, rmse, edge, n, fold orani
    per_fold: pd.DataFrame     # bir satir/(sembol,fold): fold-bazli dir_acc/rmse/n
    predictions: pd.DataFrame  # ham OOS tahminleri (symbol, fold, Date, y_true, y_pred)
    n_folds_used: int = field(default=0)
    ic: dict = field(default_factory=dict)  # gunluk cross-sectional IC ozeti


def evaluate_per_symbol(
    panel: pd.DataFrame,
    folds: Sequence[PooledFold],
    model_factory: ModelFactory,
    cfg: PerSymbolOOSConfig | None = None,
) -> PerSymbolOOSResult:
    """Fold'lari modelle tahmin et -> sembol bazinda OOS metrik dagilimi.

    Parameters
    ----------
    panel : uzun panel (PooledPanelLoader ciktisi)
    folds : PooledPurgedWalkForward.split() ciktisi
    model_factory : argumansiz cagrilinca taze sklearn-vari model dondurur
    """
    cfg = cfg or PerSymbolOOSConfig()
    feats = list(cfg.feature_cols) if cfg.feature_cols is not None else _auto_feature_cols(panel)
    if not feats:
        raise ValueError("ozellik kolonu bulunamadi (feature_cols bos)")
    missing = [c for c in feats if c not in panel.columns]
    if missing:
        raise ValueError(f"panel'de eksik ozellik kolonlari: {missing[:5]}")

    use_folds = [f for f in folds if cfg.include_holdout or not f.is_final_holdout]
    sym_all = panel[cfg.symbol_col].to_numpy()
    date_all = panel["Date"].to_numpy() if "Date" in panel.columns else np.arange(len(panel))
    y_all = pooled_target_array(panel, cfg.target_col)

    pred_rows: list[pd.DataFrame] = []
    n_used = 0
    for f in use_folds:
        tr, te = f.train_mask, f.test_mask
        if not tr.any() or not te.any():
            continue
        model = model_factory()
        X_train = pooled_feature_matrix(panel, feats, tr)
        y_train = pooled_target_array(panel, cfg.target_col, tr)
        X_test = pooled_feature_matrix(panel, feats, te)
        model.fit(X_train, y_train)
        y_hat = np.asarray(model.predict(X_test), dtype=float).ravel()
        pred_rows.append(pd.DataFrame({
            "symbol": sym_all[te],
            "fold": f.index,
            "Date": date_all[te],
            "y_true": y_all[te],
            "y_pred": y_hat,
        }))
        n_used += 1

    if not pred_rows:
        empty = pd.DataFrame()
        return PerSymbolOOSResult(empty, empty, empty, 0)
    preds = pd.concat(pred_rows, ignore_index=True)

    # --- per (symbol, fold) ---
    fold_recs = []
    for (sym, fold), g in preds.groupby(["symbol", "fold"], sort=True):
        yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        fold_recs.append({
            "symbol": sym, "fold": int(fold), "n": len(g),
            "dir_acc": _dir_acc(yt, yp, cfg.neutral_eps),
            "rmse": _rmse(yt, yp),
            "base_rate": _base_rate(yt),
        })
    per_fold = pd.DataFrame(fold_recs)

    # --- per symbol (tum fold'lar havuzlanmis) ---
    sym_recs = []
    for sym, g in preds.groupby("symbol", sort=True):
        yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        dir_acc = _dir_acc(yt, yp, cfg.neutral_eps)
        base = _base_rate(yt)
        ff = per_fold[per_fold["symbol"] == sym]
        valid = ff["dir_acc"].dropna()
        pos_ratio = float((valid >= 50.0).mean()) if len(valid) else float("nan")
        sym_recs.append({
            "symbol": sym,
            "n_oos": len(g),
            "n_folds": int(ff["fold"].nunique()),
            "dir_acc": dir_acc,
            "rmse": _rmse(yt, yp),
            "base_rate": base,
            "edge": (dir_acc - base) if np.isfinite(dir_acc) and np.isfinite(base) else float("nan"),
            "positive_fold_ratio": pos_ratio,
            "reliable": len(g) >= cfg.min_test_per_symbol,
        })
    per_symbol = pd.DataFrame(sym_recs).sort_values("symbol").reset_index(drop=True)

    ic = daily_cross_sectional_ic(preds)
    return PerSymbolOOSResult(per_symbol, per_fold, preds, n_used, ic)
