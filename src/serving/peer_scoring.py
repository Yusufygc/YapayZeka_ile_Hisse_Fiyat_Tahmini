# -*- coding: utf-8 -*-
"""Peer (cross-sectional) scoring for serving — E2 Faz 5.

Cross-sectional model bir sembolu tek basina skorlayamaz: peer_score uretmek
icin O GUNUN tum evreni siralanir. Bu modul tek bir tarihin tahmin vektorunu
peer skorlarina cevirir:
    peer_score      : merkezli sira [-1,1]  (2*(rank-0.5)/n - 1)
    peer_percentile : 0..100  ((rank-0.5)/n * 100)
    peer_label      : outperform / inline / underperform (percentile esikleri)

Serving akisi (nightly batch, src/serving/nightly_scoring.py):
  global modeli tum gecmiste egit -> en guncel tarihin evren ozelliklerini
  skorla -> rank_to_peer_scores -> per-symbol peer_scores satirlari -> DB.

Confidence burada DEGIL: `segment_confidence` (segment_IC x tradability) ayri.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PeerScoringConfig:
    pred_col: str = "y_pred"
    symbol_col: str = "symbol"
    lo_pct: float = 30.0     # bu percentilin altinda -> underperform
    hi_pct: float = 70.0     # bu percentilin ustunde -> outperform
    min_names: int = 15      # bundan az sembol -> peer skoru anlamsiz


def _label(pct: float, lo: float, hi: float) -> str:
    if not np.isfinite(pct):
        return "unknown"
    if pct >= hi:
        return "outperform"
    if pct <= lo:
        return "underperform"
    return "inline"


def rank_to_peer_scores(
    predictions: pd.DataFrame,
    as_of_date,
    cfg: PeerScoringConfig | None = None,
) -> pd.DataFrame:
    """Tek tarihin tahmin vektorunu peer skorlarina cevir.

    predictions: en az `symbol_col` + `pred_col` kolonlu (TEK tarih, her sembol
    bir satir). Returns DataFrame[symbol, raw_pred, peer_score, peer_percentile,
    peer_label, universe_size, as_of_date].
    """
    cfg = cfg or PeerScoringConfig()
    df = predictions[[cfg.symbol_col, cfg.pred_col]].copy()
    df = df.dropna(subset=[cfg.pred_col]).drop_duplicates(subset=[cfg.symbol_col])
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=[
            "symbol", "raw_pred", "peer_score", "peer_percentile",
            "peer_label", "universe_size", "as_of_date"])
    rank = df[cfg.pred_col].rank(method="average")
    pct_unit = (rank - 0.5) / n
    out = pd.DataFrame({
        "symbol": df[cfg.symbol_col].astype(str).values,
        "raw_pred": df[cfg.pred_col].astype(float).values,
        "peer_score": (2.0 * pct_unit - 1.0).values,
        "peer_percentile": (100.0 * pct_unit).values,
        "universe_size": n,
        "as_of_date": str(as_of_date),
    })
    if n < cfg.min_names:
        out["peer_label"] = "unknown"
    else:
        out["peer_label"] = out["peer_percentile"].map(
            lambda p: _label(p, cfg.lo_pct, cfg.hi_pct))
    return out[["symbol", "raw_pred", "peer_score", "peer_percentile",
                "peer_label", "universe_size", "as_of_date"]].sort_values(
        "peer_percentile", ascending=False).reset_index(drop=True)


def score_latest_universe(
    model,
    panel_latest: pd.DataFrame,
    feature_cols: list[str],
    cfg: PeerScoringConfig | None = None,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Egitilmis modelle EN GUNCEL tarihin evren satirlarini skorla -> peer.

    panel_latest: en guncel skorlama tarihinin TUM sembol satirlari (her sembol
    bir satir, `feature_cols` + symbol + Date). Tek tarih beklenir; birden fazla
    tarih varsa en yenisi alinir.
    """
    cfg = cfg or PeerScoringConfig()
    df = panel_latest
    if date_col in df.columns and df[date_col].nunique() > 1:
        latest = df[date_col].max()
        df = df[df[date_col] == latest]
    else:
        latest = df[date_col].iloc[0] if date_col in df.columns and len(df) else ""
    X = df[feature_cols].to_numpy(dtype=float)
    preds = np.asarray(model.predict(X), dtype=float).ravel()
    scored = df[[cfg.symbol_col]].copy()
    scored[cfg.pred_col] = preds
    return rank_to_peer_scores(scored, as_of_date=latest, cfg=cfg)
