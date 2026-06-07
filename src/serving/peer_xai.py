# -*- coding: utf-8 -*-
"""E2 Kol-B XAI — pooled cross-sectional model per-symbol feature attribution.

Cross-sectional model `raw_pred`'i (gün içi akran skoru) feature'lara böler:
SHAP TreeExplainer modelin LightGBM bacağına uygulanır, her sembol için en
etkili artı/eksi sürücüler çıkar. Yorum: "bu hisse akranlarına göre neden
üstte/altta" — mutlak fiyat değil, göreli sıra.

Tasarım kararları:
- Ensemble (LGB+MLP) verilirse YALNIZ LGB bacağı açıklanır; çıktıya caveat eklenir
  (MLP bacağı bu açıklamada yok). LGB-leg SHAP ucuz (<1sn/gece) ve dürüst.
- Booster bulunamazsa no-op ({} döner) — XAI atlanır, skorlama bozulmaz.
- shap kurulu değilse TabularContributionStrategy permutation_fallback'e düşer
  (approximate=True).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.xai.feature_dictionary import describe_feature, feature_group
from src.xai.narrative import contribution_sentence, direction_label
from src.xai.strategies import TabularContributionStrategy

_ENSEMBLE_CAVEAT = (
    "Bu açıklama yalnızca modelin LightGBM bacağı temellidir; "
    "MLP bacağı bu özetin dışındadır."
)


def _resolve_booster(model: Any) -> tuple[Any, bool]:
    """(booster, is_ensemble) döner. Booster yoksa (None, is_ensemble)."""
    # Ensemble: model.lgb (GlobalPooledModel) -> .booster
    leg = getattr(model, "lgb", None)
    if leg is not None:
        return getattr(leg, "booster", None), True
    # Tekil GlobalPooledModel: .booster
    return getattr(model, "booster", None), False


def compute_peer_xai(
    model: Any,
    X: np.ndarray,
    feature_cols: list[str],
    symbols: list[str],
    top_k: int = 5,
) -> dict[str, dict]:
    """Bir cross-section için per-symbol top-K feature attribution üret.

    Parameters
    ----------
    model : GlobalPooledModel | EnsemblePooledModel (LGB bacağı şart).
    X : (n_symbols, n_features) skorlama matrisi (feature_cols sırasıyla).
    feature_cols : feature adları (X kolon sırasıyla birebir).
    symbols : X satırlarıyla hizalı sembol listesi.
    top_k : sembol başına kaç sürücü (artı + eksi ayrı ayrı en fazla top_k).

    Returns
    -------
    dict[symbol -> {method, approximate, caveat, top_positive[], top_negative[]}]
    Booster yoksa boş dict (no-op).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X 2 boyutlu olmalı, alınan ndim={X.ndim}")
    if len(symbols) != X.shape[0]:
        raise ValueError(
            f"symbols ({len(symbols)}) ile X satır sayısı ({X.shape[0]}) eşleşmiyor"
        )
    if X.shape[1] != len(feature_cols):
        raise ValueError(
            f"feature_cols ({len(feature_cols)}) ile X kolon sayısı "
            f"({X.shape[1]}) eşleşmiyor"
        )
    if top_k <= 0:
        raise ValueError(f"top_k pozitif olmalı, alınan {top_k}")

    booster, is_ensemble = _resolve_booster(model)
    if booster is None:
        return {}

    strategy = TabularContributionStrategy(list(feature_cols))
    contribs, method, approximate = strategy.tree_contributions(booster, X)
    contribs = np.asarray(contribs, dtype=float)
    if contribs.ndim == 1:
        contribs = contribs.reshape(1, -1)

    caveat = _ENSEMBLE_CAVEAT if is_ensemble else ""
    out: dict[str, dict] = {}
    for row_idx, symbol in enumerate(symbols):
        row = contribs[row_idx]
        ranked = np.argsort(np.abs(row))[::-1]
        top_positive: list[dict] = []
        top_negative: list[dict] = []
        for feat_idx in ranked:
            contribution = float(row[int(feat_idx)])
            if contribution == 0.0:
                continue
            factor = _make_factor(
                feature_cols[int(feat_idx)], contribution, method, approximate
            )
            if contribution > 0 and len(top_positive) < top_k:
                top_positive.append(factor)
            elif contribution < 0 and len(top_negative) < top_k:
                top_negative.append(factor)
            if len(top_positive) >= top_k and len(top_negative) >= top_k:
                break
        out[str(symbol)] = {
            "method": method,
            "approximate": bool(approximate),
            "caveat": caveat,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }
    return out


def _make_factor(
    feature_name: str, contribution: float, method: str, approximate: bool
) -> dict:
    """XaiFactorItem ile uyumlu sözlük (API şeması bunu birebir map'ler)."""
    return {
        "feature_name": feature_name,
        "human_label": describe_feature(feature_name),
        "importance": abs(contribution),
        "direction": direction_label(contribution),
        "feature_group": feature_group(feature_name),
        "reason": contribution_sentence(feature_name, contribution, approximate),
        "method": method,
        "contribution": contribution,
        "approximate": bool(approximate),
    }
