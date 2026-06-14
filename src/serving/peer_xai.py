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
from src.xai.group_summary import build_group_summaries, group_summaries_to_dicts
from src.xai.narrative import direction_label
from src.xai.strategies import TabularContributionStrategy

_ENSEMBLE_CAVEAT = (
    "Bu aciklama ensemble skoruna permutation sensitivity uygular; "
    "LightGBM bacagi SHAP'i yalniz diagnostic alt bilgi olarak tutulur."
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
    diagnostic_method = ""
    if is_ensemble and hasattr(model, "predict"):
        contribs = strategy.permutation_contributions(model, X)
        method = "ensemble_permutation"
        approximate = True
        diagnostic_method = "lgb_leg_shap_available"
    else:
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
        group_rows: list[dict] = []
        for feat_idx in ranked:
            contribution = float(row[int(feat_idx)])
            if contribution == 0.0:
                continue
            factor = _make_factor(
                feature_cols[int(feat_idx)], contribution, method, approximate
            )
            group_rows.append(factor)
            if contribution > 0 and len(top_positive) < top_k:
                top_positive.append(factor)
            elif contribution < 0 and len(top_negative) < top_k:
                top_negative.append(factor)
        out[str(symbol)] = {
            "method": method,
            "approximate": bool(approximate),
            "caveat": caveat,
            "diagnostic_method": diagnostic_method,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "group_summaries": group_summaries_to_dicts(
                build_group_summaries(group_rows, context="peer")
            ),
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
        "reason": _peer_rank_reason(feature_name, contribution, approximate),
        "method": method,
        "contribution": contribution,
        "approximate": bool(approximate),
    }


def _peer_rank_reason(feature_name: str, contribution: float, approximate: bool) -> str:
    direction = "yukari" if contribution > 0 else "asagi"
    suffix = " Yaklasik sensitivity hesabidir." if approximate else ""
    return (
        f"{describe_feature(feature_name)} akran siralamasini {direction} iten "
        f"model sinyallerinden biri oldu.{suffix}"
    )
