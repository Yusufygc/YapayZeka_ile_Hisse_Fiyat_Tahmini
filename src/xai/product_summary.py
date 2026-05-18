# -*- coding: utf-8 -*-
"""Kullanıcıya sunulacak XAI ürün özeti.

En iyi modelin XAI çıktı dosyasını (feature_importance_*.csv) okur,
top-5 pozitif ve negatif faktörü ayırt ederek ürün payload'una uygun
bir özet döner.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.api.constants import XAI_CAVEAT
from src.xai.feature_dictionary import describe_feature

_TREE_MODELS = {"XGBoost", "Random Forest", "LightGBM Return", "Random Forest Return"}
_LINEAR_MODELS = {"Ridge Return", "ElasticNet Return"}
_SEQ_MODELS = {"LSTM", "LSTM Lite", "DLinear", "NLinear", "AttentionLSTM"}


def _model_family_caveat(model_name: str) -> str:
    if model_name in _TREE_MODELS:
        return "Tree modellerde SHAP TreeExplainer kullanılır; özellik katkıları güvenilirdir."
    if model_name in _LINEAR_MODELS:
        return "Lineer modellerde katsayı bazlı katkılar hesaplanır; yorumlama nispeten doğrudur."
    if model_name in _SEQ_MODELS:
        return (
            "Derin öğrenme modellerinde özellik katkıları yaklaşıktır; "
            "daha temkinli yorumlanmalıdır."
        )
    return "Model ailesi için açıklanabilirlik kalitesi bilinmiyor."


@dataclass
class XaiFeatureFactor:
    feature_name: str
    human_label: str
    importance: float
    direction: str  # "positive" | "negative" | "neutral"


@dataclass
class XaiProductSummary:
    available: bool
    method: str = ""
    top_positive_reasons: List[XaiFeatureFactor] = field(default_factory=list)
    top_negative_reasons: List[XaiFeatureFactor] = field(default_factory=list)
    model_family_caveat: str = ""
    caveat: str = XAI_CAVEAT


def _unavailable(reason: str = "") -> XaiProductSummary:
    return XaiProductSummary(available=False, caveat=XAI_CAVEAT)


def build_xai_product_summary(
    symbol: str,
    model_name: str,
    outputs_base: Optional[str] = None,
    top_k: int = 5,
) -> XaiProductSummary:
    """XAI ürün özeti oluştur.

    Parameters
    ----------
    symbol:
        Hisse kodu (büyük/küçük harf duyarsız).
    model_name:
        En iyi model adı.
    outputs_base:
        ``outputs/`` klasörünün kök yolu; None ise proje kökünden türetilir.
    top_k:
        Gösterilecek maksimum özellik sayısı (pozitif ve negatif her biri için).
    """
    symbol = symbol.upper()
    if outputs_base is None:
        _here = os.path.dirname(os.path.abspath(__file__))
        outputs_base = os.path.join(_here, "..", "..", "outputs")

    latest_dir = os.path.join(outputs_base, symbol, "latest", "xai")
    if not os.path.isdir(latest_dir):
        return _unavailable("xai dizini bulunamadı")

    # Model adına göre eşleşen importance dosyasını bul
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    pattern_wf = os.path.join(latest_dir, f"feature_importance_{safe_name}_wf.csv")
    pattern_fh = os.path.join(latest_dir, f"feature_importance_{safe_name}_final_holdout.csv")
    pattern_any = os.path.join(latest_dir, f"feature_importance_{safe_name}_*.csv")

    csv_path = None
    for candidate in [pattern_wf, pattern_fh]:
        if os.path.isfile(candidate):
            csv_path = candidate
            break
    if csv_path is None:
        matches = glob.glob(pattern_any)
        if matches:
            csv_path = sorted(matches)[-1]

    if csv_path is None:
        return _unavailable("özellik önem dosyası bulunamadı")

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return _unavailable("özellik önem dosyası okunamadı")

    importance_col = next(
        (c for c in df.columns if "importance" in c.lower() or "mean" in c.lower()),
        None,
    )
    feature_col = next(
        (c for c in df.columns if "feature" in c.lower()),
        None,
    )
    if importance_col is None or feature_col is None:
        return _unavailable("beklenen kolonlar bulunamadı")

    df = df[[feature_col, importance_col]].copy()
    df.columns = ["feature", "importance"]
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["importance"]).sort_values("importance", ascending=False)

    def _make_factor(row: Any, direction: str) -> XaiFeatureFactor:
        return XaiFeatureFactor(
            feature_name=str(row["feature"]),
            human_label=describe_feature(str(row["feature"])),
            importance=float(row["importance"]),
            direction=direction,
        )

    # Feature importance değeri pozitif → pozitif katkı, negatif → negatif katkı
    positives = df[df["importance"] > 0].head(top_k)
    negatives = df[df["importance"] < 0].sort_values("importance").head(top_k)

    # Tümü pozitifse (SHAP gibi signed değil, ağırlık gibi unsigned) → top N pozitif, bottom N negatif
    if negatives.empty and not positives.empty:
        top_n = positives.head(top_k)
        bottom_n = df.tail(top_k).sort_values("importance")
        top_positive = [_make_factor(row, "positive") for _, row in top_n.iterrows()]
        top_negative = [_make_factor(row, "negative") for _, row in bottom_n.iterrows()]
    else:
        top_positive = [_make_factor(row, "positive") for _, row in positives.iterrows()]
        top_negative = [_make_factor(row, "negative") for _, row in negatives.iterrows()]

    method = "SHAP TreeExplainer" if model_name in _TREE_MODELS else "Feature Importance"

    return XaiProductSummary(
        available=True,
        method=method,
        top_positive_reasons=top_positive,
        top_negative_reasons=top_negative,
        model_family_caveat=_model_family_caveat(model_name),
        caveat=XAI_CAVEAT,
    )
