# -*- coding: utf-8 -*-
"""
results.py - Pipeline degerlendiricileri icin TypedDict don\u00fc\u015f tipleri (Faz 2.3).

Motivasyon:
  EvaluationManager'nin public metodlari ham Dict[str, Any] dondurmekteydi.
  Bu TypedDict tanimlari:
    - IDE/mypy ile anahtar hatalarinin erken yakalanmasini saglar,
    - Callers'in hangi anahtarlara ihtiyaci oldugunu belgeler,
    - Arka planda dict davranisini koru (TypedDict, dict'in alt siniflandirilmasi degildir,
      ama tipik olarak sozluk olarak kullanilir).

Kullanim:
    from src.pipeline.results import SingleSplitResult, WalkForwardResult, FinalHoldoutResult
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
try:
    from typing import TypedDict
except ImportError:  # Python < 3.8 compatibility guard (not needed in practice)
    from typing_extensions import TypedDict  # type: ignore

import numpy as np
import pandas as pd


class SingleSplitResult(TypedDict):
    """evaluate_single_split() don\u00fc\u015f tipi."""

    metrics: Dict[str, Dict[str, Any]]
    """Her model icin metrik sozlugu {model_name: {metric_key: value}}."""

    y_true: np.ndarray
    """Gercek fiyat dizisi (test seti, min-uzunluk hizali)."""

    predictions: Dict[str, np.ndarray]
    """Her model icin tahmin edilen fiyat dizisi."""

    backtest: Dict[str, Any]
    """Backtest sonuclari veya {} (backtest devre disi ise)."""

    xai_payload: Optional[Dict[str, Any]]
    """XAI aciklama icerigi veya None."""

    tft_quantiles_df: Optional[pd.DataFrame]
    """TFT kantil tahmin DataFrame'i veya None."""

    quantile_predictions: Dict[str, np.ndarray]
    """Kantil tahminler {model_name: ndarray[n_samples, n_quantiles]}."""


class WalkForwardResult(TypedDict):
    """evaluate_walk_forward() don\u00fc\u015f tipi."""

    metrics: Dict[str, Dict[str, Any]]
    """Her model icin walk-forward ortalama metrikleri."""

    best_model_name: Optional[str]
    """Composite_Score + RMSE'ye gore secilen en iyi model adi."""

    y_true: Any
    """Tum fold'lardaki gercek fiyat dizisi (concatenated)."""

    predictions: Dict[str, np.ndarray]
    """Her model icin tum fold'lardaki tahminler (concatenated)."""

    backtest: Dict[str, Any]
    """Walk-forward backtest sonuclari."""

    xai_payload: Optional[Dict[str, Any]]
    """XAI aciklama icerigi veya None."""

    wf_fold_reports: Dict[str, pd.DataFrame]
    """{'fold_metrics': ..., 'worst_folds': ...} raporlari."""

    calibration_results: Dict[str, Any]
    """Sinyal kalibrasyon sonuclari."""


class FinalHoldoutResult(TypedDict):
    """evaluate_final_holdout() don\u00fc\u015f tipi."""

    metrics: Dict[str, Dict[str, Any]]
    """Secilen model icin final holdout metrikleri."""

    y_true: np.ndarray
    """Final holdout gercek fiyat dizisi."""

    predictions: Dict[str, np.ndarray]
    """Secilen model tahminleri {model_name: ndarray}."""

    quantiles_df: Optional[pd.DataFrame]
    """Kantil tahmin DataFrame'i (TFT icin) veya None."""

    quantile_price: Optional[np.ndarray]
    """Ham kantil tahmin matrisi veya None."""

    backtest: Dict[str, Any]
    """Final holdout backtest sonuclari."""

    model_name: str
    """Degerlendirilen model adi."""
