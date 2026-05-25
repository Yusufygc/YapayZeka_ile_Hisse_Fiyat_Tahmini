# -*- coding: utf-8 -*-
"""
Sprint 7 (2026-05-25) A7.3 — Analysis API icin on-the-fly PSI 30g monitor.

`compute_psi_30d(symbol_csv_path)` son 30 isgunundeki OHLCV-turevli feature
dagilimlarini, onceki 252 isgunlu pencereyle karsilastirir; max PSI doner.
Tier:
  - < 0.10            -> "stable"
  - 0.10 <= x < 0.25  -> "moderate_drift"
  - >= 0.25           -> "major_drift"  (hard block)

Cikti analysis API `data_quality` blogunda yayinlanir.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.data.quality import _psi_one_feature

_HOLDOUT_DAYS = 30
_TRAIN_DAYS = 252
_PSI_STABLE_MAX = 0.10
_PSI_MAJOR_MIN = 0.25
# 30-gun holdout sample-variance noise floor'u yuksek; coarser bin'ler
# (>=5 yanlis pozitif uretiyor). Bins=3, holdout=30 emipirik olarak
# ortalama PSI ~0.10 noise altinda kaliyor; gerek shift'lerde
# rahatlikla 0.25 esigini geciyor.
_PSI_BINS_30D = 3


@dataclass(frozen=True)
class DataQualityResult:
    psi_30d: Optional[float]
    psi_status: str  # "stable" | "moderate_drift" | "major_drift" | "unavailable"
    stale_warning: bool
    reason: Optional[str] = None


def _tier(psi: float) -> str:
    if psi < _PSI_STABLE_MAX:
        return "stable"
    if psi < _PSI_MAJOR_MIN:
        return "moderate_drift"
    return "major_drift"


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    high = df["High"].astype(float) if "High" in df.columns else close
    low = df["Low"].astype(float) if "Low" in df.columns else close
    volume = df["Volume"].astype(float) if "Volume" in df.columns else None

    # Sadece STASYONER feature'lar. Raw close levels non-stationary -> 30g
    # pencerede sahte drift uretir, PSI'i bozar. abs_log_return de kucuk
    # ornek sayisinda yuksek noise floor uretiyor (~0.25) -> drop.
    out["log_return"] = np.log(close / close.shift(1))
    out["range_pct"] = (high - low) / close.replace(0, np.nan)
    if volume is not None:
        log_vol = np.log(volume.replace(0, np.nan))
        out["volume_log_change"] = log_vol.diff()
    return out.dropna()


def compute_psi_30d(symbol_csv_path: str) -> DataQualityResult:
    """
    Sembolun CSV'sini okur; son 30 gun vs onceki 252 gun PSI doner.
    Veri yok / yetersiz ise psi_status='unavailable'.
    """
    if not os.path.exists(symbol_csv_path):
        return DataQualityResult(
            psi_30d=None,
            psi_status="unavailable",
            stale_warning=True,
            reason="csv_missing",
        )
    try:
        df = pd.read_csv(symbol_csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        return DataQualityResult(
            psi_30d=None,
            psi_status="unavailable",
            stale_warning=True,
            reason=f"csv_read_failed:{type(exc).__name__}",
        )

    if "Date" not in df.columns or "Close" not in df.columns:
        return DataQualityResult(
            psi_30d=None,
            psi_status="unavailable",
            stale_warning=True,
            reason="missing_required_columns",
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    feats = _build_features(df)
    if len(feats) < _HOLDOUT_DAYS + _TRAIN_DAYS // 4:
        return DataQualityResult(
            psi_30d=None,
            psi_status="unavailable",
            stale_warning=False,
            reason="insufficient_history",
        )

    holdout = feats.tail(_HOLDOUT_DAYS)
    train = feats.iloc[-(_HOLDOUT_DAYS + _TRAIN_DAYS): -_HOLDOUT_DAYS]
    if len(train) < _HOLDOUT_DAYS:
        return DataQualityResult(
            psi_30d=None,
            psi_status="unavailable",
            stale_warning=False,
            reason="insufficient_train_window",
        )

    psi_scores: dict[str, float] = {}
    try:
        for col in train.columns:
            t_vals = train[col].values.astype(float)
            h_vals = holdout[col].values.astype(float)
            psi_scores[col] = _psi_one_feature(t_vals, h_vals, n_bins=_PSI_BINS_30D)
    except Exception as exc:  # pragma: no cover - defensive
        return DataQualityResult(
            psi_30d=None,
            psi_status="unavailable",
            stale_warning=False,
            reason=f"psi_failed:{type(exc).__name__}",
        )

    if not psi_scores:
        return DataQualityResult(
            psi_30d=None,
            psi_status="unavailable",
            stale_warning=False,
            reason="no_numeric_columns",
        )

    psi_max = float(max(psi_scores.values()))
    return DataQualityResult(
        psi_30d=psi_max,
        psi_status=_tier(psi_max),
        stale_warning=False,
        reason=None,
    )
