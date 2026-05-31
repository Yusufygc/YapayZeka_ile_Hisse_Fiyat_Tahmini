# -*- coding: utf-8 -*-
"""Veri kalite ve distribution shift bayrakları (Adim 1.9).

compute_quality_flags(): ham OHLCV DataFrame'inden kalite sinyalleri üretir.
  - corporate_action_anomaly: Adj_Close tutarsızlığı (data_loader'dan okunur)
  - survivorship_warning    : fiyat serisi şüpheli kısalık / boşluk tespiti
  - psi_high                : train vs holdout feature PSI > 0.25
  - clip_rate               : preprocessor clip oranı (varsa df.attrs'dan)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PSI_THRESHOLD = 0.25
_PSI_N_BINS = 10
_SURVIVORSHIP_MIN_YEARS = 2.0
_SURVIVORSHIP_GAP_DAYS = 10

# Sprint 2 (2026-05-25) Plan A2.3: Audit-bazli corporate action anomaly.
# tools/audit_corporate_actions.py ciktisini son 252 isgunu icin tarar.
_AUDIT_LOOKBACK_DAYS = 252
_AUDIT_LATEST_DEFAULT = os.path.join("outputs", "_audits", "corporate_action_audit_latest.csv")


def _check_audit_anomaly(symbol: str, audit_csv: Optional[str] = None) -> bool:
    """
    Latest corporate_action_audit CSV'sinden symbol icin son 252 isgununde
    anomali olup olmadigini doner. CSV yoksa False (silent).
    """
    import pandas as pd

    audit_path = audit_csv or _AUDIT_LATEST_DEFAULT
    if not os.path.exists(audit_path):
        return False
    try:
        df = pd.read_csv(audit_path)
    except Exception as exc:
        logger.warning(f"Audit CSV okunamadi ({audit_path}): {exc}")
        return False
    if df.empty or "Symbol" not in df.columns or "Date" not in df.columns:
        return False
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        recent_cutoff = pd.Timestamp.today() - pd.Timedelta(days=_AUDIT_LOOKBACK_DAYS)
        mask = (
            (df["Symbol"].astype(str).str.upper() == symbol.upper())
            & (df["Date"] >= recent_cutoff)
        )
        return bool(mask.any())
    except Exception as exc:
        logger.warning(f"Audit anomaly check basarisiz ({symbol}): {exc}")
        return False


def _psi_one_feature(
    train_vals: np.ndarray,
    holdout_vals: np.ndarray,
    n_bins: int = _PSI_N_BINS,
) -> float:
    combined = np.concatenate([train_vals, holdout_vals])
    finite = combined[np.isfinite(combined)]
    if len(finite) < 2 * n_bins:
        return 0.0
    _, bin_edges = np.histogram(finite, bins=n_bins)
    bin_edges[0] -= 1e-9
    bin_edges[-1] += 1e-9

    def _freq(arr: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr[np.isfinite(arr)], bins=bin_edges)
        freq = counts.astype(float) / max(1, counts.sum())
        freq = np.where(freq == 0, 1e-4, freq)
        return freq

    f_train = _freq(train_vals)
    f_holdout = _freq(holdout_vals)
    psi = float(np.sum((f_holdout - f_train) * np.log(f_holdout / f_train)))
    return max(0.0, psi)


def compute_psi(
    train_df,
    holdout_df,
    exclude_cols: Optional[list] = None,
) -> Dict[str, float]:
    # Ham OHLCV/fiyat sutunlari non-stationary; trend nedeniyle PSI daima
    # yuksek cikar ve psi_high'i yaniltici sekilde tetikler. Yalnizca
    # durağan feature'lar uzerinde PSI hesapla.
    _RAW_NON_STATIONARY = {
        "Open", "High", "Low", "Close", "Adj_Close", "Volume",
        "open", "high", "low", "close", "adj_close", "volume",
    }
    exclude = (
        set(exclude_cols or [])
        | {"Date", "date", "Symbol", "symbol"}
        | _RAW_NON_STATIONARY
    )
    numeric_cols = [
        c for c in train_df.columns
        if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)
        and c in holdout_df.columns
    ]
    result: Dict[str, float] = {}
    for col in numeric_cols:
        t_vals = train_df[col].values.astype(float)
        h_vals = holdout_df[col].values.astype(float)
        result[col] = _psi_one_feature(t_vals, h_vals)
    return result


def compute_quality_flags(
    df,
    symbol: str,
    *,
    train_df=None,
    holdout_df=None,
) -> Dict[str, Any]:
    import pandas as pd

    # ── Corporate action anomaly ─────────────────────────────────────────
    # Sprint 2 A2.3: iki kaynak birlestirilir:
    #   1) data_loader.py'nin Adj_Close-dayali bayragi
    #   2) tools/audit_corporate_actions.py son 252 isgunu raporu
    ca_report = df.attrs.get("corporate_action_report", {}) if hasattr(df, "attrs") else {}
    corporate_action_anomaly = bool(ca_report.get("corporate_action_anomaly", False))
    if not corporate_action_anomaly:
        try:
            corporate_action_anomaly = _check_audit_anomaly(symbol)
        except Exception as exc:
            logger.warning(f"Audit anomaly check basarisiz ({symbol}): {exc}")

    # ── Survivorship warning ─────────────────────────────────────────────
    survivorship_warning = False
    try:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna().sort_values()
        if len(dates) >= 2:
            span_days = (dates.iloc[-1] - dates.iloc[0]).days
            span_years = span_days / 365.25
            if span_years < _SURVIVORSHIP_MIN_YEARS:
                survivorship_warning = True
            else:
                gaps = (dates.diff().dt.days.dropna())
                max_gap = int(gaps.max()) if len(gaps) > 0 else 0
                if max_gap > _SURVIVORSHIP_GAP_DAYS:
                    survivorship_warning = True
    except Exception as exc:
        logger.warning(f"Error calculating survivorship warning: {exc}")

    # ── PSI ──────────────────────────────────────────────────────────────
    psi_scores: Dict[str, float] = {}
    psi_max = 0.0
    psi_high = False
    if train_df is not None and holdout_df is not None:
        try:
            psi_scores = compute_psi(train_df, holdout_df)
            psi_max = max(psi_scores.values(), default=0.0)
            psi_high = psi_max > _PSI_THRESHOLD
        except Exception as exc:
            logger.warning(f"Error computing PSI: {exc}")

    # ── Clip rate ────────────────────────────────────────────────────────
    clip_rate = 0.0
    try:
        clip_rate = float(
            df.attrs.get("clip_report", {}).get("train_clip_rate_pct", 0.0) or 0.0
        )
    except Exception as exc:
        logger.warning(f"Error reading clip rate: {exc}")

    # Sprint 2 A2.3: data_loader survivorship_bias_report'unu da yansit.
    survivorship_report = (
        df.attrs.get("survivorship_bias_report", {}) if hasattr(df, "attrs") else {}
    )

    return {
        "corporate_action_anomaly": corporate_action_anomaly,
        "survivorship_warning": survivorship_warning,
        "survivorship_bias_report": survivorship_report,
        "psi_high": psi_high,
        "psi_max": psi_max,
        "psi_scores": psi_scores,
        "clip_rate": clip_rate,
    }
