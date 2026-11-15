# -*- coding: utf-8 -*-
"""
Sprint 7 (2026-05-25) A7.3 — Analysis API icin on-the-fly PSI 30g monitor.
2026-11-15: tahmin edici + esikler gercek veride bos-dagilimdan (placebo)
kalibre edildi.

`compute_psi_30d(symbol_csv_path)` son 30 isgunundeki OHLCV-turevli feature
dagilimlarini, onceki 252 isgunlu pencereyle karsilastirir; feature basina
PSI'nin ORTALAMASINI doner (`psi_30d`).
Tier:
  - < 0.40            -> "stable"
  - 0.40 <= x < 0.80  -> "moderate_drift"
  - >= 0.80           -> "major_drift"  (confidence -> low)

Cikti analysis API `data_quality` blogunda yayinlanir.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

_HOLDOUT_DAYS = 30
_TRAIN_DAYS = 252

# ---------------------------------------------------------------------------
# Kalibrasyon (2026-11-15, tools: scratch psi_calib.py)
# Evren: data/*.csv, 581 sembol; placebo bitis tarihleri her 21 isgununde bir
# (sembol basina ~87 pencere), toplam 50.600 placebo pencere; her biri gercek
# "30g vs onceki 252g" istatistigi. Bu istatistigin gercek veride NULL
# dagilimi (rejim kumelenmesi/volatilite kalicilik dahil) soyle:
#   mean-PSI (3 feature ortalamasi), secilen tahmin edici:
#     q50=0.171  q75=0.263  q90=0.399  q95=0.512  q99=0.795
#   (max-PSI: q50=0.277 q90=0.763 q99=1.554; ham feature q99: log_return 0.94,
#    range_pct 1.51, volume_log_change 0.42)
# Eski tahmin edici (esit-genislik 3 bin, eps=1e-4, max) icin ayni null:
#   q50=0.295 q90=0.921 q99=2.103 -> eski esikler 0.10/0.25 null MEDYANININ
#   bile altindaydi; sembollerin ~%60+'i "drift" isaretleniyordu (hepsi gurultu).
# Esikler: moderate ~ null q90 (0.399 -> 0.40), major ~ null q99 (0.795 -> 0.80).
# Not: gercek veride null, iid varsayimindan cok yuksektir (volatilite
# kumelenmesi); iid gurultude bu tahmin edici ~0.05-0.15 verir.
# ---------------------------------------------------------------------------
_PSI_STABLE_MAX = 0.40
_PSI_MAJOR_MIN = 0.80
# Referans (252g) dagiliminin 10/35/65/90. yuzdelik kesimleri -> 5 acik uclu
# bin. Esit-genislik bin'ler combined araligi uc degerlerle (fat tail) gerer ve
# 30 orneklikte bos bin'e 1e-4 eps ile log-patlamasi uretir; referans-kantil
# bin + additif (Laplace) yumusatma ikisini de giderir. Kuyruk-agirlikli
# kesimler (10/90) olcek (volatilite) kaymalarini da yakalar.
_PSI_QUANTILE_EDGES = (0.10, 0.35, 0.65, 0.90)
_PSI_SMOOTHING = 1.0  # her bin'e eklenen sahte sayim (Laplace)


@dataclass(frozen=True)
class DataQualityResult:
    psi_30d: Optional[float]
    psi_status: str  # "stable" | "moderate_drift" | "major_drift" | "unavailable"
    stale_warning: bool
    reason: Optional[str] = None


def _psi_smoothed(ref_vals: np.ndarray, cur_vals: np.ndarray) -> float:
    """
    Referans-kantil bin + Laplace yumusatmali PSI (30 orneklik pencere icin).

    Bin kenarlari yalniz referans pencereden (cur sizintisi yok) alinir; uclar
    aciktir (cur'daki aykiri deger son bine duser). Referansta ties varsa
    kenarlar tekillestirilir. Yetersiz/sonlu olmayan veri -> 0.0.
    """
    ref = ref_vals[np.isfinite(ref_vals)]
    cur = cur_vals[np.isfinite(cur_vals)]
    if len(ref) < 2 * len(_PSI_QUANTILE_EDGES) or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, _PSI_QUANTILE_EDGES))
    k = len(edges) + 1
    f_ref = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=k)
    f_cur = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=k)
    p_ref = (f_ref + _PSI_SMOOTHING) / (f_ref.sum() + k * _PSI_SMOOTHING)
    p_cur = (f_cur + _PSI_SMOOTHING) / (f_cur.sum() + k * _PSI_SMOOTHING)
    return max(0.0, float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref))))


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
    Sembolun CSV'sini okur; son 30 gun vs onceki 252 gun PSI (feature ortalamasi) doner.
    Veri yok / yetersiz ise psi_status='unavailable'.
    """
    if not os.path.exists(symbol_csv_path):
        return DataQualityResult(
            psi_30d=None,
            psi_status="unavailable",
            stale_warning=True,
            reason="csv_missing",
        )
    # Ortak temizleyici: TR/EN basliklari (Tarih/Kapanis...) Date/Close'a cevirir,
    # tarih ayristirma + tekrar + Adj Close olcekleme loader'larla aynidir.
    # Onceden ham read_csv + "Date"/"Close" kontrolu TR basliklarda hep
    # missing_required_columns donuyordu (PSI kapisi fiilen kapaliydi).
    from src.data.ohlcv_clean import clean_ohlcv_csv

    try:
        df = clean_ohlcv_csv(symbol_csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
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
            psi_scores[col] = _psi_smoothed(t_vals, h_vals)
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

    psi_mean = float(np.mean(list(psi_scores.values())))
    return DataQualityResult(
        psi_30d=psi_mean,
        psi_status=_tier(psi_mean),
        stale_warning=False,
        reason=None,
    )
