# -*- coding: utf-8 -*-
"""Olasılıksal forward interval kalibrasyonu (model-agnostik).

İki kademe sağlar:

* **B2 — residual band**: walk-forward fold residual'larından (target uzayında)
  standart sapma σ; opsiyonel rejim-koşullu σ. Band ``p50 ± z·σ·√h`` (parametrik,
  normallik varsayımı). Hızlı baseline.
* **C — conformal**: dağılımdan bağımsız, kapsama-garantili. Split-conformal
  nonconformity skorlarından (``|y_true − y_pred|``) ampirik quantile ``q̂``;
  band ``p50 ± q̂``. ACI katmanı zaman-serisi exchangeability kırılımı için
  ``q̂``'i kayan kapsama sapmasına göre online ayarlar.

Tüm fonksiyonlar saf (model bağımsız) ve hedef (target) uzayında çalışır; fiyat
uzayına çeviri çağıran tarafın sorumluluğudur (``target_to_price`` + BIST clip).
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Standart iki-taraflı normal z değerleri (level -> z). Ara değerler NormalDist
# ile hesaplanır; bu tablo yalnızca hız/okunabilirlik içindir.
_COMMON_Z: Dict[float, float] = {
    0.80: 1.2815515594965686,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
}


def _validate_level(level: float) -> float:
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError(f"level 0<level<1 olmalı, alındı: {level}")
    return level


def z_for_level(level: float) -> float:
    """İki-taraflı güven seviyesi için normal z değeri döner (örn 0.9 -> 1.645)."""
    level = _validate_level(level)
    if level in _COMMON_Z:
        return _COMMON_Z[level]
    return float(NormalDist().inv_cdf((1.0 + level) / 2.0))


def _collect_residuals(
    fold_records: Sequence[Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Fold record'larından target-uzayı residual'ları + rejim etiketlerini toplar.

    residual = y_true_target − y_pred_target. Rejim etiketi yoksa boş string.
    """
    residuals: List[float] = []
    regimes: List[str] = []
    for rec in fold_records or []:
        y_true = np.asarray(rec.get("y_true_target", []), dtype=float).ravel()
        y_pred = np.asarray(rec.get("y_pred_target", []), dtype=float).ravel()
        n = min(len(y_true), len(y_pred))
        if n == 0:
            continue
        diff = y_true[:n] - y_pred[:n]
        regime_raw = rec.get("market_regime", [])
        regime_arr = np.asarray(regime_raw).ravel() if regime_raw is not None else np.asarray([])
        for i in range(n):
            val = diff[i]
            if not np.isfinite(val):
                continue
            residuals.append(float(val))
            regimes.append(str(regime_arr[i]) if i < len(regime_arr) else "")
    return np.asarray(residuals, dtype=float), np.asarray(regimes, dtype=object)


def _std(values: np.ndarray) -> float:
    """Örneklem standart sapması (ddof=1); n<2 ise 0.0."""
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def compute_residual_calibration(
    fold_records: Sequence[Dict[str, Any]],
    *,
    levels: Sequence[float] = (0.8,),
    per_regime: bool = True,
) -> Optional[Dict[str, Any]]:
    """B2 kalibrasyonu: walk-forward residual'larından σ (+ rejim-koşullu σ).

    Args:
        fold_records: walk-forward fold record listesi (``y_true_target``,
            ``y_pred_target``, opsiyonel ``market_regime`` içerir).
        levels: desteklenecek güven seviyeleri (sidecar'a z haritası yazılır).
        per_regime: rejim-koşullu σ hesapla.

    Returns:
        Kalibrasyon sözlüğü ya da residual yoksa None.
    """
    levels = tuple(_validate_level(lv) for lv in levels)
    if not levels:
        raise ValueError("levels boş olamaz")
    residuals, regimes = _collect_residuals(fold_records)
    if residuals.size == 0:
        return None
    sigma = _std(residuals)
    sigma_by_regime: Dict[str, float] = {}
    if per_regime and regimes.size == residuals.size:
        for label in set(regimes.tolist()):
            if not label:
                continue
            mask = regimes == label
            grp = residuals[mask]
            if grp.size >= 2:
                sigma_by_regime[str(label)] = _std(grp)
    return {
        "method": "residual_b2",
        "sigma": sigma,
        "sigma_by_regime": sigma_by_regime,
        "levels": list(levels),
        "z_map": {f"{lv:.2f}": z_for_level(lv) for lv in levels},
        "n_samples": int(residuals.size),
        "mean_residual": float(np.mean(residuals)),
    }


def residual_band(
    p50_target: float,
    sigma: float,
    horizon_index: int,
    level: float = 0.8,
) -> tuple[float, float]:
    """B2 bandı: ``p50 ± z·σ·√h`` (target uzayında).

    Args:
        p50_target: merkez (medyan) hedef tahmin.
        sigma: horizon-1 residual standart sapması (≥0).
        horizon_index: 1-tabanlı horizon adımı (√h ölçeği için).
        level: güven seviyesi.

    Returns:
        (lower_target, upper_target).
    """
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError(f"sigma negatif olamaz: {sigma}")
    h = max(int(horizon_index), 1)
    z = z_for_level(level)
    half = z * sigma * math.sqrt(h)
    return (float(p50_target) - half, float(p50_target) + half)


def resolve_active_calibration(
    calibration: Optional[Dict[str, Any]],
    prefer: str = "residual_b2",
) -> Optional[Dict[str, Any]]:
    """Sidecar kalibrasyonundan aktif üreteci seçer.

    Sidecar yapısı: top-level ``method="residual_b2"`` (+ σ alanları) ve gömülü
    ``"conformal"`` alt-sözlüğü. ``prefer="conformal"`` ise conformal alt-sözlüğü
    döner (yoksa B2'ye düşer); aksi halde top-level residual döner.
    """
    if not calibration:
        return None
    if prefer == "conformal":
        conf = calibration.get("conformal")
        if conf:
            return conf
    return calibration


def sigma_for_regime(calibration: Dict[str, Any], regime: Optional[str]) -> float:
    """Rejim-koşullu σ varsa onu, yoksa global σ'yı döner."""
    base = float(calibration.get("sigma", 0.0))
    if not regime:
        return base
    by_regime = calibration.get("sigma_by_regime") or {}
    return float(by_regime.get(str(regime), base))


# --- Kademe C — conformal -------------------------------------------------


def compute_conformal_calibration(
    fold_records: Sequence[Dict[str, Any]],
    *,
    level: float = 0.9,
    mode: str = "absolute",
) -> Optional[Dict[str, Any]]:
    """C kalibrasyonu: split-conformal nonconformity quantile ``q̂``.

    Nonconformity skoru ``s_i = |y_true − y_pred|`` (target uzayı). Ampirik
    quantile rank'i ``ceil((n+1)·level)/n`` (sonlu-örneklem kapsama garantisi),
    1.0'a clip'lenir.

    Args:
        fold_records: walk-forward fold record listesi.
        level: nominal kapsama (örn 0.9).
        mode: "absolute" (|hata|) varsayılan; "cqr" çağıranın quantile model
            artıklarını verdiği durumda da aynı skor mantığı kullanılır.

    Returns:
        Kalibrasyon sözlüğü ya da skor yoksa None.
    """
    level = _validate_level(level)
    residuals, _ = _collect_residuals(fold_records)
    if residuals.size == 0:
        return None
    scores = np.abs(residuals)
    n = int(scores.size)
    rank = math.ceil((n + 1) * level) / n
    rank = min(rank, 1.0)
    q_hat = float(np.quantile(scores, rank, method="higher"))
    return {
        "method": "conformal",
        "mode": str(mode),
        "level": level,
        "q_hat": q_hat,
        "n_calib": n,
    }


def conformal_band(
    p50_target: float,
    q_hat: float,
    horizon_index: int = 1,
    horizon_scale: bool = False,
) -> tuple[float, float]:
    """C bandı: ``p50 ± q̂`` (target uzayında).

    Args:
        p50_target: merkez hedef tahmin.
        q_hat: conformal nonconformity quantile (≥0).
        horizon_index: 1-tabanlı horizon adımı.
        horizon_scale: True ise ``q̂·√h`` (birikimli belirsizlik yaklaşımı).

    Returns:
        (lower_target, upper_target).
    """
    q_hat = float(q_hat)
    if q_hat < 0.0:
        raise ValueError(f"q_hat negatif olamaz: {q_hat}")
    half = q_hat
    if horizon_scale:
        half = q_hat * math.sqrt(max(int(horizon_index), 1))
    return (float(p50_target) - half, float(p50_target) + half)


def adaptive_conformal_update(
    q_hat: float,
    recent_coverage: float,
    *,
    target_level: float = 0.9,
    gamma: float = 0.05,
) -> float:
    """ACI-lite: kayan kapsama sapmasına göre ``q̂``'i online ayarlar.

    Kapsama hedefin altındaysa band genişler (q̂ artar), üstündeyse daralır.
    ``q_new = q̂ · (1 + γ·(target − recent))``, negatif olmayacak şekilde clip.

    Args:
        q_hat: mevcut conformal quantile.
        recent_coverage: son penceredeki ampirik kapsama [0,1].
        target_level: hedef nominal kapsama.
        gamma: öğrenme hızı.

    Returns:
        Güncellenmiş q̂ (≥0).
    """
    q_hat = float(q_hat)
    if q_hat < 0.0:
        raise ValueError(f"q_hat negatif olamaz: {q_hat}")
    target_level = _validate_level(target_level)
    recent_coverage = float(min(max(recent_coverage, 0.0), 1.0))
    adjusted = q_hat * (1.0 + float(gamma) * (target_level - recent_coverage))
    return max(adjusted, 0.0)
