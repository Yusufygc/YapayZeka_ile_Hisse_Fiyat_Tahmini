# -*- coding: utf-8 -*-
"""Güven etiketi hesaplama motoru.

`GET /analysis/{symbol}` endpoint'i için `low / medium / high` güven etiketi
üretir. Karar hiyerarşisi yeniTasarim/09 Faz 1 tanımına uygundur.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, NamedTuple, Optional

ConfidenceLabelStr = Literal["low", "medium", "high"]

_SIGNAL_DIAGNOSIS_CAP_MEDIUM = {
    "insufficient_trades",
    "gate_too_strict",
    "model_signal_weak",
    "underperform_buyhold",
}

STABILITY_SCORE_THRESHOLD_LOW = -0.1
STABILITY_SCORE_THRESHOLD_HIGH = 0.5
DIR_ACC_THRESHOLD_LOW = 50.0
DIR_ACC_THRESHOLD_HIGH = 55.0
RMSE_VS_BENCHMARK_THRESHOLD = 1.0
ENSEMBLE_AGREEMENT_HIGH = 5 / 7


class ConfidenceResult(NamedTuple):
    label: ConfidenceLabelStr
    reasons: List[str]
    warnings: List[str]


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _parse_signal_diagnosis(raw: Any) -> set:
    if not raw or str(raw).strip().lower() in ("", "ok"):
        return set()
    return {lbl.strip() for lbl in str(raw).split(",") if lbl.strip()}


def compute_confidence(
    *,
    eligibility_status: str = "eligible",
    data_freshness: str = "fresh",
    directional_accuracy: Optional[float] = None,
    rmse_vs_benchmark: Optional[float] = None,
    signal_diagnosis: Optional[str] = None,
    stability_score: Optional[float] = None,
    psi_high: bool = False,
    corporate_action_anomaly: bool = False,
    model_status: str = "healthy",
    ensemble_direction_agreement: Optional[float] = None,
    regime_misalignment: bool = False,
    rolling_positive_window_ratio: Optional[float] = None,
) -> ConfidenceResult:
    """Güven etiketi hesapla.

    Parameters
    ----------
    eligibility_status:
        ``best_models.eligibility_status`` alanından gelir.
    data_freshness:
        ``"fresh"`` ya da ``"stale_data"``.
    directional_accuracy:
        Walk-forward ortalama directional accuracy (%).
    rmse_vs_benchmark:
        Model RMSE / naive benchmark RMSE. < 1 modelin iyi olduğunu gösterir.
    signal_diagnosis:
        Virgülle ayrılmış backtest sinyal tanı etiketleri.
    stability_score:
        Fold istikrar skoru (``stability_score`` experiments kolonu).
    psi_high:
        Distribution shift PSI > 0.25.
    corporate_action_anomaly:
        Kurumsal aksiyon anomalisi algılandı.
    model_status:
        ``"healthy"`` veya ``"degraded"``.
    ensemble_direction_agreement:
        0–1 arası oran; modellerin kaçı ana modelle aynı yönde (Faz 2).
    regime_misalignment:
        Tahmin yönü piyasa rejimiyle uyumsuz (Faz 2).
    rolling_positive_window_ratio:
        Rolling holdout pencerelerinin pozitif oranı (Faz 2).
    """
    reasons: List[str] = []
    warnings: List[str] = []

    # ── Hard blocks → daima low ───────────────────────────────────────────
    if eligibility_status != "eligible":
        return ConfidenceResult(
            label="low",
            reasons=[f"Model eligibility: {eligibility_status}"],
            warnings=warnings,
        )

    if data_freshness == "stale_data":
        warnings.append("Veri güncel değil; tahmin yorumlanırken dikkatli olunmalı.")
        return ConfidenceResult(label="low", reasons=["Veri tazelik durumu: stale_data"], warnings=warnings)

    dir_acc = _safe_float(directional_accuracy)
    if dir_acc is not None and dir_acc < DIR_ACC_THRESHOLD_LOW:
        return ConfidenceResult(
            label="low",
            reasons=[f"Yönsel doğruluk düşük ({dir_acc:.1f}% < {DIR_ACC_THRESHOLD_LOW}%)"],
            warnings=warnings,
        )

    rmse_ratio = _safe_float(rmse_vs_benchmark)
    if rmse_ratio is not None and rmse_ratio >= RMSE_VS_BENCHMARK_THRESHOLD:
        return ConfidenceResult(
            label="low",
            reasons=[f"RMSE benchmark'ı geçemiyor (rmse_vs_benchmark={rmse_ratio:.3f})"],
            warnings=warnings,
        )

    if psi_high:
        warnings.append("Distribution shift yüksek (PSI > 0.25); model girdileri değişmiş olabilir.")
        return ConfidenceResult(
            label="low",
            reasons=["Yüksek distribution shift tespit edildi."],
            warnings=warnings,
        )

    if corporate_action_anomaly:
        warnings.append("Kurumsal aksiyon anomalisi tespit edildi.")
        return ConfidenceResult(
            label="low",
            reasons=["Veri kalite uyarısı: kurumsal aksiyon anomalisi."],
            warnings=warnings,
        )

    if model_status == "degraded":
        warnings.append("Model son dönem canlı tahminlerde bozulma gösteriyor.")
        return ConfidenceResult(
            label="low",
            reasons=["Model canlı performansı: degraded"],
            warnings=warnings,
        )

    # ── Soft degradations → level cap veya düşürme ───────────────────────
    diagnosis_labels = _parse_signal_diagnosis(signal_diagnosis)
    capped_by_diagnosis = bool(diagnosis_labels & _SIGNAL_DIAGNOSIS_CAP_MEDIUM)
    if capped_by_diagnosis:
        active = sorted(diagnosis_labels & _SIGNAL_DIAGNOSIS_CAP_MEDIUM)
        reasons.append(f"Sinyal tanı uyarıları: {', '.join(active)}")

    stability = _safe_float(stability_score)
    stability_degraded = stability is not None and stability < STABILITY_SCORE_THRESHOLD_LOW
    if stability_degraded:
        reasons.append(f"Fold istikrarı düşük (stability_score={stability:.3f})")

    rolling_ratio = _safe_float(rolling_positive_window_ratio)
    rolling_degraded = rolling_ratio is not None and rolling_ratio < 0.5
    if rolling_degraded:
        reasons.append(f"Rolling holdout pozitif pencere oranı düşük ({rolling_ratio:.2f})")

    ensemble_low = (
        ensemble_direction_agreement is not None
        and ensemble_direction_agreement < 0.5
    )
    if ensemble_low:
        reasons.append(f"Ensemble yön uzlaşısı düşük ({ensemble_direction_agreement:.2f})")

    if regime_misalignment:
        reasons.append("Tahmin yönü piyasa rejimiyle uyumsuz.")

    # Herhangi bir soft degradation "high"a ulaşmayı engeller ve yeterince
    # fazlaysa medium'a caplar.
    any_soft = capped_by_diagnosis or stability_degraded or rolling_degraded or ensemble_low or regime_misalignment

    # ── high koşulları ────────────────────────────────────────────────────
    high_possible = (
        not any_soft
        and dir_acc is not None and dir_acc >= DIR_ACC_THRESHOLD_HIGH
        and (stability is None or stability >= STABILITY_SCORE_THRESHOLD_HIGH)
        and (rmse_ratio is None or rmse_ratio < RMSE_VS_BENCHMARK_THRESHOLD)
        and (ensemble_direction_agreement is None or ensemble_direction_agreement >= ENSEMBLE_AGREEMENT_HIGH)
    )

    if high_possible:
        return ConfidenceResult(label="high", reasons=[], warnings=warnings)

    return ConfidenceResult(label="medium", reasons=reasons, warnings=warnings)
