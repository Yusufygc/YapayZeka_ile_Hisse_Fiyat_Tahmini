# -*- coding: utf-8 -*-
"""GET /analysis/{symbol} endpoint Pydantic şemaları.

yeniTasarim/04 API sözleşmesiyle birebir uyumludur.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Pydantic yüklü değil. pip install pydantic>=2")


class DataBlock(BaseModel):
    last_observed_date: Optional[str] = None
    last_close: Optional[float] = None
    data_freshness: str = "unknown"
    staleness_days: int = 0


class ModelBlock(BaseModel):
    model_name: Optional[str] = None
    model_family: Optional[str] = None
    selection_reason: str = "Geçmiş doğrulama sonuçlarına göre en iyi eligible model"
    source_experiment_id: Optional[int] = None
    run_id: Optional[str] = None
    validation_mode: Optional[str] = None
    trained_at: Optional[str] = None
    is_trainable_model: bool = True
    is_baseline: bool = False
    is_ensemble: bool = False
    eligibility_status: str = "eligible"
    eligibility_reason: str = ""


class ForecastPoint(BaseModel):
    target_date: str
    horizon_index: int
    bounded_predicted_close: Optional[float] = None
    predicted_return: Optional[float] = None
    # Sprint 4 (2026-05-25) A4.5: model `predict_quantiles` destekliyorsa
    # advisory confidence band alanlari. Yoksa None (geriye uyumlu).
    p10_close: Optional[float] = None
    p50_close: Optional[float] = None
    p90_close: Optional[float] = None
    predicted_return_p10: Optional[float] = None
    predicted_return_p50: Optional[float] = None
    predicted_return_p90: Optional[float] = None
    lower_band: Optional[float] = None
    upper_band: Optional[float] = None
    price_tick: Optional[float] = None


class ForecastBlock(BaseModel):
    horizon_days: Optional[int] = None
    trend_label: Optional[str] = None
    weekly_expected_return: Optional[float] = None
    trend_threshold: Optional[float] = None
    ensemble_agreement: Optional[float] = None
    trend_context: Optional[Dict[str, Any]] = None
    points: List[ForecastPoint] = []


class PerformanceBlock(BaseModel):
    rmse: Optional[float] = None
    mae: Optional[float] = None
    directional_accuracy: Optional[float] = None
    hit_rate: Optional[float] = None
    sharpe: Optional[float] = None
    rmse_vs_benchmark: Optional[float] = None
    composite_score: Optional[float] = None
    stability_score: Optional[float] = None


class ConfidenceBlock(BaseModel):
    label: Literal["low", "medium", "high"] = "low"
    reasons: List[str] = []
    warnings: List[str] = []


class XaiFactorItem(BaseModel):
    feature_name: str
    human_label: str
    importance: float
    direction: str
    feature_group: Optional[str] = None
    reason: Optional[str] = None
    method: Optional[str] = None
    contribution: Optional[float] = None
    approximate: Optional[bool] = None


class XaiBlock(BaseModel):
    available: bool = False
    method: str = ""
    top_positive_reasons: List[XaiFactorItem] = []
    top_negative_reasons: List[XaiFactorItem] = []
    model_family_caveat: str = ""
    caveat: str = ""


class DataQualityBlock(BaseModel):
    """Sprint 7 (2026-05-25) A7.3 — son 30g vs onceki 252g PSI drift."""

    psi_30d: Optional[float] = None
    psi_status: Literal[
        "stable", "moderate_drift", "major_drift", "unavailable"
    ] = "unavailable"
    stale_warning: bool = False
    reason: Optional[str] = None


class PeerBlock(BaseModel):
    """E2 Faz 5 — pooled global model cross-sectional (akran-goreli) ciktisi.

    Ek (additive) blok; mevcut mutlak forecast/confidence alanlari korunur.
    Kaynak: nightly batch -> PeerStore (global_model_runs + peer_scores).
    """

    available: bool = False
    as_of_date: Optional[str] = None
    peer_score: Optional[float] = None       # -1..1 (merkezli cross-sectional sira)
    peer_percentile: Optional[float] = None  # 0..100
    peer_label: Optional[str] = None         # outperform | inline | underperform | unknown
    universe_size: Optional[int] = None
    segment_liq: Optional[str] = None
    segment_vol: Optional[str] = None
    segment_sector: Optional[str] = None
    segment_icir: Optional[float] = None
    confidence_label: Optional[str] = None   # low | medium | high (segment_IC x tradability)
    confidence_reasons: List[str] = []
    confidence_warnings: List[str] = []
    model_run_id: Optional[int] = None
    icir_overall: Optional[float] = None
    # E2 Faz 7 — peer rank -> mutlak trend egilimi (kalibre, olasiliksal).
    trend_label: Optional[str] = None             # yukarı | yatay | aşağı | belirsiz
    trend_prob_up: Optional[float] = None         # kalibre P(h-gun getiri > 0)
    trend_expected_return: Optional[float] = None  # kalibre ort. h-gun log-getiri
    # E2 Kol-B XAI — pooled modelin per-symbol feature attribution'i (SHAP, additive).
    xai_available: bool = False
    xai_method: str = ""                          # shap_tree | permutation_fallback
    xai_caveat: str = ""                          # ensemble ise LGB-leg uyarisi
    xai_top_positive: List[XaiFactorItem] = []    # sirayi yukari iten suruculer
    xai_top_negative: List[XaiFactorItem] = []    # sirayi asagi iten suruculer


class ForecastSourceBlock(BaseModel):
    type: str = "model"
    model_name: Optional[str] = None
    source_experiment_id: Optional[int] = None
    run_at: Optional[str] = None
    last_observed_date: Optional[str] = None
    method: Optional[str] = None
    members: List[str] = []
    weights: Dict[str, float] = {}
    source_experiment_ids: List[int] = []
    forecast_strategy: Optional[str] = None
    artifact_mode: Optional[str] = None
    warnings: List[str] = []


class AnalysisResponse(BaseModel):
    symbol: str
    analysis_status: str = "ok"
    generated_at: str
    data: DataBlock = DataBlock()
    model: ModelBlock = ModelBlock()
    forecast: ForecastBlock = ForecastBlock()
    performance: PerformanceBlock = PerformanceBlock()
    confidence: ConfidenceBlock = ConfidenceBlock()
    xai: XaiBlock = XaiBlock()
    disclaimer: str = ""
    refresh_status: Literal["none", "queued", "running", "completed", "failed"] = "none"
    refresh_reason: Optional[str] = None
    refresh_job_id: Optional[str] = None
    forecast_source: Optional[ForecastSourceBlock] = None
    # Sprint 7 (2026-05-25) A7.3 — son 30g PSI drift monitoru.
    data_quality: Optional[DataQualityBlock] = None
    # E2 Faz 5 — pooled global model akran-goreli ciktisi (additive).
    peer: Optional[PeerBlock] = None
