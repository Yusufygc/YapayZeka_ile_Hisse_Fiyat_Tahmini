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


class XaiBlock(BaseModel):
    available: bool = False
    method: str = ""
    top_positive_reasons: List[XaiFactorItem] = []
    top_negative_reasons: List[XaiFactorItem] = []
    model_family_caveat: str = ""
    caveat: str = ""


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
