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


class XaiGroupSummary(BaseModel):
    feature_group: str
    group_label: str
    total_importance: float
    net_contribution: float
    direction: str
    top_features: List[str] = []
    reason: str
    approximate_ratio: float = 0.0


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
    # E2 Kol-B pooled fiyat bandı
    kolb_price_p50: Optional[float] = None
    kolb_price_low: Optional[float] = None
    kolb_price_high: Optional[float] = None
    kolb_horizon_days: Optional[int] = None
    kolb_band_level: Optional[float] = None
    # Risk: havuzlu oynaklik modeli (src/risk) — 5g sigma + kalibre %80 bant
    risk_sigma_5d: Optional[float] = None         # 5g log-getiri std tahmini
    risk_sigma_naive_5d: Optional[float] = None   # naive persistence (son 21g RMS * sqrt(5))
    risk_vol_percentile: Optional[float] = None   # 0-100, gun evreni icinde
    risk_level: Optional[str] = None              # düşük | orta | yüksek (evren tertile)
    risk_z: Optional[float] = None                # konformal band z (Gaussal 1.28 DEGIL)
    risk_band_level: Optional[float] = None
    risk_horizon_days: Optional[int] = None
    risk_method: Optional[str] = None             # har_x_ridge | naive_fallback
    risk_validation: Optional[Dict[str, Any]] = None  # kosunun OOS metrikleri + kapi
    # E2 Kol-B XAI — pooled modelin per-symbol feature attribution'i (SHAP, additive).
    xai_available: bool = False
    xai_method: str = ""                          # shap_tree | permutation_fallback
    xai_caveat: str = ""                          # ensemble ise LGB-leg uyarisi
    xai_approximate: Optional[bool] = None
    xai_error: Optional[str] = None
    xai_generated_at: Optional[str] = None
    xai_top_positive: List[XaiFactorItem] = []    # sirayi yukari iten suruculer
    xai_top_negative: List[XaiFactorItem] = []    # sirayi asagi iten suruculer
    xai_group_summaries: List[XaiGroupSummary] = []


class ServingRunBlock(BaseModel):
    """Son gecelik havuz kosusunun (PeerStore) kimligi ve tazeligi."""

    run_id: Optional[int] = None
    model_name: Optional[str] = None
    as_of_date: Optional[str] = None
    age_days: Optional[int] = None
    stale: bool = False


class AnalysisResponse(BaseModel):
    """API v2: yalniz gecelik havuz (Kol-B + risk) ciktisindan uretilir.

    Per-sembol yon modeli (Kol-A) emekli oldu; `model`, `forecast`, `performance`,
    `xai` (Kol-A) ve refresh-job alanlari kaldirildi (kirici degisiklik).
    """

    api_version: str = "2"
    symbol: str
    # ok | stale_data | low_confidence | no_data
    analysis_status: str = "ok"
    generated_at: str
    data: DataBlock = DataBlock()
    confidence: ConfidenceBlock = ConfidenceBlock()
    disclaimer: str = ""
    serving_run: Optional[ServingRunBlock] = None
    # Sprint 7 (2026-05-25) A7.3 — son 30g PSI drift monitoru (yalniz piyasa verisi).
    data_quality: Optional[DataQualityBlock] = None
    # Pooled global model akran-goreli + risk ciktisi.
    peer: Optional[PeerBlock] = None
