# -*- coding: utf-8 -*-
"""GET /analysis/{symbol} servis katmanı.

AnalysisService.build(symbol) çağrısı:
  1. StockModelDB'den en iyi modeli okur.
  2. En son forecast'ı okur.
  3. Veri tazeliğini kontrol eder.
  4. Güven etiketi hesaplar.
  5. XAI ürün özetini oluşturur.
  6. AnalysisResponse payload'unu birleştirir.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.api.constants import INVESTMENT_DISCLAIMER
from src.api.schemas.analysis import (
    AnalysisResponse,
    ConfidenceBlock,
    DataBlock,
    ForecastBlock,
    ForecastPoint,
    ModelBlock,
    PerformanceBlock,
    XaiBlock,
    XaiFactorItem,
)
from src.api.services.analysis_freshness import compute_freshness
from src.pipeline.confidence_calculator import compute_confidence
from src.xai.product_summary import build_xai_product_summary

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "stock_models.db")
_OUTPUTS_BASE = os.path.join(_PROJECT_ROOT, "outputs")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _model_family(model_name: str) -> str:
    name = str(model_name or "")
    if any(k in name for k in ("XGBoost", "LightGBM", "Random Forest")):
        return "tree"
    if any(k in name for k in ("Ridge", "ElasticNet")):
        return "linear"
    if any(k in name for k in ("LSTM", "DLinear", "NLinear", "Attention")):
        return "deep"
    if any(k in name for k in ("Naive", "ARIMA", "Prophet")):
        return "baseline"
    return "unknown"


class AnalysisService:
    def __init__(
        self,
        db_path: Optional[str] = None,
        outputs_base: Optional[str] = None,
    ) -> None:
        self._db_path = db_path or _DB_PATH
        self._outputs_base = outputs_base or _OUTPUTS_BASE

    def _get_db(self):
        from src.database.stock_model_db import StockModelDB

        return StockModelDB(self._db_path)

    def build(self, symbol: str) -> AnalysisResponse:
        symbol = symbol.upper()
        generated_at = _now_iso()

        db = self._get_db()

        # ── 1. En iyi model ──────────────────────────────────────────────
        best = db.get_best_model(symbol)
        if best is None:
            return AnalysisResponse(
                symbol=symbol,
                analysis_status="no_model",
                generated_at=generated_at,
                disclaimer=INVESTMENT_DISCLAIMER,
            )

        # ── 2. En son forecast ──────────────────────────────────────────
        forecast_row = db.get_latest_forecast(symbol)
        if forecast_row is None:
            return AnalysisResponse(
                symbol=symbol,
                analysis_status="no_forecast",
                generated_at=generated_at,
                model=_build_model_block(best),
                disclaimer=INVESTMENT_DISCLAIMER,
            )

        # ── 3. Veri tazeliği ────────────────────────────────────────────
        last_observed = str(forecast_row.get("last_observed_date", "") or "")
        freshness = compute_freshness(last_observed)

        # ── 4. Performance block ─────────────────────────────────────────
        perf = _build_performance_block(best)

        # ── 5. Güven etiketi ────────────────────────────────────────────
        conf_result = compute_confidence(
            eligibility_status=str(best.get("eligibility_status", "eligible")),
            data_freshness=freshness.status,
            directional_accuracy=best.get("dir_acc"),
            rmse_vs_benchmark=None,
            signal_diagnosis=best.get("signal_diagnosis"),
            stability_score=best.get("stability_score"),
        )
        conf_block = ConfidenceBlock(
            label=conf_result.label,
            reasons=conf_result.reasons,
            warnings=conf_result.warnings,
        )

        # ── 6. XAI özeti ────────────────────────────────────────────────
        model_name = str(best.get("model_name", ""))
        xai_summary = build_xai_product_summary(
            symbol=symbol,
            model_name=model_name,
            outputs_base=self._outputs_base,
        )
        xai_block = _build_xai_block(xai_summary)

        # ── 7. Status kodu ──────────────────────────────────────────────
        status = _resolve_status(
            freshness=freshness.status,
            xai_available=xai_summary.available,
            confidence_label=conf_result.label,
        )

        return AnalysisResponse(
            symbol=symbol,
            analysis_status=status,
            generated_at=generated_at,
            data=DataBlock(
                last_observed_date=last_observed or None,
                last_close=forecast_row.get("last_close"),
                data_freshness=freshness.status,
                staleness_days=freshness.staleness_days,
            ),
            model=_build_model_block(best),
            forecast=_build_forecast_block(forecast_row),
            performance=perf,
            confidence=conf_block,
            xai=xai_block,
            disclaimer=INVESTMENT_DISCLAIMER,
        )


def _build_model_block(best: Dict[str, Any]) -> ModelBlock:
    model_name = str(best.get("model_name", ""))
    return ModelBlock(
        model_name=model_name,
        model_family=_model_family(model_name),
        source_experiment_id=best.get("experiment_id"),
        run_id=best.get("run_id"),
        validation_mode=best.get("validation_mode"),
        trained_at=best.get("updated_at"),
        is_trainable_model=True,
        is_baseline=False,
        is_ensemble=model_name.startswith("Ensemble"),
        eligibility_status=str(best.get("eligibility_status", "eligible")),
        eligibility_reason=str(best.get("eligibility_reason", "")),
    )


def _build_performance_block(best: Dict[str, Any]) -> PerformanceBlock:
    return PerformanceBlock(
        rmse=best.get("rmse"),
        mae=best.get("mae"),
        directional_accuracy=best.get("dir_acc"),
        hit_rate=best.get("hit_rate"),
        sharpe=best.get("sharpe"),
        composite_score=best.get("composite_score"),
    )


def _build_forecast_block(forecast_row: Dict[str, Any]) -> ForecastBlock:
    raw_points = forecast_row.get("points") or []
    points = [
        ForecastPoint(
            target_date=str(p.get("target_date", "")),
            horizon_index=int(p.get("horizon_index", 0)),
            bounded_predicted_close=p.get("bounded_predicted_close"),
            predicted_return=p.get("predicted_return"),
        )
        for p in raw_points
    ]
    raw_agreement = forecast_row.get("ensemble_direction_agreement")
    return ForecastBlock(
        horizon_days=forecast_row.get("horizon_days"),
        trend_label=forecast_row.get("trend_label"),
        weekly_expected_return=forecast_row.get("weekly_expected_return"),
        trend_threshold=forecast_row.get("trend_threshold"),
        ensemble_agreement=float(raw_agreement) if raw_agreement is not None else None,
        points=points,
    )


def _build_xai_block(summary) -> XaiBlock:
    if not summary.available:
        return XaiBlock(available=False, caveat=summary.caveat)
    return XaiBlock(
        available=True,
        method=summary.method,
        top_positive_reasons=[
            XaiFactorItem(
                feature_name=f.feature_name,
                human_label=f.human_label,
                importance=f.importance,
                direction=f.direction,
            )
            for f in summary.top_positive_reasons
        ],
        top_negative_reasons=[
            XaiFactorItem(
                feature_name=f.feature_name,
                human_label=f.human_label,
                importance=f.importance,
                direction=f.direction,
            )
            for f in summary.top_negative_reasons
        ],
        model_family_caveat=summary.model_family_caveat,
        caveat=summary.caveat,
    )


def _resolve_status(*, freshness: str, xai_available: bool, confidence_label: str) -> str:
    # Hiyerarşi: no_model > no_forecast > stale_data > xai_unavailable > low_confidence > ok
    if freshness == "stale_data":
        return "stale_data"
    if not xai_available:
        return "xai_unavailable"
    if confidence_label == "low":
        return "low_confidence"
    return "ok"
