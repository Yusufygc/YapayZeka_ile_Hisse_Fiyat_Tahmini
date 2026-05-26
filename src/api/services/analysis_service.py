# -*- coding: utf-8 -*-
"""GET /analysis/{symbol} service layer."""
from __future__ import annotations

import os
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.api.constants import INVESTMENT_DISCLAIMER
from src.api.schemas.analysis import (
    AnalysisResponse,
    ConfidenceBlock,
    DataBlock,
    DataQualityBlock,
    ForecastBlock,
    ForecastPoint,
    ForecastSourceBlock,
    ModelBlock,
    PerformanceBlock,
    XaiBlock,
    XaiFactorItem,
)
from src.api.services.analysis_freshness import compute_freshness
from src.api.services.data_quality_monitor import compute_psi_30d
from src.api.services.data_refresh_service import DataRefreshService, read_latest_market_row
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
        project_root: Optional[str] = None,
        enable_background_refresh: Optional[bool] = None,
        refresh_wait_timeout_seconds: Optional[float] = None,
    ) -> None:
        self._project_root = project_root or _PROJECT_ROOT
        self._db_path = db_path or os.path.join(self._project_root, "data", "stock_models.db")
        self._outputs_base = outputs_base or os.path.join(self._project_root, "outputs")
        self._enable_background_refresh = (
            self._db_path == os.path.join(self._project_root, "data", "stock_models.db")
            if enable_background_refresh is None
            else bool(enable_background_refresh)
        )
        if refresh_wait_timeout_seconds is None:
            self._refresh_wait_timeout_seconds = 90.0 if self._enable_background_refresh else 0.0
        else:
            self._refresh_wait_timeout_seconds = float(refresh_wait_timeout_seconds)

    def _get_db(self):
        from src.database.stock_model_db import StockModelDB

        return StockModelDB(self._db_path)

    def build(self, symbol: str) -> AnalysisResponse:
        if not re.match(r"^[A-Z0-9]{1,10}$", symbol.upper()):
            raise ValueError(f"Invalid symbol format: {symbol}")
        symbol = symbol.upper()
        generated_at = _now_iso()
        db = self._get_db()
        best = db.get_best_model(symbol)
        if best is None:
            return AnalysisResponse(
                symbol=symbol,
                analysis_status="no_model",
                generated_at=generated_at,
                disclaimer=INVESTMENT_DISCLAIMER,
            )

        latest = (
            _safe_latest_market_row(os.path.join(self._project_root, "data", f"{symbol}.csv"))
            if self._enable_background_refresh
            else None
        )
        forecast_row = _find_matching_forecast(
            db=db,
            symbol=symbol,
            best=best,
            latest_observed_date=None if latest is None else latest.date,
        )
        if forecast_row is None:
            job = _queue_refresh(
                db=db,
                project_root=self._project_root,
                outputs_base=self._outputs_base,
                start_background=self._enable_background_refresh,
                wait_timeout_seconds=self._refresh_wait_timeout_seconds,
                symbol=symbol,
                best=best,
                reason="missing_forecast_for_best_model",
            )
            if _job_status(job) == "completed":
                refreshed = _reload_forecast_state(
                    service=self,
                    symbol=symbol,
                    fallback_best=best,
                )
                if refreshed is not None:
                    db, best, forecast_row = refreshed
                    return _build_forecast_response(
                        db=db,
                        symbol=symbol,
                        generated_at=generated_at,
                        best=best,
                        forecast_row=forecast_row,
                        outputs_base=self._outputs_base,
                        refresh_job=job,
                        refresh_reason="missing_forecast_for_best_model",
                        project_root=self._project_root,
                    )
            return AnalysisResponse(
                symbol=symbol,
                analysis_status="no_forecast",
                generated_at=generated_at,
                model=_build_model_block(best),
                performance=_build_performance_block(best),
                disclaimer=INVESTMENT_DISCLAIMER,
                refresh_status=_job_status(job),
                refresh_reason=_refresh_reason(job, "missing_forecast_for_best_model"),
                refresh_job_id=None if job is None else job.get("job_id"),
            )

        last_observed = str(forecast_row.get("last_observed_date", "") or "")
        freshness = compute_freshness(last_observed)
        stale_job = None
        if freshness.status == "stale_data":
            stale_job = _queue_refresh(
                db=db,
                project_root=self._project_root,
                outputs_base=self._outputs_base,
                start_background=self._enable_background_refresh,
                wait_timeout_seconds=self._refresh_wait_timeout_seconds,
                symbol=symbol,
                best=best,
                reason="stale_market_data",
            )
            if _job_status(stale_job) == "completed":
                refreshed = _reload_forecast_state(
                    service=self,
                    symbol=symbol,
                    fallback_best=best,
                )
                if refreshed is not None:
                    db, best, forecast_row = refreshed
                    return _build_forecast_response(
                        db=db,
                        symbol=symbol,
                        generated_at=generated_at,
                        best=best,
                        forecast_row=forecast_row,
                        outputs_base=self._outputs_base,
                        refresh_job=stale_job,
                        refresh_reason="stale_market_data",
                        project_root=self._project_root,
                    )

        return _build_forecast_response(
            db=db,
            symbol=symbol,
            generated_at=generated_at,
            best=best,
            forecast_row=forecast_row,
            outputs_base=self._outputs_base,
            refresh_job=stale_job,
            refresh_reason=None if stale_job is None else "stale_market_data",
            project_root=self._project_root,
        )


def _safe_latest_market_row(data_file: str):
    try:
        return read_latest_market_row(data_file)
    except Exception:
        return None


def _reload_forecast_state(
    *,
    service: AnalysisService,
    symbol: str,
    fallback_best: Dict[str, Any],
):
    db = service._get_db()
    best = db.get_best_model(symbol) or fallback_best
    latest = (
        _safe_latest_market_row(os.path.join(service._project_root, "data", f"{symbol}.csv"))
        if service._enable_background_refresh
        else None
    )
    forecast_row = _find_matching_forecast(
        db=db,
        symbol=symbol,
        best=best,
        latest_observed_date=None if latest is None else latest.date,
    )
    if forecast_row is None:
        return None
    return db, best, forecast_row


def _build_forecast_response(
    *,
    db,
    symbol: str,
    generated_at: str,
    best: Dict[str, Any],
    forecast_row: Dict[str, Any],
    outputs_base: str,
    refresh_job: Optional[Dict[str, Any]] = None,
    refresh_reason: Optional[str] = None,
    project_root: Optional[str] = None,
) -> AnalysisResponse:
    last_observed = str(forecast_row.get("last_observed_date", "") or "")
    freshness = compute_freshness(last_observed)
    perf = _build_performance_block(best)
    live_model_status = _live_model_status(db, symbol)

    # Sprint 7 A7.3 — early PSI fetch (re-used in confidence + response).
    _early_dq = _build_data_quality_block(symbol=symbol, project_root=project_root)
    psi_high = _early_dq.psi_status == "major_drift"

    ensemble_agreement_raw = forecast_row.get("ensemble_direction_agreement")
    ensemble_agreement = (
        float(ensemble_agreement_raw) if ensemble_agreement_raw is not None else None
    )

    conf_result = compute_confidence(
        eligibility_status=str(best.get("eligibility_status", "eligible")),
        data_freshness=freshness.status,
        directional_accuracy=best.get("dir_acc"),
        rmse_vs_benchmark=best.get("rmse_vs_benchmark"),
        signal_diagnosis=best.get("signal_diagnosis"),
        stability_score=best.get("stability_score"),
        psi_high=psi_high,
        model_status=live_model_status,
        ensemble_direction_agreement=ensemble_agreement,
    )
    warnings = list(conf_result.warnings)
    if freshness.warning:
        warnings.append(freshness.warning)

    # Sprint 8 A8.1 — pozitif sebepleri ekle (high veya medium icin).
    reasons = list(conf_result.reasons)
    positive_reasons = _build_positive_reasons(
        best=best,
        ensemble_agreement=ensemble_agreement,
        data_quality_block=_early_dq,
    )
    if conf_result.label in {"medium", "high"}:
        reasons.extend(positive_reasons)

    # Sprint 7 A7.3 — moderate_drift soft downgrade (major_drift compute_confidence
    # icinde psi_high path'iyle zaten hard block oluyor; burada moderate'i ekle).
    conf_label = conf_result.label
    if _early_dq.psi_status == "moderate_drift":
        warnings.append(
            f"data_drift_moderate:psi_30d={_early_dq.psi_30d:.3f}"
        )
        if conf_label == "high":
            conf_label = "medium"
    elif _early_dq.psi_status == "major_drift" and _early_dq.psi_30d is not None:
        # compute_confidence zaten low veriyor; richer warning string ekle.
        warnings.append(
            f"data_drift_major:psi_30d={_early_dq.psi_30d:.3f}_>=0.25"
        )
    conf_block = ConfidenceBlock(
        label=conf_label,
        reasons=reasons,
        warnings=warnings,
    )

    model_name = str(best.get("model_name", ""))
    xai_summary = build_xai_product_summary(
        symbol=symbol,
        model_name=model_name,
        outputs_base=outputs_base,
        run_id=best.get("run_id"),
        model_path=best.get("model_path"),
    )
    xai_block = _build_xai_block(xai_summary)
    status = _resolve_status(
        freshness=freshness.status,
        xai_available=xai_summary.available,
        confidence_label=conf_result.label,
    )

    # Sprint 7 A7.3 — early hesabi tekrar kullan (cift hesap olmasin).
    data_quality_block = _early_dq

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
        refresh_status=_job_status(refresh_job),
        refresh_reason=_refresh_reason(refresh_job, refresh_reason),
        refresh_job_id=None if refresh_job is None else refresh_job.get("job_id"),
        forecast_source=_build_forecast_source_block(forecast_row),
        data_quality=data_quality_block,
    )


def _build_positive_reasons(
    *,
    best: Dict[str, Any],
    ensemble_agreement: Optional[float],
    data_quality_block: "DataQualityBlock",
) -> list:
    """
    Sprint 8 A8.1 — confidence.reasons pozitif sinyalleri.

    confidence-and-risk-policy.md uyari mapping'i: medium/high label'larda
    kullaniciya neden bu seviye verildigini anlatan kisa Turkce ifadeler.
    """
    reasons: list = []
    dir_acc = best.get("dir_acc")
    if dir_acc is not None:
        try:
            reasons.append(f"Walk-forward yonsel dogruluk: %{float(dir_acc):.1f}")
        except (TypeError, ValueError):
            pass

    hit_rate = best.get("hit_rate")
    if hit_rate is not None:
        try:
            reasons.append(f"Hit rate: %{float(hit_rate):.1f}")
        except (TypeError, ValueError):
            pass

    rmse_bench = best.get("rmse_vs_benchmark")
    if rmse_bench is not None:
        try:
            ratio = float(rmse_bench)
            if ratio < 1.0:
                reasons.append(
                    f"RMSE benchmark altinda (rmse_vs_benchmark={ratio:.3f})"
                )
        except (TypeError, ValueError):
            pass

    stability = best.get("stability_score")
    if stability is not None:
        try:
            val = float(stability)
            if val >= 0.5:
                reasons.append(f"Fold istikrari yuksek (stability_score={val:.2f})")
        except (TypeError, ValueError):
            pass

    composite = best.get("composite_score")
    if composite is not None:
        try:
            reasons.append(f"Composite score: {float(composite):.1f}/100")
        except (TypeError, ValueError):
            pass

    if ensemble_agreement is not None and ensemble_agreement >= 5 / 7:
        reasons.append(
            f"Ensemble yon uzlasisi yuksek ({ensemble_agreement:.2f})"
        )

    if data_quality_block.psi_status == "stable":
        reasons.append("Veri dagilimi stabil (PSI 30g < 0.10)")

    return reasons


def _build_data_quality_block(
    *,
    symbol: str,
    project_root: Optional[str],
) -> DataQualityBlock:
    """Sprint 7 A7.3 — on-the-fly PSI 30g hesabini API'ye tasi."""
    if not project_root:
        return DataQualityBlock(
            psi_30d=None, psi_status="unavailable", reason="project_root_missing"
        )
    csv_path = os.path.join(project_root, "data", f"{symbol}.csv")
    try:
        result = compute_psi_30d(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        return DataQualityBlock(
            psi_30d=None,
            psi_status="unavailable",
            stale_warning=True,
            reason=f"monitor_failed:{type(exc).__name__}",
        )
    return DataQualityBlock(
        psi_30d=result.psi_30d,
        psi_status=result.psi_status,
        stale_warning=result.stale_warning,
        reason=result.reason,
    )


def _find_matching_forecast(
    *,
    db,
    symbol: str,
    best: Dict[str, Any],
    latest_observed_date: Optional[str],
) -> Optional[Dict[str, Any]]:
    best_experiment_id = best.get("experiment_id")
    best_model_name = str(best.get("model_name", ""))
    for row in db.get_forecast_history(symbol, limit=50):
        if str(row.get("model_name", "")) != best_model_name:
            continue
        if best_experiment_id is not None and row.get("source_experiment_id") != best_experiment_id:
            continue
        if latest_observed_date and str(row.get("last_observed_date", ""))[:10] != latest_observed_date:
            continue
        return row
    return None


def _queue_refresh(
    *,
    db,
    project_root: str,
    outputs_base: str,
    start_background: bool,
    wait_timeout_seconds: float,
    symbol: str,
    best,
    reason: str,
):
    try:
        return DataRefreshService(
            db=db,
            project_root=project_root,
            outputs_base=outputs_base,
            start_background=start_background,
        ).ensure_refresh_job(
            symbol=symbol,
            reason=reason,
            best_model=best,
            wait_timeout_seconds=wait_timeout_seconds,
        )
    except Exception:
        return None


def _job_status(job: Optional[Dict[str, Any]]) -> str:
    if not job:
        return "none"
    status = str(job.get("status", "none"))
    return "failed" if status == "error" else status


def _job_payload(job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not job:
        return {}
    raw = job.get("payload_json")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _refresh_reason(job: Optional[Dict[str, Any]], fallback: Optional[str]) -> Optional[str]:
    payload = _job_payload(job)
    failure_reason = payload.get("failure_reason")
    if failure_reason:
        return str(failure_reason)
    return fallback


def _live_model_status(db, symbol: str) -> str:
    try:
        rolling_acc = db.get_rolling_resolution_accuracy(symbol, days=60)
        return rolling_acc.get("model_status", "healthy")
    except Exception:
        return "healthy"


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
        rmse_vs_benchmark=best.get("rmse_vs_benchmark"),
        composite_score=best.get("composite_score"),
        stability_score=best.get("stability_score"),
    )


def _build_forecast_block(forecast_row: Dict[str, Any]) -> ForecastBlock:
    raw_points = forecast_row.get("points") or []
    points = [
        ForecastPoint(
            target_date=str(p.get("target_date", "")),
            horizon_index=int(p.get("horizon_index", 0)),
            bounded_predicted_close=p.get("bounded_predicted_close"),
            predicted_return=p.get("predicted_return"),
            # Sprint 4 A4.5: quantile / band alanlari (opsiyonel).
            p10_close=p.get("p10_close"),
            p50_close=p.get("p50_close"),
            p90_close=p.get("p90_close"),
            predicted_return_p10=p.get("predicted_return_p10"),
            predicted_return_p50=p.get("predicted_return_p50"),
            predicted_return_p90=p.get("predicted_return_p90"),
            lower_band=p.get("lower_band"),
            upper_band=p.get("upper_band"),
            price_tick=p.get("price_tick"),
        )
        for p in raw_points
    ]
    raw_agreement = forecast_row.get("ensemble_direction_agreement")
    return ForecastBlock(
        horizon_days=forecast_row.get("horizon_days"),
        trend_label=_normalize_trend_label(forecast_row.get("trend_label")),
        weekly_expected_return=forecast_row.get("weekly_expected_return"),
        trend_threshold=forecast_row.get("trend_threshold"),
        ensemble_agreement=float(raw_agreement) if raw_agreement is not None else None,
        points=points,
    )


def _build_forecast_source_block(forecast_row: Dict[str, Any]) -> ForecastSourceBlock:
    ensemble_metadata = _json_dict(forecast_row.get("ensemble_metadata_json"))
    warnings = _json_list(forecast_row.get("forecast_warnings_json"))
    is_ensemble = bool(ensemble_metadata) or str(forecast_row.get("model_name", "")).startswith("Ensemble ")
    return ForecastSourceBlock(
        type="ensemble" if is_ensemble else "model",
        model_name=forecast_row.get("model_name"),
        source_experiment_id=forecast_row.get("source_experiment_id"),
        run_at=forecast_row.get("run_at"),
        last_observed_date=forecast_row.get("last_observed_date"),
        method=None if not is_ensemble else str(ensemble_metadata.get("method") or ""),
        members=list(ensemble_metadata.get("members") or []),
        weights={str(k): float(v) for k, v in dict(ensemble_metadata.get("weights") or {}).items()},
        source_experiment_ids=[
            int(v) for v in list(ensemble_metadata.get("source_experiment_ids") or []) if v is not None
        ],
        forecast_strategy=forecast_row.get("forecast_strategy"),
        artifact_mode=forecast_row.get("artifact_mode"),
        warnings=warnings,
    )


def _json_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _json_list(raw: Any) -> list:
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


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
                feature_group=f.feature_group,
                reason=f.reason,
                method=f.method,
                contribution=f.contribution,
                approximate=f.approximate,
            )
            for f in summary.top_positive_reasons
        ],
        top_negative_reasons=[
            XaiFactorItem(
                feature_name=f.feature_name,
                human_label=f.human_label,
                importance=f.importance,
                direction=f.direction,
                feature_group=f.feature_group,
                reason=f.reason,
                method=f.method,
                contribution=f.contribution,
                approximate=f.approximate,
            )
            for f in summary.top_negative_reasons
        ],
        model_family_caveat=summary.model_family_caveat,
        caveat=summary.caveat,
    )


def _normalize_trend_label(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    mapping = {"up": "up", "down": "down", "flat": "flat", "neutral": "flat"}
    return mapping.get(raw, raw or None)


def _resolve_status(*, freshness: str, xai_available: bool, confidence_label: str) -> str:
    if freshness == "stale_data":
        return "stale_data"
    if not xai_available:
        return "xai_unavailable"
    if confidence_label == "low":
        return "low_confidence"
    return "ok"
