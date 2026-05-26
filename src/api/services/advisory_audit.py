# -*- coding: utf-8 -*-
"""
Sprint 9 (2026-05-26) A9.1 — Advisory Audit Log.

Her `GET /analysis/{symbol}` yanitini disk uzerinde tutar (CSV append).
Amac: T+horizon sonrasi gercek realized return ile karsilastirip
calibration drift dashboard'una veri saglamak (backfill ayri job).

Parquet ideal (sutun tipi disiplin) ama optional `pyarrow` dependency
gerektirir. Pandas tarafindan default sira: parquet varsa parquet,
yoksa CSV append. Test edilebilirligi icin Parquet/CSV switch path-uzantili.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

_LOCK = threading.Lock()
_DEFAULT_LOG_PATH = os.path.join("data", "advisory_history.csv")


@dataclass(frozen=True)
class AdvisoryAuditRecord:
    timestamp_utc: str
    symbol: str
    horizon_days: Optional[int]
    model_name: Optional[str]
    trend_label: Optional[str]
    p50_return: Optional[float]
    p10_return: Optional[float]
    p90_return: Optional[float]
    confidence_label: Optional[str]
    analysis_status: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "symbol": self.symbol,
            "horizon_days": self.horizon_days,
            "model_name": self.model_name,
            "trend_label": self.trend_label,
            "p50_return": self.p50_return,
            "p10_return": self.p10_return,
            "p90_return": self.p90_return,
            "confidence_label": self.confidence_label,
            "analysis_status": self.analysis_status,
        }


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def build_record_from_response(response: Any) -> AdvisoryAuditRecord:
    """AnalysisResponse Pydantic objesinden audit kayit cikarir."""
    forecast = getattr(response, "forecast", None)
    confidence = getattr(response, "confidence", None)
    model = getattr(response, "model", None)
    horizon_days = getattr(forecast, "horizon_days", None) if forecast else None

    p50_ret = p10_ret = p90_ret = None
    if forecast and getattr(forecast, "points", None):
        # Sprint 4 quantile alanlari opsiyonel; ilk noktanin ortalamasi.
        p0 = forecast.points[0]
        p50_ret = getattr(p0, "predicted_return_p50", None) or getattr(p0, "predicted_return", None)
        p10_ret = getattr(p0, "predicted_return_p10", None)
        p90_ret = getattr(p0, "predicted_return_p90", None)

    return AdvisoryAuditRecord(
        timestamp_utc=_utc_now_iso(),
        symbol=getattr(response, "symbol", ""),
        horizon_days=horizon_days,
        model_name=getattr(model, "model_name", None) if model else None,
        trend_label=getattr(forecast, "trend_label", None) if forecast else None,
        p50_return=p50_ret,
        p10_return=p10_ret,
        p90_return=p90_ret,
        confidence_label=getattr(confidence, "label", None) if confidence else None,
        analysis_status=getattr(response, "analysis_status", None),
    )


def append_record(
    record: AdvisoryAuditRecord,
    *,
    log_path: Optional[str] = None,
) -> str:
    """
    Kayit ekle (CSV append). Dosya yoksa header ile yarat.

    Returns: yazilan dosyanin tam yolu.
    """
    path = log_path or _DEFAULT_LOG_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame([record.to_dict()])
    with _LOCK:
        header = not os.path.exists(path)
        df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")
    return path


def append_response(response: Any, *, log_path: Optional[str] = None) -> str:
    """Convenience: build_record_from_response + append_record."""
    return append_record(build_record_from_response(response), log_path=log_path)


def read_log(log_path: Optional[str] = None) -> pd.DataFrame:
    """Audit log'u DataFrame olarak doner. Dosya yoksa bos DataFrame."""
    path = log_path or _DEFAULT_LOG_PATH
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)
