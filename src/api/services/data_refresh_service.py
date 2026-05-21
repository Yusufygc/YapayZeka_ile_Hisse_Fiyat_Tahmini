# -*- coding: utf-8 -*-
"""Non-blocking data/forecast refresh orchestration for analysis serving."""
from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from src.api.observability import log_event
from src.data.data_updater import DataUpdater
from src.features.macro_pipeline import MacroPipeline
from src.forecasting.bist_calendar import default_calendar_path, ensure_bist_calendar


@dataclass(frozen=True)
class LatestMarketRow:
    date: str
    close: float


class DataRefreshError(RuntimeError):
    def __init__(self, reason: str, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.payload = payload or {}


_REFRESH_EXECUTOR = ThreadPoolExecutor(max_workers=4)


class DataRefreshService:
    """Queues and executes refresh work needed by the analysis endpoint."""

    def __init__(
        self,
        *,
        db,
        project_root: str,
        outputs_base: Optional[str] = None,
        horizon_days: int = 5,
        start_background: bool = True,
    ) -> None:
        self.db = db
        self.project_root = os.path.abspath(project_root)
        self.outputs_base = outputs_base or os.path.join(self.project_root, "outputs")
        self.horizon_days = int(horizon_days)
        self.start_background = bool(start_background)
        self.db_path = getattr(db, "db_path", None)

    def ensure_refresh_job(
        self,
        *,
        symbol: str,
        reason: str,
        best_model: Optional[Dict[str, Any]] = None,
        wait_timeout_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        if not re.match(r"^[A-Z0-9]{1,10}$", symbol.upper()):
            raise ValueError(f"Invalid symbol format: {symbol}")
        payload = {
            "best_model_name": None if best_model is None else best_model.get("model_name"),
            "best_experiment_id": None if best_model is None else best_model.get("experiment_id"),
            "project_root": self.project_root,
        }
        job = self.db.create_or_get_refresh_job(
            symbol=symbol,
            reason=reason,
            payload=payload,
        )
        if self.start_background and job.get("status") == "queued":
            _REFRESH_EXECUTOR.submit(self._run_job, dict(job), best_model)
        if wait_timeout_seconds > 0:
            return self.wait_for_job(
                str(job["job_id"]),
                timeout_seconds=float(wait_timeout_seconds),
            )
        return job

    def _run_job(self, job: Dict[str, Any], best_model: Optional[Dict[str, Any]]) -> None:
        job_id = str(job["job_id"])
        symbol = str(job["symbol"]).upper()
        reason = str(job.get("reason") or "")
        started = time.monotonic()
        worker_db = self._new_db()
        worker = DataRefreshService(
            db=worker_db,
            project_root=self.project_root,
            outputs_base=self.outputs_base,
            horizon_days=self.horizon_days,
            start_background=False,
        )
        try:
            worker_db.update_refresh_job(job_id, status="running")
            result = worker.refresh_symbol(symbol=symbol, best_model=best_model)
            worker_db.update_refresh_job(job_id, status="completed", payload=result, finish=True)
            log_event(
                self.project_root,
                "analysis_refresh_job",
                job_id=job_id,
                symbol=symbol,
                reason=reason,
                status="completed",
                duration_ms=int((time.monotonic() - started) * 1000),
                error=None,
            )
        except DataRefreshError as exc:
            payload = dict(exc.payload)
            payload.setdefault("failure_reason", exc.reason)
            worker_db.update_refresh_job(
                job_id,
                status="failed",
                error=str(exc),
                payload=payload,
                finish=True,
            )
            log_event(
                self.project_root,
                "analysis_refresh_job",
                job_id=job_id,
                symbol=symbol,
                reason=exc.reason,
                status="failed",
                duration_ms=int((time.monotonic() - started) * 1000),
                error=str(exc),
            )
        except Exception as exc:
            worker_db.update_refresh_job(
                job_id,
                status="failed",
                error=str(exc),
                payload={"failure_reason": "refresh_failed"},
                finish=True,
            )
            log_event(
                self.project_root,
                "analysis_refresh_job",
                job_id=job_id,
                symbol=symbol,
                reason=reason or "refresh_failed",
                status="failed",
                duration_ms=int((time.monotonic() - started) * 1000),
                error=str(exc),
            )

    def wait_for_job(self, job_id: str, *, timeout_seconds: float) -> Dict[str, Any]:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        latest = self.db.get_refresh_job(job_id) or {"job_id": job_id, "status": "queued"}
        while time.monotonic() < deadline:
            latest = self.db.get_refresh_job(job_id) or latest
            if str(latest.get("status")) in {"completed", "failed"}:
                return latest
            time.sleep(0.25)
        return self.db.get_refresh_job(job_id) or latest

    def refresh_symbol(self, *, symbol: str, best_model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data_file = self.data_file_for(symbol)
        calendar_path = default_calendar_path(self.project_root)
        try:
            ensure_bist_calendar(calendar_path, years_back=5, years_forward=1)
        except Exception as exc:
            raise DataRefreshError(
                "calendar_generation_failed",
                f"BIST calendar generation failed: {exc}",
                {"symbol": symbol, "calendar_path": calendar_path},
            ) from exc

        update_result = DataUpdater.check_and_update(data_file, symbol, interactive=False)
        update_payload = (
            update_result.to_dict()
            if hasattr(update_result, "to_dict")
            else {"status": "unknown"}
        )
        if update_payload.get("status") == "failed":
            raise DataRefreshError(
                "data_update_failed",
                str(update_payload.get("error") or "Data update failed"),
                {"symbol": symbol, "data_update": update_payload},
            )
        latest = read_latest_market_row(data_file)

        try:
            macro = MacroPipeline(cache_dir=os.path.join(self.project_root, "data", "macro"))
            macro.get_macro_features(start_date=_safe_start_date(data_file, latest.date), end_date=latest.date)
        except Exception as exc:
            raise DataRefreshError(
                "macro_refresh_failed",
                f"Macro refresh failed: {exc}",
                {"symbol": symbol, "data_update": update_payload, "latest_date": latest.date},
            ) from exc

        try:
            resolved = self.db.resolve_forecasts_from_csv(symbol, data_file)
        except Exception as exc:
            raise DataRefreshError(
                "forecast_resolution_failed",
                f"Forecast resolution failed: {exc}",
                {"symbol": symbol, "data_update": update_payload, "latest_date": latest.date},
            ) from exc

        forecast_run_id = None
        if best_model is not None:
            try:
                from src.forecasting.runner import ForecastRunner

                runner = ForecastRunner(
                    project_root=self.project_root,
                    db_path=self.db.db_path,
                    calendar_path=calendar_path,
                )
                forecast = runner.run_symbol(
                    symbol=symbol,
                    data_file=data_file,
                    horizon_days=self.horizon_days,
                    use_macro=True,
                    auto_update_data=False,
                    auto_update_interactive=False,
                )
                forecast_run_id = forecast.run_id
            except Exception as exc:
                raise DataRefreshError(
                    "forecast_generation_failed",
                    f"Forecast generation failed: {exc}",
                    {"symbol": symbol, "data_update": update_payload, "latest_date": latest.date},
                ) from exc

        return {
            "symbol": symbol,
            "latest_date": latest.date,
            "latest_close": latest.close,
            "data_update": update_payload,
            "resolved_forecast_points": resolved,
            "forecast_run_id": forecast_run_id,
        }

    def data_file_for(self, symbol: str) -> str:
        if not re.match(r"^[A-Z0-9]{1,10}$", symbol.upper()):
            raise ValueError(f"Invalid symbol format: {symbol}")
        return os.path.join(self.project_root, "data", f"{symbol.upper()}.csv")

    def _new_db(self):
        if not self.db_path:
            return self.db
        from src.database.stock_model_db import StockModelDB

        return StockModelDB(self.db_path)


def read_latest_market_row(csv_path: str) -> LatestMarketRow:
    df = pd.read_csv(csv_path)
    date_col = _find_column(df.columns, ["Date", "Tarih"])
    close_col = _find_column(df.columns, ["Close", "Kapanış", "Kapanis", "Adj Close", "Düzeltilmiş_Kapanış"])
    if date_col is None or close_col is None:
        raise ValueError(f"CSV must include date and close columns: {csv_path}")
    work = df[[date_col, close_col]].copy()
    work.columns = ["Date", "Close"]
    work["Date"] = _parse_dates(work["Date"])
    work["Close"] = pd.to_numeric(work["Close"], errors="coerce")
    work.dropna(subset=["Date", "Close"], inplace=True)
    if work.empty:
        raise ValueError(f"CSV has no valid date/close rows: {csv_path}")
    row = work.sort_values("Date").iloc[-1]
    return LatestMarketRow(
        date=pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
        close=float(row["Close"]),
    )


def _safe_start_date(csv_path: str, fallback: str) -> str:
    try:
        df = pd.read_csv(csv_path)
        date_col = _find_column(df.columns, ["Date", "Tarih"])
        if date_col is None:
            return fallback
        dates = _parse_dates(df[date_col]).dropna()
        if dates.empty:
            return fallback
        return pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
    except Exception:
        return fallback


def _find_column(columns, candidates: list[str]) -> Optional[str]:
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def _parse_dates(values) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(values, dayfirst=True, errors="coerce")
