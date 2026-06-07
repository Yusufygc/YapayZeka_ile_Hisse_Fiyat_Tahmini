# -*- coding: utf-8 -*-
"""Forward forecast çalıştırma/nokta kayıt deposu (SQLite).

Sorumluluklar:
  - ForecastRepository: forecast run ve nokta tahminlerini idempotent (run_key)
    şekilde kaydeder. Zaman damgaları UTC (timezone-aware) tutulur.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.database.repositories.helpers import _optional_float


class ForecastRepository:
    def __init__(self, db) -> None:
        self.db = db

    def log_forecast_run(
        self,
        *,
        stock_symbol: str,
        model_name: str,
        source_experiment_id: Optional[int],
        last_observed_date: str,
        last_close: float,
        horizon_days: int,
        trend_label: str,
        weekly_expected_return: float,
        trend_threshold: float,
        rules_version: str,
        points: List[Dict[str, Any]],
        status: str = "pending",
        run_at: Optional[str] = None,
        ensemble_direction_agreement: Optional[float] = None,
        forecast_strategy: Optional[str] = None,
        artifact_mode: Optional[str] = None,
        forecast_warnings: Optional[List[str]] = None,
        ensemble_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Forecast çalıştırmasını ve nokta tahminlerini idempotent kaydeder.

        `run_key` (sembol+model+kaynak+gözlem tarihi+horizon+kurallar) ile
        tekrarlı çağrılar aynı run'a düşer. `run_at` verilmezse UTC zaman damgası
        kullanılır (Sprint 9 UTC mandate).

        Returns:
            forecast_runs kaydının id'si.
        """
        stock_symbol = stock_symbol.upper()
        # Sprint 9 UTC mandate: timezone-aware (api katmaniyla tutarli).
        run_at = run_at or datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        run_key = self.db._forecast_run_key(
            stock_symbol, model_name, source_experiment_id,
            last_observed_date, horizon_days, rules_version,
        )
        with self.db._connect() as conn:
            self._upsert_forecast_run(
                conn, run_key, stock_symbol, model_name, source_experiment_id,
                run_at, last_observed_date, last_close, horizon_days,
                trend_label, weekly_expected_return, trend_threshold, rules_version, status,
                ensemble_direction_agreement=ensemble_direction_agreement,
                forecast_strategy=forecast_strategy,
                artifact_mode=artifact_mode,
                forecast_warnings=forecast_warnings,
                ensemble_metadata=ensemble_metadata,
            )
            run_id = int(conn.execute(
                "SELECT id FROM forecast_runs WHERE run_key = ?",
                (run_key,),
            ).fetchone()["id"])
            for point in points:
                self._upsert_forecast_point(conn, run_id, point)
        return run_id

    @staticmethod
    def _upsert_forecast_run(
        conn, run_key, stock_symbol, model_name, source_experiment_id, run_at,
        last_observed_date, last_close, horizon_days, trend_label,
        weekly_expected_return, trend_threshold, rules_version, status,
        ensemble_direction_agreement=None,
        forecast_strategy=None,
        artifact_mode=None,
        forecast_warnings=None,
        ensemble_metadata=None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO forecast_runs
                (run_key, stock_symbol, model_name, source_experiment_id,
                 run_at, last_observed_date, last_close, horizon_days,
                 trend_label, weekly_expected_return, trend_threshold,
                 rules_version, status, ensemble_direction_agreement,
                 forecast_strategy, artifact_mode, forecast_warnings_json,
                 ensemble_metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_key) DO UPDATE SET
                run_at                       = excluded.run_at,
                last_close                   = excluded.last_close,
                trend_label                  = excluded.trend_label,
                weekly_expected_return       = excluded.weekly_expected_return,
                trend_threshold              = excluded.trend_threshold,
                status                       = excluded.status,
                ensemble_direction_agreement = excluded.ensemble_direction_agreement,
                forecast_strategy            = excluded.forecast_strategy,
                artifact_mode                 = excluded.artifact_mode,
                forecast_warnings_json       = excluded.forecast_warnings_json,
                ensemble_metadata_json       = excluded.ensemble_metadata_json
            """,
            (
                run_key, stock_symbol, model_name, source_experiment_id, run_at,
                last_observed_date, float(last_close), int(horizon_days),
                trend_label, float(weekly_expected_return), float(trend_threshold),
                rules_version, status,
                float(ensemble_direction_agreement) if ensemble_direction_agreement is not None else None,
                forecast_strategy,
                artifact_mode,
                json.dumps(forecast_warnings or [], ensure_ascii=False),
                json.dumps(ensemble_metadata, ensure_ascii=False) if ensemble_metadata else None,
            ),
        )

    def _upsert_forecast_point(self, conn, run_id: int, point: Dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO forecast_points
                (run_id, target_date, horizon_index, raw_predicted_close,
                 bounded_predicted_close, predicted_return, lower_band,
                 upper_band, price_tick,
                 p10_close, p50_close, p90_close,
                 predicted_return_p10, predicted_return_p50, predicted_return_p90,
                 interval_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, target_date) DO UPDATE SET
                horizon_index           = excluded.horizon_index,
                raw_predicted_close     = excluded.raw_predicted_close,
                bounded_predicted_close = excluded.bounded_predicted_close,
                predicted_return        = excluded.predicted_return,
                lower_band              = excluded.lower_band,
                upper_band              = excluded.upper_band,
                price_tick              = excluded.price_tick,
                p10_close               = excluded.p10_close,
                p50_close               = excluded.p50_close,
                p90_close               = excluded.p90_close,
                predicted_return_p10    = excluded.predicted_return_p10,
                predicted_return_p50    = excluded.predicted_return_p50,
                predicted_return_p90    = excluded.predicted_return_p90,
                interval_method         = excluded.interval_method,
                actual_close            = NULL,
                actual_return           = NULL,
                abs_error               = NULL,
                direction_correct       = NULL,
                resolved_at             = NULL
            """,
            (
                run_id,
                str(point["target_date"])[:10],
                int(point["horizon_index"]),
                self.db._optional_float(point.get("raw_predicted_close")),
                self.db._optional_float(point.get("bounded_predicted_close")),
                self.db._optional_float(point.get("predicted_return")),
                self.db._optional_float(point.get("lower_band")),
                self.db._optional_float(point.get("upper_band")),
                self.db._optional_float(point.get("price_tick")),
                self.db._optional_float(point.get("p10_close")),
                self.db._optional_float(point.get("p50_close")),
                self.db._optional_float(point.get("p90_close")),
                self.db._optional_float(point.get("predicted_return_p10")),
                self.db._optional_float(point.get("predicted_return_p50")),
                self.db._optional_float(point.get("predicted_return_p90")),
                (str(point["interval_method"]) if point.get("interval_method") else None),
            ),
        )

    def get_latest_forecast(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        """En son forecast çalıştırmasını (noktalarıyla) döner; yoksa None."""
        rows = self.get_forecast_history(stock_symbol, limit=1)
        return rows[0] if rows else None

    def get_forecast_history(self, stock_symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Sembolün son `limit` forecast çalıştırmasını (her biri noktalarıyla) döner."""
        stock_symbol = stock_symbol.upper()
        with self.db._connect() as conn:
            runs = conn.execute(
                """
                SELECT *
                FROM forecast_runs
                WHERE stock_symbol = ?
                ORDER BY run_at DESC, id DESC
                LIMIT ?
                """,
                (stock_symbol, int(limit)),
            ).fetchall()
            return [self._run_with_points(conn, dict(run)) for run in runs]

    @staticmethod
    def _run_with_points(conn, run_dict: Dict[str, Any]) -> Dict[str, Any]:
        points = conn.execute(
            """
            SELECT *
            FROM forecast_points
            WHERE run_id = ?
            ORDER BY horizon_index ASC
            """,
            (run_dict["id"],),
        ).fetchall()
        summary = conn.execute(
            "SELECT * FROM forecast_accuracy_summary WHERE run_id = ?",
            (run_dict["id"],),
        ).fetchone()
        run_dict["points"] = [dict(point) for point in points]
        run_dict["accuracy_summary"] = dict(summary) if summary else None
        return run_dict
