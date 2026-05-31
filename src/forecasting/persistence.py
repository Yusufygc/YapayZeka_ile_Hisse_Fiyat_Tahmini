"""Forward forecast persistence facade'ı.

ForecastPersistence: StockModelDB üzerine ince bir kayıt katmanı; forecast run
ve nokta tahminlerini saklar.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.database.stock_model_db import StockModelDB


class ForecastPersistence:
    """Thin persistence facade for forward forecast artifacts."""

    def __init__(self, db: StockModelDB) -> None:
        self.db = db

    def save_run(
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
        ensemble_direction_agreement: Optional[float] = None,
        forecast_strategy: Optional[str] = None,
        artifact_mode: Optional[str] = None,
        forecast_warnings: Optional[List[str]] = None,
        ensemble_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self.db.log_forecast_run(
            stock_symbol=stock_symbol,
            model_name=model_name,
            source_experiment_id=source_experiment_id,
            last_observed_date=last_observed_date,
            last_close=last_close,
            horizon_days=horizon_days,
            trend_label=trend_label,
            weekly_expected_return=weekly_expected_return,
            trend_threshold=trend_threshold,
            rules_version=rules_version,
            points=points,
            status=status,
            ensemble_direction_agreement=ensemble_direction_agreement,
            forecast_strategy=forecast_strategy,
            artifact_mode=artifact_mode,
            forecast_warnings=forecast_warnings,
            ensemble_metadata=ensemble_metadata,
        )
