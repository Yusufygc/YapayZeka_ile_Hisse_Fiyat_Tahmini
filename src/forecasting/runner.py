"""Forward forecast orkestrasyonu (sembol bazlı).

Sorumluluklar:
  - En iyi üretim modelini çözer, veriyi hazırlar, recursive horizon tahmini
    üretir ve BIST kurallarıyla band-clip eder; sonucu persist eder.
  - Serve aşamasında YENİDEN EĞİTİM YOK — kayıtlı sidecar artifact kullanılır.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.database.stock_model_db import StockModelDB
from src.forecasting.bist_calendar import ensure_bist_calendar
from src.forecasting.artifacts import ModelNoLongerSupportedError
from src.forecasting.bist_rules import BistMarketRules
from src.forecasting.workflows import (
    BestModelResolver,
    ForecastContext,
    ForecastDataPreparationService,
    ForecastPointGenerator,
    ForecastSymbolWorkflow,
    LatestTargetPredictionWorkflow,
    ProductionTrainingWorkflow,
)
from src.pipeline.config import ModelConfig
from src.pipeline.model_scope import is_removed_model


@dataclass
class ForecastResult:
    run_id: int
    stock_symbol: str
    model_name: str
    trend_label: str
    weekly_expected_return: float
    trend_threshold: float
    points: list[dict[str, Any]]
    # h>1 tahmini 1-adim modelin recursion'idir ve dogrulanmamistir (bkz. workflows).
    recursive_forecast_validated: bool = False
    horizon_validation_note: Optional[str] = None


class ForecastRunner:
    """Load production artifacts and persist BIST-compliant forecasts."""

    def __init__(
        self,
        *,
        project_root: str,
        db_path: str | None = None,
        calendar_path: str | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.project_root = os.path.abspath(project_root)
        self.db = StockModelDB(db_path or os.path.join(self.project_root, "data", "stock_models.db"))
        resolved_calendar_path = calendar_path or os.path.join(self.project_root, "data", "meta", "bist_calendar.csv")
        ensure_bist_calendar(resolved_calendar_path, years_back=5, years_forward=1)
        self.rules = BistMarketRules(
            resolved_calendar_path
        )
        self.model_config = model_config or ModelConfig()
        self._init_workflows()

    def _init_workflows(self) -> None:
        # Faz 5 (E1 owner-forward epiği): owner-forward (`_OwnerBackedForecastService`)
        # yerine açık DI. ForecastContext READ-ONLY config + factory callable'ları
        # (model_config'e bağlı, runner'da kalır) + kardeş servis referanslarını taşır.
        ctx = ForecastContext(
            project_root=self.project_root,
            db=self.db,
            rules=self.rules,
            model_config=self.model_config,
            make_model_instance=self._make_model_instance,
            target_to_price=ForecastRunner._target_to_price,
        )
        self.context = ctx
        self.model_resolver = BestModelResolver(ctx)
        self.data_preparation_service = ForecastDataPreparationService(ctx)
        self.production_training_workflow = ProductionTrainingWorkflow(ctx)
        self.latest_target_prediction_workflow = LatestTargetPredictionWorkflow(ctx)
        self.forecast_point_generator = ForecastPointGenerator(ctx)
        self.symbol_workflow = ForecastSymbolWorkflow(ctx, ForecastResult)
        # Kardeş servis referanslarını context'e bağla (servisler birbirini ctx üzerinden çağırır).
        ctx.model_resolver = self.model_resolver
        ctx.data_preparation_service = self.data_preparation_service
        ctx.production_training_workflow = self.production_training_workflow
        ctx.latest_target_prediction_workflow = self.latest_target_prediction_workflow
        ctx.forecast_point_generator = self.forecast_point_generator

    def _ensure_workflows(self) -> None:
        if not all(
            hasattr(self, attr)
            for attr in (
                "model_resolver",
                "data_preparation_service",
                "production_training_workflow",
                "latest_target_prediction_workflow",
                "forecast_point_generator",
                "symbol_workflow",
            )
        ):
            self._init_workflows()

    def run_symbol(
        self,
        *,
        symbol: str,
        data_file: str,
        horizon_days: int = 5,
        force_model_name: str | None = None,
        use_macro: bool = True,
        auto_update_data: bool = True,
        auto_update_interactive: bool = False,
    ) -> ForecastResult:
        self._ensure_workflows()
        return self.symbol_workflow.run(
            symbol=symbol,
            data_file=data_file,
            horizon_days=horizon_days,
            force_model_name=force_model_name,
            use_macro=use_macro,
            auto_update_data=auto_update_data,
            auto_update_interactive=auto_update_interactive,
        )

    @staticmethod
    def _target_to_price(target: float, previous_close: float, target_mode: str) -> float:
        if target_mode == "log_return":
            return float(previous_close * np.exp(np.clip(target, -1.0, 1.0)))
        if target_mode == "return":
            return float(previous_close * (1.0 + np.clip(target, -0.95, 5.0)))
        if target_mode == "price":
            return float(target)
        raise ValueError(f"Desteklenmeyen target_mode: {target_mode}")

    def _make_model_instance(self, model_name: str, *, stage: str = "single"):
        if model_name == "Naive Last Value":
            from src.models.naive_model import NaiveLastValueModel
            return NaiveLastValueModel()
        if model_name == "Naive Zero Return":
            from src.models.naive_model import NaiveZeroReturnModel
            return NaiveZeroReturnModel()
        if model_name == "Naive Drift":
            from src.models.naive_model import NaiveDriftModel
            return NaiveDriftModel()
        if model_name == "Ridge Return":
            from src.models.linear_model import RidgeReturnModel
            return RidgeReturnModel()
        if model_name == "LightGBM Return":
            from src.models.gradient_boosting_model import LightGBMReturnModel
            return LightGBMReturnModel()
        if model_name == "XGBoost":
            from src.models.xgboost_model import XGBoostModel
            return XGBoostModel()
        if is_removed_model(model_name):
            raise ModelNoLongerSupportedError(model_name)
        raise KeyError(f"Bilinmeyen model adi: {model_name}")
