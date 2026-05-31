"""Forward forecast orkestrasyonu (sembol bazlı).

Sorumluluklar:
  - En iyi üretim modelini çözer, veriyi hazırlar, recursive horizon tahmini
    üretir ve BIST kurallarıyla band-clip eder; sonucu persist eder.
  - Serve aşamasında YENİDEN EĞİTİM YOK — kayıtlı sidecar artifact kullanılır.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.database.stock_model_db import StockModelDB
from src.forecasting.bist_calendar import ensure_bist_calendar
from src.forecasting.bist_rules import BistMarketRules
from src.forecasting.persistence import ForecastPersistence
from src.forecasting.workflows import (
    BestModelResolver,
    ForecastDataPreparationService,
    ForecastPointGenerator,
    ForecastSymbolWorkflow,
    LatestTargetPredictionWorkflow,
    ProductionTrainingWorkflow,
)
from src.pipeline.config import ModelConfig
from src.pipeline.data_manager import DataManager


@dataclass
class ForecastResult:
    run_id: int
    stock_symbol: str
    model_name: str
    trend_label: str
    weekly_expected_return: float
    trend_threshold: float
    points: list[dict[str, Any]]


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
        self.persistence = ForecastPersistence(self.db)
        self._init_workflows()

    def _init_workflows(self) -> None:
        self.model_resolver = BestModelResolver(self)
        self.data_preparation_service = ForecastDataPreparationService(self)
        self.production_training_workflow = ProductionTrainingWorkflow(self)
        self.latest_target_prediction_workflow = LatestTargetPredictionWorkflow(self)
        self.forecast_point_generator = ForecastPointGenerator(self)
        self.symbol_workflow = ForecastSymbolWorkflow(self, ForecastResult)

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

    def _train_production_model(self, model_name: str, data_manager: DataManager) -> tuple[Any, Dict[str, Any]]:
        self._ensure_workflows()
        return self.production_training_workflow.train(model_name, data_manager)

    def _predict_latest_target(self, model_name: str, model: Any, context: Dict[str, Any]) -> float:
        self._ensure_workflows()
        return self.latest_target_prediction_workflow.predict(model_name, model, context)

    def _roll_forward_points(
        self,
        *,
        predicted_target: float,
        horizon_days: int,
        last_close: float,
        last_observed_date: pd.Timestamp,
        target_mode: str,
    ) -> list[dict[str, Any]]:
        self._ensure_workflows()
        return self.forecast_point_generator.roll_forward(
            predicted_target=predicted_target,
            horizon_days=horizon_days,
            last_close=last_close,
            last_observed_date=last_observed_date,
            target_mode=target_mode,
        )

    @staticmethod
    def _make_target(close: np.ndarray, target_mode: str) -> np.ndarray:
        prev = close[:-1]
        nxt = close[1:]
        if target_mode == "log_return":
            return np.log(nxt / prev)
        if target_mode == "return":
            return (nxt / prev) - 1.0
        if target_mode == "price":
            return nxt
        raise ValueError(f"Desteklenmeyen target_mode: {target_mode}")

    @staticmethod
    def _target_to_price(target: float, previous_close: float, target_mode: str) -> float:
        if target_mode == "log_return":
            return float(previous_close * np.exp(np.clip(target, -1.0, 1.0)))
        if target_mode == "return":
            return float(previous_close * (1.0 + np.clip(target, -0.95, 5.0)))
        if target_mode == "price":
            return float(target)
        raise ValueError(f"Desteklenmeyen target_mode: {target_mode}")

    def _make_prophet(self, feature_names: list[str]):
        from src.models.prophet_model import ProphetModel

        cfg = self.model_config.model_settings.get("prophet", {})
        return ProphetModel(
            yearly_seasonality=True,
            weekly_seasonality=True,
            use_regressors=bool(cfg.get("use_regressors", False)),
            regressor_names=cfg.get("regressor_names"),
            feature_names=feature_names,
        )

    def _best_trainable_experiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._ensure_workflows()
        return self.model_resolver.best_trainable_experiment(symbol)

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
        if model_name == "ARIMA":
            from src.models.arima_model import ARIMAModel
            cfg = self.model_config.model_settings.get("arima", {})
            return ARIMAModel(
                order=tuple(cfg.get("order", (1, 0, 0))),
                auto_order=bool(cfg.get("auto_order", False)),
                candidate_orders=[tuple(order) for order in cfg.get("candidate_orders", [])] or None,
            )
        if model_name == "Ridge Return":
            from src.models.linear_model import RidgeReturnModel
            return RidgeReturnModel()
        if model_name == "ElasticNet Return":
            from src.models.linear_model import ElasticNetReturnModel
            return ElasticNetReturnModel()
        if model_name == "LightGBM Return":
            from src.models.gradient_boosting_model import LightGBMReturnModel
            return LightGBMReturnModel()
        if model_name == "XGBoost":
            from src.models.xgboost_model import XGBoostModel
            return XGBoostModel()
        if model_name == "Random Forest":
            from src.models.random_forest_model import RandomForestModel
            return RandomForestModel()
        if model_name == "DLinear":
            from src.models.linear_sequence_model import DLinearSequenceModel
            return DLinearSequenceModel()
        if model_name == "NLinear":
            from src.models.linear_sequence_model import NLinearSequenceModel
            return NLinearSequenceModel()
        if model_name == "LSTM":
            from src.models.lstm_model import AttentionLSTMModel
            cfg = self._deep_stage_config("lstm", stage)
            return AttentionLSTMModel(
                epochs=int(cfg.get("epochs", 80)),
                patience=int(cfg.get("patience", 15)),
                dropout_rate=float(cfg.get("dropout", 0.2)),
                batch_size=int(cfg.get("batch_size", 32)),
                lr_patience=int(cfg.get("lr_patience", 5)),
                validation_ratio=float(cfg.get("validation_ratio", 0.1)),
                min_val_samples=int(cfg.get("min_validation_samples", 32)),
            )
        if model_name == "LSTM Lite":
            from src.models.lstm_lite_model import LSTMLiteModel
            cfg = self._deep_stage_config("lstm_lite", stage)
            return LSTMLiteModel(
                units=int(cfg.get("units", 32)),
                dense_units=int(cfg.get("dense_units", 16)),
                epochs=int(cfg.get("epochs", 80)),
                patience=int(cfg.get("patience", 12)),
                dropout_rate=float(cfg.get("dropout", 0.25)),
                batch_size=int(cfg.get("batch_size", 32)),
                learning_rate=float(cfg.get("learning_rate", 0.0003)),
                lr_patience=int(cfg.get("lr_patience", 4)),
                validation_ratio=float(cfg.get("validation_ratio", 0.1)),
                min_val_samples=int(cfg.get("min_validation_samples", 32)),
                tune_on_fit=bool(cfg.get("tune_on_fit", False)),
                tune_n_trials=int(cfg.get("tune_n_trials", 12)),
            )
        if model_name == "AttentionLSTM v2":
            from src.models.attention_lstm_v2_model import AttentionLSTMV2Model
            cfg = self._deep_stage_config("attention_lstm_v2", stage)
            return AttentionLSTMV2Model(
                units_1=int(cfg.get("units_1", 64)),
                units_2=int(cfg.get("units_2", 32)),
                dense_units=int(cfg.get("dense_units", 32)),
                epochs=int(cfg.get("epochs", 80)),
                patience=int(cfg.get("patience", 12)),
                dropout_rate=float(cfg.get("dropout", 0.30)),
                batch_size=int(cfg.get("batch_size", 32)),
                learning_rate=float(cfg.get("learning_rate", 0.0005)),
                lr_patience=int(cfg.get("lr_patience", 4)),
                validation_ratio=float(cfg.get("validation_ratio", 0.1)),
                min_val_samples=int(cfg.get("min_validation_samples", 32)),
                tune_on_fit=bool(cfg.get("tune_on_fit", False)),
                tune_n_trials=int(cfg.get("tune_n_trials", 12)),
                loss=str(cfg.get("loss", "huber")),
            )
        raise KeyError(f"Bilinmeyen model adi: {model_name}")

    def _deep_stage_config(self, section: str, stage: str) -> dict[str, Any]:
        deep = self.model_config.model_settings.get("deep_learning", {})
        cfg = dict(deep.get(section, {}))
        stage_key = f"epochs_{stage}"
        patience_stage_key = f"patience_{stage}"
        cfg["epochs"] = cfg.get(stage_key, cfg.get("epochs_single", cfg.get("epochs", 80)))
        cfg["patience"] = cfg.get(patience_stage_key, cfg.get("patience", 15))
        cfg["validation_ratio"] = deep.get("validation_ratio", 0.1)
        cfg["min_validation_samples"] = deep.get("min_validation_samples", 32)
        return cfg
