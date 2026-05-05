from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data.preprocessor import create_sequences, scale_data
from src.database.stock_model_db import StockModelDB
from src.forecasting.bist_rules import BistMarketRules, RULES_VERSION
from src.forecasting.persistence import ForecastPersistence
from src.pipeline.config import DataConfig, ModelConfig, ValidationConfig
from src.pipeline.data_manager import DataManager

_TREE_MODELS = {"XGBoost", "Random Forest", "Ridge Return", "ElasticNet Return", "LightGBM Return"}
_SEQ_MODELS = {"LSTM", "TFT", "DLinear", "NLinear"}
_BASELINE_MODELS = {"Naive Last Value", "Naive Zero Return", "Naive Drift"}


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
    """Train the selected production model and persist BIST-compliant forecasts."""

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
        self.rules = BistMarketRules(
            calendar_path or os.path.join(self.project_root, "data", "meta", "bist_calendar.csv")
        )
        self.model_config = model_config or ModelConfig()
        self.persistence = ForecastPersistence(self.db)

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
        symbol = symbol.upper()
        best = self.db.get_best_model(symbol)
        if force_model_name:
            model_name = force_model_name
            source_experiment_id = None if best is None else best.get("experiment_id")
            target_mode = "log_return" if best is None else best.get("target_mode", "log_return")
            feature_mode = self.model_config.model_settings.get("feature_mode", "stationary_features")
            scaling_mode = "robust_x_standard_y_clip" if best is None else best.get("scaling_mode", "robust_x_standard_y_clip")
        elif best:
            model_name = str(best["model_name"])
            source_experiment_id = best.get("experiment_id")
            target_mode = best.get("target_mode", "log_return")
            feature_mode = best.get("feature_mode", "stationary_features")
            scaling_mode = best.get("scaling_mode", "robust_x_standard_y_clip")
            if model_name.startswith("Ensemble ") or model_name in _BASELINE_MODELS:
                replacement = self._best_trainable_experiment(symbol)
                if replacement is None:
                    raise ValueError(
                        f"{symbol} icin baseline/ensemble disinda train edilebilir model bulunamadi. "
                        "Once pipeline'i gercek model aileleriyle calistirin veya --model kullanin."
                    )
                model_name = str(replacement["model_name"])
                source_experiment_id = replacement.get("id")
                target_mode = replacement.get("target_mode", target_mode)
                feature_mode = replacement.get("feature_mode", feature_mode)
                scaling_mode = replacement.get("scaling_mode", scaling_mode)
        else:
            raise ValueError(f"{symbol} icin best_models kaydi yok. Once pipeline calistirin veya --model kullanin.")

        data_cfg = DataConfig(
            data_file=data_file,
            target_mode=target_mode,
            feature_mode=feature_mode,
            scaling_mode=scaling_mode,
            use_macro=use_macro,
            training_window_years=None,
            auto_update_data=auto_update_data,
            auto_update_interactive=auto_update_interactive,
        )
        data_manager = DataManager(
            data_cfg=data_cfg,
            val_cfg=ValidationConfig(validation_mode="single_split"),
            models_dir=os.path.join(self.project_root, "outputs", symbol, "forecast_models"),
            macro_cache_dir=os.path.join(self.project_root, "data", "macro"),
        )
        data_manager.ingest_and_engineer()
        if data_manager.df is None or data_manager.df.empty:
            raise ValueError(f"{symbol} icin forecast uretecek veri yok.")

        model, forecast_context = self._train_production_model(model_name, data_manager)
        predicted_target = self._predict_latest_target(model_name, model, forecast_context)
        points = self._roll_forward_points(
            predicted_target=predicted_target,
            horizon_days=horizon_days,
            last_close=forecast_context["last_close"],
            last_observed_date=forecast_context["last_observed_date"],
            target_mode=target_mode,
        )
        weekly_expected_return = (points[-1]["bounded_predicted_close"] / forecast_context["last_close"]) - 1.0
        trend_threshold = self.rules.trend_threshold(
            data_manager.df["Close"].tail(80).to_numpy(dtype=float),
            horizon_days=horizon_days,
        )
        trend_label = self.rules.trend_label(weekly_expected_return, trend_threshold)
        run_id = self.persistence.save_run(
            stock_symbol=symbol,
            model_name=model_name,
            source_experiment_id=source_experiment_id,
            last_observed_date=forecast_context["last_observed_date"].strftime("%Y-%m-%d"),
            last_close=forecast_context["last_close"],
            horizon_days=horizon_days,
            trend_label=trend_label,
            weekly_expected_return=weekly_expected_return,
            trend_threshold=trend_threshold,
            rules_version=RULES_VERSION,
            points=points,
        )
        return ForecastResult(
            run_id=run_id,
            stock_symbol=symbol,
            model_name=model_name,
            trend_label=trend_label,
            weekly_expected_return=weekly_expected_return,
            trend_threshold=trend_threshold,
            points=points,
        )

    def _train_production_model(self, model_name: str, data_manager: DataManager) -> tuple[Any, Dict[str, Any]]:
        frame = data_manager.df.copy()
        features = data_manager.feature_names
        frame = frame.dropna(subset=["Date", "Close", *features]).copy()
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if len(frame) < data_manager.data_cfg.time_steps + 2:
            raise ValueError("Production forecast icin yeterli tarihsel veri yok.")

        close = frame["Close"].to_numpy(dtype=float)
        X_all = frame[features].to_numpy(dtype=float)
        X_train = X_all[:-1]
        y_train = self._make_target(close, data_manager.data_cfg.target_mode).reshape(-1, 1)
        latest_X = X_all[-1:].copy()
        dummy_y = np.zeros((1, 1), dtype=float)
        X_train_s, latest_X_s, y_train_s, _, scaler_X, scaler_y = scale_data(
            X_train,
            latest_X,
            y_train,
            dummy_y,
            save_dir=os.path.join(self.project_root, "outputs", data_manager.stock_symbol, "forecast_models"),
            scaling_mode=data_manager.data_cfg.scaling_mode,
            save_scaler=False,
        )

        if model_name == "Prophet":
            model = self._make_prophet(features)
            model.train(X_train, y_train, dates_train=frame["Date"].iloc[:-1])
        elif model_name in _TREE_MODELS:
            model = self._make_model_instance(model_name)
            model.train(X_train_s, y_train_s)
        elif model_name in _SEQ_MODELS:
            X_all_s = scaler_X.transform(X_all)
            clip_report = getattr(scaler_X, "clip_report_", {}) or {}
            if clip_report.get("clip_low") is not None and clip_report.get("clip_high") is not None:
                X_all_s = np.clip(X_all_s, clip_report["clip_low"], clip_report["clip_high"])
            X_train_seq, y_train_seq = create_sequences(
                X_train_s,
                y_train_s,
                time_steps=data_manager.data_cfg.time_steps,
            )
            if len(X_train_seq) == 0:
                raise ValueError(f"{model_name} icin sequence sayisi yetersiz.")
            model = self._make_model_instance(model_name, stage="final")
            model.train(X_train_seq, y_train_seq)
            latest_seq = X_all_s[-data_manager.data_cfg.time_steps:].reshape(
                1,
                data_manager.data_cfg.time_steps,
                X_all_s.shape[1],
            )
        else:
            model = self._make_model_instance(model_name)
            model.train(X_train, y_train, dates_train=frame["Date"].iloc[:-1])

        context = {
            "features": features,
            "target_mode": data_manager.data_cfg.target_mode,
            "scaler_y": scaler_y,
            "latest_X": latest_X,
            "latest_X_s": latest_X_s,
            "latest_seq": locals().get("latest_seq"),
            "last_close": float(close[-1]),
            "last_observed_date": pd.to_datetime(frame["Date"].iloc[-1]).normalize(),
        }
        return model, context

    def _predict_latest_target(self, model_name: str, model: Any, context: Dict[str, Any]) -> float:
        if model_name == "Prophet":
            raw = model.predict(context["latest_X"], dates_test=[context["last_observed_date"]])
            return float(np.asarray(raw).ravel()[-1])
        if model_name in _TREE_MODELS:
            scaled = np.asarray(model.predict(context["latest_X_s"])).reshape(-1, 1)
            return float(context["scaler_y"].inverse_transform(scaled).ravel()[-1])
        if model_name in _SEQ_MODELS:
            latest_seq = context.get("latest_seq")
            if latest_seq is None:
                raise ValueError(f"{model_name} icin latest sequence yok.")
            if hasattr(model, "predict_quantiles"):
                scaled = np.asarray(model.predict_quantiles(latest_seq))
                p50_idx = scaled.shape[1] // 2 if scaled.ndim == 2 else 0
                scaled_target = scaled[:, p50_idx].reshape(-1, 1)
            else:
                scaled_target = np.asarray(model.predict(latest_seq)).reshape(-1, 1)
            return float(context["scaler_y"].inverse_transform(scaled_target).ravel()[-1])
        raw = model.predict(context["latest_X"])
        return float(np.asarray(raw).ravel()[-1])

    def _roll_forward_points(
        self,
        *,
        predicted_target: float,
        horizon_days: int,
        last_close: float,
        last_observed_date: pd.Timestamp,
        target_mode: str,
    ) -> list[dict[str, Any]]:
        dates = self.rules.next_trading_days(last_observed_date, horizon_days)
        points: list[dict[str, Any]] = []
        previous_close = float(last_close)
        for idx, target_date in enumerate(dates, start=1):
            raw_close = self._target_to_price(predicted_target, previous_close, target_mode)
            bounded_close, band = self.rules.bound_forecast_price(raw_close, previous_close)
            predicted_return = (bounded_close / previous_close) - 1.0
            points.append({
                "target_date": target_date.strftime("%Y-%m-%d"),
                "horizon_index": idx,
                "raw_predicted_close": raw_close,
                "bounded_predicted_close": bounded_close,
                "predicted_return": predicted_return,
                "lower_band": band.lower_band,
                "upper_band": band.upper_band,
                "price_tick": band.price_tick,
            })
            previous_close = bounded_close
        return points

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
        rows = self.db.get_experiments(stock_symbol=symbol, limit=500)
        candidates = [
            row for row in rows
            if (
                not str(row.get("model_name", "")).startswith("Ensemble ")
                and str(row.get("model_name", "")) not in _BASELINE_MODELS
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: float(row.get("composite_score") or float("-inf")))

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
        if model_name == "TFT":
            from src.models.tft_v2 import TFTModel
            cfg = self._deep_stage_config("tft", stage)
            return TFTModel(
                epochs=int(cfg.get("epochs", 80)),
                patience=int(cfg.get("patience", 15)),
                dropout=float(cfg.get("dropout", 0.3)),
                batch_size=int(cfg.get("batch_size", 32)),
                lr_patience=int(cfg.get("lr_patience", 5)),
                validation_ratio=float(cfg.get("validation_ratio", 0.1)),
                min_val_samples=int(cfg.get("min_validation_samples", 32)),
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
