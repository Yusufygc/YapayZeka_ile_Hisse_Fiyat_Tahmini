"""Internal forecast workflows backing ``ForecastRunner``."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data.preprocessor import create_sequences, scale_data
from src.forecasting.bist_rules import RULES_VERSION
from src.pipeline.config import DataConfig, ValidationConfig
from src.pipeline.data_manager import DataManager


_TREE_MODELS = {"XGBoost", "Random Forest", "Ridge Return", "ElasticNet Return", "LightGBM Return"}
_SEQ_MODELS = {"LSTM", "LSTM Lite", "DLinear", "NLinear"}
_BASELINE_MODELS = {"Naive Last Value", "Naive Zero Return", "Naive Drift"}


class _OwnerBackedForecastService:
    def __init__(self, owner) -> None:
        self._owner = owner

    def __getattr__(self, name: str):
        return getattr(self._owner, name)


class ForecastSymbolWorkflow(_OwnerBackedForecastService):
    def __init__(self, owner, result_cls) -> None:
        super().__init__(owner)
        self._result_cls = result_cls

    def run(
        self,
        *,
        symbol: str,
        data_file: str,
        horizon_days: int = 5,
        force_model_name: str | None = None,
        use_macro: bool = True,
        auto_update_data: bool = True,
        auto_update_interactive: bool = False,
    ):
        symbol = symbol.upper()
        selection = self.model_resolver.resolve(symbol, force_model_name)
        data_manager = self.data_preparation_service.prepare(
            symbol=symbol,
            data_file=data_file,
            target_mode=selection["target_mode"],
            feature_mode=selection["feature_mode"],
            scaling_mode=selection["scaling_mode"],
            use_macro=use_macro,
            auto_update_data=auto_update_data,
            auto_update_interactive=auto_update_interactive,
        )
        model, forecast_context = self.production_training_workflow.train(selection["model_name"], data_manager)
        predicted_target = self.latest_target_prediction_workflow.predict(
            selection["model_name"], model, forecast_context
        )
        points = self.forecast_point_generator.roll_forward(
            predicted_target=predicted_target,
            horizon_days=horizon_days,
            last_close=forecast_context["last_close"],
            last_observed_date=forecast_context["last_observed_date"],
            target_mode=selection["target_mode"],
        )
        weekly_expected_return = (points[-1]["bounded_predicted_close"] / forecast_context["last_close"]) - 1.0
        trend_threshold = self.rules.trend_threshold(
            data_manager.df["Close"].tail(80).to_numpy(dtype=float),
            horizon_days=horizon_days,
        )
        trend_label = self.rules.trend_label(weekly_expected_return, trend_threshold)
        run_id = self.persistence.save_run(
            stock_symbol=symbol,
            model_name=selection["model_name"],
            source_experiment_id=selection["source_experiment_id"],
            last_observed_date=forecast_context["last_observed_date"].strftime("%Y-%m-%d"),
            last_close=forecast_context["last_close"],
            horizon_days=horizon_days,
            trend_label=trend_label,
            weekly_expected_return=weekly_expected_return,
            trend_threshold=trend_threshold,
            rules_version=RULES_VERSION,
            points=points,
        )
        return self._result_cls(
            run_id=run_id,
            stock_symbol=symbol,
            model_name=selection["model_name"],
            trend_label=trend_label,
            weekly_expected_return=weekly_expected_return,
            trend_threshold=trend_threshold,
            points=points,
        )


class BestModelResolver(_OwnerBackedForecastService):
    def resolve(self, symbol: str, force_model_name: str | None = None) -> Dict[str, Any]:
        best = self.db.get_best_model(symbol)
        if force_model_name:
            return {
                "model_name": force_model_name,
                "source_experiment_id": None if best is None else best.get("experiment_id"),
                "target_mode": "log_return" if best is None else best.get("target_mode", "log_return"),
                "feature_mode": self.model_config.model_settings.get("feature_mode", "stationary_features"),
                "scaling_mode": "robust_x_standard_y_clip" if best is None else best.get("scaling_mode", "robust_x_standard_y_clip"),
            }
        if not best:
            raise ValueError(f"{symbol} icin best_models kaydi yok. Once pipeline calistirin veya --model kullanin.")
        return self._resolve_best_or_replacement(symbol, best)

    def _resolve_best_or_replacement(self, symbol: str, best: Dict[str, Any]) -> Dict[str, Any]:
        model_name = str(best["model_name"])
        resolved = {
            "model_name": model_name,
            "source_experiment_id": best.get("experiment_id"),
            "target_mode": best.get("target_mode", "log_return"),
            "feature_mode": best.get("feature_mode", "stationary_features"),
            "scaling_mode": best.get("scaling_mode", "robust_x_standard_y_clip"),
        }
        if model_name.startswith("Ensemble ") or model_name in _BASELINE_MODELS:
            replacement = self._best_trainable_experiment(symbol)
            if replacement is None:
                raise ValueError(
                    f"{symbol} icin baseline/ensemble disinda train edilebilir model bulunamadi. "
                    "Once pipeline'i gercek model aileleriyle calistirin veya --model kullanin."
                )
            resolved.update({
                "model_name": str(replacement["model_name"]),
                "source_experiment_id": replacement.get("id"),
                "target_mode": replacement.get("target_mode", resolved["target_mode"]),
                "feature_mode": replacement.get("feature_mode", resolved["feature_mode"]),
                "scaling_mode": replacement.get("scaling_mode", resolved["scaling_mode"]),
            })
        return resolved

    def best_trainable_experiment(self, symbol: str) -> Optional[Dict[str, Any]]:
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


class ForecastDataPreparationService(_OwnerBackedForecastService):
    def prepare(
        self,
        *,
        symbol: str,
        data_file: str,
        target_mode: str,
        feature_mode: str,
        scaling_mode: str,
        use_macro: bool,
        auto_update_data: bool,
        auto_update_interactive: bool,
    ) -> DataManager:
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
        return data_manager


class ProductionTrainingWorkflow(_OwnerBackedForecastService):
    def train(self, model_name: str, data_manager: DataManager) -> tuple[Any, Dict[str, Any]]:
        frame = self._clean_training_frame(data_manager)
        close = frame["Close"].to_numpy(dtype=float)
        features = data_manager.feature_names
        X_all = frame[features].to_numpy(dtype=float)
        X_train = X_all[:-1]
        y_train = self._make_target(close, data_manager.data_cfg.target_mode).reshape(-1, 1)
        latest_X = X_all[-1:].copy()
        X_train_s, latest_X_s, y_train_s, scaler_X, scaler_y = self._scale_train_latest(
            X_train, y_train, latest_X, data_manager
        )
        model, latest_seq = self._fit_model(
            model_name, data_manager, frame, X_all, X_train, y_train, X_train_s, y_train_s, scaler_X
        )
        return model, {
            "features": features,
            "target_mode": data_manager.data_cfg.target_mode,
            "scaler_y": scaler_y,
            "latest_X": latest_X,
            "latest_X_s": latest_X_s,
            "latest_seq": latest_seq,
            "last_close": float(close[-1]),
            "last_observed_date": pd.to_datetime(frame["Date"].iloc[-1]).normalize(),
        }

    @staticmethod
    def _clean_training_frame(data_manager: DataManager) -> pd.DataFrame:
        frame = data_manager.df.copy()
        frame = frame.dropna(subset=["Date", "Close", *data_manager.feature_names]).copy()
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if len(frame) < data_manager.data_cfg.time_steps + 2:
            raise ValueError("Production forecast icin yeterli tarihsel veri yok.")
        return frame

    def _scale_train_latest(self, X_train, y_train, latest_X, data_manager):
        X_train_s, latest_X_s, y_train_s, _, scaler_X, scaler_y = scale_data(
            X_train,
            latest_X,
            y_train,
            np.zeros((1, 1), dtype=float),
            save_dir=os.path.join(self.project_root, "outputs", data_manager.stock_symbol, "forecast_models"),
            scaling_mode=data_manager.data_cfg.scaling_mode,
            save_scaler=False,
        )
        return X_train_s, latest_X_s, y_train_s, scaler_X, scaler_y

    def _fit_model(
        self, model_name, data_manager, frame, X_all, X_train, y_train, X_train_s, y_train_s, scaler_X
    ):
        if model_name == "Prophet":
            model = self._make_prophet(data_manager.feature_names)
            model.train(X_train, y_train, dates_train=frame["Date"].iloc[:-1])
            return model, None
        if model_name in _TREE_MODELS:
            model = self._make_model_instance(model_name)
            model.train(X_train_s, y_train_s)
            return model, None
        if model_name in _SEQ_MODELS:
            return self._fit_sequence_model(model_name, data_manager, X_all, X_train_s, y_train_s, scaler_X)
        model = self._make_model_instance(model_name)
        model.train(X_train, y_train, dates_train=frame["Date"].iloc[:-1])
        return model, None

    def _fit_sequence_model(self, model_name, data_manager, X_all, X_train_s, y_train_s, scaler_X):
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
        return model, latest_seq


class LatestTargetPredictionWorkflow(_OwnerBackedForecastService):
    def predict(self, model_name: str, model: Any, context: Dict[str, Any]) -> float:
        if model_name == "Prophet":
            raw = model.predict(context["latest_X"], dates_test=[context["last_observed_date"]])
            return float(np.asarray(raw).ravel()[-1])
        if model_name in _TREE_MODELS:
            scaled = np.asarray(model.predict(context["latest_X_s"])).reshape(-1, 1)
            return float(context["scaler_y"].inverse_transform(scaled).ravel()[-1])
        if model_name in _SEQ_MODELS:
            return self._predict_sequence_target(model_name, model, context)
        raw = model.predict(context["latest_X"])
        return float(np.asarray(raw).ravel()[-1])

    @staticmethod
    def _predict_sequence_target(model_name: str, model: Any, context: Dict[str, Any]) -> float:
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


class ForecastPointGenerator(_OwnerBackedForecastService):
    def roll_forward(
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
