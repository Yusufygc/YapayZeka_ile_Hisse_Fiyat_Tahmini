"""Internal forecast workflows backing ``ForecastRunner``."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data.preprocessor import create_sequences, scale_data
from src.forecasting.artifacts import ForecastArtifactError, load_forecast_artifact_package
from src.forecasting.bist_rules import RULES_VERSION
from src.pipeline.config import DataConfig, ValidationConfig
from src.pipeline.data_manager import DataManager

_TREE_MODELS = {"XGBoost", "Random Forest", "Ridge Return", "ElasticNet Return", "LightGBM Return"}
_SEQ_MODELS = {"LSTM", "LSTM Lite", "AttentionLSTM v2", "DLinear", "NLinear"}
_BASELINE_MODELS = {"Naive Last Value", "Naive Zero Return", "Naive Drift"}
_DATE_AWARE_MODELS = {"Prophet", "Prophet-ML/DL Hybrid"}


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
        if selection["model_name"].startswith("Ensemble "):
            return self._run_ensemble(
                symbol=symbol,
                selection=selection,
                data_manager=data_manager,
                horizon_days=horizon_days,
            )
        model, forecast_context = self.production_training_workflow.train(
            selection["model_name"],
            data_manager,
            selection=selection,
        )
        points = self.forecast_point_generator.roll_forward_recursive(
            model_name=selection["model_name"],
            model=model,
            context=forecast_context,
            predictor=self.latest_target_prediction_workflow,
            horizon_days=horizon_days,
        )
        weekly_expected_return = (
            points[-1]["bounded_predicted_close"] / forecast_context["last_close"]
        ) - 1.0
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
            forecast_strategy=forecast_context.get("forecast_strategy"),
            artifact_mode=forecast_context.get("artifact_mode"),
            forecast_warnings=forecast_context.get("forecast_warnings"),
            ensemble_metadata=selection.get("ensemble_metadata"),
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

    def _run_ensemble(
        self,
        *,
        symbol: str,
        selection: Dict[str, Any],
        data_manager: DataManager,
        horizon_days: int,
    ):
        metadata = selection.get("ensemble_metadata") or {}
        members = list(metadata.get("members") or [])
        weights = dict(metadata.get("weights") or {})
        method = str(metadata.get("method") or selection["model_name"].replace("Ensemble ", ""))
        if len(members) < 2:
            raise ForecastArtifactError(
                f"{selection['model_name']} icin ensemble member metadata eksik."
            )
        member_points: Dict[str, list[dict[str, Any]]] = {}
        source_ids: list[int] = []
        for member in members:
            member_exp = self.model_resolver.latest_member_experiment(symbol, member)
            if member_exp is None:
                raise ForecastArtifactError(
                    f"Ensemble member artifact experiment bulunamadi: {member}"
                )
            source_ids.append(int(member_exp["id"]))
            member_selection = {
                "model_name": member,
                "source_experiment_id": member_exp.get("id"),
                "target_mode": member_exp.get("target_mode", selection["target_mode"]),
                "feature_mode": member_exp.get("feature_mode", selection["feature_mode"]),
                "scaling_mode": member_exp.get("scaling_mode", selection["scaling_mode"]),
                "model_path": member_exp.get("model_path", ""),
                "dataset_hash": member_exp.get("dataset_hash"),
            }
            model, context = self.production_training_workflow.train(
                member,
                data_manager,
                selection=member_selection,
            )
            member_points[member] = self.forecast_point_generator.roll_forward_recursive(
                model_name=member,
                model=model,
                context=context,
                predictor=self.latest_target_prediction_workflow,
                horizon_days=horizon_days,
            )

        points = self.forecast_point_generator.combine_member_points(
            member_points=member_points,
            weights=weights,
            method=method,
            last_observed_date=data_manager.df["Date"].iloc[-1],
            last_close=float(data_manager.df["Close"].iloc[-1]),
        )
        last_close = float(data_manager.df["Close"].iloc[-1])
        weekly_expected_return = (points[-1]["bounded_predicted_close"] / last_close) - 1.0
        trend_threshold = self.rules.trend_threshold(
            data_manager.df["Close"].tail(80).to_numpy(dtype=float),
            horizon_days=horizon_days,
        )
        trend_label = self.rules.trend_label(weekly_expected_return, trend_threshold)
        agreement = self.forecast_point_generator.member_direction_agreement(member_points)
        metadata = dict(metadata)
        metadata["source_experiment_ids"] = source_ids
        run_id = self.persistence.save_run(
            stock_symbol=symbol,
            model_name=selection["model_name"],
            source_experiment_id=selection["source_experiment_id"],
            last_observed_date=pd.to_datetime(data_manager.df["Date"].iloc[-1]).strftime(
                "%Y-%m-%d"
            ),
            last_close=last_close,
            horizon_days=horizon_days,
            trend_label=trend_label,
            weekly_expected_return=weekly_expected_return,
            trend_threshold=trend_threshold,
            rules_version=RULES_VERSION,
            points=points,
            ensemble_direction_agreement=agreement,
            forecast_strategy="ensemble_recursive_direct_target",
            artifact_mode="artifact_loaded",
            forecast_warnings=["frozen_exogenous_features"],
            ensemble_metadata=metadata,
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
                "target_mode": (
                    "log_return" if best is None else best.get("target_mode", "log_return")
                ),
                "feature_mode": self.model_config.model_settings.get(
                    "feature_mode", "stationary_features"
                ),
                "scaling_mode": (
                    "robust_x_standard_y_clip"
                    if best is None
                    else best.get("scaling_mode", "robust_x_standard_y_clip")
                ),
            }
        if not best:
            raise ValueError(
                f"{symbol} icin best_models kaydi yok. Once pipeline calistirin veya --model kullanin."
            )
        return self._resolve_best_or_replacement(symbol, best)

    def _resolve_best_or_replacement(self, symbol: str, best: Dict[str, Any]) -> Dict[str, Any]:
        model_name = str(best["model_name"])
        resolved = {
            "model_name": model_name,
            "source_experiment_id": best.get("experiment_id"),
            "target_mode": best.get("target_mode", "log_return"),
            "feature_mode": best.get("feature_mode", "stationary_features"),
            "scaling_mode": best.get("scaling_mode", "robust_x_standard_y_clip"),
            "model_path": best.get("model_path", ""),
            "dataset_hash": best.get("dataset_hash"),
            "ensemble_metadata": self._parse_ensemble_metadata(best.get("ensemble_metadata_json")),
        }
        if model_name in _BASELINE_MODELS:
            replacement = self._best_trainable_experiment(symbol)
            if replacement is None:
                raise ValueError(
                    f"{symbol} icin baseline/ensemble disinda train edilebilir model bulunamadi. "
                    "Once pipeline'i gercek model aileleriyle calistirin veya --model kullanin."
                )
            resolved.update(
                {
                    "model_name": str(replacement["model_name"]),
                    "source_experiment_id": replacement.get("id"),
                    "target_mode": replacement.get("target_mode", resolved["target_mode"]),
                    "feature_mode": replacement.get("feature_mode", resolved["feature_mode"]),
                    "scaling_mode": replacement.get("scaling_mode", resolved["scaling_mode"]),
                    "model_path": replacement.get("model_path", ""),
                    "dataset_hash": replacement.get("dataset_hash"),
                }
            )
        return resolved

    @staticmethod
    def _parse_ensemble_metadata(raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, dict):
            return raw
        if not raw:
            return None
        import json

        try:
            parsed = json.loads(str(raw))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def best_trainable_experiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        rows = self.db.get_experiments(stock_symbol=symbol, limit=500)
        candidates = [
            row
            for row in rows
            if (
                not str(row.get("model_name", "")).startswith("Ensemble ")
                and str(row.get("model_name", "")) not in _BASELINE_MODELS
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: float(row.get("composite_score") or float("-inf")))

    def latest_member_experiment(self, symbol: str, model_name: str) -> Optional[Dict[str, Any]]:
        rows = self.db.get_experiments(stock_symbol=symbol, model_name=model_name, limit=100)
        candidates = [
            row
            for row in rows
            if row.get("model_path") and os.path.isfile(str(row.get("model_path")))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: str(row.get("trained_at") or ""))


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
    def train(
        self,
        model_name: str,
        data_manager: DataManager,
        *,
        selection: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, Dict[str, Any]]:
        selection = selection or {}
        if model_name.startswith("Ensemble "):
            raise ForecastArtifactError(
                f"{model_name} ensemble forecast icin ensemble artifact workflow gerekir."
            )
        frame = self._clean_training_frame(data_manager)
        close = frame["Close"].to_numpy(dtype=float)
        artifact = load_forecast_artifact_package(
            model_name=model_name,
            model_path=self._resolve_model_path(str(selection.get("model_path") or "")),
            model_factory=lambda name: self._make_model_instance(name, stage="final"),
        )
        features = list(artifact.metadata.get("feature_names") or data_manager.feature_names)
        missing = [feature for feature in features if feature not in frame.columns]
        if missing:
            raise ForecastArtifactError(
                f"{model_name} artifact feature mismatch: missing {missing[:5]}"
            )
        X_all = frame[features].to_numpy(dtype=float)
        latest_X = X_all[-1:].copy()
        latest_X_s = self._transform_features(artifact.scaler_X, latest_X)
        latest_seq = self._latest_sequence(
            model_name=model_name,
            scaler_X=artifact.scaler_X,
            X_all=X_all,
            time_steps=data_manager.data_cfg.time_steps,
        )
        metadata_time_steps = artifact.metadata.get("time_steps")
        if metadata_time_steps and int(metadata_time_steps) != int(
            data_manager.data_cfg.time_steps
        ):
            raise ForecastArtifactError(
                f"{model_name} artifact time_steps mismatch: {metadata_time_steps} != {data_manager.data_cfg.time_steps}"
            )
        return artifact.model, {
            "features": features,
            "target_mode": data_manager.data_cfg.target_mode,
            "scaler_X": artifact.scaler_X,
            "scaler_y": artifact.scaler_y,
            "artifact_metadata": artifact.metadata,
            "latest_X": latest_X,
            "latest_X_s": latest_X_s,
            "latest_seq": latest_seq,
            "last_close": float(close[-1]),
            "last_observed_date": pd.to_datetime(frame["Date"].iloc[-1]).normalize(),
            "feature_frame": frame[["Date", "Close", *features]].copy(),
            "time_steps": data_manager.data_cfg.time_steps,
            "forecast_strategy": "recursive_direct_target",
            "artifact_mode": "artifact_loaded",
            "forecast_warnings": ["frozen_exogenous_features"],
        }

    def _resolve_model_path(self, model_path: str) -> str:
        if not model_path or os.path.isabs(model_path):
            return model_path
        return os.path.join(self.project_root, model_path)

    @staticmethod
    def _clean_training_frame(data_manager: DataManager) -> pd.DataFrame:
        frame = data_manager.df.copy()
        frame = frame.dropna(subset=["Date", "Close", *data_manager.feature_names]).copy()
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if len(frame) < data_manager.data_cfg.time_steps + 2:
            raise ValueError("Production forecast icin yeterli tarihsel veri yok.")
        return frame

    @staticmethod
    def _transform_features(scaler_X, values):
        transformed = scaler_X.transform(values)
        clip_report = getattr(scaler_X, "clip_report_", {}) or {}
        if clip_report.get("clip_low") is not None and clip_report.get("clip_high") is not None:
            transformed = np.clip(transformed, clip_report["clip_low"], clip_report["clip_high"])
        return transformed

    def _latest_sequence(self, *, model_name: str, scaler_X, X_all, time_steps: int):
        if model_name not in _SEQ_MODELS:
            return None
        if len(X_all) < time_steps:
            raise ForecastArtifactError(f"{model_name} icin recursive sequence yetersiz.")
        X_all_s = self._transform_features(scaler_X, X_all)
        return X_all_s[-time_steps:].reshape(1, time_steps, X_all_s.shape[1])

    def _fit_model(
        self,
        model_name,
        data_manager,
        frame,
        X_all,
        X_train,
        y_train,
        X_train_s,
        y_train_s,
        scaler_X,
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
            return self._fit_sequence_model(
                model_name, data_manager, X_all, X_train_s, y_train_s, scaler_X
            )
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
        latest_seq = X_all_s[-data_manager.data_cfg.time_steps :].reshape(
            1,
            data_manager.data_cfg.time_steps,
            X_all_s.shape[1],
        )
        return model, latest_seq


class LatestTargetPredictionWorkflow(_OwnerBackedForecastService):
    def predict(self, model_name: str, model: Any, context: Dict[str, Any]) -> float:
        if model_name in _DATE_AWARE_MODELS:
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

    def predict_quantiles_target(
        self,
        model_name: str,
        model: Any,
        context: Dict[str, Any],
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> Optional[Dict[float, float]]:
        """
        Sprint 4 (2026-05-25) Plan A4.4: model `predict_quantiles` destekliyorsa
        en son zaman adimi icin {quantile: target_value} doner. Aksi halde None.
        """
        if not hasattr(model, "predict_quantiles"):
            return None
        if model_name in _SEQ_MODELS:
            latest = context.get("latest_seq")
        elif model_name in _TREE_MODELS:
            latest = context.get("latest_X_s")
        else:
            latest = context.get("latest_X")
        if latest is None:
            return None
        try:
            scaled = np.asarray(model.predict_quantiles(latest))
        except TypeError:
            scaled = np.asarray(model.predict_quantiles(latest, quantiles=quantiles))
        if scaled.ndim != 2 or scaled.shape[1] == 0:
            return None
        # Son satir (latest sample) icin quantile vektoru.
        last_row_scaled = scaled[-1].reshape(-1, 1)
        scaler_y = context.get("scaler_y")
        if scaler_y is not None:
            last_row = scaler_y.inverse_transform(last_row_scaled).ravel()
        else:
            last_row = last_row_scaled.ravel()
        # Quantile sayilari modelinkilerle eslestir; kullanicinin istedigi qs
        # listesi farkliysa interpolasyon yapmak yerine modelinkileri tutariz.
        model_qs = getattr(model, "quantiles", None) or quantiles
        if len(model_qs) != len(last_row):
            return None
        return {float(q): float(v) for q, v in zip(model_qs, last_row)}


class ForecastPointGenerator(_OwnerBackedForecastService):
    def combine_member_points(
        self,
        *,
        member_points: Dict[str, list[dict[str, Any]]],
        weights: Dict[str, float],
        method: str,
        last_observed_date: Any,
        last_close: float,
    ) -> list[dict[str, Any]]:
        names = list(member_points)
        normalized = self._normalized_weights(names, weights)
        horizon = min(len(points) for points in member_points.values())
        dates = self.rules.next_trading_days(last_observed_date, horizon)
        previous_close = float(last_close)
        combined: list[dict[str, Any]] = []
        for idx in range(horizon):
            weighted_close = sum(
                float(member_points[name][idx]["bounded_predicted_close"]) * normalized[name]
                for name in names
            )
            if method == "Cash-Gated":
                returns = [float(member_points[name][idx]["predicted_return"]) for name in names]
                signs = np.sign(np.asarray(returns, dtype=float))
                agreement = max((signs > 0).sum(), (signs < 0).sum()) / float(len(signs))
                if agreement < 0.6:
                    weighted_close = previous_close
            bounded_close, band = self.rules.bound_forecast_price(weighted_close, previous_close)
            combined.append(
                {
                    "target_date": dates[idx].strftime("%Y-%m-%d"),
                    "horizon_index": idx + 1,
                    "raw_predicted_close": weighted_close,
                    "bounded_predicted_close": bounded_close,
                    "predicted_return": (
                        (bounded_close / previous_close) - 1.0 if previous_close else 0.0
                    ),
                    "lower_band": band.lower_band,
                    "upper_band": band.upper_band,
                    "price_tick": band.price_tick,
                }
            )
            previous_close = bounded_close
        return combined

    @staticmethod
    def _normalized_weights(names: list[str], weights: Dict[str, float]) -> Dict[str, float]:
        raw = {name: float(weights.get(name, 0.0)) for name in names}
        total = sum(value for value in raw.values() if value > 0)
        if total <= 0:
            return {name: 1.0 / len(names) for name in names}
        return {name: max(raw[name], 0.0) / total for name in names}

    @staticmethod
    def member_direction_agreement(
        member_points: Dict[str, list[dict[str, Any]]],
    ) -> Optional[float]:
        if not member_points:
            return None
        returns = [
            float(points[-1]["predicted_return"]) for points in member_points.values() if points
        ]
        if not returns:
            return None
        signs = np.sign(np.asarray(returns, dtype=float))
        if not np.any(signs):
            return None
        return float(max((signs > 0).sum(), (signs < 0).sum()) / len(signs))

    def roll_forward_recursive(
        self,
        *,
        model_name: str,
        model: Any,
        context: Dict[str, Any],
        predictor: LatestTargetPredictionWorkflow,
        horizon_days: int,
    ) -> list[dict[str, Any]]:
        dates = self.rules.next_trading_days(context["last_observed_date"], horizon_days)
        frame = context["feature_frame"].copy()
        points: list[dict[str, Any]] = []
        previous_close = float(context["last_close"])
        target_mode = context["target_mode"]
        for idx, target_date in enumerate(dates, start=1):
            self._refresh_latest_context(context, frame, model_name)
            predicted_target = predictor.predict(model_name, model, context)
            raw_close = self._target_to_price(
                predicted_target,
                previous_close,
                target_mode,
            )
            bounded_close, band = self.rules.bound_forecast_price(raw_close, previous_close)
            predicted_return = (bounded_close / previous_close) - 1.0
            point: dict[str, Any] = {
                "target_date": target_date.strftime("%Y-%m-%d"),
                "horizon_index": idx,
                "raw_predicted_close": raw_close,
                "bounded_predicted_close": bounded_close,
                "predicted_return": predicted_return,
                "lower_band": band.lower_band,
                "upper_band": band.upper_band,
                "price_tick": band.price_tick,
            }
            # Sprint 4 A4.4: model quantile destekliyorsa p10/p50/p90 close +
            # returns yayinla. Yoksa point sozluguna ek alan eklenmez.
            quantile_targets = predictor.predict_quantiles_target(model_name, model, context)
            if quantile_targets:
                quantile_close: dict[str, float] = {}
                quantile_returns: dict[str, float] = {}
                for q, qval in quantile_targets.items():
                    qc = self._target_to_price(float(qval), previous_close, target_mode)
                    bounded_qc, _ = self.rules.bound_forecast_price(qc, previous_close)
                    label = f"p{int(round(q * 100))}"
                    quantile_close[label] = float(bounded_qc)
                    quantile_returns[label] = (
                        (float(bounded_qc) / previous_close) - 1.0 if previous_close else 0.0
                    )
                point["quantile_close"] = quantile_close
                point["quantile_returns"] = quantile_returns
                # Convenience top-level alanlari (advisory API icin shortcut):
                if "p10" in quantile_close:
                    point["p10_close"] = quantile_close["p10"]
                    point["predicted_return_p10"] = quantile_returns["p10"]
                if "p50" in quantile_close:
                    point["p50_close"] = quantile_close["p50"]
                    point["predicted_return_p50"] = quantile_returns["p50"]
                if "p90" in quantile_close:
                    point["p90_close"] = quantile_close["p90"]
                    point["predicted_return_p90"] = quantile_returns["p90"]
            points.append(point)
            frame = self._append_recursive_row(
                frame=frame,
                target_date=target_date,
                bounded_close=bounded_close,
                previous_close=previous_close,
            )
            previous_close = bounded_close
        return points

    def _refresh_latest_context(
        self, context: Dict[str, Any], frame: pd.DataFrame, model_name: str
    ) -> None:
        features = context["features"]
        X_all = frame[features].to_numpy(dtype=float)
        latest_X = X_all[-1:].copy()
        latest_X_s = ProductionTrainingWorkflow._transform_features(context["scaler_X"], latest_X)
        context["latest_X"] = latest_X
        context["latest_X_s"] = latest_X_s
        if model_name in _SEQ_MODELS:
            time_steps = int(context["time_steps"])
            if len(X_all) < time_steps:
                raise ValueError(f"{model_name} icin recursive sequence yetersiz.")
            X_all_s = ProductionTrainingWorkflow._transform_features(context["scaler_X"], X_all)
            context["latest_seq"] = X_all_s[-time_steps:].reshape(1, time_steps, X_all_s.shape[1])

    @staticmethod
    def _append_recursive_row(
        *,
        frame: pd.DataFrame,
        target_date: pd.Timestamp,
        bounded_close: float,
        previous_close: float,
    ) -> pd.DataFrame:
        new_row = frame.iloc[-1].copy()
        new_row["Date"] = pd.to_datetime(target_date).normalize()
        new_row["Close"] = float(bounded_close)
        simple_return = (bounded_close / previous_close) - 1.0 if previous_close else 0.0
        log_return = float(np.log(bounded_close / previous_close)) if previous_close else 0.0
        if "Return" in frame.columns:
            new_row["Return"] = simple_return
        if "Log_Return" in frame.columns:
            new_row["Log_Return"] = log_return
        lag_cols = sorted(
            [col for col in frame.columns if col.startswith("LogRet_Lag_")],
            key=lambda col: int(col.rsplit("_", 1)[-1]),
        )
        previous = frame.iloc[-1]
        for col in reversed(lag_cols):
            idx = int(col.rsplit("_", 1)[-1])
            if idx == 1:
                new_row[col] = log_return
            else:
                new_row[col] = previous.get(f"LogRet_Lag_{idx - 1}", new_row.get(col, 0.0))
        return pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)

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
            points.append(
                {
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "horizon_index": idx,
                    "raw_predicted_close": raw_close,
                    "bounded_predicted_close": bounded_close,
                    "predicted_return": predicted_return,
                    "lower_band": band.lower_band,
                    "upper_band": band.upper_band,
                    "price_tick": band.price_tick,
                }
            )
            previous_close = bounded_close
        return points
