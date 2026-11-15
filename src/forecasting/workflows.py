"""Internal forecast workflows backing ``ForecastRunner``."""

from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.forecasting.artifacts import (
    ArtifactFeaturesOutdatedError,
    ForecastArtifactError,
    ModelNoLongerSupportedError,
    artifact_sidecar_paths,
    load_forecast_artifact_package,
)
from src.database.paths import resolve_model_path
from src.forecasting.bist_rules import RULES_VERSION
from src.pipeline.config import DataConfig, ValidationConfig
from src.pipeline.data_manager import DataManager
from src.pipeline.model_scope import is_removed_model

logger = logging.getLogger(__name__)

_TREE_MODELS = {"XGBoost", "Ridge Return", "LightGBM Return"}
# Production forecast icin gereken minimum temiz tarihsel satir sayisi.
_MIN_FORECAST_ROWS = 32
_BASELINE_MODELS = {"Naive Last Value", "Naive Zero Return", "Naive Drift"}
# Makro veriye bagimli (use_macro gerektiren) kurate ozellikler (2026-11 IC denetimi).
# Faiz/CPI/VIX/... eski artifact ozellikleri artik uretilmez; onlari isteyen artifact
# `_prepare_artifact_context`'te acik hata verir (feature listesi uyusmazligi).
_MACRO_FEATURES = {
    "USDTRY_Return",
    "USDTRY_Volatility7",
    "BIST100_Return",
    "BIST100_MA7",
    "Relative_Strength",
    "market_momentum_60d",
    "relative_to_market_60d",
}

# h>1 recursive forecast, 1-adim modelin kendi ciktisiyla besleyerek uretilir ve
# hicbir walk-forward/backtest ile dogrulanmamistir (yalnizca h=1 dogrulanir).
RECURSIVE_UNVALIDATED_WARNING = "recursive_horizon_unvalidated"
RECURSIVE_UNVALIDATED_NOTE = (
    "Unvalidated: h>1 forecast is a recursion of a 1-step model; "
    "only h=1 is walk-forward validated, and interval coverage/trend labels "
    "for h>1 have not been backtested."
)


def _horizon_validation_fields(horizon_days: int) -> Dict[str, Any]:
    """h>1 icin recursive_forecast_validated=False + not; h=1 icin dogrulanmis."""
    if int(horizon_days) > 1:
        return {
            "recursive_forecast_validated": False,
            "horizon_validation_note": RECURSIVE_UNVALIDATED_NOTE,
        }
    return {"recursive_forecast_validated": True, "horizon_validation_note": None}


def _merge_warnings(*groups: Any) -> list[str]:
    """Uyari listelerini sirayi koruyarak tekillestirir."""
    merged: list[str] = []
    for group in groups:
        for item in group or []:
            if item not in merged:
                merged.append(item)
    return merged


@dataclass
class ForecastContext:
    """Forecast servisleri için açık READ-ONLY bağımlılık taşıyıcısı (Faz 5 DI).

    Owner-forward (`_OwnerBackedForecastService.__getattr__`) yerine geçer:
    config/identity + `ForecastRunner` factory callable'ları + kardeş servis
    referansları. Kardeş referanslar servisler kurulduktan sonra atanır.
    """

    project_root: str
    db: Any
    rules: Any
    model_config: Any
    # ForecastRunner factory callable'ları (model_config'e bağlı; runner'da kalır).
    make_model_instance: Callable[..., Any]
    target_to_price: Callable[[float, float, str], float]
    # Kardeş servis referansları (servisler kurulduktan sonra runner tarafından bağlanır).
    model_resolver: Any = None
    data_preparation_service: Any = None
    production_training_workflow: Any = None
    latest_target_prediction_workflow: Any = None
    forecast_point_generator: Any = None


class ForecastSymbolWorkflow:
    def __init__(self, ctx: ForecastContext, result_cls) -> None:
        self.ctx = ctx
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
        """Tek sembol için uçtan uca forward forecast üretir.

        Modeli çözer, veriyi hazırlar; ensemble seçilmişse ensemble dalına,
        aksi halde tek-model recursive horizon üretimine yönlendirir. Sonuç
        BIST kurallarıyla band-clip edilir ve persist edilir.

        Args:
            symbol: Hisse sembolü.
            data_file: OHLCV CSV yolu.
            horizon_days: İleriye dönük işlem günü sayısı.
            force_model_name: Verilirse seçim guard'ını atlayıp bu modeli kullanır.
            use_macro: Makro feature'ları dahil et.
            auto_update_data: Eksik/eski veriyi otomatik güncelle.
            auto_update_interactive: Güncellemede etkileşimli onay iste.

        Returns:
            Forecast noktaları ve metadata içeren sonuç sözlüğü.
        """
        symbol = symbol.upper()
        selection = self.ctx.model_resolver.resolve(symbol, force_model_name)
        use_macro = bool(use_macro or self._selection_requires_macro(symbol, selection))
        data_manager = self.ctx.data_preparation_service.prepare(
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
        model, forecast_context = self.ctx.production_training_workflow.train(
            selection["model_name"],
            data_manager,
            selection=selection,
        )
        points = self.ctx.forecast_point_generator.roll_forward_recursive(
            model_name=selection["model_name"],
            model=model,
            context=forecast_context,
            predictor=self.ctx.latest_target_prediction_workflow,
            horizon_days=horizon_days,
        )
        weekly_expected_return = (
            points[-1]["bounded_predicted_close"] / forecast_context["last_close"]
        ) - 1.0
        trend_threshold = self.ctx.rules.trend_threshold(
            data_manager.df["Close"].tail(80).to_numpy(dtype=float),
            horizon_days=horizon_days,
        )
        trend_label = self.ctx.rules.trend_label(weekly_expected_return, trend_threshold)
        horizon_fields = _horizon_validation_fields(horizon_days)
        extra_warnings = [] if horizon_fields["recursive_forecast_validated"] else [
            RECURSIVE_UNVALIDATED_WARNING
        ]
        run_id = self.ctx.db.log_forecast_run(
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
            forecast_warnings=_merge_warnings(
                forecast_context.get("forecast_warnings"), extra_warnings
            ),
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
            **horizon_fields,
        )

    def _selection_requires_macro(self, symbol: str, selection: Dict[str, Any]) -> bool:
        model_path = selection.get("model_path")
        print(f"[DEBUG] _selection_requires_macro symbol={symbol} model_path={model_path}")
        features = self._artifact_feature_names(model_path)
        print(f"[DEBUG] _selection_requires_macro loaded features={features}")
        if any(feature in _MACRO_FEATURES for feature in features):
            return True
        metadata = selection.get("ensemble_metadata") or {}
        for member in metadata.get("members") or []:
            member_exp = self.ctx.model_resolver.latest_member_experiment(symbol, member)
            if member_exp and any(
                feature in _MACRO_FEATURES
                for feature in self._artifact_feature_names(member_exp.get("model_path"))
            ):
                return True
        return False

    def _artifact_feature_names(self, model_path: Any) -> list[str]:
        model_path = self._resolve_model_path(str(model_path or ""))
        print(f"[DEBUG] _artifact_feature_names resolved model_path={model_path}")
        if not model_path:
            return []
        metadata_path = artifact_sidecar_paths(model_path)["metadata"]
        print(f"[DEBUG] _artifact_feature_names metadata_path={metadata_path} exists={os.path.isfile(metadata_path)}")
        if not os.path.isfile(metadata_path):
            return []
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception as e:
            print(f"[DEBUG] _artifact_feature_names json load failed: {e}")
            return []
        return [str(feature) for feature in metadata.get("feature_names") or []]

    def _resolve_model_path(self, model_path: str) -> str:
        # Göreceli yol proje köküne, başka makineden kalan mutlak `...\outputs\...`
        # yolu yerel outputs köküne çözülür (src/database/paths.py).
        return resolve_model_path(model_path, project_root=self.ctx.project_root)

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
        if len(members) < 2:
            raise ForecastArtifactError(
                f"{selection['model_name']} icin ensemble member metadata eksik."
            )
        member_points: Dict[str, list[dict[str, Any]]] = {}
        member_warnings: list[list[str]] = []
        source_ids: list[int] = []
        for member in members:
            member_exp = self.ctx.model_resolver.latest_member_experiment(symbol, member)
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
            model, context = self.ctx.production_training_workflow.train(
                member,
                data_manager,
                selection=member_selection,
            )
            member_points[member] = self.ctx.forecast_point_generator.roll_forward_recursive(
                model_name=member,
                model=model,
                context=context,
                predictor=self.ctx.latest_target_prediction_workflow,
                horizon_days=horizon_days,
            )
            member_warnings.append(list(context.get("forecast_warnings") or []))

        points = self.ctx.forecast_point_generator.combine_member_points(
            member_points=member_points,
            weights=weights,
            last_observed_date=data_manager.df["Date"].iloc[-1],
            last_close=float(data_manager.df["Close"].iloc[-1]),
        )
        last_close = float(data_manager.df["Close"].iloc[-1])
        weekly_expected_return = (points[-1]["bounded_predicted_close"] / last_close) - 1.0
        trend_threshold = self.ctx.rules.trend_threshold(
            data_manager.df["Close"].tail(80).to_numpy(dtype=float),
            horizon_days=horizon_days,
        )
        trend_label = self.ctx.rules.trend_label(weekly_expected_return, trend_threshold)
        agreement = self.ctx.forecast_point_generator.member_direction_agreement(member_points)
        metadata = dict(metadata)
        metadata["source_experiment_ids"] = source_ids
        horizon_fields = _horizon_validation_fields(horizon_days)
        extra_warnings = [] if horizon_fields["recursive_forecast_validated"] else [
            RECURSIVE_UNVALIDATED_WARNING
        ]
        run_id = self.ctx.db.log_forecast_run(
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
            forecast_warnings=_merge_warnings(
                ["projected_exogenous_features"], *member_warnings, extra_warnings
            ),
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
            **horizon_fields,
        )


class BestModelResolver:
    def __init__(self, ctx: ForecastContext) -> None:
        self.ctx = ctx

    def resolve(self, symbol: str, force_model_name: str | None = None) -> Dict[str, Any]:
        """Forecast için kullanılacak model + mod (target/feature/scaling) seçimini döner.

        `force_model_name` verilmezse kayıtlı best model kullanılır.
        """
        best = self.ctx.db.get_best_model(symbol)
        self._reject_removed_model(
            symbol, force_model_name or (None if best is None else best.get("model_name"))
        )
        if force_model_name:
            # Once zorlanan modele ait, artifact dosyasi diskte mevcut olan en guncel
            # deneyi ara. Bulunursa forecast dogrudan o artifact'i yukler (model_path
            # dolu) ve egitim konfigurasyonu (target/feature/scaling/dataset_hash) o
            # deneyden gelir. Aksi halde eski davranis korunur: best metadata ile devam
            # edilir, model_path bos kalir (artifact'i olmayan model zorlanmissa cagiran
            # taraf net hata verir).
            member = self.latest_member_experiment(symbol, force_model_name)
            if member is not None:
                return {
                    "model_name": force_model_name,
                    "source_experiment_id": member.get("id"),
                    "target_mode": member.get("target_mode", "log_return"),
                    "feature_mode": member.get(
                        "feature_mode",
                        self.ctx.model_config.model_settings.get(
                            "feature_mode", "stationary_features"
                        ),
                    ),
                    "scaling_mode": member.get(
                        "scaling_mode", "robust_x_standard_y_clip"
                    ),
                    "model_path": member.get("model_path", ""),
                    "dataset_hash": member.get("dataset_hash"),
                    "ensemble_metadata": None,
                }
            return {
                "model_name": force_model_name,
                "source_experiment_id": None if best is None else best.get("experiment_id"),
                "target_mode": (
                    "log_return" if best is None else best.get("target_mode", "log_return")
                ),
                "feature_mode": self.ctx.model_config.model_settings.get(
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

    @staticmethod
    def _reject_removed_model(symbol: str, model_name: Any) -> None:
        """Silinen modeller icin yapisal 'yeniden egit' hatasi (import hatasi degil)."""
        if is_removed_model(model_name):
            raise ModelNoLongerSupportedError(str(model_name), symbol=symbol)

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
            replacement = self.best_trainable_experiment(symbol)
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
        """Ensemble/baseline dışı, en yüksek composite_score'lu deneyi döner (yoksa None)."""
        rows = self.ctx.db.get_experiments(stock_symbol=symbol, limit=500)
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
        """Model adı için artifact dosyası mevcut olan en güncel deneyi döner (yoksa None)."""
        rows = self.ctx.db.get_experiments(stock_symbol=symbol, model_name=model_name, limit=100)
        candidates = [
            row
            for row in rows
            if row.get("model_path")
            and os.path.isfile(resolve_model_path(row.get("model_path"), project_root=self.ctx.project_root))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: str(row.get("trained_at") or ""))


class ForecastDataPreparationService:
    def __init__(self, ctx: ForecastContext) -> None:
        self.ctx = ctx

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
        """Forecast için veriyi yükleyip feature'ları üreten DataManager döner.

        single_split modunda ingest + feature engineering çalıştırır; veri boşsa
        ValueError yükselir.
        """
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
            models_dir=os.path.join(self.ctx.project_root, "outputs", symbol, "forecast_models"),
            macro_cache_dir=os.path.join(self.ctx.project_root, "data", "macro"),
        )
        data_manager.ingest_and_engineer()
        if data_manager.df is None or data_manager.df.empty:
            raise ValueError(f"{symbol} icin forecast uretecek veri yok.")
        return data_manager


class ProductionTrainingWorkflow:
    def __init__(self, ctx: ForecastContext) -> None:
        self.ctx = ctx

    def train(
        self,
        model_name: str,
        data_manager: DataManager,
        *,
        selection: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, Dict[str, Any]]:
        """Üretim forecast'i için tek modeli tüm veri üzerinde yeniden eğitir.

        Ensemble adları reddedilir (ayrı ensemble artifact workflow gerekir).

        Returns:
            (egitilmis_model, artifact_metadata) ikilisi.
        """
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
            model_factory=lambda name: self.ctx.make_model_instance(name, stage="final"),
        )
        features = list(artifact.metadata.get("feature_names") or data_manager.feature_names)
        self._validate_artifact_features(
            model_name=model_name,
            features=features,
            frame_columns=frame.columns,
            current_features=data_manager.feature_names,
            scaler_X=artifact.scaler_X,
        )
        X_all = frame[features].to_numpy(dtype=float)
        latest_X = X_all[-1:].copy()
        latest_X_s = self._transform_features(artifact.scaler_X, latest_X)
        return artifact.model, {
            "features": features,
            "target_mode": data_manager.data_cfg.target_mode,
            "feature_mode": data_manager.data_cfg.feature_mode,
            # Olasılıksal interval kalibrasyonu (B2/C). Artifact sidecar'ında varsa
            # dolu; yoksa None -> interval üretilmez (geriye uyumlu).
            "interval_calibration": artifact.metadata.get("interval_calibration"),
            # Aktif üreteç tercihi (residual_b2 | conformal). Config'ten gelir.
            "interval_method_pref": str(
                self.ctx.model_config.model_settings.get("interval_method", "residual_b2")
            ),
            "scaler_X": artifact.scaler_X,
            "scaler_y": artifact.scaler_y,
            "artifact_metadata": artifact.metadata,
            "latest_X": latest_X,
            "latest_X_s": latest_X_s,
            "last_close": float(close[-1]),
            "last_observed_date": pd.to_datetime(frame["Date"].iloc[-1]).normalize(),
            "feature_frame": self._recursive_feature_frame(frame, features),
            "forecast_strategy": "recursive_direct_target",
            "artifact_mode": "artifact_loaded",
            "forecast_warnings": ["projected_exogenous_features"],
        }

    def _resolve_model_path(self, model_path: str) -> str:
        # Göreceli yol proje köküne, başka makineden kalan mutlak `...\outputs\...`
        # yolu yerel outputs köküne çözülür (src/database/paths.py).
        return resolve_model_path(model_path, project_root=self.ctx.project_root)

    @staticmethod
    def _recursive_feature_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """Recursive roll-forward icin feature frame'i: Date/Close + OHLV + ozellikler.

        Open/High/Low/Volume ozellik OLMASA bile tutulur: NATR_14 (High/Low) ve diger
        close-bagimli gostergelerin her adimda yeniden hesaplanabilmesi icin gerekir.
        """
        ohlv = [c for c in ("Open", "High", "Low", "Volume") if c in frame.columns]
        cols: list[str] = []
        for col in ["Date", "Close", *ohlv, *features]:
            if col not in cols:
                cols.append(col)
        return frame[cols].copy()

    @staticmethod
    def _validate_artifact_features(
        *,
        model_name: str,
        features: list[str],
        frame_columns,
        current_features: list[str],
        scaler_X,
    ) -> None:
        """Artifact ozellik listesi mevcut feature matrisiyle uyusmuyorsa ACIK hata verir.

        2026-11 IC denetimi kurate ozellik setine gecti: eski artifact'lar kaldirilan
        ozellikleri (EMA_*, VIX_*, Rate_*, ...) bekler. Eksik kolon -> yeniden egitim
        gerekir (migrasyon yok). Ayrica scaler ozellik sayisi artifact listesiyle
        uyusmazsa (metadata listesi eksik/eski) sessiz kolon kaymasi yerine hata verilir.
        """
        available = set(frame_columns)
        missing = [feature for feature in features if feature not in available]
        if missing:
            raise ArtifactFeaturesOutdatedError(
                model_name,
                f"{model_name} artifact feature mismatch: artifact bekledigi "
                f"{len(missing)}/{len(features)} ozellik mevcut feature setinde yok "
                f"(ilk 10: {missing[:10]}). Artifact eski ozellik seti ile egitilmis; "
                f"modeli yeniden egitin. Mevcut set ({len(current_features)}): "
                f"{list(current_features)}"
            )
        n_scaler = getattr(scaler_X, "n_features_in_", None)
        if n_scaler is not None and int(n_scaler) != len(features):
            raise ArtifactFeaturesOutdatedError(
                model_name,
                f"{model_name} artifact feature mismatch: scaler {int(n_scaler)} ozellik "
                f"bekliyor, artifact listesi {len(features)} ozellik iceriyor "
                f"(metadata feature_names eksik/eski). Modeli yeniden egitin."
            )

    @staticmethod
    def _clean_training_frame(data_manager: DataManager) -> pd.DataFrame:
        frame = data_manager.df.copy()
        frame = frame.dropna(subset=["Date", "Close", *data_manager.feature_names]).copy()
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if len(frame) < _MIN_FORECAST_ROWS:
            raise ValueError("Production forecast icin yeterli tarihsel veri yok.")
        return frame

    @staticmethod
    def _transform_features(scaler_X, values):
        transformed = scaler_X.transform(values)
        clip_report = getattr(scaler_X, "clip_report_", {}) or {}
        if clip_report.get("clip_low") is not None and clip_report.get("clip_high") is not None:
            transformed = np.clip(transformed, clip_report["clip_low"], clip_report["clip_high"])
        return transformed


class LatestTargetPredictionWorkflow:
    def __init__(self, ctx: ForecastContext) -> None:
        self.ctx = ctx

    def predict(self, model_name: str, model: Any, context: Dict[str, Any]) -> float:
        """En son gözlem penceresinden tek-adım hedef tahmini üretir.

        Model tipine göre (tree / ham tabular) uygun giriş tensörünü seçer ve
        son adımın skaler tahminini döner.
        """
        if model_name in _TREE_MODELS:
            scaled = np.asarray(model.predict(context["latest_X_s"])).reshape(-1, 1)
            return float(context["scaler_y"].inverse_transform(scaled).ravel()[-1])
        raw = model.predict(context["latest_X"])
        return float(np.asarray(raw).ravel()[-1])


class ForecastPointGenerator:
    def __init__(self, ctx: ForecastContext) -> None:
        self.ctx = ctx

    def combine_member_points(
        self,
        *,
        member_points: Dict[str, list[dict[str, Any]]],
        weights: Dict[str, float],
        last_observed_date: Any,
        last_close: float,
    ) -> list[dict[str, Any]]:
        """Ensemble üye nokta tahminlerini ağırlıklı birleştirip tek seriye indirger.

        `weights` ile her horizon adımında üye getirilerini harmanlar ve
        `last_close`'tan başlayarak fiyat serisini yeniden kurar.
        """
        names = list(member_points)
        normalized = self._normalized_weights(names, weights)
        horizon = min(len(points) for points in member_points.values())
        dates = self.ctx.rules.next_trading_days(last_observed_date, horizon)
        previous_close = float(last_close)
        combined: list[dict[str, Any]] = []
        for idx in range(horizon):
            weighted_close = sum(
                float(member_points[name][idx]["bounded_predicted_close"]) * normalized[name]
                for name in names
            )
            bounded_close, band = self.ctx.rules.bound_forecast_price(weighted_close, previous_close)
            point = {
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
            # Olasılıksal interval: üyelerin p10/p90'ını ağırlıklı birleştir (varsa).
            self._combine_member_interval(
                point=point,
                member_points=member_points,
                names=names,
                normalized=normalized,
                idx=idx,
                previous_close=previous_close,
                p50_close=bounded_close,
            )
            combined.append(point)
            previous_close = bounded_close
        return combined

    @staticmethod
    def _combine_member_interval(
        *,
        point: Dict[str, Any],
        member_points: Dict[str, list[dict[str, Any]]],
        names: list[str],
        normalized: Dict[str, float],
        idx: int,
        previous_close: float,
        p50_close: float,
    ) -> None:
        """Üye p10/p90'larını ağırlıklı birleştirir (yerinde yazar).

        Yalnızca p10_close/p90_close dolu üyeler katkı verir; ağırlıklar bu alt küme
        üzerinde yeniden normalize edilir. Hiç üyede interval yoksa hiçbir alan
        eklenmez (geriye uyumlu). p50 = ensemble bounded close; band p50 etrafında
        clamp edilir.
        """
        avail = [
            name
            for name in names
            if member_points[name][idx].get("p10_close") is not None
            and member_points[name][idx].get("p90_close") is not None
        ]
        if not avail:
            return
        total_w = sum(normalized.get(name, 0.0) for name in avail)
        if total_w <= 0.0:
            renorm = {name: 1.0 / len(avail) for name in avail}
        else:
            renorm = {name: normalized.get(name, 0.0) / total_w for name in avail}
        lower = sum(float(member_points[name][idx]["p10_close"]) * renorm[name] for name in avail)
        upper = sum(float(member_points[name][idx]["p90_close"]) * renorm[name] for name in avail)
        if lower > upper:
            lower, upper = upper, lower
        # Band p50 etrafında tutarlı olsun (alt ≤ p50 ≤ üst).
        lower = min(lower, p50_close)
        upper = max(upper, p50_close)
        methods = {
            str(member_points[name][idx].get("interval_method"))
            for name in avail
            if member_points[name][idx].get("interval_method")
        }
        method = methods.pop() if len(methods) == 1 else "ensemble"
        point["p10_close"] = lower
        point["p50_close"] = float(p50_close)
        point["p90_close"] = upper
        point["predicted_return_p10"] = (lower / previous_close) - 1.0 if previous_close else 0.0
        point["predicted_return_p50"] = float(point.get("predicted_return", 0.0))
        point["predicted_return_p90"] = (upper / previous_close) - 1.0 if previous_close else 0.0
        point["interval_method"] = method

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
        """Ensemble üyelerinin yön (işaret) uzlaşı oranını [0.5, 1.0] döner (yoksa None).

        Son horizon adımının `predicted_return` işaretlerine bakar; çoğunluk
        oranını verir.
        """
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
        """Close-bağımlı feature'ları yeniden hesaplayarak recursive horizon tahmini üretir.

        Her BIST işlem günü için: tahmin -> fiyat -> feature frame güncelle -> bir
        sonraki adım. Nedensel zincir korunur (gelecek bilgisi sızmaz).

        Returns:
            Horizon boyunca tarih/fiyat/getiri nokta sözlüklerinin listesi.
        """
        dates = self.ctx.rules.next_trading_days(context["last_observed_date"], horizon_days)
        frame = context["feature_frame"].copy()
        # Adim bazli uyarilar (recompute_failed_step_k ...) context'e yazilir; liste
        # kopyalanir ki paylasilan/yeniden kullanilan liste mutasyona ugramasin.
        context["forecast_warnings"] = list(context.get("forecast_warnings") or [])
        points: list[dict[str, Any]] = []
        previous_close = float(context["last_close"])
        target_mode = context["target_mode"]
        for idx, target_date in enumerate(dates, start=1):
            self._refresh_latest_context(context, frame, model_name)
            predicted_target = predictor.predict(model_name, model, context)
            raw_close = self.ctx.target_to_price(
                predicted_target,
                previous_close,
                target_mode,
            )
            bounded_close, band = self.ctx.rules.bound_forecast_price(raw_close, previous_close)
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
            # Model-agnostik olasiliksal interval (B2 residual / C conformal).
            # Kalibrasyon yoksa band yayinlanmaz (kalibrasyonsuz band yayinlamaktan iyi).
            self._apply_model_agnostic_interval(
                point=point,
                predicted_target=predicted_target,
                previous_close=previous_close,
                target_mode=target_mode,
                calibration=context.get("interval_calibration"),
                horizon_index=idx,
                frame=frame,
                prefer=context.get("interval_method_pref", "residual_b2"),
            )
            points.append(point)
            frame = self._append_recursive_row(
                frame=frame,
                target_date=target_date,
                bounded_close=bounded_close,
                previous_close=previous_close,
            )
            # Sprint 5 A5.1: close-bagimli teknik gostergeleri yeniden hesapla.
            # Macro + sector + lag dokunulmaz (lag _append_recursive_row icinde
            # ele alindi). Hata olursa loglanir + forecast_warnings'e yazilir.
            frame = self._recompute_close_dependent_safe(frame, context, step=idx)
            # Sprint 5 A5.2: macro getiri/degisim -> 0, seviye -> son deger.
            frame = self._apply_macro_forward_projection_safe(
                frame, context, target_date, step=idx
            )
            # Momentum (60g) close + BIST100 projeksiyonuna bagli: projeksiyondan SONRA tazele.
            frame = self._recompute_momentum_features_safe(frame, context, step=idx)
            # Relative_Strength = Return - BIST100_Return (yeni satir icin tazele).
            frame = self._recompute_relative_strength(frame)
            previous_close = bounded_close
        return points

    def _apply_model_agnostic_interval(
        self,
        *,
        point: Dict[str, Any],
        predicted_target: float,
        previous_close: float,
        target_mode: str,
        calibration: Optional[Dict[str, Any]],
        horizon_index: int,
        frame: pd.DataFrame,
        prefer: str = "residual_b2",
    ) -> None:
        """B2/C kalibrasyonundan p10/p50/p90 close + return üretir (yerinde yazar).

        Kalibrasyon yoksa veya geçersizse hiçbir alan eklenmez (geriye uyumlu).
        `prefer` aktif üreteci seçer (residual_b2 | conformal). Band hedef (target)
        uzayında kurulur, fiyata çevrilir, BIST clip uygulanır. p50 = mevcut bounded
        tahmin (point), p10/p90 = band sınırları.
        """
        from src.forecasting.interval_calibration import (
            COVERAGE_GUARANTEE,
            conformal_band,
            residual_band,
            resolve_active_calibration,
            sigma_for_regime,
        )

        calibration = resolve_active_calibration(calibration, prefer)
        if not calibration:
            return

        method = str(calibration.get("method", ""))
        levels = calibration.get("levels") or [0.8]
        level = float(levels[0]) if levels else 0.8
        try:
            if method == "residual_b2":
                regime = self._latest_regime(frame)
                sigma = sigma_for_regime(calibration, regime)
                lower_t, upper_t = residual_band(
                    float(predicted_target), sigma, horizon_index, level
                )
            elif method == "conformal":
                q_hat = float(calibration.get("q_hat", 0.0))
                level = float(calibration.get("level", level))
                lower_t, upper_t = conformal_band(
                    float(predicted_target), q_hat, horizon_index, horizon_scale=True
                )
            else:
                return
        except (ValueError, TypeError):
            return

        lower_close = self._target_to_bounded_price(lower_t, previous_close, target_mode)
        upper_close = self._target_to_bounded_price(upper_t, previous_close, target_mode)
        # Band sıralaması garanti (target_to_price monoton ama clip karıştırabilir).
        if lower_close > upper_close:
            lower_close, upper_close = upper_close, lower_close
        p50_close = float(point["bounded_predicted_close"])
        point["p10_close"] = lower_close
        point["p50_close"] = p50_close
        point["p90_close"] = upper_close
        point["predicted_return_p10"] = (
            (lower_close / previous_close) - 1.0 if previous_close else 0.0
        )
        point["predicted_return_p50"] = float(point.get("predicted_return", 0.0))
        point["predicted_return_p90"] = (
            (upper_close / previous_close) - 1.0 if previous_close else 0.0
        )
        point["interval_method"] = method
        point["interval_level"] = level
        point["coverage_guarantee"] = COVERAGE_GUARANTEE

    def _target_to_bounded_price(
        self, target: float, previous_close: float, target_mode: str
    ) -> float:
        raw = self.ctx.target_to_price(float(target), previous_close, target_mode)
        bounded, _ = self.ctx.rules.bound_forecast_price(raw, previous_close)
        return float(bounded)

    @staticmethod
    def _latest_regime(frame: pd.DataFrame) -> Optional[str]:
        for col in ("Market_Regime", "market_regime"):
            if col in frame.columns and len(frame):
                val = frame[col].iloc[-1]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return str(val)
        return None

    def _refresh_latest_context(
        self, context: Dict[str, Any], frame: pd.DataFrame, model_name: str
    ) -> None:
        features = context["features"]
        X_all = frame[features].to_numpy(dtype=float)
        latest_X = X_all[-1:].copy()
        latest_X_s = ProductionTrainingWorkflow._transform_features(context["scaler_X"], latest_X)
        context["latest_X"] = latest_X
        context["latest_X_s"] = latest_X_s

    @staticmethod
    def _record_warning(context: Dict[str, Any], warning: str) -> None:
        warnings_list = context.setdefault("forecast_warnings", [])
        if warning not in warnings_list:
            warnings_list.append(warning)

    def _recompute_close_dependent_safe(
        self,
        frame: pd.DataFrame,
        context: Dict[str, Any],
        step: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Sprint 5 A5.1: FeaturePipeline.recompute_close_dependent() cagrisi.
        Basarisiz olursa frame degismez (forecast yine uretilir) AMA sessiz degil:
        WARNING loglanir ve ``recompute_failed_step_{k}`` forecast_warnings'e eklenir.
        """
        tag = f"recompute_failed_step_{step}" if step is not None else "recompute_failed"
        feature_mode = context.get("feature_mode", "stationary_features")
        try:
            from src.features.feature_pipeline import FeaturePipeline

            fp = FeaturePipeline(feature_mode=feature_mode)
            updated = fp.recompute_close_dependent(frame)
            # NaN guard: yeni satirda NaN olursa eski deger ile doldur (ffill).
            return updated.ffill().bfill()
        except Exception as exc:  # noqa: BLE001 - forecast uretimi bozulmamali; kayit altinda
            logger.warning(
                "recompute_close_dependent basarisiz (step=%s): %s: %s -- teknik gostergeler "
                "bu adimda bayat kaliyor.",
                step,
                type(exc).__name__,
                exc,
            )
            self._record_warning(context, tag)
            return frame

    def _recompute_momentum_features_safe(
        self,
        frame: pd.DataFrame,
        context: Dict[str, Any],
        step: Optional[int] = None,
    ) -> pd.DataFrame:
        """momentum_60d / market_momentum_60d / relative_to_market_60d'yi tazeler.

        Hata olursa frame degismez; loglanir + forecast_warnings'e yazilir.
        """
        if "momentum_60d" not in frame.columns:
            return frame
        try:
            from src.features.feature_pipeline import FeaturePipeline

            return FeaturePipeline().recompute_momentum_features(frame).ffill().bfill()
        except Exception as exc:  # noqa: BLE001 - forecast uretimi bozulmamali; kayit altinda
            logger.warning(
                "momentum recompute basarisiz (step=%s): %s: %s",
                step,
                type(exc).__name__,
                exc,
            )
            self._record_warning(
                context,
                f"momentum_recompute_failed_step_{step}"
                if step is not None
                else "momentum_recompute_failed",
            )
            return frame

    def _apply_macro_forward_projection_safe(
        self,
        frame: pd.DataFrame,
        context: Dict[str, Any],
        target_date: pd.Timestamp,
        step: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Sprint 5 A5.2: Son satir macro feature'lerini projekte eder (getiri/degisim
        -> 0, seviye -> son deger). Hata olursa frame degismez; loglanir + uyari.
        """
        try:
            from src.features.macro_forward_projection import MacroForwardProjector

            projector = MacroForwardProjector()
            return projector.project_last_row(frame, target_date=pd.to_datetime(target_date))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "macro forward projection basarisiz (step=%s): %s: %s",
                step,
                type(exc).__name__,
                exc,
            )
            self._record_warning(
                context,
                f"macro_projection_failed_step_{step}"
                if step is not None
                else "macro_projection_failed",
            )
            return frame

    @staticmethod
    def _recompute_relative_strength(frame: pd.DataFrame) -> pd.DataFrame:
        """Son satirin Relative_Strength'ini Return - BIST100_Return ile tazeler.

        feature_pipeline._merge_macro ile ayni formul; yeni (sentetik) satirda
        onceki satirdan kopya kalan degeri duzeltir. Sutunlar yoksa dokunmaz.
        """
        needed = {"Relative_Strength", "Return", "BIST100_Return"}
        if frame.empty or not needed.issubset(frame.columns):
            return frame
        out = frame.copy()
        last = out.index[-1]
        ret = out.at[last, "Return"]
        mkt = out.at[last, "BIST100_Return"]
        out.at[last, "Relative_Strength"] = float(0.0 if pd.isna(ret) else ret) - float(
            0.0 if pd.isna(mkt) else mkt
        )
        return out

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
        # Takvim ozelligi: son satirdan kopya kalmasin, yeni tarihten turet (0=Pazartesi).
        if "day_of_week" in frame.columns:
            new_row["day_of_week"] = float(pd.Timestamp(new_row["Date"]).dayofweek)
        # Sentetik bar: gun-ici bilgi yok -> durust sifir-aralikli bar.
        # Open = onceki kapanis; High/Low = Open ve Close'un max/min'i (Low<=Close<=High);
        # Volume = son 20 bar medyani. (Son gercek barin OHLV'si KOPYALANMAZ.)
        if "Open" in frame.columns:
            new_row["Open"] = float(previous_close)
        if "High" in frame.columns:
            new_row["High"] = max(float(previous_close), float(bounded_close))
        if "Low" in frame.columns:
            new_row["Low"] = min(float(previous_close), float(bounded_close))
        if "Volume" in frame.columns:
            trailing = pd.to_numeric(frame["Volume"].iloc[-20:], errors="coerce").dropna()
            if len(trailing):
                new_row["Volume"] = float(trailing.median())
        simple_return = (bounded_close / previous_close) - 1.0 if previous_close else 0.0
        if "Return" in frame.columns:
            new_row["Return"] = simple_return
        # LogRet_Lag_k[t+1] = log-getiri[t+1-k]: egitimdeki tanimla (shift(k)) ayni.
        # Lag_1 = SON GERCEK/ONCEKI adimin log-getirisi (log(previous_close / Close[-2]));
        # yeni adimin kendi getirisi lag-0'dir (Return kolonu) ve lag'a SIZDIRILMAZ.
        # Lag_k (k>=2) = onceki satirin Lag_{k-1}'i.
        prev_prev_close = float(frame["Close"].iloc[-2]) if len(frame) >= 2 else 0.0
        prev_log_return = (
            float(np.log(previous_close / prev_prev_close))
            if previous_close and prev_prev_close
            else 0.0
        )
        lag_cols = sorted(
            [col for col in frame.columns if col.startswith("LogRet_Lag_")],
            key=lambda col: int(col.rsplit("_", 1)[-1]),
        )
        previous = frame.iloc[-1]
        for col in reversed(lag_cols):
            idx = int(col.rsplit("_", 1)[-1])
            if idx == 1:
                new_row[col] = prev_log_return
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
        """Sabit tek hedef tahminini horizon boyunca tekrarlayarak fiyat serisi üretir.

        Feature recompute yapmayan basit ileri-yuvarlama; tahmini her adımda aynı
        `predicted_target` ile uygular ve `last_close`'tan fiyatları kurar.

        Returns:
            Horizon boyunca tarih/fiyat/getiri nokta sözlüklerinin listesi.
        """
        dates = self.ctx.rules.next_trading_days(last_observed_date, horizon_days)
        points: list[dict[str, Any]] = []
        previous_close = float(last_close)
        for idx, target_date in enumerate(dates, start=1):
            raw_close = self.ctx.target_to_price(predicted_target, previous_close, target_mode)
            bounded_close, band = self.ctx.rules.bound_forecast_price(raw_close, previous_close)
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
