# -*- coding: utf-8 -*-
"""
data_manager.py — Veri Hazırlama Orkestratörü
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SRP: Veri indirme, özellik mühendisliği (teknik + makro),
train/test ayırma ve tensör hazırlama işlemlerinden sorumludur.
"""

import os
import json
import hashlib
from dataclasses import fields
import numpy as np
import pandas as pd

from src.data.data_updater import DataUpdater
from src.data.data_loader import load_data
from src.data.preprocessor import scale_data, create_sequences
from src.utils.data_splitter import TimeSeriesSplitter
from src.features.feature_pipeline import FeaturePipeline
from src.features.feature_cache import FeatureCache
from src.features.macro_pipeline import MacroPipeline
from src.pipeline.config import DataConfig, ValidationConfig
from src.pipeline.data_services import (
    DataIngestionService,
    DataQualityReportingService,
    TensorPreparationService,
    ValidationSplitService,
)


class DataManager:
    """
    Args:
        data_cfg       : Veri yükleme ve özellik mühendisliği konfigürasyonu.
        val_cfg        : Validasyon protokolü konfigürasyonu.
        models_dir     : Scaler objelerinin kaydedileceği dizin.
        macro_cache_dir: Makro CSV cache dizini.
    """

    def __init__(
        self,
        data_cfg: DataConfig | None = None,
        val_cfg: ValidationConfig | None = None,
        models_dir: str | None = None,
        macro_cache_dir: str = None,
        **legacy_kwargs,
    ):
        data_cfg, val_cfg, models_dir = self._normalize_constructor_args(
            data_cfg=data_cfg,
            val_cfg=val_cfg,
            models_dir=models_dir,
            legacy_kwargs=legacy_kwargs,
        )
        self.data_cfg = data_cfg
        self.val_cfg = val_cfg
        self.models_dir = models_dir

        self.stock_symbol = os.path.splitext(os.path.basename(self.data_cfg.data_file))[0]

        # Makro cache dizini: varsayılan olarak proje kökü altında data/macro/
        if macro_cache_dir is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            macro_cache_dir = os.path.join(project_root, "data", "macro")
        self.macro_cache_dir = macro_cache_dir

        # State variables
        self.df: pd.DataFrame = None
        self.feature_names: list = []
        self.tensors: dict = {}
        self.wf_splits: list = []
        self.selection_df: pd.DataFrame | None = None
        self.final_holdout_df: pd.DataFrame | None = None

        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.universe_file = self.data_cfg.universe_file
        if self.universe_file and not os.path.isabs(self.universe_file):
            self.universe_file = os.path.join(self.project_root, self.universe_file)
            self.data_cfg.universe_file = self.universe_file

        if self.universe_file and getattr(self.data_cfg, "universe_auto_sync", True):
            try:
                from src.data.universe_sync import sync_universe

                sync_universe(os.path.join(self.project_root, "data"), self.universe_file)
            except Exception as _exc:
                print(f"  [UNIVERSE] sync atlandi: {_exc}")

        effective_wf_embargo_size = (
            self.data_cfg.time_steps
            if self.val_cfg.wf_embargo_size is None
            else max(0, int(self.val_cfg.wf_embargo_size))
        )

        wf_max_train_size = self.val_cfg.wf_max_train_size
        if self.val_cfg.wf_window_type not in {"sliding", "expanding"}:
            raise ValueError("wf_window_type 'sliding' veya 'expanding' olmalidir.")
        if self.val_cfg.wf_window_type == "expanding":
            wf_max_train_size = None

        self.validation_config = {
            "wf_n_splits": self.val_cfg.wf_n_splits,
            "wf_min_train_size": self.val_cfg.wf_min_train_size,
            "wf_test_size": self.val_cfg.wf_test_size,
            "wf_max_train_size": wf_max_train_size,
            "wf_window_type": self.val_cfg.wf_window_type,
            "wf_embargo_size": effective_wf_embargo_size,
            "final_holdout_size": self.val_cfg.final_holdout_size,
        }
        self.dataset_metadata: dict = {}
        self.dataset_hash: str = "N/A"
        self.corporate_action_report: dict = {}
        self.feature_groups: dict[str, str] = {}
        self.feature_pruning_report: dict = {}
        self.sector_mapping_report: dict = {}
        self.survivorship_bias_report: dict = {}
        self.training_window_report: dict = {}
        self.scaling_reports: list[dict] = []
        self._prepare_tensors_call_idx = 0
        # Walk-forward modunda True: fold başına scaler diske yazılmaz.
        # Aynı path'e tekrar tekrar yazılması (overwrite) inference için
        # hangi fold'un scaler'ının geçerli olduğunu belirsizleştirir.
        self._wf_mode: bool = False
        self._init_pipeline_services()

    def _init_pipeline_services(self) -> None:
        self.data_ingestion_service = DataIngestionService(self)
        self.tensor_preparation_service = TensorPreparationService(self)
        self.validation_split_service = ValidationSplitService(self)
        self.data_quality_service = DataQualityReportingService(self)

    def _ensure_pipeline_services(self) -> None:
        if not all(
            hasattr(self, attr)
            for attr in (
                "data_ingestion_service",
                "tensor_preparation_service",
                "validation_split_service",
                "data_quality_service",
            )
        ):
            self._init_pipeline_services()

    @staticmethod
    def _config_kwargs(config_cls, values: dict) -> dict:
        field_names = {field.name for field in fields(config_cls)}
        return {key: value for key, value in values.items() if key in field_names}

    @classmethod
    def _normalize_constructor_args(
        cls,
        *,
        data_cfg: DataConfig | None,
        val_cfg: ValidationConfig | None,
        models_dir: str | None,
        legacy_kwargs: dict,
    ) -> tuple[DataConfig, ValidationConfig, str]:
        """
        Backward-compatible constructor adapter.

        Older callers passed flat arguments such as data_file/test_ratio/time_steps
        directly to DataManager. Normalize those into config objects immediately so
        the rest of the class can keep using self.data_cfg/self.val_cfg.
        """
        if data_cfg is None:
            data_values = cls._config_kwargs(DataConfig, legacy_kwargs)
            if "data_file" not in data_values:
                raise TypeError("DataManager requires data_cfg or legacy data_file.")
            data_cfg = DataConfig(**data_values)
        elif not isinstance(data_cfg, DataConfig):
            raise TypeError("data_cfg must be a DataConfig instance.")

        if val_cfg is None:
            val_values = cls._config_kwargs(ValidationConfig, legacy_kwargs)
            val_cfg = ValidationConfig(**val_values)
        elif not isinstance(val_cfg, ValidationConfig):
            raise TypeError("val_cfg must be a ValidationConfig instance.")

        if models_dir is None:
            models_dir = legacy_kwargs.get("models_dir")
        if models_dir is None:
            models_dir = os.path.join("outputs", "_models")

        unknown = sorted(
            set(legacy_kwargs)
            - {field.name for field in fields(DataConfig)}
            - {field.name for field in fields(ValidationConfig)}
            - {"models_dir"}
        )
        if unknown:
            raise TypeError(f"Unsupported DataManager arguments: {unknown}")

        return data_cfg, val_cfg, models_dir

    def _ensure_config_objects(self) -> None:
        """
        Normalize legacy __new__-constructed test objects before using methods.
        Production construction goes through __init__; this protects older tests and
        external scripts that set flat attributes and call methods directly.
        """
        if not hasattr(self, "data_cfg"):
            self.data_cfg = DataConfig(
                data_file=getattr(self, "data_file", ""),
                test_ratio=getattr(self, "test_ratio", 0.20),
                time_steps=getattr(self, "time_steps", 30),
                target_mode=getattr(self, "target_mode", "log_return"),
                scaling_mode=getattr(self, "scaling_mode", "robust_x_standard_y_clip"),
                use_macro=getattr(self, "use_macro", True),
                universe_file=getattr(self, "universe_file", "data/bist_universe.csv"),
                clip_shift_warning_threshold_pct=getattr(
                    self, "clip_shift_warning_threshold_pct", 1.0
                ),
                training_window_years=getattr(self, "training_window_years", 5),
                window_candidates=getattr(self, "window_candidates", None) or [3, 5, 7, 10, None],
                min_history_days=getattr(self, "min_history_days", 504),
                new_listing_min_days=getattr(self, "new_listing_min_days", 252),
                auto_update_data=getattr(self, "auto_update_data", False),
                auto_update_interactive=getattr(self, "auto_update_interactive", False),
            )
        if not hasattr(self, "val_cfg"):
            self.val_cfg = ValidationConfig(
                validation_mode=getattr(self, "validation_mode", "single_split"),
                final_holdout_size=getattr(self, "final_holdout_size", 60),
            )
        if not hasattr(self, "models_dir"):
            self.models_dir = getattr(self, "models_dir", os.path.join("outputs", "_models"))
        if not hasattr(self, "scaling_reports"):
            self.scaling_reports = []
        if not hasattr(self, "_prepare_tensors_call_idx"):
            self._prepare_tensors_call_idx = 0
        if not hasattr(self, "_wf_mode"):
            self._wf_mode = False
        if not hasattr(self, "validation_config"):
            effective_wf_embargo_size = (
                self.data_cfg.time_steps
                if self.val_cfg.wf_embargo_size is None
                else max(0, int(self.val_cfg.wf_embargo_size))
            )
            wf_max_train_size = (
                None
                if self.val_cfg.wf_window_type == "expanding"
                else self.val_cfg.wf_max_train_size
            )
            self.validation_config = {
                "wf_n_splits": self.val_cfg.wf_n_splits,
                "wf_min_train_size": self.val_cfg.wf_min_train_size,
                "wf_test_size": self.val_cfg.wf_test_size,
                "wf_max_train_size": wf_max_train_size,
                "wf_window_type": self.val_cfg.wf_window_type,
                "wf_embargo_size": effective_wf_embargo_size,
                "final_holdout_size": self.val_cfg.final_holdout_size,
            }

    # ── Veri Yükleme & Özellik Mühendisliği ──────────────────────────────────
    def ingest_and_engineer(self) -> None:
        self._ensure_pipeline_services()
        return self.data_ingestion_service.run()

    def _apply_training_window(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_pipeline_services()
        return self.data_ingestion_service.apply_training_window(raw_df)

    def _format_window_candidates(self) -> list[str]:
        self._ensure_pipeline_services()
        return self.data_ingestion_service.format_window_candidates()

    def _check_survivorship_bias(self) -> dict:
        self._ensure_pipeline_services()
        return self.data_quality_service.check_survivorship_bias()

    def _fetch_macro(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_pipeline_services()
        return self.data_ingestion_service.fetch_macro(raw_df)

    def _refresh_dataset_metadata(self) -> None:
        self._ensure_pipeline_services()
        return self.data_ingestion_service.refresh_dataset_metadata()

    def build_run_metadata(
        self, validation_mode: str, model_config: dict | None = None
    ) -> tuple[dict, str]:
        run_metadata = dict(self.dataset_metadata)
        run_metadata["validation_mode"] = validation_mode
        run_metadata["selection_set"] = self._frame_metadata(self.selection_df, "selection")
        run_metadata["evaluation_set"] = self._frame_metadata(
            self.final_holdout_df, "final_holdout"
        )
        run_metadata["model_config"] = model_config or {}
        run_metadata["scaling_reports"] = self.scaling_reports
        run_metadata["nested_model_selection"] = {
            "hyperparameter_tuning_scope": "train_only_temporal_cv",
            "walk_forward_test_used_for_selection": validation_mode == "walk_forward",
            "final_holdout_used_for_selection": False,
        }
        if self.final_holdout_df is not None and not self.final_holdout_df.empty:
            run_metadata["final_holdout"] = {
                "rows": len(self.final_holdout_df),
                "date_start": pd.to_datetime(self.final_holdout_df["Date"].iloc[0]).strftime(
                    "%Y-%m-%d"
                ),
                "date_end": pd.to_datetime(self.final_holdout_df["Date"].iloc[-1]).strftime(
                    "%Y-%m-%d"
                ),
                "used_for_model_selection": False,
            }
        else:
            run_metadata["final_holdout"] = {
                "rows": 0,
                "used_for_model_selection": False,
            }
        payload = json.dumps(run_metadata, ensure_ascii=False, sort_keys=True)
        run_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return run_metadata, run_hash

    @staticmethod
    def _frame_metadata(df: pd.DataFrame | None, label: str) -> dict:
        if df is None or df.empty:
            return {"label": label, "rows": 0}
        return {
            "label": label,
            "rows": len(df),
            "date_start": (
                pd.to_datetime(df["Date"].iloc[0]).strftime("%Y-%m-%d")
                if "Date" in df.columns
                else "N/A"
            ),
            "date_end": (
                pd.to_datetime(df["Date"].iloc[-1]).strftime("%Y-%m-%d")
                if "Date" in df.columns
                else "N/A"
            ),
        }

    # ── Tensör Hazırlama ──────────────────────────────────────────────────────
    def _build_target_series(self, close_values: np.ndarray) -> np.ndarray:
        self._ensure_pipeline_services()
        return self.tensor_preparation_service.build_target_series(close_values)

    @staticmethod
    def _scale_data_compat(*args, **kwargs):
        return TensorPreparationService.scale_data_compat(*args, **kwargs)

    def prepare_tensors(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
    ) -> dict:
        self._ensure_pipeline_services()
        return self.tensor_preparation_service.prepare_tensors(
            train_df, test_df, context_df=context_df
        )

    def _record_scaling_report(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, scaler_X: object
    ) -> None:
        self._ensure_pipeline_services()
        return self.tensor_preparation_service.record_scaling_report(train_df, test_df, scaler_X)

    def split_data(self, validation_mode: str) -> None:
        self._ensure_pipeline_services()
        return self.validation_split_service.split_data(validation_mode)

    def get_validation_protocol_data(self) -> pd.DataFrame:
        self._ensure_pipeline_services()
        return self.validation_split_service.get_validation_protocol_data()

    def get_data_quality_reports(self) -> dict:
        self._ensure_pipeline_services()
        return self.data_quality_service.get_data_quality_reports()
