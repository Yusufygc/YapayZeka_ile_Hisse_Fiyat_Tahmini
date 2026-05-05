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
        data_cfg:        DataConfig | None = None,
        val_cfg:         ValidationConfig | None = None,
        models_dir:      str | None = None,
        macro_cache_dir: str  = None,
        **legacy_kwargs,
    ):
        data_cfg, val_cfg, models_dir = self._normalize_constructor_args(
            data_cfg=data_cfg,
            val_cfg=val_cfg,
            models_dir=models_dir,
            legacy_kwargs=legacy_kwargs,
        )
        self.data_cfg   = data_cfg
        self.val_cfg    = val_cfg
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
        self.df:            pd.DataFrame = None
        self.feature_names: list         = []
        self.tensors:       dict         = {}
        self.wf_splits:     list         = []
        self.selection_df: pd.DataFrame | None = None
        self.final_holdout_df: pd.DataFrame | None = None

        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.universe_file = self.data_cfg.universe_file
        if self.universe_file and not os.path.isabs(self.universe_file):
            self.universe_file = os.path.join(self.project_root, self.universe_file)

        effective_wf_embargo_size = self.data_cfg.time_steps if self.val_cfg.wf_embargo_size is None else max(0, int(self.val_cfg.wf_embargo_size))
        
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
        self.dataset_metadata: dict      = {}
        self.dataset_hash:  str          = "N/A"
        self.corporate_action_report: dict = {}
        self.feature_groups: dict[str, str] = {}
        self.feature_pruning_report: dict = {}
        self.survivorship_bias_report: dict = {}
        self.training_window_report: dict = {}
        self.scaling_reports: list[dict] = []
        self._prepare_tensors_call_idx = 0
        # Walk-forward modunda True: fold başına scaler diske yazılmaz.
        # Aynı path'e tekrar tekrar yazılması (overwrite) inference için
        # hangi fold'un scaler'ının geçerli olduğunu belirsizleştirir.
        self._wf_mode: bool = False

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
                clip_shift_warning_threshold_pct=getattr(self, "clip_shift_warning_threshold_pct", 1.0),
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
            wf_max_train_size = None if self.val_cfg.wf_window_type == "expanding" else self.val_cfg.wf_max_train_size
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
        print("\n" + "=" * 60)
        print("  ADIM 1 | Veri Yükleme & Özellik Mühendisliği (DataManager)")
        print("=" * 60)

        # Ham hisse verisi
        if self.data_cfg.auto_update_data:
            DataUpdater.check_and_update(
                self.data_cfg.data_file,
                self.stock_symbol,
                interactive=self.data_cfg.auto_update_interactive,
            )
        raw_df = load_data(self.data_cfg.data_file)
        self.corporate_action_report = dict(raw_df.attrs.get("corporate_action_report", {}))
        raw_df = self._apply_training_window(raw_df)

        # Makro veri (isteğe bağlı)
        macro_df = None
        if self.data_cfg.use_macro:
            macro_df = self._fetch_macro(raw_df)

        # Teknik + makro özellikler (cache destekli)
        feature_cache_dir = os.path.join(self.project_root, "data", "feature_cache")
        _cache = FeatureCache(cache_dir=feature_cache_dir, ttl_hours=24.0)
        _cache_key = _cache.make_key(self.data_cfg.data_file, self.data_cfg)
        _cache_hit = _cache.get(_cache_key)

        if _cache_hit is not None:
            self.df, _meta = _cache_hit
            self.feature_names = _meta["feature_names"]
            self.feature_groups = _meta.get("feature_groups", {})
            self.feature_pruning_report = _meta.get("feature_pruning_report", {})
            print("  [CACHE] Ozellik muhendisligi cache\'den yuklendi.")
        else:
            feature_pipeline = FeaturePipeline(
                feature_mode=self.data_cfg.feature_mode,
                prune_correlated_features=self.data_cfg.prune_correlated_features,
                correlation_threshold=self.data_cfg.correlation_threshold,
                lag_feature_count=self.data_cfg.lag_feature_count,
            )
            self.df = feature_pipeline.engineer_features(raw_df, macro_df=macro_df)
            self.feature_names = feature_pipeline.feature_names
            self.feature_groups = feature_pipeline.feature_groups
            self.feature_pruning_report = feature_pipeline.pruning_report
            _cache.put(_cache_key, self.df, {
                "feature_names": self.feature_names,
                "feature_groups": self.feature_groups,
                "feature_pruning_report": self.feature_pruning_report,
            })

        self.survivorship_bias_report = self._check_survivorship_bias()

        # Özet
        has_rel_str  = "Relative_Strength" in self.feature_names
        macro_base   = len(MacroPipeline.macro_feature_names(include_rates=True))
        macro_count  = macro_base + (1 if has_rel_str else 0)
        tech_count   = len(self.feature_names) - (macro_count if self.data_cfg.use_macro and macro_df is not None and not macro_df.empty else 0)

        print(f"  Veri boyutu      : {self.df.shape[0]} satır × {self.df.shape[1]} sütun")
        print(f"  Teknik özellikler: {tech_count}")
        if self.data_cfg.use_macro and macro_df is not None and not macro_df.empty:
            print(f"  Makro özellikler : {macro_count}  "
                  f"(USDTRY_Return, USDTRY_MA7, USDTRY_Volatility7, "
                  f"BIST100_Norm, BIST100_Return, BIST100_MA7, "
                  f"Rate_Level, Rate_Change, CPI_YoY, CPI_MoM, Real_Rate, "
                  f"Relative_Strength)")
        print(f"  Toplam özellik   : {len(self.feature_names)}")
        if self.corporate_action_report.get("warning"):
            print(f"  [DATA] Uyari       : {self.corporate_action_report['warning']}")
        elif self.corporate_action_report:
            print(
                "  [DATA] Price source : "
                f"{self.corporate_action_report.get('price_source')} "
                f"(max Close/Adj diff={self.corporate_action_report.get('max_abs_adj_close_diff_pct', 0):.4f}%)"
            )
        if self.feature_pruning_report.get("enabled"):
            print(
                "  [FEATURE] Pruning   : "
                f"{len(self.feature_pruning_report.get('dropped_features', []))} feature dropped"
            )
        if self.training_window_report:
            print(
                "  [DATA] Train window  : "
                f"{self.training_window_report.get('effective_training_window_years_label')} "
                f"({self.training_window_report.get('effective_date_start')} -> "
                f"{self.training_window_report.get('effective_date_end')}, "
                f"{self.training_window_report.get('history_days')} satir)"
            )
        self._refresh_dataset_metadata()

    def _apply_training_window(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        cfg = getattr(self, "data_cfg", self)
        if raw_df is None or raw_df.empty:
            self.training_window_report = {
                "status": "empty_dataset",
                "requested_training_window_years": cfg.training_window_years,
                "window_candidates": self._format_window_candidates(),
                "min_history_days": cfg.min_history_days,
                "new_listing_min_days": cfg.new_listing_min_days,
            }
            return raw_df

        df = raw_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.attrs.update(raw_df.attrs)

        raw_start = pd.to_datetime(df["Date"].iloc[0]).normalize()
        raw_end = pd.to_datetime(df["Date"].iloc[-1]).normalize()
        raw_history_days = int(len(df))
        new_listing_mode = raw_history_days < cfg.new_listing_min_days
        insufficient_history = raw_history_days < cfg.min_history_days

        effective_df = df
        effective_years = cfg.training_window_years
        effective_label = "all"
        cutoff_date = None
        status = "all_data_used"

        if cfg.training_window_years is not None:
            window_years = int(cfg.training_window_years)
            cutoff_date = raw_end - pd.DateOffset(years=window_years)
            candidate_df = df[df["Date"] >= cutoff_date].copy()
            if len(candidate_df) >= cfg.min_history_days and len(candidate_df) < len(df):
                effective_df = candidate_df
                effective_df.reset_index(drop=True, inplace=True)
                effective_df.attrs.update(raw_df.attrs)
                effective_label = f"{window_years}y"
                status = "window_applied"
            elif len(candidate_df) < cfg.min_history_days:
                effective_years = None
                status = "window_skipped_min_history"
            else:
                effective_years = None
                effective_label = "all"
                status = "window_not_needed"
        else:
            effective_years = None

        effective_start = pd.to_datetime(effective_df["Date"].iloc[0]).normalize()
        effective_end = pd.to_datetime(effective_df["Date"].iloc[-1]).normalize()
        self.training_window_report = {
            "status": status,
            "raw_date_start": raw_start.strftime("%Y-%m-%d"),
            "raw_date_end": raw_end.strftime("%Y-%m-%d"),
            "raw_history_days": raw_history_days,
            "requested_training_window_years": cfg.training_window_years,
            "effective_training_window_years": effective_years,
            "effective_training_window_years_label": effective_label,
            "effective_date_start": effective_start.strftime("%Y-%m-%d"),
            "effective_date_end": effective_end.strftime("%Y-%m-%d"),
            "history_days": int(len(effective_df)),
            "min_history_days": cfg.min_history_days,
            "new_listing_min_days": cfg.new_listing_min_days,
            "new_listing_mode": bool(new_listing_mode),
            "insufficient_history_warning": bool(insufficient_history),
            "window_candidates": self._format_window_candidates(),
            "cutoff_date": "" if cutoff_date is None else pd.to_datetime(cutoff_date).strftime("%Y-%m-%d"),
            "filter_stage": "raw_after_load_before_feature_engineering",
        }
        return effective_df

    def _format_window_candidates(self) -> list[str]:
        cfg = getattr(self, "data_cfg", self)
        return ["all" if years is None else f"{int(years)}y" for years in cfg.window_candidates]

    def _check_survivorship_bias(self) -> dict:
        if self.df is None or self.df.empty:
            return {"survivorship_bias_warning": True, "status": "empty_dataset"}

        date_start = pd.to_datetime(self.df["Date"].iloc[0]).normalize()
        date_end = pd.to_datetime(self.df["Date"].iloc[-1]).normalize()
        report = {
            "universe_file": self.universe_file,
            "symbol": self.stock_symbol,
            "date_start": date_start.strftime("%Y-%m-%d"),
            "date_end": date_end.strftime("%Y-%m-%d"),
            "required_schema": "Symbol,Listed_Date,Delisted_Date,Status",
        }

        if not self.universe_file or not os.path.exists(self.universe_file):
            report.update({
                "universe_file_exists": False,
                "symbol_found": False,
                "coverage_ok": None,
                "survivorship_bias_warning": True,
                "status": "missing_universe_file",
            })
            print("  [DATA] Survivorship bias kontrolu: universe dosyasi yok, uyari kaydedildi.")
            return report

        try:
            universe = pd.read_csv(self.universe_file)
            required = {"Symbol", "Listed_Date", "Delisted_Date", "Status"}
            missing = sorted(required - set(universe.columns))
            if missing:
                report.update({
                    "universe_file_exists": True,
                    "symbol_found": False,
                    "coverage_ok": False,
                    "survivorship_bias_warning": True,
                    "status": f"invalid_schema_missing_{','.join(missing)}",
                })
                return report

            symbol_rows = universe[universe["Symbol"].astype(str).str.upper() == self.stock_symbol.upper()].copy()
            if symbol_rows.empty:
                report.update({
                    "universe_file_exists": True,
                    "symbol_found": False,
                    "coverage_ok": False,
                    "survivorship_bias_warning": True,
                    "status": "symbol_not_found",
                })
                return report

            row = symbol_rows.iloc[0]
            listed = pd.to_datetime(row.get("Listed_Date"), errors="coerce")
            delisted = pd.to_datetime(row.get("Delisted_Date"), errors="coerce")
            listed_ok = pd.isna(listed) or listed.normalize() <= date_start
            delisted_ok = pd.isna(delisted) or delisted.normalize() >= date_end
            coverage_ok = bool(listed_ok and delisted_ok)
            report.update({
                "universe_file_exists": True,
                "symbol_found": True,
                "listed_date": "" if pd.isna(listed) else listed.strftime("%Y-%m-%d"),
                "delisted_date": "" if pd.isna(delisted) else delisted.strftime("%Y-%m-%d"),
                "status_value": row.get("Status", ""),
                "coverage_ok": coverage_ok,
                "survivorship_bias_warning": not coverage_ok,
                "status": "covered" if coverage_ok else "symbol_not_listed_for_full_period",
            })
            return report
        except Exception as exc:
            report.update({
                "universe_file_exists": True,
                "symbol_found": False,
                "coverage_ok": False,
                "survivorship_bias_warning": True,
                "status": f"universe_read_failed: {exc}",
            })
            return report

    def _fetch_macro(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """MacroPipeline'ı çağırır; başarısız olursa boş DataFrame döner."""
        try:
            dates = pd.to_datetime(raw_df["Date"])
            start = dates.min().strftime("%Y-%m-%d")
            end   = dates.max().strftime("%Y-%m-%d")

            mp = MacroPipeline(
                cache_dir=self.macro_cache_dir,
                rate_release_lag_days=self.data_cfg.macro_rate_lag_days,
                cpi_release_lag_days=self.data_cfg.macro_cpi_lag_days,
            )
            macro_df = mp.get_macro_features(start_date=start, end_date=end)

            if macro_df.empty:
                print("  [MACRO] Makro veri boş döndü, teknik özellikler tek başına kullanılacak.")
            return macro_df

        except Exception as exc:
            print(f"  [MACRO] Makro veri alınamadı ({exc}), devam ediliyor.")
            return pd.DataFrame()

    def _refresh_dataset_metadata(self) -> None:
        if self.df is None or self.df.empty:
            self.dataset_metadata = {}
            self.dataset_hash = "N/A"
            return

        date_start = pd.to_datetime(self.df["Date"].iloc[0]).strftime("%Y-%m-%d")
        date_end = pd.to_datetime(self.df["Date"].iloc[-1]).strftime("%Y-%m-%d")
        self.dataset_metadata = {
            "stock_symbol": self.stock_symbol,
            "target_mode": self.data_cfg.target_mode,
            "feature_mode": self.data_cfg.feature_mode,
            "scaling_mode": self.data_cfg.scaling_mode,
            "target_semantics": "X[t] uses information known after close t; y[t] is t+1 return/price.",
            "execution_lag": "Signals generated after close t are applied to the aligned next realized bar.",
            "macro_release_lag": {
                "rate_days": self.data_cfg.macro_rate_lag_days,
                "cpi_days": self.data_cfg.macro_cpi_lag_days,
            },
            "validation_config": self.validation_config,
            "date_range": f"{date_start}:{date_end}",
            "features_count": len(self.feature_names),
            "features": self.feature_names,
            "feature_groups": self.feature_groups,
            "feature_pruning": self.feature_pruning_report,
            "corporate_action": self.corporate_action_report,
            "survivorship_bias": self.survivorship_bias_report,
            "training_window": self.training_window_report,
        }
        payload = json.dumps(self.dataset_metadata, ensure_ascii=False, sort_keys=True)
        self.dataset_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def build_run_metadata(self, validation_mode: str, model_config: dict | None = None) -> tuple[dict, str]:
        run_metadata = dict(self.dataset_metadata)
        run_metadata["validation_mode"] = validation_mode
        run_metadata["selection_set"] = self._frame_metadata(self.selection_df, "selection")
        run_metadata["evaluation_set"] = self._frame_metadata(self.final_holdout_df, "final_holdout")
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
                "date_start": pd.to_datetime(self.final_holdout_df["Date"].iloc[0]).strftime("%Y-%m-%d"),
                "date_end": pd.to_datetime(self.final_holdout_df["Date"].iloc[-1]).strftime("%Y-%m-%d"),
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
            "date_start": pd.to_datetime(df["Date"].iloc[0]).strftime("%Y-%m-%d") if "Date" in df.columns else "N/A",
            "date_end": pd.to_datetime(df["Date"].iloc[-1]).strftime("%Y-%m-%d") if "Date" in df.columns else "N/A",
        }

    # ── Tensör Hazırlama ──────────────────────────────────────────────────────
    def _build_target_series(
        self,
        close_values: np.ndarray,
    ) -> np.ndarray:
        self._ensure_config_objects()
        if self.data_cfg.target_mode == "log_return":
            return np.log(close_values[1:] / close_values[:-1])
        if self.data_cfg.target_mode == "return":
            return (close_values[1:] / close_values[:-1]) - 1.0
        if self.data_cfg.target_mode == "price":
            return close_values[1:]
        raise ValueError(
            f"Desteklenmeyen target_mode: {self.data_cfg.target_mode}. "
            "Beklenen: price, return, log_return"
        )

    @staticmethod
    def _scale_data_compat(*args, **kwargs):
        try:
            return scale_data(*args, **kwargs)
        except TypeError as exc:
            if "save_scaler" not in str(exc) or "save_scaler" not in kwargs:
                raise
            legacy_kwargs = dict(kwargs)
            legacy_kwargs.pop("save_scaler", None)
            return scale_data(*args, **legacy_kwargs)

    def prepare_tensors(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Train/test DataFrame'lerini model eğitimine uygun tensörlere çevirir.
        Scaler yalnızca train verisi üzerine fit edilir.

        Semantik (v3):
          X[t] = t gününün sonunda bilinen özellikler
          y[t] = t+1 gününün log-getirisi

        Böylece aynı gün bilgisinden aynı gün hedef üretilmez. Tree ve sequence
        modeller aynı tahmin problemi üzerinde çalışır.
        """
        self._ensure_config_objects()
        exclude = {"Date", "Close"}
        features = [c for c in train_df.columns if c not in exclude]

        train_close = train_df["Close"].values.astype(float)
        test_close = test_df["Close"].values.astype(float)
        context_df = context_df.copy() if context_df is not None and not context_df.empty else None

        if len(train_df) < 2 or len(test_df) < 2:
            raise ValueError("t+1 hedefi için train ve test setlerinde en az 2 satır gerekir.")

        # X[t] -> y[t] = hedef(Close[t+1], Close[t])
        X_train = train_df[features].iloc[:-1].values
        X_test = test_df[features].iloc[:-1].values

        train_target = self._build_target_series(train_close)
        test_target = self._build_target_series(test_close)
        test_prev_close = test_close[:-1]

        y_train = train_target.reshape(-1, 1)
        y_test = test_target.reshape(-1, 1)
        if "Market_Regime_SMA200" in test_df.columns:
            market_regime_test = test_df["Market_Regime_SMA200"].iloc[:-1].to_numpy(dtype=float)
        else:
            market_regime_test = np.zeros(len(y_test), dtype=float)

        X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = self._scale_data_compat(
            X_train, X_test, y_train, y_test,
            save_dir=self.models_dir,
            scaling_mode=self.data_cfg.scaling_mode,
            save_scaler=not self._wf_mode,  # WF fold'larında diske yazma
        )
        self._record_scaling_report(train_df, test_df, scaler_X)

        X_train_seq, y_train_seq = create_sequences(
            X_train_s, y_train_s, time_steps=self.data_cfg.time_steps
        )

        # İlk test örneği için son time_steps-1 train günü + ilk test günü kullanılır.
        prefix_len = max(0, self.data_cfg.time_steps - 1)
        if context_df is not None and prefix_len > 0:
            prefix_source = pd.concat([train_df.tail(prefix_len), context_df], ignore_index=True)
            X_prefix_raw = prefix_source[features].tail(prefix_len).values
            X_prefix_s = scaler_X.transform(X_prefix_raw)
            clip_report = getattr(scaler_X, "clip_report_", {}) or {}
            if clip_report.get("clip_low") is not None and clip_report.get("clip_high") is not None:
                X_prefix_s = np.clip(X_prefix_s, clip_report["clip_low"], clip_report["clip_high"])
            y_prefix_s = np.zeros((len(X_prefix_s), 1), dtype=float)
        else:
            X_prefix_s = X_train_s[-prefix_len:] if prefix_len else np.empty((0, X_train_s.shape[1]))
            y_prefix_s = y_train_s[-prefix_len:] if prefix_len else np.empty((0, 1))

        X_test_input = np.vstack((X_prefix_s, X_test_s))
        y_test_input = np.vstack((y_prefix_s, y_test_s))
        X_test_seq, y_test_seq = create_sequences(
            X_test_input, y_test_input, time_steps=self.data_cfg.time_steps
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_train_s": X_train_s,
            "y_train_s": y_train_s,
            "X_test_s": X_test_s,
            "y_test_s": y_test_s,
            "X_train_seq": X_train_seq,
            "y_train_seq": y_train_seq,
            "X_test_seq": X_test_seq,
            "y_test_seq": y_test_seq,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "clip_report": getattr(scaler_X, "clip_report_", {}),
            "context_rows": 0 if context_df is None else len(context_df),
            "original_y_test_aligned": test_close[1:],
            "prev_close_test": test_prev_close,
            "market_regime_test": market_regime_test,
            "train_close_last": float(train_close[-1]),
            "target_mode": self.data_cfg.target_mode,
            "dates_train": train_df["Date"].iloc[1:] if "Date" in train_df.columns else None,
            "dates_prediction": test_df["Date"].iloc[:-1] if "Date" in test_df.columns else None,
            "dates_test": test_df["Date"].iloc[1:] if "Date" in test_df.columns else None,
        }

    def _record_scaling_report(self, train_df: pd.DataFrame, test_df: pd.DataFrame, scaler_X: object) -> None:
        self._ensure_config_objects()
        clip_report = dict(getattr(scaler_X, "clip_report_", {}) or {})
        if not clip_report:
            clip_report = {
                "train_clip_rate_pct": 0.0,
                "test_clip_rate_pct": 0.0,
                "clip_low": None,
                "clip_high": None,
            }

        self._prepare_tensors_call_idx += 1
        train_clip = float(clip_report.get("train_clip_rate_pct", 0.0) or 0.0)
        test_clip = float(clip_report.get("test_clip_rate_pct", 0.0) or 0.0)
        warning = ""
        if test_clip >= self.data_cfg.clip_shift_warning_threshold_pct and test_clip > train_clip:
            warning = (
                "distribution_shift_warning: test clip rate is high relative to train "
                f"({test_clip:.3f}% vs {train_clip:.3f}%)."
            )
            print(f"  [WARN] {warning}")

        self.scaling_reports.append({
            "call_idx": self._prepare_tensors_call_idx,
            "scaler_fit_start": pd.to_datetime(train_df["Date"].iloc[0]).strftime("%Y-%m-%d") if "Date" in train_df.columns else "",
            "scaler_fit_end": pd.to_datetime(train_df["Date"].iloc[-1]).strftime("%Y-%m-%d") if "Date" in train_df.columns else "",
            "test_start": pd.to_datetime(test_df["Date"].iloc[0]).strftime("%Y-%m-%d") if "Date" in test_df.columns else "",
            "test_end": pd.to_datetime(test_df["Date"].iloc[-1]).strftime("%Y-%m-%d") if "Date" in test_df.columns else "",
            "train_clip_rate_pct": train_clip,
            "test_clip_rate_pct": test_clip,
            "clip_low": clip_report.get("clip_low"),
            "clip_high": clip_report.get("clip_high"),
            "warning": warning,
            "scaler_fit_scope": "train_only",
        })

    # ── Train/Test Bölme ──────────────────────────────────────────────────────
    def split_data(self, validation_mode: str) -> None:
        self._ensure_config_objects()
        print("\n" + "=" * 60)
        print("  ADIM 2 | Train/Test Split (DataManager)")
        print("=" * 60)

        if validation_mode == "single_split":
            self._wf_mode = False  # single split: scaler diske yazılır
            train_df, test_df, _, _ = TimeSeriesSplitter.single_split(
                self.df, test_ratio=self.data_cfg.test_ratio
            )
            self.selection_df = train_df.copy()
            self.final_holdout_df = test_df.copy()
            self.tensors = self.prepare_tensors(train_df, test_df)

        elif validation_mode == "walk_forward":
            self._wf_mode = True   # WF fold'larında scaler diske yazılmaz
            wf_source_df = self.df
            holdout_size = int(self.validation_config.get("final_holdout_size", 0) or 0)
            min_required = (
                self.validation_config["wf_min_train_size"]
                + self.validation_config["wf_embargo_size"]
                + self.validation_config["wf_test_size"] * self.validation_config["wf_n_splits"]
                + holdout_size
            )
            if holdout_size > 0 and len(self.df) >= min_required:
                wf_source_df = self.df.iloc[:-holdout_size].copy()
                self.selection_df = wf_source_df.copy()
                self.final_holdout_df = self.df.iloc[-holdout_size:].copy()
                print(
                    "  [INFO] Final untouched holdout ayrildi "
                    f"({len(self.final_holdout_df)} satir): "
                    f"{self.final_holdout_df['Date'].iloc[0]} -> {self.final_holdout_df['Date'].iloc[-1]}"
                )
            else:
                self.selection_df = wf_source_df.copy()
                self.final_holdout_df = None
                print("  [WARN] Final holdout icin yeterli veri yok; tum veri walk-forward seciminde kullanilacak.")

            self.wf_splits = TimeSeriesSplitter.walk_forward_splits(
                wf_source_df,
                n_splits=self.validation_config["wf_n_splits"],
                min_train_size=self.validation_config["wf_min_train_size"],
                test_size=self.validation_config["wf_test_size"],
                # Sliding window: yalnızca son ~2.7 yıl (~700 iş günü) kullanılır.
                # Durağan olmayan (trend gösteren) BIST hisselerinde MinMaxScaler'ın
                # neden olduğu sistematik küçük tahmini engeller.
                max_train_size=self.validation_config["wf_max_train_size"],
                embargo_size=self.validation_config["wf_embargo_size"],
            )
            print(
                "  [INFO] Walk-Forward splitleri olusturuldu "
                f"({len(self.wf_splits)} adet, {self.validation_config['wf_window_type']} window, "
                f"test={self.validation_config['wf_test_size']}, "
                f"max_train={self.validation_config['wf_max_train_size']})."
            )

    def get_validation_protocol_data(self) -> pd.DataFrame:
        rows = []
        for split in self.wf_splits:
            train_df = split["train"]
            test_df = split["test"]
            rows.append({
                "Split": split["split_idx"],
                "Protocol": "walk_forward",
                "Window_Type": self.validation_config["wf_window_type"],
                "Train_Rows": len(train_df),
                "Test_Rows": len(test_df),
                "Train_Date_Start": split.get("train_date_start"),
                "Train_Date_End": split.get("train_date_end"),
                "Embargo_Rows": len(split.get("embargo_context", [])),
                "Embargo_Date_Start": split.get("embargo_date_start"),
                "Embargo_Date_End": split.get("embargo_date_end"),
                "Test_Date_Start": split.get("test_date_start"),
                "Test_Date_End": split.get("test_date_end"),
                "Effective_Train_End": split.get("effective_train_end"),
                "Test_Start": split.get("test_start"),
                "Scaler_Fit_Start": split.get("train_date_start"),
                "Scaler_Fit_End": split.get("train_date_end"),
                "Features_Count": len(self.feature_names),
                "Features": ",".join(self.feature_names),
                "Selection_Set": "walk_forward_train_windows",
                "Evaluation_Set": "walk_forward_test_window",
                "Final_Holdout_Used_For_Selection": False,
            })

        if self.final_holdout_df is not None and not self.final_holdout_df.empty:
            rows.append({
                "Split": "final_holdout",
                "Protocol": "final_holdout",
                "Window_Type": self.validation_config["wf_window_type"],
                "Train_Rows": len(self.selection_df) if self.selection_df is not None else 0,
                "Test_Rows": len(self.final_holdout_df),
                "Train_Date_Start": self.selection_df["Date"].iloc[0] if self.selection_df is not None and not self.selection_df.empty else None,
                "Train_Date_End": self.selection_df["Date"].iloc[-1] if self.selection_df is not None and not self.selection_df.empty else None,
                "Test_Date_Start": self.final_holdout_df["Date"].iloc[0],
                "Test_Date_End": self.final_holdout_df["Date"].iloc[-1],
                "Scaler_Fit_Start": self.selection_df["Date"].iloc[0] if self.selection_df is not None and not self.selection_df.empty else None,
                "Scaler_Fit_End": self.selection_df["Date"].iloc[-1] if self.selection_df is not None and not self.selection_df.empty else None,
                "Features_Count": len(self.feature_names),
                "Features": ",".join(self.feature_names),
                "Selection_Set": "full_selection_period",
                "Evaluation_Set": "untouched_final_holdout",
                "Final_Holdout_Used_For_Selection": False,
            })

        return pd.DataFrame(rows)

    def get_data_quality_reports(self) -> dict:
        reports = {}

        reports["corporate_actions"] = [self.corporate_action_report or {}]

        feature_rows = [
            {"Feature": feature, "Feature_Group": group}
            for feature, group in sorted(self.feature_groups.items())
        ]
        reports["feature_groups"] = feature_rows

        dropped = self.feature_pruning_report.get("dropped_features", []) if self.feature_pruning_report else []
        pruning_rows = dropped or [{
            "feature": "",
            "correlated_with": "",
            "abs_corr": "",
            "enabled": bool(self.feature_pruning_report.get("enabled", False)) if self.feature_pruning_report else False,
            "threshold": self.data_cfg.correlation_threshold,
        }]
        reports["feature_pruning"] = pruning_rows

        reports["scaling_clip"] = self.scaling_reports
        reports["survivorship"] = [self.survivorship_bias_report or {}]
        reports["training_window"] = [self.training_window_report or {}]

        return reports
