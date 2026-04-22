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
import numpy as np
import pandas as pd

from src.data_updater import DataUpdater
from src.data_loader import load_data
from src.preprocessor import scale_data, create_sequences
from src.utils.data_splitter import TimeSeriesSplitter
from src.features.feature_pipeline import FeaturePipeline
from src.features.macro_pipeline import MacroPipeline


class DataManager:
    """
    Args:
        data_file      : Hisse senedi CSV dosyasının tam yolu.
        test_ratio     : Test seti oranı (0-1).
        time_steps     : Sequence uzunluğu (LSTM/TFT için).
        models_dir     : Scaler objelerinin kaydedileceği dizin.
        use_macro      : True ise USD/TRY + BIST100 makro özellikler eklenir.
        macro_cache_dir: Makro CSV cache dizini.
    """

    def __init__(
        self,
        data_file:       str,
        test_ratio:      float,
        time_steps:      int,
        models_dir:      str,
        use_macro:       bool = True,
        macro_cache_dir: str  = None,
        target_mode:     str  = "log_return",
        feature_mode:    str  = "stationary_features",
        scaling_mode:    str  = "robust_x_standard_y_clip",
        macro_rate_lag_days: int = 1,
        macro_cpi_lag_days: int = 15,
        wf_n_splits: int = 12,
        wf_min_train_size: int = 504,
        wf_test_size: int = 21,
        wf_max_train_size: int | None = 756,
        wf_window_type: str = "sliding",
        final_holdout_size: int = 60,
        prune_correlated_features: bool = False,
        correlation_threshold: float = 0.98,
        clip_shift_warning_threshold_pct: float = 1.0,
    ):
        self.data_file  = data_file
        self.test_ratio = test_ratio
        self.time_steps = time_steps
        self.models_dir = models_dir
        self.use_macro  = use_macro

        self.stock_symbol = os.path.splitext(os.path.basename(self.data_file))[0]

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
        self.target_mode:   str          = target_mode
        self.feature_mode:  str          = feature_mode
        self.scaling_mode:  str          = scaling_mode
        self.macro_rate_lag_days = macro_rate_lag_days
        self.macro_cpi_lag_days = macro_cpi_lag_days
        self.prune_correlated_features = prune_correlated_features
        self.correlation_threshold = correlation_threshold
        self.clip_shift_warning_threshold_pct = clip_shift_warning_threshold_pct
        if wf_window_type not in {"sliding", "expanding"}:
            raise ValueError("wf_window_type 'sliding' veya 'expanding' olmalidir.")
        if wf_window_type == "expanding":
            wf_max_train_size = None

        self.validation_config = {
            "wf_n_splits": wf_n_splits,
            "wf_min_train_size": wf_min_train_size,
            "wf_test_size": wf_test_size,
            "wf_max_train_size": wf_max_train_size,
            "wf_window_type": wf_window_type,
            "final_holdout_size": final_holdout_size,
        }
        self.dataset_metadata: dict      = {}
        self.dataset_hash:  str          = "N/A"
        self.corporate_action_report: dict = {}
        self.feature_groups: dict[str, str] = {}
        self.feature_pruning_report: dict = {}
        self.scaling_reports: list[dict] = []
        self._prepare_tensors_call_idx = 0

    # ── Veri Yükleme & Özellik Mühendisliği ──────────────────────────────────
    def ingest_and_engineer(self) -> None:
        print("\n" + "=" * 60)
        print("  ADIM 1 | Veri Yükleme & Özellik Mühendisliği (DataManager)")
        print("=" * 60)

        # Ham hisse verisi
        DataUpdater.check_and_update(self.data_file, self.stock_symbol)
        raw_df = load_data(self.data_file)
        self.corporate_action_report = dict(raw_df.attrs.get("corporate_action_report", {}))

        # Makro veri (isteğe bağlı)
        macro_df = None
        if self.use_macro:
            macro_df = self._fetch_macro(raw_df)

        # Teknik + makro özellikler
        feature_pipeline  = FeaturePipeline(
            feature_mode=self.feature_mode,
            prune_correlated_features=self.prune_correlated_features,
            correlation_threshold=self.correlation_threshold,
        )
        self.df           = feature_pipeline.engineer_features(raw_df, macro_df=macro_df)
        self.feature_names = feature_pipeline.feature_names
        self.feature_groups = feature_pipeline.feature_groups
        self.feature_pruning_report = feature_pipeline.pruning_report

        # Özet
        has_rel_str  = "Relative_Strength" in self.feature_names
        macro_base   = len(MacroPipeline.macro_feature_names(include_rates=True))
        macro_count  = macro_base + (1 if has_rel_str else 0)
        tech_count   = len(self.feature_names) - (macro_count if self.use_macro and macro_df is not None and not macro_df.empty else 0)

        print(f"  Veri boyutu      : {self.df.shape[0]} satır × {self.df.shape[1]} sütun")
        print(f"  Teknik özellikler: {tech_count}")
        if self.use_macro and macro_df is not None and not macro_df.empty:
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
        self._refresh_dataset_metadata()

    def _fetch_macro(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """MacroPipeline'ı çağırır; başarısız olursa boş DataFrame döner."""
        try:
            dates = pd.to_datetime(raw_df["Date"])
            start = dates.min().strftime("%Y-%m-%d")
            end   = dates.max().strftime("%Y-%m-%d")

            mp = MacroPipeline(
                cache_dir=self.macro_cache_dir,
                rate_release_lag_days=self.macro_rate_lag_days,
                cpi_release_lag_days=self.macro_cpi_lag_days,
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
            "target_mode": self.target_mode,
            "feature_mode": self.feature_mode,
            "scaling_mode": self.scaling_mode,
            "target_semantics": "X[t] uses information known after close t; y[t] is t+1 return/price.",
            "execution_lag": "Signals generated after close t are evaluated on the next realized bar.",
            "macro_release_lag": {
                "rate_days": self.macro_rate_lag_days,
                "cpi_days": self.macro_cpi_lag_days,
            },
            "validation_config": self.validation_config,
            "date_range": f"{date_start}:{date_end}",
            "features_count": len(self.feature_names),
            "features": self.feature_names,
            "feature_groups": self.feature_groups,
            "feature_pruning": self.feature_pruning_report,
            "corporate_action": self.corporate_action_report,
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
        if self.target_mode == "log_return":
            return np.log(close_values[1:] / close_values[:-1])
        if self.target_mode == "return":
            return (close_values[1:] / close_values[:-1]) - 1.0
        if self.target_mode == "price":
            return close_values[1:]
        raise ValueError(
            f"Desteklenmeyen target_mode: {self.target_mode}. "
            "Beklenen: price, return, log_return"
        )

    def prepare_tensors(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """
        Train/test DataFrame'lerini model eğitimine uygun tensörlere çevirir.
        Scaler yalnızca train verisi üzerine fit edilir.

        Semantik (v3):
          X[t] = t gününün sonunda bilinen özellikler
          y[t] = t+1 gününün log-getirisi

        Böylece aynı gün bilgisinden aynı gün hedef üretilmez. Tree ve sequence
        modeller aynı tahmin problemi üzerinde çalışır.
        """
        exclude = {"Date", "Close"}
        features = [c for c in train_df.columns if c not in exclude]

        train_close = train_df["Close"].values.astype(float)
        test_close = test_df["Close"].values.astype(float)

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

        X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = scale_data(
            X_train, X_test, y_train, y_test,
            save_dir=self.models_dir,
            scaling_mode=self.scaling_mode,
        )
        self._record_scaling_report(train_df, test_df, scaler_X)

        X_train_seq, y_train_seq = create_sequences(
            X_train_s, y_train_s, time_steps=self.time_steps
        )

        # İlk test örneği için son time_steps-1 train günü + ilk test günü kullanılır.
        prefix_len = max(0, self.time_steps - 1)
        X_test_input = np.vstack((X_train_s[-prefix_len:], X_test_s))
        y_test_input = np.vstack((y_train_s[-prefix_len:], y_test_s))
        X_test_seq, y_test_seq = create_sequences(
            X_test_input, y_test_input, time_steps=self.time_steps
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
            "original_y_test_aligned": test_close[1:],
            "prev_close_test": test_prev_close,
            "train_close_last": float(train_close[-1]),
            "target_mode": self.target_mode,
            "dates_train": train_df["Date"].iloc[1:] if "Date" in train_df.columns else None,
            "dates_prediction": test_df["Date"].iloc[:-1] if "Date" in test_df.columns else None,
            "dates_test": test_df["Date"].iloc[1:] if "Date" in test_df.columns else None,
        }

    def _record_scaling_report(self, train_df: pd.DataFrame, test_df: pd.DataFrame, scaler_X: object) -> None:
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
        if test_clip >= self.clip_shift_warning_threshold_pct and test_clip > train_clip:
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
        print("\n" + "=" * 60)
        print("  ADIM 2 | Train/Test Split (DataManager)")
        print("=" * 60)

        if validation_mode == "single_split":
            train_df, test_df, _, _ = TimeSeriesSplitter.single_split(
                self.df, test_ratio=self.test_ratio
            )
            self.selection_df = train_df.copy()
            self.final_holdout_df = test_df.copy()
            self.tensors = self.prepare_tensors(train_df, test_df)

        elif validation_mode == "walk_forward":
            wf_source_df = self.df
            holdout_size = int(self.validation_config.get("final_holdout_size", 0) or 0)
            min_required = (
                self.validation_config["wf_min_train_size"]
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
            )
            print(
                "  [INFO] Walk-Forward splitleri olusturuldu "
                f"({len(self.wf_splits)} adet, {self.validation_config['wf_window_type']} window, "
                f"test={self.validation_config['wf_test_size']}, "
                f"max_train={self.validation_config['wf_max_train_size']})."
            )

    def save_validation_protocol_report(self, outputs_dir: str) -> str:
        os.makedirs(outputs_dir, exist_ok=True)
        report_path = os.path.join(outputs_dir, "validation_protocol_v1.csv")
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
                "Test_Date_Start": split.get("test_date_start"),
                "Test_Date_End": split.get("test_date_end"),
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

        pd.DataFrame(rows).to_csv(report_path, sep=";", index=False, encoding="utf-8-sig")
        print(f"  [INFO] Validation protocol raporu kaydedildi -> {report_path}")
        return report_path

    def save_data_quality_reports(self, outputs_dir: str) -> dict:
        os.makedirs(outputs_dir, exist_ok=True)
        paths = {}

        corporate_path = os.path.join(outputs_dir, "data_quality_corporate_actions_v1.csv")
        pd.DataFrame([self.corporate_action_report or {}]).to_csv(
            corporate_path,
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )
        paths["corporate_actions"] = corporate_path

        feature_rows = [
            {"Feature": feature, "Feature_Group": group}
            for feature, group in sorted(self.feature_groups.items())
        ]
        feature_path = os.path.join(outputs_dir, "feature_groups_v1.csv")
        pd.DataFrame(feature_rows).to_csv(feature_path, sep=";", index=False, encoding="utf-8-sig")
        paths["feature_groups"] = feature_path

        pruning_path = os.path.join(outputs_dir, "feature_pruning_v1.csv")
        dropped = self.feature_pruning_report.get("dropped_features", []) if self.feature_pruning_report else []
        pruning_rows = dropped or [{
            "feature": "",
            "correlated_with": "",
            "abs_corr": "",
            "enabled": bool(self.feature_pruning_report.get("enabled", False)) if self.feature_pruning_report else False,
            "threshold": self.correlation_threshold,
        }]
        pd.DataFrame(pruning_rows).to_csv(pruning_path, sep=";", index=False, encoding="utf-8-sig")
        paths["feature_pruning"] = pruning_path

        scaling_path = os.path.join(outputs_dir, "scaling_clip_report_v1.csv")
        pd.DataFrame(self.scaling_reports).to_csv(scaling_path, sep=";", index=False, encoding="utf-8-sig")
        paths["scaling_clip"] = scaling_path

        print(f"  [INFO] Data quality raporlari kaydedildi -> {outputs_dir}")
        return paths
