# -*- coding: utf-8 -*-
"""Service-boundary tests for DataManager Phase 3 decomposition."""

import os

import numpy as np
import pandas as pd

from src.pipeline.config import DataConfig, ValidationConfig
from src.pipeline.data_manager import DataManager
from src.pipeline.data_services import (
    DataIngestionService,
    DataQualityReportingService,
    TensorPreparationService,
    ValidationSplitService,
)
from src.utils.data_splitter import TimeSeriesSplitter


def _frame(n: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(100.0, 120.0, n)
    return pd.DataFrame({
        "Date": dates,
        "Close": close,
        "Feature_A": np.linspace(0.0, 1.0, n),
        "Feature_B": np.sin(np.linspace(0.0, 3.0, n)),
    })


def _manager(tmp_path, *, val_cfg: ValidationConfig | None = None) -> DataManager:
    return DataManager(
        data_cfg=DataConfig(
            data_file=os.path.join(str(tmp_path), "TEST.csv"),
            use_macro=False,
            time_steps=3,
            prune_correlated_features=False,
        ),
        val_cfg=val_cfg or ValidationConfig(),
        models_dir=os.path.join(str(tmp_path), "models"),
    )


def test_data_manager_composes_owner_backed_services(tmp_path):
    manager = _manager(tmp_path)

    assert isinstance(manager.data_ingestion_service, DataIngestionService)
    assert isinstance(manager.tensor_preparation_service, TensorPreparationService)
    assert isinstance(manager.validation_split_service, ValidationSplitService)
    assert isinstance(manager.data_quality_service, DataQualityReportingService)


def test_prepare_tensors_records_train_only_scaler_scope(tmp_path):
    manager = _manager(tmp_path)
    df = _frame(30)

    tensors = manager.prepare_tensors(df.iloc[:20].copy(), df.iloc[20:].copy())

    assert tensors["X_train"].shape[0] == 19
    assert tensors["X_test"].shape[0] == 9
    assert manager.scaling_reports
    assert manager.scaling_reports[-1]["scaler_fit_scope"] == "train_only"
    assert manager.scaling_reports[-1]["scaler_fit_end"] == df["Date"].iloc[19].strftime("%Y-%m-%d")


def test_walk_forward_split_excludes_final_holdout_from_selection(monkeypatch, tmp_path):
    val_cfg = ValidationConfig(
        validation_mode="walk_forward",
        wf_n_splits=2,
        wf_min_train_size=10,
        wf_test_size=4,
        wf_max_train_size=20,
        # wf_embargo_size None/0/negatif birakilirsa _resolve_wf_embargo_size
        # max(200, time_steps)=200'e cikar; bu fixture'da (40 satir)
        # min_required=222>40 olur ve holdout ayrilamaz. Test holdout-exclusion
        # davranisini olcer (embargo'yu degil) ve walk_forward_splits zaten
        # monkeypatch'li (embargo_context 2 satir), bu yuzden embargo'yu 2'ye
        # sabitliyoruz: min_required=10+2+8+4=24<=40, holdout ayrilir.
        wf_embargo_size=2,
        final_holdout_size=4,
    )
    manager = _manager(tmp_path, val_cfg=val_cfg)
    manager.df = _frame(40)
    manager.feature_names = ["Feature_A", "Feature_B"]
    captured = {}

    def fake_walk_forward_splits(source_df, **kwargs):
        captured["source_rows"] = len(source_df)
        return [{
            "split_idx": 1,
            "train": source_df.iloc[:10].copy(),
            "test": source_df.iloc[12:16].copy(),
            "embargo_context": source_df.iloc[10:12].copy(),
            "train_date_start": source_df["Date"].iloc[0],
            "train_date_end": source_df["Date"].iloc[9],
            "test_date_start": source_df["Date"].iloc[12],
            "test_date_end": source_df["Date"].iloc[15],
        }]

    monkeypatch.setattr(TimeSeriesSplitter, "walk_forward_splits", fake_walk_forward_splits)

    manager.split_data("walk_forward")
    protocol = manager.get_validation_protocol_data()

    assert captured["source_rows"] == 36
    assert len(manager.selection_df) == 36
    assert len(manager.final_holdout_df) == 4
    assert not protocol["Final_Holdout_Used_For_Selection"].any()
    assert "final_holdout" in set(protocol["Split"])


# --------------------------------------------------------------------------- #
#  Feature-cache makro zehirlenmesi (regresyon)                               #
# --------------------------------------------------------------------------- #


class _StubFeaturePipeline:
    """macro_df doluysa makro ozellik (Rate_Level) ekleyen sahte FeaturePipeline."""

    def __init__(self, *args, **kwargs) -> None:
        self.feature_names = []
        self.feature_groups = {}
        self.pruning_report = {}
        self.sector_mapping_report = {}

    def engineer_features(self, raw_df, macro_df=None, symbol=None,
                          sector_mapping=None, prune_fit_tail=0):
        out = raw_df.copy()
        if macro_df is not None and not macro_df.empty:
            out["Rate_Level"] = 0.0
            self.feature_names = ["Feature_A", "Rate_Level"]
        else:
            self.feature_names = ["Feature_A"]
        return out


def _macro_manager(tmp_path) -> DataManager:
    manager = DataManager(
        data_cfg=DataConfig(
            data_file=os.path.join(str(tmp_path), "TEST.csv"),
            use_macro=True,
            time_steps=3,
            prune_correlated_features=False,
        ),
        val_cfg=ValidationConfig(),
        models_dir=os.path.join(str(tmp_path), "models"),
    )
    manager.project_root = str(tmp_path)
    return manager


def _cache_pkls(tmp_path) -> list:
    cache_dir = os.path.join(str(tmp_path), "data", "feature_cache")
    if not os.path.isdir(cache_dir):
        return []
    return [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]


def test_degraded_macro_frame_not_cached(tmp_path, monkeypatch):
    """use_macro=True ama makro alinamadiysa makrosuz frame cache'lenmemeli."""
    import src.pipeline.data_services as ds

    monkeypatch.setattr(ds, "FeaturePipeline", _StubFeaturePipeline)
    manager = _macro_manager(tmp_path)
    svc = manager.data_ingestion_service

    svc._engineer_features_cached(_frame(30), pd.DataFrame(), {})

    assert "Feature_A" in manager.feature_names
    assert "Rate_Level" not in manager.feature_names
    assert _cache_pkls(tmp_path) == []  # degraded -> cache yok


def test_macro_frame_cached_and_poisoned_entry_self_heals(tmp_path, monkeypatch):
    """Makro mevcutsa cache yazilir; eski makrosuz (zehirli) kayit self-heal edilir."""
    import src.pipeline.data_services as ds
    from src.features.feature_cache import FeatureCache

    monkeypatch.setattr(ds, "FeaturePipeline", _StubFeaturePipeline)
    manager = _macro_manager(tmp_path)
    svc = manager.data_ingestion_service
    raw = _frame(30)

    # Zehirli (makrosuz) kaydi dogrudan use_macro=True anahtari altina yaz.
    cache_dir = os.path.join(str(tmp_path), "data", "feature_cache")
    cache = FeatureCache(cache_dir=cache_dir)
    key = cache.make_key(manager.data_cfg.data_file, manager.data_cfg)
    cache.put(key, raw.copy(), {
        "feature_names": ["Feature_A"],
        "feature_groups": {},
        "feature_pruning_report": {},
        "sector_mapping_report": {},
    })

    macro_df = pd.DataFrame({"Date": raw["Date"], "Rate_Level": 0.0})
    svc._engineer_features_cached(raw, macro_df, {})

    # Stale kayit yerine makrolu frame uretildi + cache'e yazildi.
    assert "Rate_Level" in manager.feature_names
    assert len(_cache_pkls(tmp_path)) == 1
