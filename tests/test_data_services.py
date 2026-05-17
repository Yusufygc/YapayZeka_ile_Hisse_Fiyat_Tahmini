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
