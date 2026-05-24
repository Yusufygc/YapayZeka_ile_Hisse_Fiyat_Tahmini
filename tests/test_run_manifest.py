# -*- coding: utf-8 -*-
"""Run manifest oluşturma testleri (Adim 1.7)."""

import json
import os
import tempfile

import pytest


def _make_minimal_orchestrator(tmp_dir: str):
    """Gerçek pipeline çalıştırmadan _write_run_manifest() test etmek için minimal mock."""
    from unittest.mock import MagicMock
    import types

    # Minimal signal config
    sc = MagicMock()
    sc.quality_gate_mode = "soft"
    sc.entry_cost_multiplier = 1.0
    sc.volatility_multiplier = 1.0
    sc.min_holding_bars = 1
    sc.max_holding_bars = 10
    sc.take_profit_vol_multiplier = 2.0
    sc.stop_loss_vol_multiplier = 1.5
    sc.min_directional_accuracy = 52.0
    sc.max_rmse_vs_benchmark = 1.0
    sc.min_composite_score = 0.0
    sc.emergency_stop_overrides_min_hold = True

    exe_cfg = MagicMock()
    exe_cfg.signal_config = sc
    exe_cfg.backtest_enabled = False
    exe_cfg.commission_bps = 5
    exe_cfg.slippage_bps = 3
    exe_cfg.initial_capital = 10000
    exe_cfg.signal_mode = "default"
    exe_cfg.report_detail_level = "standard"

    data_cfg = MagicMock()
    data_cfg.data_file = os.path.join(tmp_dir, "TEST.csv")
    data_cfg.target_mode = "return"
    data_cfg.feature_mode = "v2"
    data_cfg.scaling_mode = "standard"
    data_cfg.macro_rate_lag_days = 1
    data_cfg.macro_cpi_lag_days = 1
    data_cfg.prune_correlated_features = True
    data_cfg.correlation_threshold = 0.95
    data_cfg.lag_feature_count = 5
    data_cfg.clip_shift_warning_threshold_pct = 5
    data_cfg.training_window_years = None
    data_cfg.window_candidates = None
    data_cfg.min_history_days = 252
    data_cfg.new_listing_min_days = 60
    data_cfg.universe_file = None

    val_cfg = MagicMock()
    val_cfg.validation_mode = "walk_forward"
    val_cfg.wf_n_splits = 5
    val_cfg.wf_min_train_size = 252
    val_cfg.wf_test_size = 63
    val_cfg.wf_max_train_size = None
    val_cfg.wf_window_type = "expanding"
    val_cfg.wf_embargo_size = 5
    val_cfg.final_holdout_size = 63

    model_cfg = MagicMock()
    model_cfg.selected_models = ["XGBoost"]
    model_cfg.disabled_models = []
    model_cfg.require_available = False
    model_cfg.ensemble_enabled = False
    model_cfg.ensemble_eligibility_overrides = {}
    model_cfg.model_settings = {"prophet": {"use_regressors": False}}

    cfg = MagicMock()
    cfg.data = data_cfg
    cfg.validation = val_cfg
    cfg.models = model_cfg
    cfg.execution = exe_cfg

    # Minimal CSV dosyası
    with open(data_cfg.data_file, "w") as f:
        f.write("Date,Close\n2025-01-01,100\n")

    from src.pipeline.orchestrator import ForecastingPipeline

    pipeline = ForecastingPipeline.__new__(ForecastingPipeline)

    pipeline.data_file = data_cfg.data_file
    pipeline.validation_mode = "walk_forward"
    pipeline.selected_models = ["XGBoost"]
    pipeline.target_mode = "return"
    pipeline.feature_mode = "v2"
    pipeline.scaling_mode = "standard"
    pipeline.backtest_enabled = False
    pipeline.commission_bps = 5
    pipeline.slippage_bps = 3
    pipeline.initial_capital = 10000
    pipeline.signal_mode = "default"
    pipeline.quality_gate_mode = "soft"
    pipeline.signal_entry_cost_multiplier = 1.0
    pipeline.signal_volatility_multiplier = 1.0
    pipeline.min_holding_bars = 1
    pipeline.max_holding_bars = 10
    pipeline.take_profit_vol_multiplier = 2.0
    pipeline.stop_loss_vol_multiplier = 1.5
    pipeline.min_directional_accuracy = 52.0
    pipeline.max_rmse_vs_benchmark = 1.0
    pipeline.min_composite_score = 0.0
    pipeline.emergency_stop_overrides_min_hold = True
    pipeline.macro_rate_lag_days = 1
    pipeline.macro_cpi_lag_days = 1
    pipeline.wf_n_splits = 5
    pipeline.wf_min_train_size = 252
    pipeline.wf_test_size = 63
    pipeline.wf_max_train_size = None
    pipeline.wf_window_type = "expanding"
    pipeline.wf_embargo_size = 5
    pipeline.final_holdout_size = 63
    pipeline.prune_correlated_features = True
    pipeline.correlation_threshold = 0.95
    pipeline.lag_feature_count = 5
    pipeline.clip_shift_warning_threshold_pct = 5
    pipeline.training_window_years = None
    pipeline.window_candidates = [3, 5, 7, 10, None]
    pipeline.min_history_days = 252
    pipeline.new_listing_min_days = 60
    pipeline.use_prophet_macro_regressors = False
    pipeline.ensemble_enabled = False
    pipeline.universe_file = None
    pipeline.model_config = {"prophet": {"use_regressors": False}}
    pipeline._cfg = cfg
    pipeline.stock_symbol = "TEST"
    pipeline.project_root = tmp_dir
    pipeline.disabled_models = []
    pipeline.require_available = False
    pipeline.ensemble_eligibility_overrides = {}
    pipeline.candidate_models = {"XGBoost"}
    pipeline.benchmark_models = set()
    pipeline.run_id = "20260101_TEST_wf"
    pipeline.output_root = os.path.join(tmp_dir, "out", "TEST")
    pipeline.outputs_dir = os.path.join(tmp_dir, "out", "TEST", "runs", pipeline.run_id)
    pipeline.latest_dir = os.path.join(tmp_dir, "out", "TEST", "latest")
    pipeline.models_dir = os.path.join(pipeline.outputs_dir, "models")
    pipeline.report_detail_level = "standard"
    os.makedirs(pipeline.outputs_dir, exist_ok=True)

    return pipeline


class TestRunManifest:
    def test_manifest_file_created(self, tmp_path):
        pipeline = _make_minimal_orchestrator(str(tmp_path))
        pipeline._write_run_manifest()
        manifest_path = os.path.join(pipeline.outputs_dir, "run_manifest.json")
        assert os.path.exists(manifest_path), "run_manifest.json oluşturulmalı"

    def test_latest_sync_uses_temp_dir_and_replaces_existing_latest(self, tmp_path):
        pipeline = _make_minimal_orchestrator(str(tmp_path))
        os.makedirs(os.path.join(pipeline.outputs_dir, "csv"), exist_ok=True)
        with open(
            os.path.join(pipeline.outputs_dir, "run_manifest.json"), "w", encoding="utf-8"
        ) as fh:
            fh.write('{"run_id": "new"}')
        with open(
            os.path.join(pipeline.outputs_dir, "csv", "backtest_report_wf.csv"),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write("Model;Net_Return\nXGBoost;0.1\n")

        os.makedirs(pipeline.latest_dir, exist_ok=True)
        with open(os.path.join(pipeline.latest_dir, "old.txt"), "w", encoding="utf-8") as fh:
            fh.write("old")

        pipeline._sync_latest_output()

        assert os.path.exists(os.path.join(pipeline.latest_dir, "run_manifest.json"))
        assert os.path.exists(os.path.join(pipeline.latest_dir, "csv", "backtest_report_wf.csv"))
        assert not os.path.exists(os.path.join(pipeline.latest_dir, "old.txt"))
        assert not os.path.exists(os.path.join(pipeline.output_root, ".latest_sync.lock"))
        assert not os.path.exists(
            os.path.join(pipeline.output_root, f"latest.__tmp__{pipeline.run_id}")
        )

    def test_manifest_required_fields(self, tmp_path):
        pipeline = _make_minimal_orchestrator(str(tmp_path))
        pipeline.final_holdout_status = {"status": "failed", "error_type": "RuntimeError"}
        pipeline._write_run_manifest()
        manifest_path = os.path.join(pipeline.outputs_dir, "run_manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

        required = [
            "run_id",
            "generated_at",
            "stock_symbol",
            "data_hash",
            "feature_pipeline_version",
            "model_config_hash",
            "signal_config_hash",
            "random_seed",
            "model_list",
            "validation_protocol",
            "final_holdout_status",
            "git_commit",
            "python_version",
            "lib_versions",
        ]
        for field in required:
            assert field in manifest, f"Zorunlu alan eksik: {field}"
        assert manifest["final_holdout_status"]["status"] == "failed"
        assert manifest["final_holdout_status"]["error_type"] == "RuntimeError"

    def test_manifest_values_correct(self, tmp_path):
        pipeline = _make_minimal_orchestrator(str(tmp_path))
        pipeline._write_run_manifest()
        manifest_path = os.path.join(pipeline.outputs_dir, "run_manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

        assert manifest["run_id"] == pipeline.run_id
        assert manifest["stock_symbol"] == "TEST"
        assert manifest["random_seed"] == 42
        assert manifest["validation_protocol"] == "walk_forward"
        assert "XGBoost" in manifest["model_list"]
        assert manifest["data_hash"] != "unavailable"

    def test_manifest_records_research_policy_metadata(self, tmp_path):
        pipeline = _make_minimal_orchestrator(str(tmp_path))
        pipeline.research_policy = "V3"
        pipeline.research_phase = "plan1"
        pipeline.research_metadata = {
            "history_bucket": "mid_history",
            "sector": "Technology",
        }
        pipeline._write_run_manifest()
        manifest_path = os.path.join(pipeline.outputs_dir, "run_manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

        assert manifest["research_policy"] == "V3"
        assert manifest["research_phase"] == "plan1"
        assert manifest["research_metadata"]["history_bucket"] == "mid_history"
        assert manifest["uses_final_holdout_for_selection"] is False

    def test_manifest_lib_versions_dict(self, tmp_path):
        pipeline = _make_minimal_orchestrator(str(tmp_path))
        pipeline._write_run_manifest()
        manifest_path = os.path.join(pipeline.outputs_dir, "run_manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

        assert isinstance(manifest["lib_versions"], dict)
        assert "numpy" in manifest["lib_versions"]
        assert "pandas" in manifest["lib_versions"]

    def test_manifest_valid_json(self, tmp_path):
        pipeline = _make_minimal_orchestrator(str(tmp_path))
        pipeline._write_run_manifest()
        manifest_path = os.path.join(pipeline.outputs_dir, "run_manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            content = fh.read()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
