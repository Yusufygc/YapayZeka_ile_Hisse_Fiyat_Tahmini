# -*- coding: utf-8 -*-
"""Silinen modeller (LSTM/Prophet/ARIMA/...) icin eski DB satirlari.

Eski ``experiments``/``best_models`` satirlari silinmis modelleri adlandirabilir.
Okuma yollari (GET /analysis, leaderboard) cokmemeli; forecast uretimi import hatasi
veya traceback yerine yapisal "model artik desteklenmiyor, yeniden egitin" sonucu vermeli.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest

from src.forecasting.artifacts import (
    MODEL_NO_LONGER_SUPPORTED_REASON,
    ModelNoLongerSupportedError,
)
from src.pipeline.model_scope import is_removed_model

REMOVED = ("LSTM", "LSTM Lite", "AttentionLSTM v2", "DLinear", "NLinear", "Prophet", "ARIMA")
# Audit-sonrasi kaldirilanlar: tabular duplikatlar + zayif ensemble varyantlari.
REMOVED_AUDIT = (
    "ElasticNet Return",
    "Random Forest",
    "Ensemble Equal Weight",
    "Ensemble Sharpe-Weighted",
    "Ensemble Risk-Parity",
    "Ensemble Hierarchical",
    "Ensemble Meta-Stacker",
    "Ensemble Cash-Gated",
)
_METRICS = {"MAE": 1.0, "RMSE": 2.0, "Dir_Acc": 56.0, "Sharpe": 0.5, "Trade_Count": 10}


def _db_with_best(tmp_path, model_name: str, *, with_forecast: bool = True):
    from src.database.stock_model_db import StockModelDB

    db = StockModelDB(os.path.join(str(tmp_path), "stock_models.db"))
    db.log_experiment(
        stock_symbol="TUPRS",
        model_name=model_name,
        metrics=dict(_METRICS),
        dataset_hash="hash1",
        validation_mode="final_holdout",
        is_production_candidate=True,
        run_id="run001",
    )
    if with_forecast:
        db.log_forecast_run(
            stock_symbol="TUPRS",
            model_name=model_name,
            source_experiment_id=1,
            last_observed_date=datetime.now().date().isoformat(),
            last_close=100.0,
            horizon_days=5,
            trend_label="up",
            weekly_expected_return=0.02,
            trend_threshold=0.01,
            rules_version="v1",
            points=[
                {"target_date": "2026-05-20", "horizon_index": 1,
                 "bounded_predicted_close": 102.0, "predicted_return": 0.02}
            ],
        )
    return db


@pytest.mark.parametrize("name", REMOVED + REMOVED_AUDIT)
def test_removed_model_names_are_flagged(name):
    assert is_removed_model(name)


@pytest.mark.parametrize("name", ["XGBoost", "Ridge Return", "Naive Last Value", "Ensemble Inverse RMSE"])
def test_supported_model_names_are_not_flagged(name):
    assert not is_removed_model(name)


@pytest.mark.parametrize("name", ["LSTM", "Prophet", "ARIMA"])
def test_analysis_endpoint_reads_legacy_removed_model_rows(tmp_path, name):
    from src.api.services.analysis_service import AnalysisService

    db = _db_with_best(tmp_path, name)
    svc = AnalysisService(db_path=db.db_path, outputs_base=tempfile.mkdtemp())

    result = svc.build("TUPRS")

    assert result.model.model_name == name
    assert result.model.model_family == "removed"
    assert result.forecast is not None and len(result.forecast.points) == 1


def test_resolver_rejects_removed_best_model_with_structured_error(tmp_path):
    from src.forecasting.runner import ForecastRunner

    db = _db_with_best(tmp_path, "LSTM Lite", with_forecast=False)
    runner = ForecastRunner(project_root=str(tmp_path), db_path=db.db_path, calendar_path=None)

    with pytest.raises(ModelNoLongerSupportedError) as exc:
        runner.model_resolver.resolve("TUPRS")

    assert exc.value.model_name == "LSTM Lite"
    assert exc.value.reason == MODEL_NO_LONGER_SUPPORTED_REASON
    assert exc.value.as_payload()["action"] == "retrain"
    assert "retrain" in str(exc.value)


def test_resolver_rejects_forced_removed_model(tmp_path):
    from src.forecasting.runner import ForecastRunner

    db = _db_with_best(tmp_path, "XGBoost", with_forecast=False)
    runner = ForecastRunner(project_root=str(tmp_path), db_path=db.db_path, calendar_path=None)

    with pytest.raises(ModelNoLongerSupportedError):
        runner.model_resolver.resolve("TUPRS", force_model_name="Prophet")


def test_run_symbol_fails_fast_before_touching_data(tmp_path):
    from src.forecasting.runner import ForecastRunner

    db = _db_with_best(tmp_path, "ARIMA", with_forecast=False)
    runner = ForecastRunner(project_root=str(tmp_path), db_path=db.db_path, calendar_path=None)

    # data_file yok: model reddi veri hazirligindan ONCE olmali (FileNotFoundError degil).
    with pytest.raises(ModelNoLongerSupportedError):
        runner.run_symbol(
            symbol="TUPRS",
            data_file=os.path.join(str(tmp_path), "missing.csv"),
            auto_update_data=False,
        )


def test_make_model_instance_raises_structured_error_for_removed_model(tmp_path):
    from src.forecasting.runner import ForecastRunner

    db = _db_with_best(tmp_path, "XGBoost", with_forecast=False)
    runner = ForecastRunner(project_root=str(tmp_path), db_path=db.db_path, calendar_path=None)

    with pytest.raises(ModelNoLongerSupportedError):
        runner._make_model_instance("DLinear")
    with pytest.raises(KeyError):
        runner._make_model_instance("__NoSuchModel__")


def test_refresh_service_reports_model_no_longer_supported(tmp_path, monkeypatch):
    from src.api.services import data_refresh_service as module
    from src.api.services.data_refresh_service import DataRefreshError, DataRefreshService

    # Veri guncellemesi artik retrain sonucundan ONCE calisir (agsiz sahte updater).
    class _Update:
        def to_dict(self):
            return {"status": "up_to_date"}

    monkeypatch.setattr(
        module.DataUpdater, "check_and_update", staticmethod(lambda *a, **k: _Update())
    )
    monkeypatch.setattr(module, "ensure_bist_calendar", lambda *a, **k: None)
    db = _db_with_best(tmp_path, "LSTM", with_forecast=False)
    service = DataRefreshService(db=db, project_root=str(tmp_path), start_background=False)

    with pytest.raises(DataRefreshError) as exc:
        service.refresh_symbol(symbol="TUPRS", best_model={"model_name": "LSTM", "experiment_id": 1})

    assert exc.value.reason == MODEL_NO_LONGER_SUPPORTED_REASON
    assert exc.value.payload["action"] == "retrain"
    assert exc.value.payload["model_name"] == "LSTM"


def test_failed_refresh_job_surfaces_retrain_reason(tmp_path):
    """Arka plan refresh job'i basarisiz olunca reason analysis cevabina yansir (_refresh_reason)."""
    db = _db_with_best(tmp_path, "LSTM", with_forecast=False)
    job = db.create_or_get_refresh_job(symbol="TUPRS", reason="missing_forecast_for_best_model")
    db.update_refresh_job(
        job["job_id"],
        status="failed",
        error="x",
        payload={"failure_reason": MODEL_NO_LONGER_SUPPORTED_REASON, "action": "retrain"},
        finish=True,
    )

    from src.api.services import analysis_service as module

    reason = module._refresh_reason(db.get_refresh_job(job["job_id"]), "missing_forecast_for_best_model")
    assert reason == MODEL_NO_LONGER_SUPPORTED_REASON
    assert module._job_status(db.get_refresh_job(job["job_id"])) == "failed"


@pytest.mark.parametrize("name", REMOVED_AUDIT)
def test_resolver_rejects_audit_removed_best_model(tmp_path, name):
    from src.forecasting.runner import ForecastRunner

    db = _db_with_best(tmp_path, name, with_forecast=False)
    runner = ForecastRunner(project_root=str(tmp_path), db_path=db.db_path, calendar_path=None)

    with pytest.raises(ModelNoLongerSupportedError) as exc:
        runner.model_resolver.resolve("TUPRS")
    assert exc.value.model_name == name
    assert exc.value.as_payload()["action"] == "retrain"


@pytest.mark.parametrize("name", ["ElasticNet Return", "Random Forest", "Ensemble Cash-Gated"])
def test_analysis_endpoint_reads_audit_removed_rows(tmp_path, name):
    from src.api.services.analysis_service import AnalysisService

    db = _db_with_best(tmp_path, name)
    svc = AnalysisService(db_path=db.db_path, outputs_base=tempfile.mkdtemp())

    result = svc.build("TUPRS")

    assert result.model.model_name == name
    assert result.model.model_family == "removed"

