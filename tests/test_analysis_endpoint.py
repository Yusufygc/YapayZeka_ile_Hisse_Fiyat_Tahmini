# -*- coding: utf-8 -*-
"""GET /analysis/{symbol} endpoint testleri — tüm status kod senaryoları."""
import tempfile
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI yüklü değil", allow_module_level=True)

from src.api.services.analysis_service import AnalysisService
from src.api.schemas.analysis import AnalysisResponse


def _make_service(db=None, outputs=None) -> AnalysisService:
    tmpdir = tempfile.mkdtemp()
    return AnalysisService(
        db_path=db,
        outputs_base=outputs or tmpdir,
    )


def _make_db_with_model(
    *,
    symbol: str = "TUPRS",
    model_name: str = "XGBoost",
    dir_acc: float = 56.0,
    with_forecast: bool = True,
    freshness_date: str = None,
):
    from src.database.stock_model_db import StockModelDB
    from datetime import datetime

    if freshness_date is None:
        freshness_date = datetime.now().date().isoformat()

    tmp = tempfile.mktemp(suffix=".db")
    db = StockModelDB(tmp)
    db.log_experiment(
        stock_symbol=symbol,
        model_name=model_name,
        metrics={
            "MAE": 1.0, "RMSE": 2.0, "Dir_Acc": dir_acc,
            "Sharpe": 0.5, "Trade_Count": 10,
        },
        dataset_hash="hash1",
        validation_mode="final_holdout",
        is_production_candidate=True,
        run_id="run001",
    )
    if with_forecast:
        db.log_forecast_run(
            stock_symbol=symbol,
            model_name=model_name,
            source_experiment_id=1,
            last_observed_date=freshness_date,
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
    return tmp


class TestAnalysisService:
    def test_no_model_status(self):
        svc = _make_service()
        # Boş DB
        import tempfile as _tf
        from src.database.stock_model_db import StockModelDB
        tmp = _tf.mktemp(suffix=".db")
        StockModelDB(tmp)  # schema oluştur
        svc2 = AnalysisService(db_path=tmp, outputs_base=_tf.mkdtemp())
        result = svc2.build("UNKNOWN")
        assert result.analysis_status == "no_model"
        assert result.disclaimer != ""

    def test_no_forecast_status(self):
        tmp_db = _make_db_with_model(with_forecast=False)
        svc = AnalysisService(db_path=tmp_db, outputs_base=tempfile.mkdtemp())
        result = svc.build("TUPRS")
        assert result.analysis_status == "no_forecast"
        assert result.model.model_name == "XGBoost"
        assert result.refresh_status == "queued"
        assert result.refresh_reason == "missing_forecast_for_best_model"
        assert result.refresh_job_id

    def test_forecast_must_match_best_model_source_experiment(self):
        from src.database.stock_model_db import StockModelDB

        tmp = tempfile.mktemp(suffix=".db")
        db = StockModelDB(tmp)
        exp_id = db.log_experiment(
            stock_symbol="ASELS",
            model_name="LSTM Lite",
            metrics={"MAE": 1.0, "RMSE": 2.0, "Dir_Acc": 58.0, "Sharpe": 0.4, "Trade_Count": 10},
            validation_mode="final_holdout",
            is_production_candidate=True,
            run_id="lstm-run",
        )
        arima_exp_id = db.log_experiment(
            stock_symbol="ASELS",
            model_name="ARIMA",
            metrics={"MAE": 2.0, "RMSE": 3.0, "Dir_Acc": 48.0, "Sharpe": -0.1, "Trade_Count": 10},
            validation_mode="walk_forward",
            is_production_candidate=False,
            run_id="arima-run",
        )
        db.log_forecast_run(
            stock_symbol="ASELS",
            model_name="ARIMA",
            source_experiment_id=arima_exp_id,
            last_observed_date="2026-05-19",
            last_close=100.0,
            horizon_days=5,
            trend_label="UP",
            weekly_expected_return=0.02,
            trend_threshold=0.01,
            rules_version="v1",
            points=[{"target_date": "2026-05-20", "horizon_index": 1, "bounded_predicted_close": 102.0}],
        )

        result = AnalysisService(db_path=tmp, outputs_base=tempfile.mkdtemp()).build("ASELS")
        assert exp_id
        assert result.analysis_status == "no_forecast"
        assert result.model.model_name == "LSTM Lite"
        assert result.forecast.points == []

    def test_stale_data_status(self):
        # 2020'den kalma bir tarih → kesinlikle stale
        tmp_db = _make_db_with_model(freshness_date="2020-01-01")
        svc = AnalysisService(db_path=tmp_db, outputs_base=tempfile.mkdtemp())
        result = svc.build("TUPRS")
        assert result.analysis_status == "stale_data"
        assert result.data.data_freshness == "stale_data"

    def test_xai_unavailable_status(self):
        # Forecast var, XAI dosyası yok, dir_acc yeterli
        tmp_db = _make_db_with_model(dir_acc=56.0)
        svc = AnalysisService(db_path=tmp_db, outputs_base=tempfile.mkdtemp())
        result = svc.build("TUPRS")
        # Fresh veri, XAI yok → xai_unavailable ya da low_confidence (dir_acc güven mekanizmasına bağlı)
        assert result.analysis_status in ("xai_unavailable", "low_confidence", "ok")

    def test_analysis_uses_best_model_run_id_for_xai_lookup(self):
        from src.database.stock_model_db import StockModelDB

        tmp = tempfile.mktemp(suffix=".db")
        outputs = Path(tempfile.mkdtemp())
        db = StockModelDB(tmp)
        exp_id = db.log_experiment(
            stock_symbol="ASELS",
            model_name="LSTM",
            metrics={"MAE": 1.0, "RMSE": 2.0, "Dir_Acc": 62.0, "Sharpe": 0.8, "Trade_Count": 10},
            validation_mode="final_holdout",
            is_production_candidate=True,
            run_id="lstm-run",
        )
        db.log_forecast_run(
            stock_symbol="ASELS",
            model_name="LSTM",
            source_experiment_id=exp_id,
            last_observed_date="2026-05-19",
            last_close=100.0,
            horizon_days=5,
            trend_label="flat",
            weekly_expected_return=0.0,
            trend_threshold=0.01,
            rules_version="v1",
            points=[{"target_date": "2026-05-20", "horizon_index": 1, "bounded_predicted_close": 100.0}],
        )
        latest_csv = outputs / "ASELS" / "latest" / "xai" / "csv"
        run_csv = outputs / "ASELS" / "runs" / "lstm-run" / "xai" / "csv"
        latest_csv.mkdir(parents=True)
        run_csv.mkdir(parents=True)
        latest_csv.joinpath("xai_top_reasons_wf.csv").write_text(
            "Model;Feature;Readable_Feature;Feature_Group;Importance;Contribution;Direction;Reason;Method;Approximate\n"
            "NLinear;WrongFeature;Wrong feature;technical;0.4;0.02;positive;wrong reason;sequence;True\n",
            encoding="utf-8",
        )
        run_csv.joinpath("xai_top_reasons_wf.csv").write_text(
            "Model;Feature;Readable_Feature;Feature_Group;Importance;Contribution;Direction;Reason;Method;Approximate\n"
            "LSTM;BestRunFeature;Best run feature;technical;0.5;0.03;positive;best run reason;sequence;True\n",
            encoding="utf-8",
        )

        result = AnalysisService(db_path=tmp, outputs_base=str(outputs)).build("ASELS")

        assert result.xai.available is True
        assert result.xai.top_positive_reasons[0].feature_name == "BestRunFeature"
        assert result.xai.top_positive_reasons[0].feature_group == "technical"
        assert result.xai.top_positive_reasons[0].reason == "best run reason"
        assert result.xai.top_positive_reasons[0].contribution == pytest.approx(0.03)
        assert result.xai.top_positive_reasons[0].approximate is True

    def test_ensemble_forecast_source_metadata_surfaces(self):
        from src.database.stock_model_db import StockModelDB

        tmp = tempfile.mktemp(suffix=".db")
        db = StockModelDB(tmp)
        exp_id = db.log_experiment(
            stock_symbol="ASELS",
            model_name="Ensemble Inverse RMSE",
            metrics={
                "MAE": 1.0,
                "RMSE": 2.0,
                "Dir_Acc": 58.0,
                "Sharpe": 0.4,
                "Trade_Count": 10,
                "RMSE_vs_benchmark": 0.95,
                "Ensemble_Method": "Inverse RMSE",
                "Ensemble_Weights": '{"Ridge Return": 0.4, "LSTM": 0.6}',
            },
            validation_mode="walk_forward",
            is_production_candidate=True,
            selection_source="walk_forward_production_ensemble",
            run_id="ens-run",
        )
        db.log_forecast_run(
            stock_symbol="ASELS",
            model_name="Ensemble Inverse RMSE",
            source_experiment_id=exp_id,
            last_observed_date="2026-05-19",
            last_close=100.0,
            horizon_days=5,
            trend_label="UP",
            weekly_expected_return=0.02,
            trend_threshold=0.01,
            rules_version="v1",
            points=[{"target_date": "2026-05-20", "horizon_index": 1, "bounded_predicted_close": 102.0}],
            forecast_strategy="ensemble_recursive_direct_target",
            artifact_mode="artifact_loaded",
            forecast_warnings=["projected_exogenous_features"],
            ensemble_metadata={
                "method": "Inverse RMSE",
                "members": ["Ridge Return", "LSTM"],
                "weights": {"Ridge Return": 0.4, "LSTM": 0.6},
                "source_experiment_ids": [1, 2],
            },
        )

        result = AnalysisService(db_path=tmp, outputs_base=tempfile.mkdtemp()).build("ASELS")

        assert result.forecast_source is not None
        assert result.forecast_source.type == "ensemble"
        assert result.forecast_source.method == "Inverse RMSE"
        assert result.forecast_source.weights["LSTM"] == 0.6
        assert result.forecast_source.forecast_strategy == "ensemble_recursive_direct_target"

    def test_disclaimer_always_present(self):
        tmp_db = _make_db_with_model()
        svc = AnalysisService(db_path=tmp_db, outputs_base=tempfile.mkdtemp())
        result = svc.build("TUPRS")
        assert "yatırım tavsiyesi" in result.disclaimer.lower()

    def test_symbol_normalized_uppercase(self):
        tmp_db = _make_db_with_model(symbol="TUPRS")
        svc = AnalysisService(db_path=tmp_db, outputs_base=tempfile.mkdtemp())
        result = svc.build("tuprs")
        assert result.symbol == "TUPRS"

    def test_confidence_block_present(self):
        tmp_db = _make_db_with_model()
        svc = AnalysisService(db_path=tmp_db, outputs_base=tempfile.mkdtemp())
        result = svc.build("TUPRS")
        assert result.confidence.label in ("low", "medium", "high")

    def test_forecast_block_has_points(self):
        tmp_db = _make_db_with_model()
        svc = AnalysisService(db_path=tmp_db, outputs_base=tempfile.mkdtemp())
        result = svc.build("TUPRS")
        if result.analysis_status not in ("no_model", "no_forecast"):
            assert result.forecast.horizon_days == 5
            assert len(result.forecast.points) == 1

    def test_refresh_job_deduplicates_queued_work(self):
        from src.database.stock_model_db import StockModelDB

        tmp = tempfile.mktemp(suffix=".db")
        db = StockModelDB(tmp)
        first = db.create_or_get_refresh_job(symbol="TUPRS", reason="missing_forecast_for_best_model")
        second = db.create_or_get_refresh_job(symbol="TUPRS", reason="missing_forecast_for_best_model")
        assert first["job_id"] == second["job_id"]
        assert db.get_refresh_job(first["job_id"])["status"] == "queued"

    def test_missing_forecast_completed_refresh_reloads_forecast(self, monkeypatch):
        from src.database.stock_model_db import StockModelDB

        project_root = Path(tempfile.mkdtemp())
        data_dir = project_root / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "ASELS.csv").write_text("Date,Close\n2026-05-19,100.0\n", encoding="utf-8")

        tmp = tempfile.mktemp(suffix=".db")
        db = StockModelDB(tmp)
        exp_id = db.log_experiment(
            stock_symbol="ASELS",
            model_name="LSTM Lite",
            metrics={"MAE": 1.0, "RMSE": 2.0, "Dir_Acc": 60.0, "Sharpe": 0.5, "Trade_Count": 10},
            validation_mode="final_holdout",
            is_production_candidate=True,
            run_id="lstm-run",
        )

        def fake_refresh(self, *, symbol, reason, best_model=None, wait_timeout_seconds=0.0):
            self.db.log_forecast_run(
                stock_symbol=symbol,
                model_name=best_model["model_name"],
                source_experiment_id=best_model["experiment_id"],
                last_observed_date="2026-05-19",
                last_close=100.0,
                horizon_days=5,
                trend_label="up",
                weekly_expected_return=0.02,
                trend_threshold=0.01,
                rules_version="v1",
                points=[{"target_date": "2026-05-20", "horizon_index": 1, "bounded_predicted_close": 102.0}],
            )
            return {"job_id": "job-sync", "status": "completed", "reason": reason}

        monkeypatch.setattr(
            "src.api.services.analysis_service.DataRefreshService.ensure_refresh_job",
            fake_refresh,
        )
        result = AnalysisService(
            db_path=tmp,
            outputs_base=tempfile.mkdtemp(),
            project_root=str(project_root),
            enable_background_refresh=True,
            refresh_wait_timeout_seconds=90,
        ).build("ASELS")

        assert exp_id
        assert result.analysis_status != "no_forecast"
        assert result.refresh_status == "completed"
        assert result.forecast_source.source_experiment_id == exp_id
        assert len(result.forecast.points) == 1

    def test_missing_forecast_timeout_keeps_refresh_status(self, monkeypatch):
        project_root = Path(tempfile.mkdtemp())
        data_dir = project_root / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "ASELS.csv").write_text("Date,Close\n2026-05-19,100.0\n", encoding="utf-8")

        tmp_db = _make_db_with_model(symbol="ASELS", model_name="LSTM Lite", with_forecast=False)

        def fake_refresh(self, *, symbol, reason, best_model=None, wait_timeout_seconds=0.0):
            return {"job_id": "job-running", "status": "running", "reason": reason}

        monkeypatch.setattr(
            "src.api.services.analysis_service.DataRefreshService.ensure_refresh_job",
            fake_refresh,
        )
        result = AnalysisService(
            db_path=tmp_db,
            outputs_base=tempfile.mkdtemp(),
            project_root=str(project_root),
            enable_background_refresh=True,
            refresh_wait_timeout_seconds=0.01,
        ).build("ASELS")

        assert result.analysis_status == "no_forecast"
        assert result.refresh_status == "running"
        assert result.refresh_job_id == "job-running"

    def test_stale_forecast_completed_refresh_reloads_current_forecast(self, monkeypatch):
        project_root = Path(tempfile.mkdtemp())
        data_dir = project_root / "data"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "ASELS.csv"
        csv_path.write_text("Date,Close\n2020-01-01,100.0\n", encoding="utf-8")

        tmp_db = _make_db_with_model(
            symbol="ASELS",
            model_name="LSTM Lite",
            with_forecast=True,
            freshness_date="2020-01-01",
        )

        def fake_refresh(self, *, symbol, reason, best_model=None, wait_timeout_seconds=0.0):
            csv_path.write_text("Date,Close\n2026-05-19,110.0\n", encoding="utf-8")
            self.db.log_forecast_run(
                stock_symbol=symbol,
                model_name=best_model["model_name"],
                source_experiment_id=best_model["experiment_id"],
                last_observed_date="2026-05-19",
                last_close=110.0,
                horizon_days=5,
                trend_label="flat",
                weekly_expected_return=0.0,
                trend_threshold=0.01,
                rules_version="v1",
                points=[{"target_date": "2026-05-20", "horizon_index": 1, "bounded_predicted_close": 110.0}],
            )
            return {"job_id": "job-stale-sync", "status": "completed", "reason": reason}

        monkeypatch.setattr(
            "src.api.services.analysis_service.DataRefreshService.ensure_refresh_job",
            fake_refresh,
        )
        result = AnalysisService(
            db_path=tmp_db,
            outputs_base=tempfile.mkdtemp(),
            project_root=str(project_root),
            enable_background_refresh=True,
            refresh_wait_timeout_seconds=90,
        ).build("ASELS")

        assert result.refresh_status == "completed"
        assert result.data.last_observed_date == "2026-05-19"
        assert result.data.last_close == 110.0

    def test_missing_forecast_failed_refresh_surfaces_failure_reason(self, monkeypatch):
        project_root = Path(tempfile.mkdtemp())
        data_dir = project_root / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "ASELS.csv").write_text("Date,Close\n2026-05-19,100.0\n", encoding="utf-8")

        tmp_db = _make_db_with_model(symbol="ASELS", model_name="LSTM Lite", with_forecast=False)

        def fake_refresh(self, *, symbol, reason, best_model=None, wait_timeout_seconds=0.0):
            return {
                "job_id": "job-failed",
                "status": "failed",
                "reason": reason,
                "payload_json": '{"failure_reason":"data_update_failed"}',
            }

        monkeypatch.setattr(
            "src.api.services.analysis_service.DataRefreshService.ensure_refresh_job",
            fake_refresh,
        )
        result = AnalysisService(
            db_path=tmp_db,
            outputs_base=tempfile.mkdtemp(),
            project_root=str(project_root),
            enable_background_refresh=True,
            refresh_wait_timeout_seconds=90,
        ).build("ASELS")

        assert result.analysis_status == "no_forecast"
        assert result.refresh_status == "failed"
        assert result.refresh_reason == "data_update_failed"


class TestAnalysisRouter:
    def test_endpoint_importable(self):
        from src.api.routers.analysis import router
        assert router is not None

    def test_get_analysis_path(self):
        from src.api.routers.analysis import router
        routes = {r.path for r in router.routes}
        assert "/analysis/{symbol}" in routes

    def test_cors_defaults_are_local_only(self):
        from src.api.runtime_config import get_cors_settings

        settings = get_cors_settings()
        assert "*" not in settings.allow_origins
        assert settings.allow_origin_regex
        assert settings.mode in {"local-only", "local-plus-env"}
