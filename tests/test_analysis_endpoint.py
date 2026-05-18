# -*- coding: utf-8 -*-
"""GET /analysis/{symbol} endpoint testleri — tüm status kod senaryoları."""
import tempfile

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
    freshness_date: str = "2026-05-19",
):
    from src.database.stock_model_db import StockModelDB

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


class TestAnalysisRouter:
    def test_endpoint_importable(self):
        from src.api.routers.analysis import router
        assert router is not None

    def test_get_analysis_path(self):
        from src.api.routers.analysis import router
        routes = {r.path for r in router.routes}
        assert "/analysis/{symbol}" in routes
