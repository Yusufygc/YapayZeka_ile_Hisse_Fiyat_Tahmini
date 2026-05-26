# -*- coding: utf-8 -*-
"""
Sprint 8 (2026-05-25) A8.7 — Analysis API Faz 2 testleri.

- confidence.reasons: medium/high label icin pozitif sinyaller doldurulur
- confidence.warnings: psi moderate_drift/major_drift warning string'leri
- xai.top_positive_reasons: CSV varsa dolu
- forecast.ensemble_agreement: ensemble forecast'larda float, tek model None
- forecast.points: Sprint 4 quantile alanlari (var/null)
- disclaimer: her durumda dolu
- /v1/analysis/{symbol} alias /analysis ile ayni davranis
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

try:
    import fastapi  # noqa: F401
    _HAVE_FASTAPI = True
except Exception:
    _HAVE_FASTAPI = False

try:
    from fastapi.testclient import TestClient
    _HAVE_TEST_CLIENT = True
except Exception:  # httpx eksikse de import patlayabilir
    _HAVE_TEST_CLIENT = False

try:
    from src.api.services.analysis_service import AnalysisService
except Exception as exc:  # pragma: no cover
    pytest.skip(f"AnalysisService import basarisiz: {exc}", allow_module_level=True)


def _today_iso() -> str:
    return datetime.now().date().isoformat()


def _make_db_with_high_quality_model():
    """Composite + dir_acc + stability yuksek -> medium/high confidence."""
    from src.database.stock_model_db import StockModelDB

    tmp = tempfile.mktemp(suffix=".db")
    db = StockModelDB(tmp)
    exp_id = db.log_experiment(
        stock_symbol="TUPRS",
        model_name="XGBoost",
        metrics={
            "MAE": 1.0, "RMSE": 2.0,
            "Dir_Acc": 62.0, "Hit_Rate": 58.0,
            "Sharpe": 0.8, "Trade_Count": 30,
            "RMSE_vs_benchmark": 0.85,
            "Composite_Score": 78.0,
            "Stability_Score": 0.65,
        },
        validation_mode="final_holdout",
        is_production_candidate=True,
        run_id="run-good",
    )
    db.log_forecast_run(
        stock_symbol="TUPRS",
        model_name="XGBoost",
        source_experiment_id=exp_id,
        last_observed_date=_today_iso(),
        last_close=100.0,
        horizon_days=5,
        trend_label="up",
        weekly_expected_return=0.02,
        trend_threshold=0.01,
        rules_version="v1",
        points=[
            {
                "target_date": "2026-05-26",
                "horizon_index": 1,
                "bounded_predicted_close": 102.0,
                "predicted_return": 0.02,
                "p10_close": 100.5,
                "p50_close": 102.0,
                "p90_close": 103.5,
                "predicted_return_p10": 0.005,
                "predicted_return_p50": 0.020,
                "predicted_return_p90": 0.035,
            }
        ],
    )
    return tmp


def _make_db_with_ensemble():
    """Ensemble forecast — ensemble_direction_agreement icermeli."""
    from src.database.stock_model_db import StockModelDB

    tmp = tempfile.mktemp(suffix=".db")
    db = StockModelDB(tmp)
    exp_id = db.log_experiment(
        stock_symbol="ASELS",
        model_name="Ensemble Inverse RMSE",
        metrics={
            "MAE": 1.0, "RMSE": 2.0,
            "Dir_Acc": 60.0, "Hit_Rate": 55.0, "Sharpe": 0.7,
            "Trade_Count": 20,
            "Composite_Score": 72.0, "Stability_Score": 0.55,
            "Ensemble_Method": "Inverse RMSE",
            "Ensemble_Weights": '{"Ridge Return": 0.4, "LSTM": 0.6}',
            "production_method": "Inverse RMSE",
        },
        # Ensemble best_model upsert path: validation_mode != final_holdout
        # ama ensemble_metadata.production_method varsa promotion OK.
        validation_mode="final_holdout",
        is_production_candidate=True,
        selection_source="walk_forward_production_ensemble",
        run_id="ens-run",
    )
    db.log_forecast_run(
        stock_symbol="ASELS",
        model_name="Ensemble Inverse RMSE",
        source_experiment_id=exp_id,
        last_observed_date=_today_iso(),
        last_close=100.0,
        horizon_days=5,
        trend_label="up",
        weekly_expected_return=0.02,
        trend_threshold=0.01,
        rules_version="v1",
        points=[
            {"target_date": "2026-05-26", "horizon_index": 1,
             "bounded_predicted_close": 102.0, "predicted_return": 0.02}
        ],
        ensemble_direction_agreement=0.83,
        ensemble_metadata={
            "method": "Inverse RMSE",
            "members": ["Ridge Return", "LSTM"],
            "weights": {"Ridge Return": 0.4, "LSTM": 0.6},
            "source_experiment_ids": [1, 2],
        },
    )
    return tmp


def test_confidence_reasons_populated_for_medium_high():
    """A8.1 — medium/high label'da pozitif sebepler doldurulur."""
    tmp_db = _make_db_with_high_quality_model()
    svc = AnalysisService(
        db_path=tmp_db,
        outputs_base=tempfile.mkdtemp(),
        enable_background_refresh=False,
    )
    result = svc.build("TUPRS")
    # Label medium veya high olmali (degil low)
    assert result.confidence.label in ("medium", "high")
    # Reasons en az 1 pozitif sinyal icermeli
    assert len(result.confidence.reasons) >= 1
    joined = " ".join(result.confidence.reasons).lower()
    # En az birinin Walk-forward dir_acc / composite gibi pozitif olmasi gerek
    has_positive = any(
        kw in joined for kw in ("dogruluk", "composite", "hit rate", "stability")
    )
    assert has_positive, f"Pozitif neden bulunamadi: {result.confidence.reasons}"


def test_quantile_fields_in_forecast_points(monkeypatch):
    """A4.5 + A8 — forecast.points quantile alanlari yayinlanir.

    DB schema su an p10/p50/p90 sutunlarini persist etmiyor (Sprint 4
    point-level alanlari workflows.py icinde uretti ancak schema ekledi-
    rilmedi). Bu testte forecast_row builder'a quantile dictini direkt
    enjekte edip API'nin onu opsiyonel alanlara dogru aktardigini
    dogrularir.
    """
    from src.api.services import analysis_service as svc_mod

    tmp_db = _make_db_with_high_quality_model()
    svc = AnalysisService(
        db_path=tmp_db,
        outputs_base=tempfile.mkdtemp(),
        enable_background_refresh=False,
    )

    original_find = svc_mod._find_matching_forecast

    def patched_find(**kwargs):
        row = original_find(**kwargs)
        if row is None:
            return None
        pts = list(row.get("points") or [])
        if pts:
            pts[0] = {
                **pts[0],
                "p10_close": 100.5,
                "p50_close": 102.0,
                "p90_close": 103.5,
                "predicted_return_p10": 0.005,
                "predicted_return_p50": 0.020,
                "predicted_return_p90": 0.035,
            }
            row["points"] = pts
        return row

    monkeypatch.setattr(svc_mod, "_find_matching_forecast", patched_find)
    result = svc.build("TUPRS")
    assert len(result.forecast.points) == 1
    pt = result.forecast.points[0]
    assert pt.p10_close == pytest.approx(100.5)
    assert pt.p50_close == pytest.approx(102.0)
    assert pt.p90_close == pytest.approx(103.5)
    assert pt.predicted_return_p10 == pytest.approx(0.005)
    assert pt.predicted_return_p90 == pytest.approx(0.035)


def test_ensemble_agreement_surfaces_when_persisted():
    """A8.4 — ensemble forecast'lar ensemble_direction_agreement doldurur."""
    tmp_db = _make_db_with_ensemble()
    svc = AnalysisService(
        db_path=tmp_db,
        outputs_base=tempfile.mkdtemp(),
        enable_background_refresh=False,
    )
    result = svc.build("ASELS")
    assert result.forecast.ensemble_agreement is not None
    assert result.forecast.ensemble_agreement == pytest.approx(0.83)


def test_single_model_ensemble_agreement_is_none():
    """Tek model forecast'larda ensemble_agreement None."""
    tmp_db = _make_db_with_high_quality_model()
    svc = AnalysisService(
        db_path=tmp_db,
        outputs_base=tempfile.mkdtemp(),
        enable_background_refresh=False,
    )
    result = svc.build("TUPRS")
    assert result.forecast.ensemble_agreement is None


def test_disclaimer_always_present_in_all_paths():
    """A8.5 — disclaimer her yanitta dolu."""
    tmp_db = _make_db_with_high_quality_model()
    svc = AnalysisService(
        db_path=tmp_db,
        outputs_base=tempfile.mkdtemp(),
        enable_background_refresh=False,
    )
    result = svc.build("TUPRS")
    assert result.disclaimer
    assert "yatırım tavsiyesi" in result.disclaimer.lower()


@pytest.mark.skipif(not _HAVE_TEST_CLIENT, reason="httpx/TestClient yuklu degil")
def test_v1_alias_returns_same_response_as_unversioned():
    """A8.6 — /v1/analysis/{symbol} /analysis ile ayni payload."""
    from src.api.main import app
    from src.api.routers.analysis import _service as router_service

    tmp_db = _make_db_with_high_quality_model()
    new_svc = AnalysisService(
        db_path=tmp_db,
        outputs_base=tempfile.mkdtemp(),
        enable_background_refresh=False,
    )
    # Router'daki module-level service'i geciici degistir
    import src.api.routers.analysis as analysis_module
    original = analysis_module._service
    analysis_module._service = new_svc
    try:
        client = TestClient(app)
        r1 = client.get("/analysis/TUPRS")
        r2 = client.get("/v1/analysis/TUPRS")
        assert r1.status_code == 200
        assert r2.status_code == 200
        b1 = r1.json()
        b2 = r2.json()
        # generated_at saniye saniye degisebilir; o haric esit
        b1.pop("generated_at", None)
        b2.pop("generated_at", None)
        assert b1 == b2
    finally:
        analysis_module._service = original


@pytest.mark.skipif(not _HAVE_FASTAPI, reason="FastAPI yuklu degil")
def test_v1_alias_route_registered():
    """A8.6 — /v1 alias route FastAPI'de kayitli."""
    from src.api.main import app
    paths = {route.path for route in app.routes}
    assert "/v1/analysis/{symbol}" in paths
    assert "/analysis/{symbol}" in paths


def test_data_quality_block_propagated():
    """A7.3 + A8 — data_quality block yanit icinde."""
    tmp_db = _make_db_with_high_quality_model()
    svc = AnalysisService(
        db_path=tmp_db,
        outputs_base=tempfile.mkdtemp(),
        enable_background_refresh=False,
    )
    result = svc.build("TUPRS")
    assert result.data_quality is not None
    assert result.data_quality.psi_status in (
        "stable", "moderate_drift", "major_drift", "unavailable",
    )
