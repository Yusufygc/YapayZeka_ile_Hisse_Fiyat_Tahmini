# -*- coding: utf-8 -*-
"""Forward interval: roll_forward üretimi + persistence round-trip + coverage."""
from __future__ import annotations

import os
import shutil

import pandas as pd
import pytest

from src.database.stock_model_db import StockModelDB
from src.forecasting.workflows import ForecastPointGenerator


def _workspace_tmp(name: str) -> str:
    path = os.path.abspath(os.path.join("outputs", "_test_interval", name))
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


# --- roll_forward model-agnostik interval dalı --------------------------

class _Band:
    lower_band = None
    upper_band = None
    price_tick = None


class _Rules:
    def bound_forecast_price(self, raw, prev):  # passthrough (clip yok)
        return float(raw), _Band()


class _Ctx:
    rules = _Rules()

    @staticmethod
    def target_to_price(target, prev_close, target_mode):
        # return modu: prev*(1+t)
        return prev_close * (1.0 + float(target))


def _generator():
    return ForecastPointGenerator(_Ctx())


def test_residual_interval_branch_sets_quantiles():
    gen = _generator()
    point = {"bounded_predicted_close": 102.0, "predicted_return": 0.02}
    calib = {
        "method": "residual_b2",
        "sigma": 0.05,
        "levels": [0.8],
        "sigma_by_regime": {},
    }
    frame = pd.DataFrame({"Market_Regime": ["calm"]})
    gen._apply_model_agnostic_interval(
        point=point,
        predicted_target=0.02,
        previous_close=100.0,
        target_mode="return",
        calibration=calib,
        horizon_index=1,
        frame=frame,
    )
    assert point["interval_method"] == "residual_b2"
    assert point["p50_close"] == pytest.approx(102.0)
    assert point["p10_close"] < point["p50_close"] < point["p90_close"]
    assert point["predicted_return_p10"] < point["predicted_return_p90"]


def test_conformal_interval_branch():
    gen = _generator()
    point = {"bounded_predicted_close": 100.0, "predicted_return": 0.0}
    calib = {"method": "conformal", "q_hat": 0.05, "level": 0.9}
    frame = pd.DataFrame({"Market_Regime": ["calm"]})
    gen._apply_model_agnostic_interval(
        point=point,
        predicted_target=0.0,
        previous_close=100.0,
        target_mode="return",
        calibration=calib,
        horizon_index=1,
        frame=frame,
    )
    assert point["interval_method"] == "conformal"
    assert point["p10_close"] == pytest.approx(95.0)
    assert point["p90_close"] == pytest.approx(105.0)


def test_no_calibration_is_noop():
    gen = _generator()
    point = {"bounded_predicted_close": 100.0, "predicted_return": 0.0}
    gen._apply_model_agnostic_interval(
        point=point,
        predicted_target=0.0,
        previous_close=100.0,
        target_mode="return",
        calibration=None,
        horizon_index=1,
        frame=pd.DataFrame(),
    )
    assert "p10_close" not in point
    assert "interval_method" not in point


# --- persistence round-trip + coverage ----------------------------------

def _interval_points():
    return [
        {
            "target_date": "2026-05-04",
            "horizon_index": 1,
            "raw_predicted_close": 101.0,
            "bounded_predicted_close": 101.0,
            "predicted_return": 0.01,
            "lower_band": 90.0,
            "upper_band": 110.0,
            "price_tick": 0.05,
            "p10_close": 99.0,
            "p50_close": 101.0,
            "p90_close": 103.0,
            "predicted_return_p10": -0.01,
            "predicted_return_p50": 0.01,
            "predicted_return_p90": 0.03,
            "interval_method": "residual_b2",
        },
        {
            "target_date": "2026-05-05",
            "horizon_index": 2,
            "raw_predicted_close": 102.0,
            "bounded_predicted_close": 102.0,
            "predicted_return": 0.0099,
            "lower_band": 90.9,
            "upper_band": 111.1,
            "price_tick": 0.05,
            "p10_close": 100.0,
            "p50_close": 102.0,
            "p90_close": 104.0,
            "predicted_return_p10": -0.0098,
            "predicted_return_p50": 0.0099,
            "predicted_return_p90": 0.0196,
            "interval_method": "residual_b2",
        },
    ]


def _log_kwargs(points):
    return dict(
        stock_symbol="ITEST",
        model_name="Ridge Return",
        source_experiment_id=None,
        last_observed_date="2026-04-30",
        last_close=100.0,
        horizon_days=2,
        trend_label="UP",
        weekly_expected_return=0.02,
        trend_threshold=0.005,
        rules_version="test_rules",
        points=points,
    )


def test_interval_roundtrip_and_coverage():
    tmp = _workspace_tmp("db")
    db = StockModelDB(os.path.join(tmp, "stock_models.db"))
    db.log_forecast_run(**_log_kwargs(_interval_points()))

    latest = db.get_latest_forecast("ITEST")
    pt0 = latest["points"][0]
    assert pt0["p10_close"] == 99.0
    assert pt0["p90_close"] == 103.0
    assert pt0["interval_method"] == "residual_b2"

    # actual'lar bandın İÇİNDE -> coverage %100
    resolved = db.resolve_forecasts("ITEST", {"2026-05-04": 101.5, "2026-05-05": 103.0})
    assert resolved == 2
    latest = db.get_latest_forecast("ITEST")
    assert latest["accuracy_summary"]["interval_coverage"] == pytest.approx(100.0)
    assert latest["accuracy_summary"]["interval_avg_width"] == pytest.approx(4.0)


def test_interval_coverage_partial():
    tmp = _workspace_tmp("db2")
    db = StockModelDB(os.path.join(tmp, "stock_models.db"))
    db.log_forecast_run(**_log_kwargs(_interval_points()))
    # ikinci nokta band DIŞINDA (104 üst sınır, actual 120) -> coverage %50
    db.resolve_forecasts("ITEST", {"2026-05-04": 101.5, "2026-05-05": 120.0})
    latest = db.get_latest_forecast("ITEST")
    assert latest["accuracy_summary"]["interval_coverage"] == pytest.approx(50.0)


def test_coverage_report_tabulates_method():
    tmp = _workspace_tmp("db_report")
    db_path = os.path.join(tmp, "stock_models.db")
    db = StockModelDB(db_path)
    db.log_forecast_run(**_log_kwargs(_interval_points()))
    db.resolve_forecasts("ITEST", {"2026-05-04": 101.5, "2026-05-05": 103.0})

    from tools.interval_coverage_report import build_report

    rows = build_report(db_path)
    methods = {r["method"]: r for r in rows}
    assert "residual_b2" in methods
    assert methods["residual_b2"]["avg_coverage"] == pytest.approx(100.0)
    assert methods["residual_b2"]["resolved_runs"] == 1


def test_legacy_points_without_interval_are_null():
    tmp = _workspace_tmp("db3")
    db = StockModelDB(os.path.join(tmp, "stock_models.db"))
    legacy = [
        {
            "target_date": "2026-05-04",
            "horizon_index": 1,
            "raw_predicted_close": 101.0,
            "bounded_predicted_close": 101.0,
            "predicted_return": 0.01,
            "lower_band": 90.0,
            "upper_band": 110.0,
            "price_tick": 0.05,
        }
    ]
    db.log_forecast_run(**_log_kwargs(legacy))
    latest = db.get_latest_forecast("ITEST")
    pt0 = latest["points"][0]
    assert pt0["p10_close"] is None
    assert pt0["interval_method"] is None
