# -*- coding: utf-8 -*-
"""P1 deep-audit (2026-11): forecast interval / recursive roll-forward regression tests.

Kapsam:
  1. Quantile-model bantlari model-agnostik kalibrasyondan (B2/C) gecer;
     modelin kendi bandi yalnizca ``raw_model_quantiles`` teshis metadatasi.
  2. ``cqr`` modu reddedilir; ``adaptive_conformal_update`` kaldirildi;
     ``coverage_guarantee`` payload'da.
  3. Sentetik recursive satir: Low <= Close <= High, Volume = trailing medyan,
     recompute hatasi loglanir + ``recompute_failed_step_{k}`` uyarisi.
  4. h>1 icin ``recursive_forecast_validated=False`` + not.
  5. MacroForwardProjector: *_Return/*_Change -> 0, seviye -> son deger (ARIMA yok).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.features.macro_forward_projection import MacroForwardProjector
from src.forecasting import interval_calibration as ic
from src.forecasting.interval_calibration import (
    compute_conformal_calibration,
    compute_residual_calibration,
    z_for_level,
)
from src.forecasting.runner import ForecastResult
from src.forecasting.workflows import ForecastPointGenerator, ForecastSymbolWorkflow


# --------------------------------------------------------------------------
# Ortak stub'lar
# --------------------------------------------------------------------------


class _Band:
    lower_band = None
    upper_band = None
    price_tick = None


class _Rules:
    def bound_forecast_price(self, raw, prev):  # clip yok
        return float(raw), _Band()

    def next_trading_days(self, last, n):
        return list(pd.bdate_range(pd.Timestamp(last) + pd.Timedelta(days=1), periods=n))

    def trend_threshold(self, closes, horizon_days=5):
        return 0.01

    def trend_label(self, ret, thr):
        return "flat"


class _Ctx:
    rules = _Rules()

    @staticmethod
    def target_to_price(target, prev_close, target_mode):
        return prev_close * (1.0 + float(target))


class _IdentityScaler:
    def transform(self, x):
        return np.asarray(x, dtype=float)


class _QuantilePredictor:
    """Quantile destekli modelin predictor'u: median 0.01, ham band cok dar/farkli."""

    MEDIAN = 0.01

    def predict(self, model_name, model, context):
        return self.MEDIAN

    def predict_quantiles_target(self, model_name, model, context):
        return {0.1: -0.001, 0.5: self.MEDIAN, 0.9: 0.021}


def _rec_context(calibration, *, pref="residual_b2"):
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-01", periods=30),
            "Close": np.linspace(100.0, 100.0, 30),
            "f1": np.zeros(30),
        }
    )
    return {
        "features": ["f1"],
        "target_mode": "return",
        "scaler_X": _IdentityScaler(),
        "feature_frame": frame,
        "last_close": 100.0,
        "last_observed_date": frame["Date"].iloc[-1],
        "time_steps": 5,
        "interval_calibration": calibration,
        "interval_method_pref": pref,
        "forecast_warnings": ["projected_exogenous_features"],
    }


def _roll(calibration, horizon=3, pref="residual_b2"):
    gen = ForecastPointGenerator(_Ctx())
    ctx = _rec_context(calibration, pref=pref)
    pts = gen.roll_forward_recursive(
        model_name="Ridge Return",
        model=object(),
        context=ctx,
        predictor=_QuantilePredictor(),
        horizon_days=horizon,
    )
    return pts, ctx


# --------------------------------------------------------------------------
# 1. Quantile-model bantlari kalibrasyondan gecer
# --------------------------------------------------------------------------


def test_quantile_model_band_equals_calibrated_residual_path():
    sigma = 0.02
    calib = {"method": "residual_b2", "sigma": sigma, "levels": [0.8], "sigma_by_regime": {}}
    pts, _ = _roll(calib, horizon=1)
    p = pts[0]
    z = z_for_level(0.8)
    half = z * sigma  # h=1
    assert p["interval_method"] == "residual_b2"
    assert p["p50_close"] == pytest.approx(100.0 * (1.0 + 0.01))
    assert p["p10_close"] == pytest.approx(100.0 * (1.0 + 0.01 - half))
    assert p["p90_close"] == pytest.approx(100.0 * (1.0 + 0.01 + half))
    # Modelin ham bandi (-0.001/0.021) YAYINLANAN band degil.
    assert p["p10_close"] != pytest.approx(100.0 * (1.0 - 0.001))
    # Ham model quantile'lari yalniz teshis olarak saklanir.
    raw = p["raw_model_quantiles"]
    assert raw["quantile_close"]["p10"] == pytest.approx(100.0 * (1.0 - 0.001))
    assert raw["quantile_close"]["p90"] == pytest.approx(100.0 * (1.0 + 0.021))
    assert "quantile_close" not in p  # top-level uncalibrated band kalmadi


def test_quantile_model_band_widens_with_horizon():
    calib = {"method": "residual_b2", "sigma": 0.02, "levels": [0.8], "sigma_by_regime": {}}
    pts, _ = _roll(calib, horizon=4)
    widths = [(p["p90_close"] - p["p10_close"]) / p["p50_close"] for p in pts]
    assert widths[1] > widths[0] * 1.2
    assert widths[3] > widths[1]


def test_quantile_model_uses_conformal_when_preferred():
    calib = {
        "method": "residual_b2",
        "sigma": 0.02,
        "levels": [0.8],
        "conformal": {"method": "conformal", "q_hat": 0.05, "level": 0.9},
    }
    pts, _ = _roll(calib, horizon=1, pref="conformal")
    assert pts[0]["interval_method"] == "conformal"
    assert pts[0]["p10_close"] == pytest.approx(100.0 * (1.0 + 0.01 - 0.05))
    assert pts[0]["p90_close"] == pytest.approx(100.0 * (1.0 + 0.01 + 0.05))


def test_quantile_model_without_calibration_publishes_no_uncalibrated_band():
    pts, _ = _roll(None, horizon=1)
    p = pts[0]
    assert "p10_close" not in p and "p90_close" not in p
    assert p.get("interval_method") is None
    assert p["raw_model_quantiles"]["quantile_close"]["p90"] == pytest.approx(102.1)


# --------------------------------------------------------------------------
# 2. interval_calibration temizligi + coverage_guarantee
# --------------------------------------------------------------------------


def _records():
    rng = np.random.default_rng(0)
    return [{"y_true_target": rng.normal(0, 0.02, 200), "y_pred_target": np.zeros(200)}]


def test_cqr_mode_is_rejected():
    with pytest.raises(NotImplementedError):
        compute_conformal_calibration(_records(), level=0.9, mode="cqr")


def test_conformal_default_mode_still_works():
    out = compute_conformal_calibration(_records(), level=0.9)
    assert out is not None and out["q_hat"] > 0


def test_adaptive_conformal_update_removed():
    assert not hasattr(ic, "adaptive_conformal_update")


def test_coverage_guarantee_in_calibration_and_points():
    expected = (
        "split-conformal, h=1, exchangeability assumed; "
        "h>1 widened by sqrt(h) heuristic, no guarantee"
    )
    assert ic.COVERAGE_GUARANTEE == expected
    assert compute_conformal_calibration(_records())["coverage_guarantee"] == expected
    assert compute_residual_calibration(_records())["coverage_guarantee"] == expected
    calib = {"method": "residual_b2", "sigma": 0.02, "levels": [0.8], "sigma_by_regime": {}}
    pts, _ = _roll(calib, horizon=1)
    assert pts[0]["coverage_guarantee"] == expected


# --------------------------------------------------------------------------
# 3. Sentetik recursive satir
# --------------------------------------------------------------------------


def _ohlcv_frame(n=40):
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-01", periods=n),
            "Open": close + 0.3,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": rng.integers(1_000, 90_000, n).astype(float),
            "Return": np.zeros(n),
            "Log_Return": np.zeros(n),
        }
    )


@pytest.mark.parametrize("move", [1.05, 0.93])
def test_synthetic_row_is_internally_consistent(move):
    frame = _ohlcv_frame()
    prev = float(frame["Close"].iloc[-1])
    new_close = prev * move
    out = ForecastPointGenerator._append_recursive_row(
        frame=frame,
        target_date=pd.Timestamp("2026-06-01"),
        bounded_close=new_close,
        previous_close=prev,
    )
    row = out.iloc[-1]
    assert row["Open"] == pytest.approx(prev)
    assert row["Low"] <= row["Close"] <= row["High"]
    assert row["High"] == pytest.approx(max(prev, new_close))
    assert row["Low"] == pytest.approx(min(prev, new_close))
    assert row["Volume"] == pytest.approx(float(np.median(frame["Volume"].iloc[-20:])))


def test_recompute_relative_strength_uses_new_return_and_projected_market():
    frame = pd.DataFrame(
        {
            "Return": [0.01, 0.02],
            "BIST100_Return": [0.005, 0.0],
            "Relative_Strength": [0.005, 0.0],  # son satir bayat (kopya)
        }
    )
    out = ForecastPointGenerator._recompute_relative_strength(frame)
    assert out["Relative_Strength"].iloc[-1] == pytest.approx(0.02)


def test_recompute_failure_logs_and_flags_warning(monkeypatch, caplog):
    from src.features import feature_pipeline

    def _boom(self, frame):
        raise RuntimeError("recompute exploded")

    monkeypatch.setattr(feature_pipeline.FeaturePipeline, "recompute_close_dependent", _boom)
    gen = ForecastPointGenerator(_Ctx())
    frame = _ohlcv_frame()
    context = {"forecast_warnings": ["projected_exogenous_features"]}
    with caplog.at_level(logging.WARNING, logger="src.forecasting.workflows"):
        out = gen._recompute_close_dependent_safe(frame, context, step=2)
    assert out is frame
    assert "recompute_failed_step_2" in context["forecast_warnings"]
    assert "projected_exogenous_features" in context["forecast_warnings"]
    assert any(
        r.levelno == logging.WARNING and "recompute" in r.getMessage().lower()
        for r in caplog.records
    )


def test_roll_forward_records_recompute_failure_per_step():
    # feature_frame'de High/Low/Volume yok -> recompute gercekten basarisiz olur.
    calib = {"method": "residual_b2", "sigma": 0.02, "levels": [0.8], "sigma_by_regime": {}}
    _, ctx = _roll(calib, horizon=2)
    warns = ctx["forecast_warnings"]
    assert "recompute_failed_step_1" in warns
    assert "recompute_failed_step_2" in warns


# --------------------------------------------------------------------------
# 4. h>1 dogrulanmamis uyarisi
# --------------------------------------------------------------------------


def _workflow(horizon_warnings_sink):
    point = {
        "horizon_index": 1,
        "target_date": "2026-05-04",
        "raw_predicted_close": 101.0,
        "bounded_predicted_close": 101.0,
        "predicted_return": 0.01,
        "lower_band": 90.0,
        "upper_band": 110.0,
        "price_tick": 0.05,
    }

    class _Resolver:
        def resolve(self, symbol, force_model_name=None):
            return {
                "model_name": "Random Forest",
                "source_experiment_id": 1,
                "target_mode": "log_return",
                "feature_mode": "stationary_features",
                "scaling_mode": "robust_x_standard_y_clip",
                "model_path": "",
                "ensemble_metadata": None,
            }

    class _Prep:
        def prepare(self, **kw):
            return SimpleNamespace(
                df=pd.DataFrame(
                    {
                        "Date": pd.bdate_range("2026-01-01", periods=90),
                        "Close": np.linspace(100.0, 110.0, 90),
                    }
                )
            )

    class _Train:
        def train(self, name, dm, *, selection=None):
            return object(), {
                "last_close": 100.0,
                "last_observed_date": pd.Timestamp("2026-04-30"),
                "forecast_strategy": "recursive_direct_target",
                "artifact_mode": "artifact_loaded",
                "forecast_warnings": ["projected_exogenous_features"],
            }

    class _Gen:
        def roll_forward_recursive(self, **kw):
            return [dict(point)]

    class _Db:
        def log_forecast_run(self, **kw):
            horizon_warnings_sink.append(kw.get("forecast_warnings"))
            return 7

    ctx = SimpleNamespace(
        project_root=".",
        rules=_Rules(),
        model_resolver=_Resolver(),
        data_preparation_service=_Prep(),
        production_training_workflow=_Train(),
        forecast_point_generator=_Gen(),
        db=_Db(),
        latest_target_prediction_workflow=object(),
    )
    return ForecastSymbolWorkflow(ctx, ForecastResult)


def test_horizon_gt_1_marks_recursive_forecast_unvalidated():
    sink: list = []
    res = _workflow(sink).run(symbol="T", data_file="x.csv", horizon_days=5)
    assert res.recursive_forecast_validated is False
    assert res.horizon_validation_note and "unvalidated" in res.horizon_validation_note.lower()
    assert "recursive_horizon_unvalidated" in sink[0]
    assert "projected_exogenous_features" in sink[0]


def test_horizon_1_has_no_unvalidated_warning():
    sink: list = []
    res = _workflow(sink).run(symbol="T", data_file="x.csv", horizon_days=1)
    assert res.recursive_forecast_validated is True
    assert res.horizon_validation_note is None
    assert "recursive_horizon_unvalidated" not in sink[0]


# --------------------------------------------------------------------------
# 5. MacroForwardProjector
# --------------------------------------------------------------------------


def _macro_frame(n=60):
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2025-01-01", periods=n),
            "Close": 100.0 + np.cumsum(rng.normal(0, 1, n)),
            "Return": rng.normal(0, 0.01, n),
            "Log_Return": rng.normal(0, 0.01, n),
            "USDTRY_Return": rng.normal(0.002, 0.01, n),
            "BIST_XUSIN_Return": rng.normal(0, 0.01, n),  # whitelist'te yok -> sonek ile yakalanir
            "Rate_Change": rng.normal(0, 0.1, n),
            "USDTRY": 30.0 + np.cumsum(rng.normal(0.01, 0.05, n)),
            "VIX": 18.0 + rng.normal(0, 2, n),
        }
    )


def test_projection_zero_for_return_and_change_last_value_for_levels():
    df = _macro_frame()
    out = MacroForwardProjector().project_last_row(df)
    last = out.index[-1]
    for col in ("USDTRY_Return", "BIST_XUSIN_Return", "Rate_Change"):
        assert out.at[last, col] == 0.0, col
    # Seviye: random walk -> son GOZLENEN (history, son satir haric) deger.
    assert out.at[last, "USDTRY"] == pytest.approx(df["USDTRY"].iloc[-2])
    assert out.at[last, "VIX"] == pytest.approx(df["VIX"].iloc[-2])
    # Hissenin kendi getirileri makro degildir -> dokunulmaz.
    assert out.at[last, "Return"] == pytest.approx(df["Return"].iloc[-1])
    assert out.at[last, "Log_Return"] == pytest.approx(df["Log_Return"].iloc[-1])
    # Gecmis satirlar degismez.
    pd.testing.assert_frame_equal(out.iloc[:-1], df.iloc[:-1])


def test_projection_has_no_arima_machinery():
    p = MacroForwardProjector()
    assert not hasattr(p, "_arima_forecast_one")
    assert not hasattr(p, "arima_order")
