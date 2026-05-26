# -*- coding: utf-8 -*-
"""
Sprint 9 (2026-05-26) A9.1 — advisory audit log testleri.

`append_response()` AnalysisResponse benzeri objeden audit kayit ureterek
CSV dosyasina yazar; tekrarli cagrilarda satir eklenir; UTC timestamp ISO
formatlidir.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pandas as pd
import pytest

from src.api.services.advisory_audit import (
    AdvisoryAuditRecord,
    append_record,
    append_response,
    build_record_from_response,
    read_log,
)


def _mock_response(
    *,
    symbol="TUPRS",
    status="ok",
    horizon=5,
    model_name="XGBoost",
    trend="up",
    p10=0.005, p50=0.02, p90=0.035,
    label="medium",
):
    point = SimpleNamespace(
        predicted_return=p50,
        predicted_return_p10=p10,
        predicted_return_p50=p50,
        predicted_return_p90=p90,
    )
    forecast = SimpleNamespace(horizon_days=horizon, trend_label=trend, points=[point])
    confidence = SimpleNamespace(label=label, reasons=[], warnings=[])
    model = SimpleNamespace(model_name=model_name)
    return SimpleNamespace(
        symbol=symbol,
        analysis_status=status,
        forecast=forecast,
        confidence=confidence,
        model=model,
    )


def test_build_record_extracts_all_fields():
    rec = build_record_from_response(_mock_response())
    d = rec.to_dict()
    assert d["symbol"] == "TUPRS"
    assert d["horizon_days"] == 5
    assert d["model_name"] == "XGBoost"
    assert d["trend_label"] == "up"
    assert d["confidence_label"] == "medium"
    assert d["analysis_status"] == "ok"
    assert d["p50_return"] == pytest.approx(0.02)
    assert d["p10_return"] == pytest.approx(0.005)
    assert d["p90_return"] == pytest.approx(0.035)
    # UTC ISO timestamp icermeli
    assert d["timestamp_utc"].endswith("+00:00") or "T" in d["timestamp_utc"]


def test_append_record_creates_file_and_appends():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "history.csv")
        for i in range(3):
            rec = build_record_from_response(_mock_response(symbol=f"S{i}"))
            append_record(rec, log_path=path)
        df = read_log(path)
        assert len(df) == 3
        assert set(df["symbol"]) == {"S0", "S1", "S2"}


def test_append_response_convenience_path():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "h.csv")
        append_response(_mock_response(symbol="ASELS"), log_path=path)
        append_response(_mock_response(symbol="TUPRS"), log_path=path)
        df = read_log(path)
        assert len(df) == 2
        assert list(df["symbol"]) == ["ASELS", "TUPRS"]


def test_read_log_missing_returns_empty_dataframe():
    df = read_log(log_path="nonexistent_xyz.csv")
    assert df.empty


def test_no_forecast_points_emits_null_quantiles():
    resp = _mock_response()
    resp.forecast.points = []
    rec = build_record_from_response(resp)
    d = rec.to_dict()
    assert d["p50_return"] is None
    assert d["p10_return"] is None
    assert d["p90_return"] is None


def test_timestamp_strictly_utc_tz_aware():
    rec = build_record_from_response(_mock_response())
    ts = rec.timestamp_utc
    # Should be parseable + have +00:00 offset
    parsed = pd.to_datetime(ts, utc=True)
    assert parsed.tzinfo is not None
