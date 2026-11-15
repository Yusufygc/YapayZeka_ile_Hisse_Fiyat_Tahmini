# -*- coding: utf-8 -*-
"""API v2: /analysis, /symbols, /health yalniz gecici PeerStore DB'sinden (Kol-A yok)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import src.api.main as api_main  # noqa: E402
from src.api.routers import analysis as analysis_router  # noqa: E402
from src.api.schemas.analysis import PeerBlock  # noqa: E402
from src.api.services.analysis_service import AnalysisService  # noqa: E402
from src.api.services.response_cache import ResponseCache  # noqa: E402
from src.api.services.serving_health import serving_pool_status  # noqa: E402
from src.api.services.symbols_service import list_serving_symbols  # noqa: E402
from src.serving.peer_store import GlobalRunMeta, PeerStore  # noqa: E402

_IST = timezone(timedelta(hours=3))
_NOW = datetime(2026, 11, 16, 12, 0, tzinfo=_IST)  # pazartesi
_AS_OF = "2026-11-15"


def _peer_row(symbol: str, label: str = "high", **extra) -> dict:
    row = {
        "symbol": symbol, "as_of_date": _AS_OF, "peer_score": 0.4, "peer_percentile": 70.0,
        "peer_label": "outperform", "raw_pred": 0.01, "universe_size": 500,
        "segment_sector": "Industrials", "segment_icir": 1.2,
        "confidence_label": label, "confidence_reasons": ["Segment sinyali guclu"],
        "confidence_warnings": [], "trend_label": "yukarı", "trend_prob_up": 0.56,
        "trend_expected_return": 0.004, "kolb_price_p50": 101.0, "kolb_price_low": 97.0,
        "kolb_price_high": 105.0, "kolb_horizon_days": 5, "kolb_band_level": 0.8,
    }
    row.update(extra)
    return row


def _make_pool(path, rows: list[dict], as_of: str = _AS_OF) -> str:
    store = PeerStore(str(path))
    rid = store.insert_run(
        GlobalRunMeta(model_name="EnsemblePooled", as_of_date=as_of, n_symbols=len(rows), icir=1.4)
    )
    df = pd.DataFrame(rows)
    store.insert_peer_scores(rid, df)
    return str(path)


def _write_market(data_dir, symbol: str, last_date: str, n: int = 40) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(end=last_date, periods=n)
    close = 100.0 + np.arange(n) * 0.1
    pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"), "Open": close, "High": close * 1.01,
        "Low": close * 0.99, "Close": close, "Volume": np.full(n, 1e6),
    }).to_csv(data_dir / f"{symbol}.csv", index=False)


class _Env:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.pool = str(tmp_path / "serving_pool.db")
        self.data = tmp_path / "data"
        self.monkeypatch = monkeypatch

    def service(self, peer_override: Optional[PeerBlock] = None) -> AnalysisService:
        svc = AnalysisService(
            serving_db_path=self.pool, data_dir=str(self.data),
            calendar_path=str(self.tmp / "no_calendar.csv"), now_fn=lambda: _NOW,
        )
        if peer_override is not None:
            svc._peers = type("P", (), {"get_peer_block": lambda s, sym: peer_override})()
        return svc

    def client(self, svc: Optional[AnalysisService] = None) -> TestClient:
        svc = svc or self.service()
        self.monkeypatch.setattr(analysis_router, "_service", svc)
        self.monkeypatch.setattr(analysis_router, "_cache", ResponseCache(ttl_seconds=0))
        self.monkeypatch.setattr(analysis_router, "log_event", lambda *a, **k: None)
        self.monkeypatch.setattr(api_main, "serving_pool_status", lambda: serving_pool_status(self.pool, today=_NOW.date()))
        self.monkeypatch.setattr(api_main, "list_serving_symbols", lambda: list_serving_symbols(self.pool))
        return TestClient(api_main.app)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    return _Env(tmp_path, monkeypatch)


# ------------------------------------------------------------------ /analysis
def test_ok_response_has_v2_shape_and_no_kol_a_blocks(env):
    _make_pool(env.pool, [_peer_row("TUPRS", "high")])
    _write_market(env.data, "TUPRS", "2026-11-16")
    body = env.client().get("/analysis/tuprs").json()
    assert body["api_version"] == "2" and body["symbol"] == "TUPRS"
    assert body["analysis_status"] == "ok"
    assert body["confidence"]["label"] == "high"
    assert "Segment sinyali guclu" in body["confidence"]["reasons"]
    assert body["peer"]["available"] is True and body["peer"]["peer_label"] == "outperform"
    assert body["peer"]["kolb_price_low"] == 97.0 and body["peer"]["kolb_price_high"] == 105.0
    assert body["data"]["last_observed_date"] == "2026-11-16" and body["data"]["last_close"] > 0
    assert body["data"]["data_freshness"] == "fresh"
    assert body["serving_run"]["as_of_date"] == _AS_OF and body["serving_run"]["stale"] is False
    assert body["disclaimer"]
    for removed in ("model", "forecast", "performance", "xai", "forecast_source",
                    "refresh_status", "refresh_reason", "refresh_job_id"):
        assert removed not in body


def test_v1_alias_matches_unversioned(env):
    _make_pool(env.pool, [_peer_row("TUPRS", "medium")])
    _write_market(env.data, "TUPRS", "2026-11-16")
    client = env.client()
    a = client.get("/analysis/TUPRS").json()
    b = client.get("/v1/analysis/TUPRS").json()
    a.pop("generated_at"), b.pop("generated_at")
    assert a == b


def test_symbol_not_in_latest_run_is_no_data(env):
    _make_pool(env.pool, [_peer_row("TUPRS")])
    resp = env.client().get("/analysis/ZZZZ")
    assert resp.status_code == 200
    body = resp.json()
    assert body["analysis_status"] == "no_data" and body["peer"] is None
    assert body["confidence"]["label"] == "low" and body["confidence"]["warnings"]
    assert body["serving_run"]["run_id"] == 1


def test_missing_serving_db_is_no_data_and_not_created(env):
    body = env.client().get("/analysis/TUPRS").json()
    assert body["analysis_status"] == "no_data" and body["serving_run"] is None
    assert not (env.tmp / "serving_pool.db").exists()


def test_stale_market_data_sets_stale_status_and_low_confidence(env):
    _make_pool(env.pool, [_peer_row("TUPRS", "high")])
    _write_market(env.data, "TUPRS", "2026-11-02")  # 10 islem gunu eski
    body = env.client().get("/analysis/TUPRS").json()
    assert body["analysis_status"] == "stale_data"
    assert body["data"]["data_freshness"] == "stale_data" and body["data"]["staleness_days"] >= 5
    assert body["confidence"]["label"] == "low"
    assert any("Piyasa verisi" in w for w in body["confidence"]["warnings"])


def test_stale_serving_run_sets_stale_status(env):
    _make_pool(env.pool, [_peer_row("TUPRS", "high", as_of_date="2026-10-01")], as_of="2026-10-01")
    _write_market(env.data, "TUPRS", "2026-11-16")
    body = env.client().get("/analysis/TUPRS").json()
    assert body["analysis_status"] == "stale_data"
    assert body["serving_run"]["stale"] is True and body["serving_run"]["age_days"] > 7
    assert any("Gecelik havuz" in w for w in body["confidence"]["warnings"])


def test_low_peer_confidence_is_low_confidence_status(env):
    _make_pool(env.pool, [_peer_row("TUPRS", "low")])
    _write_market(env.data, "TUPRS", "2026-11-16")
    body = env.client().get("/analysis/TUPRS").json()
    assert body["analysis_status"] == "low_confidence"
    assert body["confidence"]["label"] == "low"


def test_missing_market_csv_falls_back_to_serving_as_of(env):
    _make_pool(env.pool, [_peer_row("TUPRS", "medium")])
    body = env.client().get("/analysis/TUPRS").json()
    assert body["data"]["last_observed_date"] == _AS_OF and body["data"]["last_close"] is None
    assert body["analysis_status"] == "ok"


class _RiskPeer(PeerBlock):
    """Risk modeli alanlari (diger dalin PeerBlock ekleri) getattr ile okunur."""

    risk_method: Optional[str] = None
    risk_level: Optional[str] = None


def _risk_peer(**kw) -> PeerBlock:
    base = dict(available=True, as_of_date=_AS_OF, confidence_label="high",
                confidence_reasons=["Segment sinyali guclu"], peer_label="outperform")
    base.update(kw)
    return _RiskPeer(**base)


def test_naive_fallback_risk_lowers_confidence_one_step(env):
    _make_pool(env.pool, [_peer_row("TUPRS")])
    _write_market(env.data, "TUPRS", "2026-11-16")
    peer = _risk_peer(risk_method="naive_fallback", risk_level="orta")
    body = env.client(env.service(peer)).get("/analysis/TUPRS").json()
    assert body["confidence"]["label"] == "medium"
    assert body["analysis_status"] == "ok"
    assert any("naive" in w for w in body["confidence"]["warnings"])


def test_naive_fallback_on_medium_becomes_low_confidence_status(env):
    _make_pool(env.pool, [_peer_row("TUPRS")])
    _write_market(env.data, "TUPRS", "2026-11-16")
    peer = _risk_peer(risk_method="naive_fallback", confidence_label="medium")
    body = env.client(env.service(peer)).get("/analysis/TUPRS").json()
    assert body["confidence"]["label"] == "low" and body["analysis_status"] == "low_confidence"


def test_regular_risk_method_keeps_label_and_adds_risk_reason(env):
    _make_pool(env.pool, [_peer_row("TUPRS")])
    _write_market(env.data, "TUPRS", "2026-11-16")
    peer = _risk_peer(risk_method="pooled_lgbm", risk_level="yüksek")
    body = env.client(env.service(peer)).get("/analysis/TUPRS").json()
    assert body["confidence"]["label"] == "high"
    assert "Risk seviyesi: yüksek." in body["confidence"]["reasons"]


def test_invalid_symbol_is_400(env):
    assert env.client().get("/analysis/BAD-SYMBOL!!").status_code == 400


# ---------------------------------------------------------- /symbols, /health
def test_symbols_lists_latest_run_only(env):
    _make_pool(env.pool, [_peer_row("TUPRS"), _peer_row("AKBNK"), _peer_row("ASELS")])
    body = env.client().get("/symbols").json()
    assert body == {"count": 3, "symbols": ["AKBNK", "ASELS", "TUPRS"]}


def test_symbols_empty_without_db(env):
    body = env.client().get("/symbols").json()
    assert body == {"count": 0, "symbols": []}
    assert not (env.tmp / "serving_pool.db").exists()


def test_health_reports_serving_pool_without_kol_a_db_fields(env, monkeypatch):
    _make_pool(env.pool, [_peer_row("TUPRS"), _peer_row("AKBNK")])
    monkeypatch.setattr(api_main, "nightly_run_status", lambda: {"available": False})
    body = env.client().get("/health").json()
    assert body["api_version"] == "2" and body["status"] == "ok"
    assert body["serving_pool"]["available"] and body["serving_pool"]["as_of_date"] == _AS_OF
    assert body["registered_symbols"] == 2
    for dropped in ("db_accessible", "schema", "latest_refresh_job", "db_path"):
        assert dropped not in body


def test_health_degraded_when_pool_missing_or_stale(env, monkeypatch):
    monkeypatch.setattr(api_main, "nightly_run_status", lambda: {"available": False})
    assert env.client().get("/health").json()["status"] == "degraded"
    _make_pool(env.pool, [_peer_row("TUPRS")], as_of="2026-10-01")
    assert env.client().get("/health").json()["status"] == "degraded"


# ----------------------------------------------------------- removed endpoints
@pytest.mark.parametrize("method,path", [
    ("get", "/best-model/TUPRS"), ("get", "/experiments/TUPRS"), ("get", "/metrics/TUPRS"),
    ("get", "/leaderboard"), ("post", "/run/TUPRS"), ("get", "/run/status/abc"),
    ("get", "/refresh/status/abc"),
])
def test_removed_kol_a_endpoints_return_404(env, method, path):
    client = env.client()
    resp = client.post(path, json={}) if method == "post" else client.get(path)
    assert resp.status_code == 404
