# -*- coding: utf-8 -*-
"""Serving sertlestirme (2026-11): salt-okunur okuyucular, 503/no_data ayrimi, risk durumu,
as_of geriligi, kapi degisimi, band merkezi, --horizon, Istanbul parmak izi, guvenlik."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import src.api.main as api_main  # noqa: E402
from src.api.routers import analysis as analysis_router  # noqa: E402
from src.api.schemas.analysis import PeerBlock  # noqa: E402
from src.api.services import response_cache as rc_mod  # noqa: E402
from src.api.services.analysis_service import AnalysisService  # noqa: E402
from src.api.services.peer_service import PeerEnrichmentService  # noqa: E402
from src.api.services.response_cache import ResponseCache, compute_freshness_fingerprint  # noqa: E402
from src.api.services.serving_health import nightly_run_status, serving_pool_status  # noqa: E402
from src.api.services.symbols_service import list_serving_symbols  # noqa: E402
from src.gui.core.peer_reader import PeerReader  # noqa: E402
from src.risk.serving import gate_flip_warning, previous_risk_method, risk_run_status  # noqa: E402
from src.serving import nightly as nl  # noqa: E402
from src.serving.nightly_risk import run_risk_step  # noqa: E402
from src.serving.nightly_scoring import attach_risk_and_band  # noqa: E402
from src.serving.peer_store import GlobalRunMeta, PeerStore  # noqa: E402
from src.serving.run_status import NightlyRunStatus, read_status, status_path_for_db  # noqa: E402

_IST = timezone(timedelta(hours=3))
_NOW = datetime(2026, 11, 16, 12, 0, tzinfo=_IST)  # pazartesi
_AS_OF = "2026-11-15"

_OLD_SCHEMA = """
CREATE TABLE global_model_runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
  as_of_date TEXT, model_name TEXT, data_snapshot_hash TEXT, n_symbols INTEGER, n_rows INTEGER,
  horizon INTEGER, ic_mean REAL, icir REAL, pct_ic_positive REAL, config_json TEXT);
CREATE TABLE peer_scores (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
  symbol TEXT NOT NULL, as_of_date TEXT, peer_score REAL, peer_percentile REAL, peer_label TEXT,
  raw_pred REAL, universe_size INTEGER, segment_liq TEXT, segment_vol TEXT, segment_sector TEXT,
  segment_icir REAL, confidence_label TEXT, confidence_reasons TEXT, confidence_warnings TEXT,
  UNIQUE(run_id, symbol));
INSERT INTO global_model_runs (created_at, as_of_date, model_name) VALUES ('2026-01-01', '2026-11-15', 'old');
INSERT INTO peer_scores (run_id, symbol, as_of_date, peer_score, confidence_label)
  VALUES (1, 'OLD1', '2026-11-15', 0.5, 'high');
"""


def _old_db(tmp_path) -> str:
    path = str(tmp_path / "old.db")
    con = sqlite3.connect(path)
    con.executescript(_OLD_SCHEMA)
    con.commit()
    con.close()
    return path


def _state(path: str):
    con = sqlite3.connect(path)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(peer_scores)")]
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        con.close()
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return cols, mode, digest, sorted(os.listdir(os.path.dirname(path)))


# --------------------------------------------------------- 1. salt-okunur okuyucular
def test_readers_never_migrate_or_change_journal_mode_on_old_schema_db(tmp_path):
    db = _old_db(tmp_path)
    before = _state(db)
    assert before[1] == "delete" and "risk_method" not in before[0]

    block = PeerEnrichmentService(db).get_peer_block("OLD1")
    assert block is not None and block.risk_method is None and block.risk_sigma_5d is None
    assert list_serving_symbols(db) == ["OLD1"]
    snap = PeerReader(db).load()
    assert snap.error is None and snap.run is not None and len(snap.rows) == 1
    assert nightly_run_status(db)["available"] is False
    assert serving_pool_status(db, today=date(2026, 11, 16))["available"] is True

    assert _state(db) == before  # sutunlar, journal_mode, dosya baytlari, dizin icerigi AYNI


def test_readonly_store_rejects_writes_and_does_not_create_file(tmp_path):
    db = _old_db(tmp_path)
    store = PeerStore.open_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        store.insert_run(GlobalRunMeta(model_name="x", as_of_date="2026-11-16"))
    missing = str(tmp_path / "sub" / "none.db")
    with pytest.raises(sqlite3.Error):
        PeerStore.open_readonly(missing).latest_run()
    assert not os.path.exists(missing) and not os.path.exists(os.path.dirname(missing))


def test_readonly_works_on_wal_db_written_by_writer(tmp_path):
    db = str(tmp_path / "wal.db")
    w = PeerStore(db)
    rid = w.insert_run(GlobalRunMeta(model_name="m", as_of_date="2026-11-15"))
    w.insert_peer_scores(rid, pd.DataFrame({"symbol": ["AAA"], "as_of_date": ["2026-11-15"]}))
    assert PeerStore.open_readonly(db).get_peer_score("AAA")["symbol"] == "AAA"
    assert PeerStore.open_readonly(db).get_run(rid)["model_name"] == "m"


# ---------------------------------------------------------------- helpers (API)
def _peer_row(symbol: str, **extra) -> dict:
    row = {
        "symbol": symbol, "as_of_date": _AS_OF, "peer_score": 0.4, "peer_percentile": 70.0,
        "peer_label": "outperform", "raw_pred": 0.01, "universe_size": 500,
        "confidence_label": "high", "confidence_reasons": ["Segment sinyali guclu"],
        "confidence_warnings": [], "trend_label": "yukarı", "trend_expected_return": 0.004,
        "kolb_price_low": 97.0, "kolb_price_high": 105.0,
        "risk_method": "har_x_ridge", "risk_sigma_5d": 0.03, "risk_z": 1.6,
    }
    row.update(extra)
    return row


def _make_pool(path, rows, as_of: str = _AS_OF, config: Optional[dict] = None) -> str:
    store = PeerStore(str(path))
    rid = store.insert_run(GlobalRunMeta(
        model_name="EnsemblePooled", as_of_date=as_of, n_symbols=len(rows), icir=1.4, config=config))
    store.insert_peer_scores(rid, pd.DataFrame(rows))
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
        self.tmp, self.monkeypatch = tmp_path, monkeypatch
        self.pool = str(tmp_path / "serving_pool.db")
        self.data = tmp_path / "data"
        self.cache = ResponseCache(ttl_seconds=3600)

    def client(self) -> TestClient:
        svc = AnalysisService(serving_db_path=self.pool, data_dir=str(self.data),
                              calendar_path=str(self.tmp / "no_cal.csv"), now_fn=lambda: _NOW)
        mp = self.monkeypatch
        mp.setattr(analysis_router, "_service", svc)
        mp.setattr(analysis_router, "_cache", self.cache)
        mp.setattr(analysis_router, "log_event", lambda *a, **k: None)
        mp.setattr(api_main, "serving_pool_status", lambda: serving_pool_status(self.pool, today=_NOW.date()))
        mp.setattr(api_main, "list_serving_symbols", lambda: list_serving_symbols(self.pool))
        return TestClient(api_main.app)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    return _Env(tmp_path, monkeypatch)


# --------------------------------------------- 2. unavailable (503) vs no_data, onbellek
def test_locked_db_gives_503_unavailable_and_is_not_cached(env, monkeypatch):
    _make_pool(env.pool, [_peer_row("TUPRS")])
    _write_market(env.data, "TUPRS", "2026-11-16")

    def _locked(self, symbol, run_id=None):
        raise sqlite3.OperationalError("database is locked")

    client = env.client()
    monkeypatch.setattr(PeerStore, "get_peer_score", _locked)
    resp = client.get("/analysis/TUPRS")
    assert resp.status_code == 503 and resp.json()["analysis_status"] == "unavailable"
    assert len(env.cache) == 0
    monkeypatch.undo()  # DB tekrar okunabilir -> 503 yapismamali
    client = env.client()
    ok = client.get("/analysis/TUPRS")
    assert ok.status_code == 200 and ok.json()["analysis_status"] == "ok"


def test_corrupt_db_is_503(env):
    with open(env.pool, "wb") as fh:
        fh.write(b"garbage" * 100)
    assert env.client().get("/analysis/TUPRS").status_code == 503


def test_unknown_symbols_are_no_data_200_and_never_cached(env):
    _make_pool(env.pool, [_peer_row("TUPRS")])
    client = env.client()
    for sym in ("AAAA", "BBBB", "CCCC"):
        r = client.get(f"/analysis/{sym}")
        assert r.status_code == 200 and r.json()["analysis_status"] == "no_data"
    assert len(env.cache) == 0
    _write_market(env.data, "TUPRS", "2026-11-16")
    assert client.get("/analysis/TUPRS").json()["analysis_status"] == "ok"
    assert len(env.cache) == 1


def test_response_cache_is_lru_capped():
    cache = ResponseCache(ttl_seconds=3600, max_entries=3)
    for i in range(3):
        cache.set(f"S{i}", i)
    assert cache.get("S0") == 0            # S0 en yeni oldu
    cache.set("S3", 3)                     # en eski (S1) atilir
    assert len(cache) == 3
    assert cache.get("S1") is None and cache.get("S0") == 0 and cache.get("S3") == 3


# ------------------------------------------------- 3. risk durumu: nightly -> /health -> /analysis
def test_risk_run_status_values():
    assert risk_run_status({"method": "har_x_ridge"})["status"] == "ok"
    assert risk_run_status({"method": "naive_fallback"})["status"] == "fallback"
    failed = risk_run_status({"error": "ValueError: boom"})
    assert failed["status"] == "failed" and failed["method"] is None and "boom" in failed["error"]


def test_run_risk_step_writes_failed_status_to_status_file(tmp_path, monkeypatch):
    import src.serving.nightly_risk as nr

    monkeypatch.setattr(nr, "compute_risk_safe", lambda panel, as_of, log=print, quarantined=None: (None, {"error": "RuntimeError: x"}))
    logs: list[str] = []
    db = str(tmp_path / "serving_pool.db")
    status = NightlyRunStatus(status_path_for_db(db))
    table, summary, warns = run_risk_step(pd.DataFrame(), pd.Timestamp("2026-11-13"), None,
                                          log=logs.append, status=status)
    assert table is None and summary["error"] and warns == []
    on_disk = read_status(status_path_for_db(db))
    assert on_disk["risk"]["status"] == "failed" and "RuntimeError" in on_disk["risk"]["error"]
    assert any("risk adimi basarisiz" in m for m in logs)
    assert nightly_run_status(db)["risk"]["status"] == "failed"


def test_health_is_degraded_when_nightly_risk_failed(env, monkeypatch):
    _make_pool(env.pool, [_peer_row("TUPRS")])
    healthy = {"available": True, "interrupted": False, "healthy": True}
    monkeypatch.setattr(api_main, "nightly_run_status", lambda: {**healthy, "risk": {"status": "ok"}})
    assert env.client().get("/health").json()["status"] == "ok"
    monkeypatch.setattr(api_main, "nightly_run_status", lambda: {**healthy, "risk": {"status": "fallback"}})
    assert env.client().get("/health").json()["status"] == "ok"
    monkeypatch.setattr(api_main, "nightly_run_status",
                        lambda: {**healthy, "risk": {"status": "failed", "error": "x"}})
    body = env.client().get("/health").json()
    assert body["status"] == "degraded" and body["nightly_run"]["risk"]["status"] == "failed"


def test_missing_risk_method_run_level_failure_warns_and_downgrades(env):
    _make_pool(env.pool, [_peer_row("TUPRS", risk_method=None, risk_sigma_5d=None, risk_z=None)],
               config={"risk": {"error": "RuntimeError: x"}})
    _write_market(env.data, "TUPRS", "2026-11-16")
    body = env.client().get("/analysis/TUPRS").json()
    assert body["confidence"]["label"] == "medium"          # high -> medium (tek basamak)
    assert any("bu gece calismadi" in w for w in body["confidence"]["warnings"])


def test_missing_risk_method_short_history_warns_differently(env):
    _make_pool(env.pool, [_peer_row("TUPRS", risk_method=None, risk_sigma_5d=None, risk_z=None)],
               config={"risk": {"method": "har_x_ridge", "gate_passed": True}})
    _write_market(env.data, "TUPRS", "2026-11-16")
    body = env.client().get("/analysis/TUPRS").json()
    warns = " ".join(body["confidence"]["warnings"])
    assert body["confidence"]["label"] == "medium"
    assert "risk tahmini yok" in warns and "calismadi" not in warns


# ------------------------------------------------------------- 4. as_of geriligi
def test_as_of_behind_csv_by_more_than_one_trading_day_is_stale(env):
    # as_of cuma 11-06 degil: pazartesi 11-09; CSV 11-16 -> 5 islem gunu
    _make_pool(env.pool, [_peer_row("TUPRS", as_of_date="2026-11-09")], as_of="2026-11-09")
    _write_market(env.data, "TUPRS", "2026-11-16")
    body = env.client().get("/analysis/TUPRS").json()
    assert body["analysis_status"] == "stale_data" and body["confidence"]["label"] == "low"
    assert any("2026-11-09 kapanisina" in w and "fiyat" in w.lower() for w in body["confidence"]["warnings"])


def test_as_of_one_trading_day_behind_is_not_flagged(env):
    # as_of cuma 11-13, CSV pazartesi 11-16 -> 1 islem gunu (esik asilmaz)
    _make_pool(env.pool, [_peer_row("TUPRS", as_of_date="2026-11-13")], as_of="2026-11-13")
    _write_market(env.data, "TUPRS", "2026-11-16")
    body = env.client().get("/analysis/TUPRS").json()
    assert body["analysis_status"] == "ok"
    assert not any("kapanisina" in w for w in body["confidence"]["warnings"])


# ------------------------------------------------------------ 5. kapi degisimi
def test_gate_flip_warning_and_previous_method_parsing():
    assert gate_flip_warning("har_x_ridge", "naive_fallback")
    assert gate_flip_warning("naive_fallback", "har_x_ridge")
    assert gate_flip_warning("har_x_ridge", "har_x_ridge") is None
    assert gate_flip_warning(None, "har_x_ridge") is None
    assert previous_risk_method({"config_json": json.dumps({"risk": {"method": "har_x_ridge"}})}) == "har_x_ridge"
    assert previous_risk_method({"config_json": "{bozuk"}) is None
    assert previous_risk_method(None) is None


def test_run_risk_step_logs_and_records_flip(monkeypatch):
    import src.serving.nightly_risk as nr

    monkeypatch.setattr(nr, "compute_risk_safe",
                        lambda panel, as_of, log=print, quarantined=None: (pd.DataFrame(), {"method": "naive_fallback", "gate_passed": False}))
    prev = {"config_json": json.dumps({"risk": {"method": "har_x_ridge"}})}
    logs: list[str] = []
    _, summary, warns = run_risk_step(pd.DataFrame(), pd.Timestamp("2026-11-13"), prev, log=logs.append)
    assert len(warns) == 1 and "har_x_ridge -> naive_fallback" in warns[0]
    assert summary["gate_flip"] is True and summary["previous_method"] == "har_x_ridge"
    assert any("risk kapisi degisti" in m for m in logs)


# ------------------------------------------------------------------ 6. band merkezi
def test_band_uses_centre_zero_when_expected_return_not_finite():
    merged = pd.DataFrame({"symbol": ["A"], "trend_expected_return": [np.nan]})
    risk = pd.DataFrame({"symbol": ["A"], "risk_sigma_5d": [0.05], "risk_sigma_naive_5d": [0.04],
                         "risk_vol_percentile": [50.0], "risk_level": ["orta"], "risk_z": [1.5],
                         "risk_band_level": [0.8], "risk_horizon_days": [5], "risk_method": ["har_x_ridge"]})
    out = attach_risk_and_band(merged, risk, {"A": 100.0}).iloc[0]
    assert out["kolb_price_low"] == pytest.approx(100 * np.exp(-0.075))
    assert out["kolb_price_high"] == pytest.approx(100 * np.exp(0.075))
    assert np.isnan(out["kolb_price_p50"])


# ----------------------------------------------------------------- 7. --horizon
@pytest.mark.parametrize("bad", ["3", "10", "0"])
def test_horizon_other_than_5_is_rejected_with_cli_error(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        nl.build_parser().parse_args(["--horizon", bad])
    assert exc.value.code == 2
    assert "yalniz 5" in capsys.readouterr().err


def test_horizon_default_and_explicit_5_accepted():
    assert nl.build_parser().parse_args([]).horizon == 5
    assert nl.build_parser().parse_args(["--horizon", "5"]).horizon == 5


# -------------------------------------------------- 9. parmak izi Istanbul tarihi
def test_fingerprint_day_uses_istanbul_not_utc(monkeypatch):
    class _FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 15, 22, 0, tzinfo=timezone.utc).astimezone(tz)  # IST 11-16 01:00

    monkeypatch.setattr(rc_mod, "datetime", _FakeDT)
    fp = compute_freshness_fingerprint("TUPRS", serving_db_path="", data_dir="")
    assert fp[2] == "2026-11-16"   # UTC gunu 11-15 olurdu


# ----------------------------------------------------------------- 10. guvenlik
def test_500_does_not_leak_exception_text(env, monkeypatch):
    _make_pool(env.pool, [_peer_row("TUPRS")])
    client = env.client()

    def _boom(self, symbol):
        raise RuntimeError("secret C:\\internal\\path.db")

    monkeypatch.setattr(AnalysisService, "build", _boom)
    resp = client.get("/analysis/TUPRS")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal server error"}


def test_symbols_500_and_503_are_generic(env, monkeypatch):
    client = env.client()
    monkeypatch.setattr(api_main, "list_serving_symbols", lambda: (_ for _ in ()).throw(RuntimeError("secret")))
    r = client.get("/symbols")
    assert r.status_code == 500 and "secret" not in r.text
    monkeypatch.setattr(api_main, "list_serving_symbols",
                        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("locked /x/y.db")))
    r = client.get("/symbols")
    assert r.status_code == 503 and "/x/y.db" not in r.text


def test_health_has_no_internal_paths(env, monkeypatch):
    _make_pool(env.pool, [_peer_row("TUPRS")])
    monkeypatch.setattr(api_main, "nightly_run_status", lambda: {"available": False})
    text = env.client().get("/health").text
    for leak in ("serving_db_path", "project_root", "sys.executable", "python.exe", "serving_pool.db"):
        assert leak not in text
    assert "runtime" not in json.loads(text)


def test_bind_address_defaults_to_loopback_and_env_overrides(monkeypatch):
    monkeypatch.delenv("AI_CORE_API_HOST", raising=False)
    monkeypatch.delenv("AI_CORE_API_PORT", raising=False)
    assert api_main._bind_address() == ("127.0.0.1", 8000)
    monkeypatch.setenv("AI_CORE_API_HOST", "0.0.0.0")
    monkeypatch.setenv("AI_CORE_API_PORT", "9001")
    assert api_main._bind_address() == ("0.0.0.0", 9001)


# ------------------------------------- 13. risk tahmini olmayan sembol: neden siniflama
def _gap_panel():
    """S00: olcek-kirigi karantinasini taklit eden uzun bosluk; S01: yalniz 30 satir."""
    from risk_synth import make_panel

    base = make_panel(n_sym=12, n_days=300, seed=3)
    dates = sorted(base["Date"].unique())
    drop_dates = dates[150:270]                       # ~karantina: 120 gun panelden atildi
    base = base[~((base["symbol"] == "S00") & base["Date"].isin(drop_dates))]
    base = base[~((base["symbol"] == "S01") & (base["Date"] < dates[-30]))]
    return base.reset_index(drop=True), pd.Timestamp(dates[-1])


def test_quarantine_gap_nans_har_windows_root_cause_of_missing_risk():
    from src.risk.features import RISK_FEATURE_COLS, build_risk_frame

    panel, as_of = _gap_panel()
    day = build_risk_frame(panel, 5).query("Date == @as_of").set_index("symbol")
    assert day.loc["S00", ["rv_5", "rv_21"]].notna().all()        # kisa pencereler dolu
    assert np.isnan(day.loc["S00", "rv_63"])                      # rv_63: <%80 gecerli gun
    assert day.loc["S05", list(RISK_FEATURE_COLS)].notna().all()  # gaps olmayan sembol saglam


def test_explain_missing_risk_classifies_reasons():
    from src.risk.serving import explain_missing_risk

    panel, as_of = _gap_panel()
    table = pd.DataFrame({"symbol": ["S00", "S01", "S02", "S05"],
                          "risk_sigma_5d": [np.nan, np.nan, np.nan, 0.05]})
    out = explain_missing_risk(table, panel, as_of, quarantined={"S00": 69})
    assert out == {"S00": "quarantine", "S01": "short_history", "S02": "gap"}
    assert explain_missing_risk(None, panel, as_of) == {}


@pytest.mark.parametrize("reason,needle", [
    ("quarantine", "olcek kirigi"), ("short_history", "yetersiz"), ("gap", "eksik/islemsiz")])
def test_api_missing_risk_warning_names_the_reason(env, reason, needle):
    _make_pool(env.pool, [_peer_row("ASELS", risk_method=None, risk_sigma_5d=None, risk_z=None)],
               config={"risk": {"method": "har_x_ridge", "missing": {"ASELS": reason}}})
    _write_market(env.data, "ASELS", "2026-11-16")
    body = env.client().get("/analysis/ASELS").json()
    assert body["confidence"]["label"] == "medium"
    assert any(needle in w for w in body["confidence"]["warnings"])
