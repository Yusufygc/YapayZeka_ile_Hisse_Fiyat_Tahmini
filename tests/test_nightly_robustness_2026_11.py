# -*- coding: utf-8 -*-
"""Gecelik is dayanikliligi (2026-11): durum dosyasi, bayat-oran kapisi, ag zaman asimi, ISO tarih.

Ag yok; zaman/saglayici sahte.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import src.data.market_providers as mp
import src.serving.nightly as nl
from src.api.services.serving_health import nightly_run_status
from src.data.data_updater import DataUpdater, DataUpdateResult
from src.data.freshness import _today
from src.data.updater_frames import _parse_date_series, _parse_last_date
from src.serving.run_status import NightlyRunStatus, read_status, status_path_for_db

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ durum dosyasi
def _args(tmp_path, *extra):
    return ["--skip-trading-gate", "--db", str(tmp_path / "pool.db"),
            "--status-file", str(tmp_path / "st.json"), *extra]


def test_status_path_is_next_to_db(tmp_path):
    assert status_path_for_db(str(tmp_path / "x" / "serving_pool.db")) == str(
        tmp_path / "x" / "nightly_status.json")


def test_status_written_at_start_as_incomplete_and_atomic(tmp_path):
    path = str(tmp_path / "st.json")
    st = NightlyRunStatus(path, model="lgb")
    data = read_status(path)
    assert data["completed"] is False and data["finished_at"] is None and data["model"] == "lgb"
    st.update(phase="refresh")
    assert read_status(path)["phase"] == "refresh"
    st.finish(0)
    done = read_status(path)
    assert done["completed"] is True and done["exit_code"] == 0 and done["finished_at"]
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []   # gecici dosya kalmaz


def test_status_corrupt_file_raises_value_error(tmp_path):
    p = tmp_path / "st.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        read_status(str(p))


def test_main_status_is_incomplete_while_scoring_runs_and_crash_keeps_false(monkeypatch, tmp_path):
    seen = {}

    def _scoring(a):
        seen["during"] = read_status(str(tmp_path / "st.json"))
        raise RuntimeError("OOM benzeri")

    monkeypatch.setattr(nl, "run_scoring", _scoring)
    with pytest.raises(RuntimeError):
        nl.main(_args(tmp_path, "--skip-data"))
    assert seen["during"]["completed"] is False and seen["during"]["phase"] == "scoring"
    final = read_status(str(tmp_path / "st.json"))
    assert final["completed"] is False and final["phase"] == "crashed"
    assert "OOM" in final["error"] and final["finished_at"]


def test_main_success_marks_completed(monkeypatch, tmp_path):
    monkeypatch.setattr(nl, "run_scoring", lambda a: 0)
    assert nl.main(_args(tmp_path, "--skip-data")) == 0
    data = read_status(str(tmp_path / "st.json"))
    assert data["completed"] is True and data["exit_code"] == 0 and data["scoring"] == "done"


def test_non_trading_day_does_not_touch_status(monkeypatch, tmp_path):
    prev = tmp_path / "st.json"
    prev.write_text(json.dumps({"completed": False, "marker": "onceki"}), encoding="utf-8")
    monkeypatch.setattr(nl, "is_trading_day", lambda d, calendar="XIST": False)
    assert nl.main(["--db", str(tmp_path / "pool.db"), "--status-file", str(prev)]) == 0
    assert json.loads(prev.read_text(encoding="utf-8"))["marker"] == "onceki"


# ------------------------------------------------------------ bayat-oran kapisi
def _counts(total, stale):
    return {"updated": 0, "total": total, "stale_after": stale,
            "stale_fraction": stale / total, "failed": stale}


def test_stale_refresh_skips_scoring_and_exits_3(monkeypatch, tmp_path):
    monkeypatch.setattr(nl, "refresh_universe", lambda *a, **k: _counts(100, 73))
    monkeypatch.setattr(nl, "run_scoring",
                        lambda a: (_ for _ in ()).throw(AssertionError("skorlama atlanmaliydi")))
    assert nl.main(_args(tmp_path)) == nl.EXIT_STALE_REFRESH == 3
    data = read_status(str(tmp_path / "st.json"))
    assert data["completed"] is True and data["exit_code"] == 3
    assert data["scoring"] == "skipped_stale_refresh" and data["stale_symbols_after_refresh"] == 73


def test_small_stale_share_still_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(nl, "refresh_universe", lambda *a, **k: _counts(100, 5))
    ran = []
    monkeypatch.setattr(nl, "run_scoring", lambda a: ran.append(1) or 0)
    assert nl.main(_args(tmp_path)) == 0 and ran == [1]


def test_threshold_is_cli_overridable(monkeypatch, tmp_path):
    monkeypatch.setattr(nl, "refresh_universe", lambda *a, **k: _counts(100, 40))
    monkeypatch.setattr(nl, "run_scoring", lambda a: 0)
    assert nl.main(_args(tmp_path, "--max-stale-fraction", "0.5")) == 0
    assert nl.main(_args(tmp_path, "--max-stale-fraction", "0.3")) == 3


def test_refresh_universe_counts_failed_and_stale_after_refresh(monkeypatch, tmp_path):
    for n in ("AAA", "BBB", "CCC", "DDD"):
        (tmp_path / f"{n}.csv").write_text("Tarih\n2026-01-01\n", encoding="utf-8")
    today = _today().strftime("%Y-%m-%d")
    res = {
        "AAA": DataUpdateResult(status="updated", latest_date_after=today, rows_added=2),
        "BBB": DataUpdateResult(status="up_to_date", latest_date_after=today),
        "CCC": DataUpdateResult(status="skipped", latest_date_after="2026-06-02"),   # hala bayat
        "DDD": DataUpdateResult(status="failed", latest_date_after="2026-06-02", error="x"),
    }
    seen = {}

    def _fake(targets, **kw):
        seen.update(kw)
        return {s: res[s] for _, s in targets}

    monkeypatch.setattr(nl.DataUpdater, "update_many", staticmethod(_fake))
    counts = nl.refresh_universe(str(tmp_path), deadline_s=123.0)
    assert seen["deadline_s"] == 123.0
    assert counts["total"] == 4 and counts["stale_after"] == 2 and counts["failed"] == 1
    assert counts["stale_fraction"] == 0.5 and counts["failed_sample"] == ["DDD"]


# --------------------------------------------------------------- /health ozeti
def _write_status(tmp_path, **kw):
    base = {"started_at": datetime.now(timezone.utc).isoformat(), "completed": False, "phase": "refresh",
            "exit_code": None}
    (tmp_path / "nightly_status.json").write_text(json.dumps({**base, **kw}), encoding="utf-8")
    return str(tmp_path / "pool.db")


def test_health_nightly_status_variants(tmp_path):
    db = str(tmp_path / "pool.db")
    assert nightly_run_status(db) == {"available": False, "reason": "no_status_file"}
    _write_status(tmp_path, completed=True, exit_code=0, phase="finished", stale_fraction=0.01)
    ok = nightly_run_status(db)
    assert ok["healthy"] is True and ok["interrupted"] is False and ok["stale_fraction"] == 0.01
    _write_status(tmp_path, completed=True, exit_code=3, phase="finished")
    assert nightly_run_status(db)["healthy"] is False
    _write_status(tmp_path)                                  # simdi basladi, calisiyor
    assert nightly_run_status(db)["interrupted"] is False
    old = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
    _write_status(tmp_path, started_at=old)                  # 10 saat once basladi, bitmedi
    assert nightly_run_status(db)["interrupted"] is True
    _write_status(tmp_path, phase="crashed", error="boom")
    assert nightly_run_status(db)["interrupted"] is True
    (tmp_path / "nightly_status.json").write_text("{bozuk", encoding="utf-8")
    assert nightly_run_status(db) == {"available": False, "reason": "unreadable"}


# ------------------------------------------------------------ ag zaman asimlari
def test_call_with_deadline_abandons_hung_call():
    release = threading.Event()
    try:
        with pytest.raises(mp.ProviderTimeout):
            mp.call_with_deadline(lambda: release.wait(30), 0.2, "hung")
    finally:
        release.set()
    assert mp.call_with_deadline(lambda: 7, 1.0) == 7
    with pytest.raises(KeyError):
        mp.call_with_deadline(lambda: {}["x"], 1.0)
    assert mp.call_with_deadline(lambda: 5, None) == 5       # sinirsiz = dogrudan


def test_yfinance_provider_passes_request_timeout_and_bounds_hang(monkeypatch):
    seen = {}

    def _ok(*a, **kw):
        seen.update(kw)
        return pd.DataFrame()

    monkeypatch.setattr(mp.yf, "download", _ok)
    prov = mp.YFinanceProvider()
    prov.download("AAA.IS", start="2026-01-01", end="2026-01-05")
    assert seen["timeout"] == mp.DEFAULT_REQUEST_TIMEOUT_S and seen["auto_adjust"] is True
    seen.clear()
    prov.download_many(["AAA.IS", "BBB.IS"], start="2026-01-01", end="2026-01-05")
    assert seen["timeout"] == mp.DEFAULT_REQUEST_TIMEOUT_S

    release = threading.Event()
    monkeypatch.setattr(mp.yf, "download", lambda *a, **kw: release.wait(30))
    hung = mp.YFinanceProvider()
    hung.SINGLE_CALL_DEADLINE_S = 0.2
    try:
        with pytest.raises(mp.ProviderTimeout):
            hung.download("AAA.IS", start="2026-01-01", end="2026-01-05")
    finally:
        release.set()


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_download_with_retries_stops_at_deadline():
    clock = _Clock()
    calls = []

    class _P:
        def download_many(self, tickers, *, start, end):
            calls.append(list(tickers))
            clock.t += 100.0                      # her chunk 100 sn "surer"
            return {t: pd.DataFrame({"Close": [1.0]}) for t in tickers}

    frames, errors = mp.download_with_retries(
        _P(), ["A", "B", "C", "D"], start="2026-01-01", end="2026-01-05", chunk_size=2,
        max_retries=3, sleep=lambda s: None, deadline=50.0, clock=clock)
    # ilk chunk 100 sn surdu -> sinir (50) asildi; ikinci chunk HIC baslatilmaz
    assert calls == [["A", "B"]]
    assert set(frames) == {"A", "B"}
    assert errors == {"C": mp.DEADLINE_ERROR, "D": mp.DEADLINE_ERROR}


def _csv(path, last: str, fmt: str, n: int = 40) -> None:
    dates = pd.bdate_range(end=last, periods=n)
    c = 100.0 + (dates - pd.Timestamp("2026-01-01")).days.to_numpy() * 0.01
    pd.DataFrame({"Date": dates.strftime(fmt), "Open": c, "High": c, "Low": c, "Close": c,
                  "Adj Close": c, "Volume": np.full(n, 1e6)}).to_csv(path, index=False)


def test_update_many_marks_remaining_failed_after_deadline_and_keeps_csv(tmp_path):
    p = tmp_path / "AAA.csv"
    _csv(p, "2026-11-10", "%Y-%m-%d")
    before = p.read_bytes()
    clock = _Clock()
    clock.t = 1_000.0

    class _P:
        def download_many(self, tickers, *, start, end):
            raise AssertionError("sinir asilmisken indirme yapilmamali")

    # deadline_s=0 -> baslangicta zaten dolmus
    res = DataUpdater.update_many([(str(p), "AAA")], provider=_P(), now=datetime(2026, 11, 17, 18, 30),
                                  deadline_s=0.0, clock=clock, sleep=lambda s: None)
    assert res["AAA"].status == "failed" and mp.DEADLINE_ERROR in res["AAA"].error
    assert p.read_bytes() == before


# ------------------------------------------------------------------ ISO tarih
def test_parse_date_series_mixed_iso_first_then_dmy():
    s = pd.Series(["2024-01-02", "2024-01-03", "04/01/2024", "05.01.2024"])
    out = _parse_date_series(s)
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    assert _parse_last_date(s) == pd.Timestamp("2024-01-05")


def test_parse_date_series_mixed_dmy_first_then_iso_and_pure_formats():
    s = pd.Series(["02/01/2024", "2024-01-03", "13/01/2024"])
    assert list(_parse_date_series(s).dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-13"]
    assert _parse_last_date(pd.Series(["29/10/2015", "30/10/2015"])) == pd.Timestamp("2015-10-30")
    assert _parse_last_date(pd.Series(["2024-02-01", "2024-03-01"])) == pd.Timestamp("2024-03-01")
    assert _parse_date_series(pd.Series([None, None], dtype=object)).empty


def test_append_writes_iso_even_when_existing_file_is_dmy(tmp_path):
    p = tmp_path / "AAA.csv"
    _csv(p, "2026-11-13", "%d/%m/%Y")
    dates = pd.bdate_range("2026-11-06", "2026-11-17")
    c = 100.0 + (dates - pd.Timestamp("2026-01-01")).days.to_numpy() * 0.01
    frame = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6},
                         index=pd.DatetimeIndex(dates, name="Date"))

    class _P:
        def download(self, ticker, *, start, end):
            return frame

    res = DataUpdater.check_and_update(str(p), "AAA", interactive=False, provider=_P(),
                                       now=datetime(2026, 11, 17, 18, 30))
    assert res.status == "updated" and res.rows_added == 2
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[-1].startswith("2026-11-17,") and lines[-2].startswith("2026-11-16,")
    assert lines[-3].startswith("13/11/2026,")           # eski satirlara dokunulmaz
    assert _parse_last_date(pd.read_csv(p)["Date"]) == pd.Timestamp("2026-11-17")


# ----------------------------------------------------------- PowerShell betikleri
def _read(name):
    with open(os.path.join(_REPO, "scripts", name), encoding="utf-8") as fh:
        return fh.read()


def test_register_script_uses_register_scheduledtask_with_explicit_settings():
    txt = _read("register_nightly_task.ps1")
    code = " ".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
    assert "Register-ScheduledTask" in code and "schtasks" not in code
    for needle in ("-AllowStartIfOnBatteries", "-DontStopIfGoingOnBatteries", "-StartWhenAvailable",
                   "-MultipleInstances IgnoreNew", "New-TimeSpan -Hours 2", 'ts_forecasting_nightly',
                   '[string]$Time = "21:00"', "nightly_serving.ps1"):
        assert needle in txt, needle


def test_wrapper_script_has_finally_end_line_and_utf8_log():
    txt = _read("nightly_serving.ps1")
    assert "try {" in txt and "finally {" in txt and "nightly_serving bitis" in txt
    code = " ".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
    assert "*>>" not in code and ">> $Log" not in code      # PS 5.1'de UTF-16 uretir
    assert "UTF8Encoding" in txt and "-m src.serving.nightly" in txt
