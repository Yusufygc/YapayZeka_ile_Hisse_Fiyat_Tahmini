# -*- coding: utf-8 -*-
"""E2 Faz 8 — gecelik pipeline orkestratoru tests (ag yok)."""

from datetime import date, datetime

import pandas as pd
import pytest

import src.serving.nightly as nl

from src.data.data_updater import DataUpdateResult  # noqa: E402


# ----------------------------------------------------------- symbol discovery
def test_universe_excludes_helper_csvs(tmp_path):
    for name in ["AAA", "BBB", "bist_universe", "advisory_history"]:
        (tmp_path / f"{name}.csv").write_text("Tarih\n2026-01-01\n", encoding="utf-8")
    syms = nl._universe_symbols(str(tmp_path))
    assert syms == ["AAA", "BBB"]


# --------------------------------------------------------------- refresh agg
def _make_csvs(tmp_path, names):
    for n in names:
        (tmp_path / f"{n}.csv").write_text("Tarih\n2026-01-01\n", encoding="utf-8")


def test_refresh_counts_aggregate(monkeypatch, tmp_path):
    _make_csvs(tmp_path, ["AAA", "BBB", "CCC", "DDD"])
    status_by = {
        "AAA": DataUpdateResult(status="updated", rows_added=3),
        "BBB": DataUpdateResult(status="up_to_date"),
        "CCC": DataUpdateResult(status="skipped"),
        "DDD": DataUpdateResult(status="updated", rows_added=2),
    }
    monkeypatch.setattr(nl.DataUpdater, "check_and_update",
                        staticmethod(lambda path, sym, interactive=False: status_by[sym]))
    counts = nl.refresh_universe(str(tmp_path), sleep_s=0.0)
    assert counts["updated"] == 2
    assert counts["up_to_date"] == 1
    assert counts["skipped"] == 1
    assert counts["rows_added"] == 5


def test_refresh_continues_on_exception(monkeypatch, tmp_path):
    _make_csvs(tmp_path, ["AAA", "BOOM", "CCC"])

    def _cu(path, sym, interactive=False):
        if sym == "BOOM":
            raise RuntimeError("yfinance patladi")
        return DataUpdateResult(status="up_to_date")

    monkeypatch.setattr(nl.DataUpdater, "check_and_update", staticmethod(_cu))
    counts = nl.refresh_universe(str(tmp_path), sleep_s=0.0)
    assert counts["failed"] == 1
    assert counts["up_to_date"] == 2  # digerleri sayildi


def test_refresh_limit(monkeypatch, tmp_path):
    _make_csvs(tmp_path, ["AAA", "BBB", "CCC", "DDD"])
    monkeypatch.setattr(nl.DataUpdater, "check_and_update",
                        staticmethod(lambda path, sym, interactive=False:
                                     DataUpdateResult(status="up_to_date")))
    counts = nl.refresh_universe(str(tmp_path), sleep_s=0.0, limit=2)
    assert counts["up_to_date"] == 2


# ----------------------------------------------------------- trading-day gate
class _FakeCal:
    def __init__(self, valid_dates):
        self._valid = valid_dates

    def valid_days(self, start_date, end_date):
        return pd.DatetimeIndex([pd.Timestamp(d) for d in self._valid])


def _patch_mcal(monkeypatch, valid_dates):
    import pandas_market_calendars as mcal
    monkeypatch.setattr(mcal, "get_calendar", lambda name: _FakeCal(valid_dates))


def test_trading_day_true(monkeypatch):
    pytest.importorskip("pandas_market_calendars")
    _patch_mcal(monkeypatch, ["2026-06-03"])
    assert nl.is_trading_day(date(2026, 6, 3)) is True


def test_trading_day_false(monkeypatch):
    pytest.importorskip("pandas_market_calendars")
    _patch_mcal(monkeypatch, ["2026-06-05"])  # hedef gun listede yok (tatil)
    assert nl.is_trading_day(date(2026, 6, 4)) is False


def test_trading_day_real_xist_weekend():
    """Gercek XIST takvimi: hafta sonu islem gunu degil."""
    pytest.importorskip("pandas_market_calendars")
    assert nl.is_trading_day(date(2026, 6, 6)) is False  # cumartesi
    assert nl.is_trading_day(date(2026, 6, 3)) is True   # carsamba (dogrulandi)


def test_trading_day_fallback_weekday(monkeypatch):
    """Lib/takvim hatasi -> hafta-ici fallback (Cmt False, Cuma True)."""
    import pandas_market_calendars as mcal
    monkeypatch.setattr(mcal, "get_calendar",
                        lambda name: (_ for _ in ()).throw(RuntimeError("takvim yok")))
    assert nl.is_trading_day(date(2026, 6, 6)) is False  # cumartesi
    assert nl.is_trading_day(date(2026, 6, 5)) is True   # cuma


# --------------------------------------------------------- gate target date
def test_gate_target_evening_is_today():
    # 21:00 aksam kosusu -> bugunun seansi (kapanis sonrasi).
    assert nl.gate_target_date(datetime(2026, 6, 4, 21, 0)) == date(2026, 6, 4)


def test_gate_target_morning_is_yesterday():
    # 03:00 sabah kosusu -> en son seans dun.
    assert nl.gate_target_date(datetime(2026, 6, 4, 3, 0)) == date(2026, 6, 3)


def test_gate_target_boundary_close_hour():
    # tam kapanis saati (19) -> bugun (>=).
    assert nl.gate_target_date(datetime(2026, 6, 4, 19, 0)) == date(2026, 6, 4)
    assert nl.gate_target_date(datetime(2026, 6, 4, 18, 59)) == date(2026, 6, 3)


# ------------------------------------------------------------------ main flow
def test_main_skips_on_non_trading_day(monkeypatch):
    monkeypatch.setattr(nl, "is_trading_day", lambda d, calendar="XIST": False)
    called = {"score": False}
    monkeypatch.setattr(nl, "run_scoring",
                        lambda args: called.__setitem__("score", True) or 0)
    rc = nl.main(["--skip-data"])
    assert rc == 0
    assert called["score"] is False  # skip -> skorlama cagrilmadi


def test_main_runs_scoring_on_trading_day(monkeypatch):
    seen = {}
    monkeypatch.setattr(nl, "run_scoring",
                        lambda a: seen.update(db=a.db, boost=a.boost, model=a.model,
                                              fb=a.fallback_dir, qj=a.quarantine_jumps) or 0)
    rc = nl.main(["--skip-data", "--skip-trading-gate"])
    assert rc == 0
    assert seen["db"] == "data/serving_pool.db" and seen["boost"] == 400
    assert seen["model"] == "lgb" and seen["fb"] == [] and seen["qj"] is True


def test_main_skip_data_does_not_refresh(monkeypatch):
    monkeypatch.setattr(nl, "run_scoring", lambda a: 0)
    monkeypatch.setattr(nl, "refresh_universe",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cagrilmamali")))
    assert nl.main(["--skip-data", "--skip-trading-gate"]) == 0
