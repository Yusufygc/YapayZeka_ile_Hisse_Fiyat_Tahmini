# -*- coding: utf-8 -*-
"""DataUpdater 2026-11 duzeltmeleri: seans takvimi, kismi mum, toplu indirme, kilit, yedek.

Ag yok; zaman `now` parametresiyle dondurulur (naive -> Europe/Istanbul).
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.csv_safety import CsvLockTimeout, backup_before_rewrite, csv_lock
from src.data.data_updater import DataUpdater
from src.data.market_providers import YFinanceProvider, split_batch_frame
from src.data.session_clock import latest_completed_session


def _close(dates: pd.DatetimeIndex) -> np.ndarray:
    return 100.0 + (dates - pd.Timestamp("2026-01-01")).days.to_numpy() * 0.01


def _frame(start: str, end_incl: str, ratio: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, end_incl)
    c = _close(dates) * ratio
    return pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(len(dates), 1e6)},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _write_csv(path, last: str, n: int = 40) -> None:
    dates = pd.bdate_range(end=last, periods=n)
    c = _close(dates)
    pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"), "Open": c, "High": c, "Low": c,
        "Close": c, "Adj Close": c, "Volume": np.full(n, 1e6),
    }).to_csv(path, index=False)


class _Provider:
    """Tekli indirme; `through` tarihine kadar (dahil, kismi gun dahil) satir uretir."""

    def __init__(self, through: str, ratio: float = 1.0, boom: bool = False):
        self.through, self.ratio, self.boom = through, ratio, boom
        self.calls: list[tuple[str, str, str]] = []

    def download(self, ticker, *, start, end):
        self.calls.append((ticker, start, end))
        if self.boom:
            raise AssertionError("up_to_date iken indirme yapilmamali")
        return _frame(start, self.through, self.ratio)


TUE = lambda h, m=0: datetime(2026, 11, 17, h, m)  # noqa: E731  (Sali)


# ----------------------------------------------------------------- takvim / bayatlik
def test_latest_completed_session_rules():
    assert latest_completed_session(TUE(12)) == pd.Timestamp("2026-11-16")
    assert latest_completed_session(TUE(18, 9)) == pd.Timestamp("2026-11-16")
    assert latest_completed_session(TUE(18, 10)) == pd.Timestamp("2026-11-17")
    assert latest_completed_session(datetime(2026, 11, 15, 12)) == pd.Timestamp("2026-11-13")  # Pazar
    # 29 Ekim (Cumhuriyet Bayrami, takvimde kapali): 30 Ekim sabahi -> 28 Ekim
    assert latest_completed_session(datetime(2026, 10, 30, 10)) == pd.Timestamp("2026-10-28")


def test_yesterday_bar_before_close_is_up_to_date(tmp_path):
    csv = str(tmp_path / "AAA.csv")
    _write_csv(csv, "2026-11-16")
    prov = _Provider("2026-11-17", boom=True)
    res = DataUpdater.check_and_update(csv, "AAA", interactive=False, provider=prov, now=TUE(12))
    assert res.status == "up_to_date" and not prov.calls


def test_yesterday_bar_after_close_triggers_update(tmp_path):
    csv = str(tmp_path / "AAA.csv")
    _write_csv(csv, "2026-11-16")
    prov = _Provider("2026-11-17")
    res = DataUpdater.check_and_update(csv, "AAA", interactive=False, provider=prov, now=TUE(18, 30))
    assert res.status == "updated" and res.rows_added == 1
    assert res.latest_date_after == "2026-11-17"
    assert pd.read_csv(csv)["Date"].iloc[-1] == "2026-11-17"


def test_monday_morning_with_friday_bar_is_up_to_date(tmp_path):
    csv = str(tmp_path / "AAA.csv")
    _write_csv(csv, "2026-11-13")
    res = DataUpdater.check_and_update(
        csv, "AAA", interactive=False, provider=_Provider("2026-11-16", boom=True),
        now=datetime(2026, 11, 16, 10),
    )
    assert res.status == "up_to_date"


# ----------------------------------------------------------------- kismi mum
def test_partial_today_bar_is_dropped_on_append(tmp_path):
    csv = str(tmp_path / "AAA.csv")
    _write_csv(csv, "2026-11-13")
    prov = _Provider("2026-11-17")  # 17'sinin kismi mumu da gelir; seans 17:00'de acik
    res = DataUpdater.check_and_update(csv, "AAA", interactive=False, provider=prov, now=TUE(17))
    after = pd.read_csv(csv)
    assert res.status == "updated"
    assert after["Date"].iloc[-1] == "2026-11-16"
    assert "2026-11-17" not in set(after["Date"])


def test_partial_today_bar_is_dropped_on_full_refetch(tmp_path):
    csv = str(tmp_path / "AAA.csv")
    _write_csv(csv, "2026-11-13")
    prov = _Provider("2026-11-17", ratio=0.5)  # dikis -> tam yeniden indirme
    res = DataUpdater.check_and_update(csv, "AAA", interactive=False, provider=prov, now=TUE(17))
    after = pd.read_csv(csv)
    assert res.status == "updated" and res.full_refetch
    assert after["Date"].iloc[-1] == "2026-11-16"


# ----------------------------------------------------------------- toplu indirme
class _BatchProvider:
    """download_many: ilk cagrida 'BBB.IS' hic donmez, ikincide gelir; CCC.IS ilk bos."""

    AUTO_ADJUST = True

    def __init__(self):
        self.batches: list[list[str]] = []
        self.attempt: dict[str, int] = {}

    def download(self, ticker, *, start, end):  # tekli yol (full refetch icin)
        return _frame(start, "2026-11-17")

    def download_many(self, tickers, *, start, end):
        self.batches.append(list(tickers))
        out = {}
        for t in tickers:
            n = self.attempt[t] = self.attempt.get(t, 0) + 1
            if t == "BBB.IS" and n == 1:
                continue  # basarisiz
            out[t] = pd.DataFrame() if (t == "CCC.IS" and n == 1) else _frame(start, "2026-11-17")
        return out


def test_batch_path_retries_failed_and_empty_tickers(tmp_path):
    targets = []
    for sym in ("AAA", "BBB", "CCC"):
        p = str(tmp_path / f"{sym}.csv")
        _write_csv(p, "2026-11-13")
        targets.append((p, sym))
    prov, sleeps = _BatchProvider(), []
    res = DataUpdater.update_many(
        targets, provider=prov, now=TUE(18, 30), chunk_size=2, backoff_s=2.0, sleep=sleeps.append,
    )
    assert {s: r.status for s, r in res.items()} == {"AAA": "updated", "BBB": "updated", "CCC": "updated"}
    assert prov.attempt == {"AAA.IS": 1, "BBB.IS": 2, "CCC.IS": 2}
    assert sleeps == [2.0, 2.0]  # iki grup (2+1), her birinde bir yeniden deneme
    assert all(len(b) <= 2 for b in prov.batches)
    for p, _ in targets:
        assert pd.read_csv(p)["Date"].iloc[-1] == "2026-11-17"


def test_batch_gives_up_after_max_retries(tmp_path):
    p = str(tmp_path / "ZZZ.csv")
    _write_csv(p, "2026-11-13")

    class _Dead(_BatchProvider):
        def download_many(self, tickers, *, start, end):
            raise ConnectionError("ag yok")

    sleeps: list[float] = []
    res = DataUpdater.update_many(
        [(p, "ZZZ")], provider=_Dead(), now=TUE(18, 30), max_retries=2, backoff_s=1.0, sleep=sleeps.append,
    )
    assert res["ZZZ"].status == "failed" and "ag yok" in (res["ZZZ"].error or "")
    assert sleeps == [1.0, 2.0]  # ustel geri cekilme


def test_batch_skips_up_to_date_symbols_without_download(tmp_path):
    p = str(tmp_path / "AAA.csv")
    _write_csv(p, "2026-11-16")
    prov = _BatchProvider()
    res = DataUpdater.update_many([(p, "AAA")], provider=prov, now=TUE(12), sleep=lambda s: None)
    assert res["AAA"].status == "up_to_date" and prov.batches == []


def test_split_batch_frame_handles_multiindex_and_missing_ticker():
    a, b = _frame("2026-11-02", "2026-11-04"), _frame("2026-11-02", "2026-11-04")
    raw = pd.concat({"AAA.IS": a, "BBB.IS": b}, axis=1)
    raw.loc[raw.index[-1], "BBB.IS"] = np.nan  # BBB'nin son satiri tamamen NaN
    out = split_batch_frame(raw, ["AAA.IS", "BBB.IS", "MISSING.IS"])
    assert len(out["AAA.IS"]) == 3 and len(out["BBB.IS"]) == 2
    assert "MISSING.IS" not in out
    assert YFinanceProvider.AUTO_ADJUST is True


# ----------------------------------------------------------------- kilit
def test_lock_times_out_while_held_and_leaves_csv_untouched(tmp_path):
    csv = str(tmp_path / "AAA.csv")
    _write_csv(csv, "2026-11-13")
    before = open(csv, "rb").read()
    with csv_lock(csv, timeout=5):
        res = DataUpdater.check_and_update(
            csv, "AAA", interactive=False, provider=_Provider("2026-11-17"),
            now=TUE(18, 30), lock_timeout=0.3,
        )
    assert res.status == "failed" and "kilid" in (res.error or "")
    assert open(csv, "rb").read() == before
    with pytest.raises(CsvLockTimeout):
        with csv_lock(csv, timeout=5):
            with csv_lock(csv, timeout=0.2):
                pass


def test_second_writer_waits_for_lock_release(tmp_path):
    csv = str(tmp_path / "AAA.csv")
    _write_csv(csv, "2026-11-13")
    out: dict = {}
    held = threading.Event()
    release = threading.Event()

    def holder():
        with csv_lock(csv, timeout=5):
            held.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    held.wait(5)

    def writer():
        out["res"] = DataUpdater.check_and_update(
            csv, "AAA", interactive=False, provider=_Provider("2026-11-17"),
            now=TUE(18, 30), lock_timeout=10,
        )

    w = threading.Thread(target=writer)
    w.start()
    time.sleep(0.4)
    assert "res" not in out  # kilit tutuldukca bekler
    release.set()
    t.join(5)
    w.join(10)
    assert out["res"].status == "updated"
    assert pd.read_csv(csv)["Date"].is_unique


# ----------------------------------------------------------------- yedek
def test_full_refetch_creates_backup_with_original_content(tmp_path):
    csv = str(tmp_path / "AAA.csv")
    _write_csv(csv, "2026-11-13")
    original = open(csv, "rb").read()
    res = DataUpdater.check_and_update(
        csv, "AAA", interactive=False, provider=_Provider("2026-11-16", ratio=0.5), now=TUE(12),
    )
    assert res.full_refetch
    bak = tmp_path / "yedek" / "AAA.csv.bak_20261117"
    assert bak.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_backup_rotation_keeps_three_newest(tmp_path):
    csv = str(tmp_path / "AAA.csv")
    _write_csv(csv, "2026-11-13")
    for day in ("2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14"):
        backup_before_rewrite(csv, today=pd.Timestamp(day))
    names = sorted(os.listdir(tmp_path / "yedek"))
    assert names == ["AAA.csv.bak_20261112", "AAA.csv.bak_20261113", "AAA.csv.bak_20261114"]
    # baska sembolun yedegine dokunmaz
    other = str(tmp_path / "BBB.csv")
    _write_csv(other, "2026-11-13")
    backup_before_rewrite(other, today=pd.Timestamp("2026-11-14"))
    assert len(os.listdir(tmp_path / "yedek")) == 4
