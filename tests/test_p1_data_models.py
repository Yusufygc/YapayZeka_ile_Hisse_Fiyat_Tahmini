# -*- coding: utf-8 -*-
"""P1 derin denetim (2026-11) — veri katmani duzeltmeleri.

1. DataUpdater: overlap'li indirme; split/temettu sonrasi tam yeniden indirme.
2. data_loader: Close=Adj_Close iken O/H/L ayni faktorle olceklenir.

Ag erisimi yok: indirici sahte provider ile enjekte edilir.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  4) DataUpdater                                                              #
# --------------------------------------------------------------------------- #
class _FakeProvider:
    """Deterministik fiyat uretir; `ratio` her indirilen satira uygulanir."""

    def __init__(self, ratio: float = 1.0, extra_future_rows: int = 0):
        self.ratio = ratio
        self.extra_future_rows = extra_future_rows
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def base_close(dates: pd.DatetimeIndex) -> np.ndarray:
        return 100.0 + (dates - pd.Timestamp("2020-01-01")).days.to_numpy() * 0.01

    def download(self, ticker, *, start, end):
        self.calls.append((start, end))
        last = pd.Timestamp(datetime.today().date()) + pd.Timedelta(days=self.extra_future_rows)
        dates = pd.bdate_range(start=start, end=last)
        dates = dates[dates < pd.Timestamp(end)] if self.extra_future_rows == 0 else dates
        close = self.base_close(dates) * self.ratio
        frame = pd.DataFrame({
            "Open": close, "High": close, "Low": close, "Close": close,
            "Volume": np.full(len(dates), 1e6),
        }, index=pd.DatetimeIndex(dates, name="Date"))
        return frame


def _write_cache_csv(path, n_old: int = 40, lag_days: int = 20):
    last = pd.Timestamp(datetime.today().date()) - pd.Timedelta(days=lag_days)
    dates = pd.bdate_range(end=last, periods=n_old)
    close = _FakeProvider.base_close(dates)
    pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": close, "High": close, "Low": close, "Close": close,
        "Adj Close": close, "Volume": np.full(n_old, 1e6),
    }).to_csv(path, index=False)
    return dates


def test_updater_appends_when_overlap_matches(tmp_path):
    from src.data.data_updater import DataUpdater

    csv = str(tmp_path / "TEST.csv")
    old_dates = _write_cache_csv(csv)
    before = pd.read_csv(csv)
    provider = _FakeProvider(ratio=1.0)

    res = DataUpdater.check_and_update(csv, "TEST", interactive=False, provider=provider)

    after = pd.read_csv(csv)
    assert res.status == "updated"
    assert len(provider.calls) == 1  # tam yeniden indirme yok
    # overlap ile indirildi: start <= son tarih - 7 gun
    assert pd.Timestamp(provider.calls[0][0]) <= old_dates[-1] - pd.Timedelta(days=7)
    # eski satirlar degismedi, yalniz yeni satirlar eklendi (cift yok)
    pd.testing.assert_frame_equal(after.iloc[: len(before)], before)
    assert after["Date"].is_unique
    assert pd.to_datetime(after["Date"].iloc[len(before):]).min() > old_dates[-1]
    assert res.rows_added == len(after) - len(before) > 0


def test_updater_full_refetch_on_adjustment_seam(tmp_path):
    from src.data.data_updater import DataUpdater

    csv = str(tmp_path / "TEST.csv")
    old_dates = _write_cache_csv(csv)
    provider = _FakeProvider(ratio=0.5)  # split sonrasi gecmis yeniden olceklendi

    res = DataUpdater.check_and_update(csv, "TEST", interactive=False, provider=provider)

    after = pd.read_csv(csv)
    assert res.status == "updated"
    assert len(provider.calls) == 2  # overlap + tam yeniden indirme
    assert pd.Timestamp(provider.calls[1][0]) == old_dates[0]
    # tum gecmis yeni (olceklenmis) seriyle degisti: dikis yok
    expected = _FakeProvider.base_close(pd.DatetimeIndex(pd.to_datetime(after["Date"]))) * 0.5
    np.testing.assert_allclose(after["Close"].to_numpy(), expected, rtol=1e-9)
    assert after["Date"].is_unique
    assert not list(tmp_path.glob("*.tmp"))  # gecici dosya birakilmadi


def test_updater_never_appends_future_dated_rows(tmp_path):
    from src.data.data_updater import DataUpdater

    csv = str(tmp_path / "TEST.csv")
    _write_cache_csv(csv)
    provider = _FakeProvider(ratio=1.0, extra_future_rows=10)

    DataUpdater.check_and_update(csv, "TEST", interactive=False, provider=provider)

    after = pd.read_csv(csv)
    assert pd.to_datetime(after["Date"]).max() <= pd.Timestamp(datetime.today().date())


# --------------------------------------------------------------------------- #
#  5) data_loader OHLC tutarliligi                                             #
# --------------------------------------------------------------------------- #
def test_loader_scales_ohl_with_adj_close_factor(tmp_path):
    from src.data.data_loader import load_and_clean

    n = 6
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    raw_close = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    adj = raw_close * np.array([0.5, 0.5, 0.5, 1.0, 1.0, np.nan])  # son satir NaN
    csv = tmp_path / "X.csv"
    pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": raw_close * 1.01, "High": raw_close * 1.02, "Low": raw_close * 0.99,
        "Close": raw_close, "Adj_Close": adj, "Volume": np.full(n, 1000.0),
    }).to_csv(csv, index=False)

    out = load_and_clean(str(csv))

    factor = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(out["Close"], raw_close * factor)
    np.testing.assert_allclose(out["Open"], raw_close * 1.01 * factor)
    np.testing.assert_allclose(out["High"], raw_close * 1.02 * factor)
    np.testing.assert_allclose(out["Low"], raw_close * 0.99 * factor)
    np.testing.assert_allclose(out["Volume"], 1000.0)  # hacim olceklenmez
    # tutarlilik: Low <= Close <= High korunur
    assert (out["Low"] <= out["Close"]).all() and (out["Close"] <= out["High"]).all()
