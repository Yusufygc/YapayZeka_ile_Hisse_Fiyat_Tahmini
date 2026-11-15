# -*- coding: utf-8 -*-
"""2026-11 veri dogruluk duzeltmeleri: tarih ayristirma, bozuk CSV, tazelik, onarim araci."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

import src.data.pooled_loader as pl
from src.data.dates import parse_dates
from src.data.freshness import check_freshness, stale_symbols
from src.data.ohlcv_clean import read_ohlcv_csv
from src.data.pooled_loader import PooledLoaderConfig, PooledPanelLoader
from src.database.repositories.forecast_resolution import ForecastResolutionRepository

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
import repair_symbol_csv as rsc  # noqa: E402


# ------------------------------------------------------------------ parse_dates
def test_parse_dates_iso_not_swapped():
    out = parse_dates(["2024-01-02", "2024-12-03"])
    assert out.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-03")]


def test_parse_dates_dmy_slash_and_dot():
    out = parse_dates(["02/01/2024", "03.02.2024", "31/12/2023"])
    assert out.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-03"),
                            pd.Timestamp("2023-12-31")]


def test_parse_dates_mixed_series_per_element():
    s = pd.Series(["2024-01-02", "02/01/2024", "2024-03-04 00:00:00", "04.03.2024", "bozuk", None],
                  index=list("abcdef"))
    out = parse_dates(s)
    assert list(out.index) == list("abcdef")
    assert out["a"] == out["b"] == pd.Timestamp("2024-01-02")
    assert out["c"] == out["d"] == pd.Timestamp("2024-03-04")
    assert pd.isna(out["e"]) and pd.isna(out["f"])


def test_parse_dates_timestamp_passthrough():
    ts = pd.Timestamp("2024-05-06")
    assert parse_dates([ts]).iloc[0] == ts
    ser = pd.Series(pd.to_datetime(["2024-01-02", "2024-02-01"]))
    assert parse_dates(ser).tolist() == ser.tolist()
    assert list(pd.DatetimeIndex(parse_dates(pd.DatetimeIndex(["2024-01-02"])))) == [pd.Timestamp("2024-01-02")]


def test_parse_dates_errors_raise():
    with pytest.raises(ValueError):
        parse_dates(["not a date"], errors="raise")


# ------------------------------------------------------- forecast_resolution CSV
def test_forecast_resolution_csv_iso_dates_not_swapped(tmp_path):
    csv_path = tmp_path / "X.csv"
    csv_path.write_text("Date,Close\n2024-01-02,10.5\n2024-03-04,11.0\n", encoding="utf-8")
    repo = ForecastResolutionRepository(db=object())
    captured = {}
    repo.resolve_forecasts = lambda sym, actuals: captured.update(actuals) or len(actuals)
    repo.resolve_forecasts_from_csv("X", str(csv_path))
    assert captured == {"2024-01-02": 10.5, "2024-03-04": 11.0}


def test_forecast_resolution_csv_dmy_dates_unchanged(tmp_path):
    csv_path = tmp_path / "X.csv"
    csv_path.write_text("Tarih,Kapanış\n02/01/2024,10.5\n04/03/2024,11.0\n", encoding="utf-8")
    repo = ForecastResolutionRepository(db=object())
    captured = {}
    repo.resolve_forecasts = lambda sym, actuals: captured.update(actuals) or len(actuals)
    repo.resolve_forecasts_from_csv("X", str(csv_path))
    assert captured == {"2024-01-02": 10.5, "2024-03-04": 11.0}


# ----------------------------------------------------------------- pooled loader
class _StubFP:
    def __init__(self, *a, **k) -> None:
        pass

    def engineer_features(self, raw, macro_df=None, symbol=None, **k):
        out = pd.DataFrame({"Date": raw["Date"]})
        out["feat_a"] = pd.to_numeric(raw["Close"], errors="coerce").pct_change().values
        return out


_HEADER = "Tarih,Açılış,Yüksek,Düşük,Kapanış,Düzeltilmiş_Kapanış,Hacim"


def _write_symbol(path, n=80, extra_rows=()):
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = np.linspace(10.0, 20.0, n)
    lines = [_HEADER]
    for d, c in zip(dates, close):
        lines.append(f"{d:%d/%m/%Y},{c},{c},{c},{c},{c},1000000")
    lines.extend(extra_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _panel(tmp_path, monkeypatch, extra_rows):
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    data = tmp_path / "data"
    data.mkdir()
    _write_symbol(data / "AAA.csv", extra_rows=extra_rows)
    cfg = PooledLoaderConfig(data_dir=str(data), universe_file=str(data / "none.csv"),
                             target_horizon=5, min_rows=60, keep_unlabeled=True,
                             freshness_today=pd.Timestamp("2024-06-01"))
    loader = PooledPanelLoader(cfg)
    return loader, loader.load()


def test_pooled_rejects_six_field_row_without_shifting(tmp_path, monkeypatch):
    # 6 alanli satir: Hacim (999999999) Close'a kaymamali.
    bad = ("01/06/2024,15.0,15.0,15.0,15.0,999999999",)
    loader, panel = _panel(tmp_path, monkeypatch, bad)
    assert loader.rejected_rows == {"AAA": 1}
    assert panel["Close"].max() < 100
    assert pd.Timestamp("2024-06-01") not in set(panel["Date"])


def test_pooled_dedupe_zero_volume_and_ohl_rescale(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=80)
    close = np.linspace(10.0, 20.0, 80)
    lines = [_HEADER]
    for i, (d, c) in enumerate(zip(dates, close)):
        adj = c * 0.5 if i == 10 else c      # tek satirda duzeltme faktoru 0.5
        vol = 0 if i == 20 else 1000000       # sifir-hacim satiri
        lines.append(f"{d:%d/%m/%Y},{c},{c},{c},{c},{adj},{vol}")
    lines.append(f"{dates[30]:%d/%m/%Y},1,1,1,1,1,1000000")   # tekrar tarih: SON satir kalir
    data = tmp_path / "d"
    data.mkdir()
    (data / "AAA.csv").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    from src.data.ohlcv_clean import clean_ohlcv_csv

    df = clean_ohlcv_csv(str(data / "AAA.csv"))
    assert df["Date"].is_unique
    assert dates[20] not in set(df["Date"])                         # sifir hacim atildi
    assert df.loc[df["Date"] == dates[30], "Close"].iloc[0] == 1.0  # son tekrar tutuldu
    row = df[df["Date"] == dates[10]].iloc[0]
    assert row["Close"] == pytest.approx(close[10] * 0.5)
    assert row["Open"] == pytest.approx(close[10] * 0.5)            # OHL ayni faktorle olceklendi
    assert row["High"] == pytest.approx(close[10] * 0.5)


def test_pooled_flags_stale_symbols(tmp_path, monkeypatch):
    loader, _ = _panel(tmp_path, monkeypatch, ())
    assert "AAA" in loader.stale_symbols   # son tarih ~Nisan 2024, bugun 2024-06-01


def test_read_ohlcv_csv_counts_rejected(tmp_path):
    p = tmp_path / "S.csv"
    p.write_text(_HEADER + "\n02/01/2024,1,1,1,1,1,5\n03/01/2024,1,1,1,1,9\n", encoding="utf-8-sig")
    df = read_ohlcv_csv(str(p))
    assert len(df) == 1 and df.attrs["rejected_rows"] == 1
    assert df["Volume"].iloc[0] == 5


# --------------------------------------------------------------------- freshness
def test_freshness_fresh_and_stale():
    today = pd.Timestamp("2026-11-16")  # Pazartesi
    assert not check_freshness("2026-11-13", today).stale           # Cuma -> 1 is gunu
    res = check_freshness("2026-06-02", today)
    assert res.stale and res.lag_business_days > 100 and "BAYAT" in res.message
    assert check_freshness("2026-11-09", today, max_business_days=4).stale is True
    assert check_freshness("2026-11-09", today, max_business_days=5).stale is False


def test_freshness_missing_date_is_stale():
    res = check_freshness(pd.NaT, today="2026-11-16")
    assert res.stale and res.lag_business_days is None


def test_stale_symbols_filters():
    out = stale_symbols({"A": "2026-11-13", "B": "2026-06-02"}, today="2026-11-16")
    assert set(out) == {"B"}


# ---------------------------------------------------------------- repair script
def test_repair_script_fixes_rows_and_backs_up(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    src = _HEADER + "\r\n02/01/2024,1,2,0.5,1.5,1.4,100\r\n03/01/2024,1,2,0.5,1.7,500\r\n"
    p = data / "ZZZ.csv"
    p.write_bytes(b"\xef\xbb\xbf" + src.encode("utf-8"))
    dry = rsc.run(str(data), None, str(data / "yedek"), "bak_t", dry_run=True)
    assert dry[0].status == "would_repair" and not (data / "yedek").exists()
    rep = rsc.run(str(data), None, str(data / "yedek"), "bak_t", dry_run=False)
    assert rep[0].fixed_rows == 1
    out = p.read_bytes().decode("utf-8-sig")
    assert "03/01/2024,1,2,0.5,1.7,1.7,500\r\n" in out
    assert (data / "yedek" / "ZZZ.csv.bak_t").read_bytes().decode("utf-8-sig") == src
    assert rsc.run(str(data), None, str(data / "yedek"), "bak_t", dry_run=True)[0].status == "ok"


def test_repair_script_normalizes_minority_date_format(tmp_path):
    text = _HEADER + "\n02/01/2024,1,1,1,1,1,5\n03/01/2024,1,1,1,1,1,5\n2024-01-04,1,1,1,1,1,5\n"
    out, rep = rsc.analyze_text(text)
    assert rep.date_normalized == 1 and "04/01/2024,1,1,1,1,1,5" in out
