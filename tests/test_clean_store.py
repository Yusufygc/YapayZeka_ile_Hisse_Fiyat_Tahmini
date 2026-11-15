# -*- coding: utf-8 -*-
"""CleanStore (temizlenmis OHLCV Parquet deposu) testleri."""

import logging
import os

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import src.data.clean_store as cs
import src.data.pooled_loader as pl
from src.data.clean_store import CleanStore, CleanStoreError, panel_content_hash
from src.data.ohlcv_clean import clean_ohlcv_csv
from src.data.pooled_loader import PooledLoaderConfig, PooledPanelLoader

STD = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _write_csv(path, n=90, seed=0, bad_row=False, zero_vol=True, dup=True) -> None:
    rng = np.random.default_rng(seed)
    dates = list(pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"))
    close = 10 + np.cumsum(rng.normal(size=n)) ** 2 / 50
    vol = rng.integers(1000, 5000, n).astype(float)
    if zero_vol:
        vol[3] = 0.0
    adj = close * 0.98
    df = pd.DataFrame({
        "Tarih": dates, "Açılış": close * 0.99, "Yüksek": close * 1.01, "Düşük": close * 0.97,
        "Kapanış": close, "Düzeltilmiş_Kapanış": adj, "Hacim": vol})
    if dup:
        df = pd.concat([df, df.iloc[[10]]], ignore_index=True)   # tekrar tarih
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if bad_row:
        with open(path, "a", encoding="utf-8-sig") as fh:
            fh.write("2030-01-01,1,2,3\n")                        # alan sayisi uyusmaz


def _paths(tmp_path, names=("AAA", "BBB", "CCC")) -> dict[str, str]:
    d = tmp_path / "csv"
    d.mkdir(exist_ok=True)
    out = {}
    for i, name in enumerate(names):
        p = str(d / f"{name}.csv")
        _write_csv(p, seed=i, bad_row=(name == "BBB"))
        out[name] = p
    return out


def _store(tmp_path) -> CleanStore:
    return CleanStore(str(tmp_path / "clean"))


def _count_cleans(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = cs.clean_ohlcv_csv

    def spy(path, label=None, **kw):
        calls.append(label)
        return real(path, label=label, **kw)

    monkeypatch.setattr(cs, "clean_ohlcv_csv", spy)
    return calls


def test_exact_equality_with_clean_ohlcv_csv(tmp_path):
    paths = _paths(tmp_path)
    store = _store(tmp_path)
    rep = store.refresh(paths)
    assert (rep.new, rep.unchanged, rep.changed, rep.removed) == (3, 0, 0, 0)
    frames = store.load()
    assert sorted(frames) == ["AAA", "BBB", "CCC"]
    for sym, path in paths.items():
        ref = clean_ohlcv_csv(path, label=sym)
        got = frames[sym]
        assert_frame_equal(got, ref[STD], check_exact=True)
        assert got.attrs == ref.attrs
        assert isinstance(got.index, pd.RangeIndex)
    assert frames["BBB"].attrs["rejected_rows"] == 1
    assert frames["AAA"].attrs["rejected_rows"] == 0
    assert len(frames["AAA"]) == 89          # tekrar tarih + sifir hacim atildi


def test_load_subset_and_unknown_symbols(tmp_path):
    store = _store(tmp_path)
    store.refresh(_paths(tmp_path))
    got = store.load(["CCC", "AAA", "ZZZ"])
    assert sorted(got) == ["AAA", "CCC"]


def test_unchanged_files_not_recleaned(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    store = _store(tmp_path)
    store.refresh(paths)
    calls = _count_cleans(monkeypatch)
    rep = store.refresh(paths)
    assert calls == []
    assert rep.unchanged == 3 and rep.hashed == 0 and rep.rebuilt is None


def test_modified_file_recleans_only_that_symbol(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    store = _store(tmp_path)
    store.refresh(paths)
    _write_csv(paths["CCC"], n=100, seed=9)
    calls = _count_cleans(monkeypatch)
    rep = store.refresh(paths)
    assert calls == ["CCC"]
    assert (rep.changed, rep.unchanged, rep.new) == (1, 2, 0)
    frames = store.load()
    for sym, path in paths.items():
        assert_frame_equal(frames[sym], clean_ohlcv_csv(path)[STD], check_exact=True)
    assert len(frames["CCC"]) == 99


def test_new_symbol_added_and_removed_symbol_dropped(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    store = _store(tmp_path)
    store.refresh(paths)
    extra = str(tmp_path / "csv" / "DDD.csv")
    _write_csv(extra, seed=5)
    calls = _count_cleans(monkeypatch)
    rep = store.refresh({**paths, "DDD": extra})
    assert calls == ["DDD"] and rep.new == 1
    smaller = {k: v for k, v in {**paths, "DDD": extra}.items() if k != "BBB"}
    rep = store.refresh(smaller)
    assert rep.removed == 1 and calls == ["DDD"]      # yeniden temizleme yok
    frames = store.load()
    assert sorted(frames) == ["AAA", "CCC", "DDD"]
    assert_frame_equal(frames["CCC"], clean_ohlcv_csv(paths["CCC"])[STD], check_exact=True)


def test_mtime_change_same_content_no_reclean(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    store = _store(tmp_path)
    store.refresh(paths)
    st = os.stat(paths["AAA"])
    os.utime(paths["AAA"], ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    calls = _count_cleans(monkeypatch)
    rep = store.refresh(paths)
    assert calls == [] and rep.hashed == 1 and rep.unchanged == 3
    rep2 = store.refresh(paths)                       # stat manifestte tazelendi -> hizli yol
    assert rep2.hashed == 0


def test_cleaner_fingerprint_change_rebuilds_everything(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    store = _store(tmp_path)
    store.refresh(paths)
    h_before = store.content_hash()
    calls = _count_cleans(monkeypatch)
    monkeypatch.setattr(cs, "cleaner_fingerprint", lambda: "yeni-parmak-izi")
    rep = store.refresh(paths)
    assert sorted(calls) == ["AAA", "BBB", "CCC"]
    assert rep.rebuilt and "parmak izi" in rep.rebuilt and rep.new == 3
    assert store.content_hash() != h_before


def test_content_hash_changes_on_data_and_is_stable(tmp_path):
    paths = _paths(tmp_path)
    store = _store(tmp_path)
    store.refresh(paths)
    h1 = store.content_hash()
    store.refresh(paths)
    assert store.content_hash() == h1
    st = os.stat(paths["AAA"])
    os.utime(paths["AAA"], ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    store.refresh(paths)
    assert store.content_hash() == h1                 # yalniz mtime: icerik ayni
    _write_csv(paths["AAA"], n=95, seed=3)
    store.refresh(paths)
    h2 = store.content_hash()
    assert h2 != h1
    assert store.content_hash(["BBB", "CCC"]) != store.content_hash(["AAA", "BBB", "CCC"])
    assert CleanStore(store.root).content_hash() == h2   # yeni nesne, ayni depo


@pytest.mark.parametrize("victim", ["panel.parquet", "manifest.parquet"])
def test_corrupt_store_file_rebuilds_with_warning(tmp_path, monkeypatch, caplog, victim):
    paths = _paths(tmp_path)
    store = _store(tmp_path)
    store.refresh(paths)
    with open(os.path.join(store.root, victim), "wb") as fh:
        fh.write(b"not a parquet file")
    calls = _count_cleans(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=cs.logger.name):
        rep = store.refresh(paths)
    assert rep.rebuilt and sorted(calls) == ["AAA", "BBB", "CCC"]
    assert any("clean_store" in r.message for r in caplog.records)
    for sym, path in paths.items():
        assert_frame_equal(store.load()[sym], clean_ohlcv_csv(path)[STD], check_exact=True)


def test_truncated_panel_mid_refresh_falls_back_to_rebuild(tmp_path, caplog):
    paths = _paths(tmp_path)
    store = _store(tmp_path)
    store.refresh(paths)
    other = _store(tmp_path)
    _write_csv(paths["AAA"], n=120, seed=4)
    # manifest saglam ama panel baska panel_id ile degistirildi (yarim kalmis yazim benzetimi)
    import pyarrow.parquet as pq
    tbl = pq.read_table(store.panel_path).replace_schema_metadata({b"tsfl.panel_id": b"zzz"})
    pq.write_table(tbl, store.panel_path)
    with caplog.at_level(logging.WARNING, logger=cs.logger.name):
        rep = other.refresh(paths)
    assert rep.rebuilt and rep.new == 3
    assert_frame_equal(other.load()["AAA"], clean_ohlcv_csv(paths["AAA"])[STD], check_exact=True)


def test_load_without_refresh_raises(tmp_path):
    with pytest.raises(CleanStoreError):
        _store(tmp_path).load()


def test_unreadable_symbol_not_stored_and_reported(tmp_path):
    paths = _paths(tmp_path)
    with open(paths["CCC"], "wb") as fh:
        fh.write(b"\xff\xfe\x00bad\xff")             # gecersiz utf-8
    store = _store(tmp_path)
    rep = store.refresh(paths)
    assert rep.failed == ["CCC"] and rep.new == 2
    assert sorted(store.load()) == ["AAA", "BBB"]


def test_mixed_dtype_and_missing_columns_roundtrip(tmp_path):
    d = tmp_path / "csv"
    d.mkdir()
    # tam sayi fiyatlar (int64 kalir), Volume yok
    pd.DataFrame({"Date": pd.bdate_range("2021-01-01", periods=5).strftime("%Y-%m-%d"),
                  "Open": [1, 2, 3, 4, 5], "High": [2, 3, 4, 5, 6], "Low": [1, 1, 2, 3, 4],
                  "Close": [2, 2, 3, 4, 5]}).to_csv(d / "INT.csv", index=False)
    # bos CSV
    (d / "EMPTY.csv").write_text("", encoding="utf-8")
    paths = {"INT": str(d / "INT.csv"), "EMPTY": str(d / "EMPTY.csv")}
    _write_csv(str(d / "OK.csv"))
    paths["OK"] = str(d / "OK.csv")
    store = _store(tmp_path)
    store.refresh(paths)
    frames = store.load()
    for sym, path in paths.items():
        ref = clean_ohlcv_csv(path)
        assert_frame_equal(frames[sym], ref[[c for c in STD if c in ref.columns]],
                           check_exact=True)
        assert frames[sym].attrs == ref.attrs
    assert frames["INT"]["Close"].dtype == np.int64
    assert frames["EMPTY"].empty


def test_panel_content_hash_detects_change():
    df = pd.DataFrame({"symbol": ["A", "A", "B"], "Date": pd.to_datetime(["2020-01-01"] * 3),
                       "Close": [1.0, 2.0, 3.0], "target": [0.1, 0.2, np.nan]})
    h = panel_content_hash(df)
    assert panel_content_hash(df.copy()) == h
    df.loc[1, "Close"] = 2.5
    assert panel_content_hash(df) != h


# ------------------------------------------------------------- loader entegrasyonu
class _StubFP:
    def __init__(self, *a, **k) -> None:
        pass

    def engineer_features(self, raw, macro_df=None, symbol=None, **k):
        out = pd.DataFrame({"Date": raw["Date"]})
        out["feat_a"] = pd.to_numeric(raw["Close"], errors="coerce").pct_change().values
        return out


def _loader(tmp_path, monkeypatch, store_dir, **over) -> PooledPanelLoader:
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    data_dir = tmp_path / "data"
    if not data_dir.exists():
        data_dir.mkdir()
        for i, name in enumerate(("AAA", "BBB", "CCC")):
            _write_csv(str(data_dir / f"{name}.csv"), n=120, seed=i, bad_row=name == "BBB")
        pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "Sector": ["X", "Y", "Z"]}).to_csv(
            data_dir / "bist_universe.csv", index=False, encoding="utf-8-sig")
    return PooledPanelLoader(PooledLoaderConfig(
        data_dir=str(data_dir), universe_file=str(data_dir / "bist_universe.csv"),
        min_rows=60, clean_store_dir=store_dir, **over))


def test_loader_store_on_equals_store_off(tmp_path, monkeypatch):
    off = _loader(tmp_path, monkeypatch, None)
    on = _loader(tmp_path, monkeypatch, str(tmp_path / "store"))
    p_off, p_on = off.load(), on.load()
    assert_frame_equal(p_off, p_on, check_exact=True)
    assert off.report == on.report and off.rejected_rows == on.rejected_rows == {"BBB": 1}
    assert on.clean_store_info["enabled"] and len(on.clean_store_info["content_hash"]) == 64
    assert off.clean_store_info == {"enabled": False}
    # sicak kosu: ayni sonuc, yeniden temizleme yok
    warm = _loader(tmp_path, monkeypatch, str(tmp_path / "store"))
    assert_frame_equal(p_off, warm.load(), check_exact=True)
    assert "unchanged=3" in warm.clean_store_info["refresh"]
    assert warm.clean_store_info["content_hash"] == on.clean_store_info["content_hash"]


def test_loader_limit_does_not_shrink_store(tmp_path, monkeypatch):
    ld = _loader(tmp_path, monkeypatch, str(tmp_path / "store"))
    ld._symbols = lambda: ["AAA"]
    ld.load()
    assert sorted(CleanStore(str(tmp_path / "store")).load()) == ["AAA", "BBB", "CCC"]
