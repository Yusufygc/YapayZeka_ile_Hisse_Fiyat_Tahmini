# -*- coding: utf-8 -*-
"""Pooled panel sembol-paralel hesap (n_jobs): sekansiyelle BIREBIR sonuc, sira, yan bilgi."""

import logging
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import src.data.pooled_loader as pl
import src.data.pooled_parallel as pp
from src.data.pooled_loader import PooledLoaderConfig, PooledPanelLoader


def _write_csv(path: str, n: int, seed: int, *, jump: bool = False, bad_row: bool = False) -> None:
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    if jump:
        close[n // 2:] *= 3.0                              # olcek kirigi -> K8 karantina
    dates = pd.bdate_range("2019-01-01", periods=n).strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "Tarih": dates, "Açılış": close * 0.99, "Yüksek": close * 1.01,
        "Düşük": close * 0.98, "Kapanış": close, "Düzeltilmiş_Kapanış": close,
        "Hacim": rng.integers(1_000, 90_000, n).astype("float64"),
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if bad_row:                                            # alan sayisi yanlis satir -> rejected_rows
        with open(path, "a", encoding="utf-8-sig") as fh:
            fh.write("2030-01-01,1,2,3,4,5,6,7,8,9\n")


def _setup(tmp_path) -> tuple[str, str]:
    data_dir = os.path.join(str(tmp_path), "data")
    os.makedirs(data_dir)
    _write_csv(os.path.join(data_dir, "AAA.csv"), 330, 1)
    _write_csv(os.path.join(data_dir, "BBB.csv"), 330, 2, jump=True)
    _write_csv(os.path.join(data_dir, "CCC.csv"), 330, 3, bad_row=True)
    _write_csv(os.path.join(data_dir, "DDD.csv"), 330, 4)        # delisted
    _write_csv(os.path.join(data_dir, "EEE.csv"), 330, 5)
    _write_csv(os.path.join(data_dir, "TINY.csv"), 20, 6)        # too_short
    uni = os.path.join(data_dir, "bist_universe.csv")
    pd.DataFrame({
        "Symbol": ["AAA", "BBB", "CCC", "DDD", "EEE", "TINY"],
        "Sector": ["Industrials", "Financial Services", "", "Energy", "Energy", "Energy"],
        "Delisted_Date": ["", "", "", "2020-03-02", "", ""],
    }).to_csv(uni, index=False, encoding="utf-8-sig")
    return data_dir, uni


def _cfg(tmp_path, **over) -> PooledLoaderConfig:
    data_dir, uni = _setup(tmp_path) if not os.path.exists(os.path.join(str(tmp_path), "data")) \
        else (os.path.join(str(tmp_path), "data"), os.path.join(str(tmp_path), "data", "bist_universe.csv"))
    base = dict(data_dir=data_dir, universe_file=uni, target_horizon=5, min_rows=60,
                quarantine_jumps=True, keep_unlabeled=True)
    base.update(over)
    return PooledLoaderConfig(**base)


def _assert_same(a: PooledPanelLoader, pa: pd.DataFrame, b: PooledPanelLoader, pb: pd.DataFrame):
    pd.testing.assert_frame_equal(pa, pb, check_exact=True)
    assert list(a.report.items()) == list(b.report.items())          # icerik + sira
    assert list(a.rejected_rows.items()) == list(b.rejected_rows.items())
    assert list(a.quarantined_days.items()) == list(b.quarantined_days.items())
    assert a.feature_cache_stats == b.feature_cache_stats
    assert a.stale_symbols == b.stale_symbols


def test_config_validation():
    assert PooledLoaderConfig(data_dir="d", universe_file="u").n_jobs == 1
    for bad in (0, -2):
        with pytest.raises(ValueError):
            PooledLoaderConfig(data_dir="d", universe_file="u", n_jobs=bad)
    with pytest.raises(ValueError):
        PooledLoaderConfig(data_dir="d", universe_file="u", parallel_chunk_size=-1)


def test_worker_and_chunk_resolution():
    assert pp.resolve_workers(1, 100) == 1
    assert pp.resolve_workers(4, 100) == 4
    assert pp.resolve_workers(4, 2) == 2                     # is sayisini asmaz
    assert 1 <= pp.resolve_workers(-1, 1000) <= pp.MAX_AUTO_WORKERS
    assert pp.resolve_chunk_size(7, 1000, 8) == 7
    assert pp.AUTO_CHUNK_MIN <= pp.resolve_chunk_size(0, 581, 8) <= pp.AUTO_CHUNK_MAX
    assert pp.resolve_chunk_size(0, 3, 8) == pp.AUTO_CHUNK_MIN


def test_parallel_matches_sequential_real_pipeline(tmp_path, monkeypatch):
    """Gercek FeaturePipeline + loky surec havuzu: panel, report, rejected/quarantine birebir."""
    calls = []
    real = pl.run_parallel
    monkeypatch.setattr(pl, "run_parallel", lambda *a: calls.append(a[3:]) or real(*a))
    seq = PooledPanelLoader(_cfg(tmp_path, n_jobs=1))
    p_seq = seq.load()
    par = PooledPanelLoader(_cfg(tmp_path, n_jobs=2, parallel_chunk_size=2))
    p_par = par.load()
    assert calls == [(2, 2)]                                 # havuz GERCEKTEN kullanildi
    assert not p_seq.empty
    _assert_same(seq, p_seq, par, p_par)
    # senaryolar gercekten olustu
    assert seq.report["TINY"].startswith("too_short")
    assert seq.rejected_rows.get("CCC", 0) >= 1
    assert seq.quarantined_days.get("BBB", 0) > 0
    assert p_seq.loc[p_seq["symbol"] == "DDD", "Date"].max() <= pd.Timestamp("2020-03-02")
    assert list(p_par["symbol"].unique()) == list(p_seq["symbol"].unique())


def test_parallel_feature_cache_stats_aggregated(tmp_path):
    cdir = os.path.join(str(tmp_path), "fc")
    cold = PooledPanelLoader(_cfg(tmp_path, n_jobs=2, parallel_chunk_size=2, feature_cache_dir=cdir))
    p_cold = cold.load()
    n_ok = sum(1 for v in cold.report.values() if v == "ok" or v.startswith("few_usable"))
    stats = cold.feature_cache_stats
    assert stats["enabled"] and stats["hits"] == 0 and stats["errors"] == 0
    assert stats["misses"] == n_ok >= 4
    warm = PooledPanelLoader(_cfg(tmp_path, n_jobs=2, parallel_chunk_size=3, feature_cache_dir=cdir))
    p_warm = warm.load()
    assert warm.feature_cache_stats["hits"] == stats["misses"]
    assert warm.feature_cache_stats["misses"] == 0
    # sekansiyel + onbellek acik (sicak) ve onbelleksiz sekansiyel ile ayni panel
    seq = PooledPanelLoader(_cfg(tmp_path, n_jobs=1, feature_cache_dir=cdir))
    pd.testing.assert_frame_equal(p_cold, p_warm, check_exact=True)
    pd.testing.assert_frame_equal(seq.load(), p_warm, check_exact=True)
    assert seq.feature_cache_stats["hits"] == stats["misses"]


# ----------------------------------------------------- surec-ici (stub) senaryolar
class _StubFP:
    """Sembole gore davranan sahte FeaturePipeline: BAD hata verir, WARN uyari loglar."""

    def __init__(self, *a, **k) -> None:
        pass

    def engineer_features(self, raw, macro_df=None, symbol=None, **k):
        if symbol == "BAD":
            raise RuntimeError("sahte ozellik hatasi " + "x" * 100)
        if symbol == "WARN":
            logging.getLogger("tsfl.test_worker").warning("uyari sembol=%s", symbol)
        out = pd.DataFrame({"Date": raw["Date"]})
        out["feat_a"] = pd.to_numeric(raw["Close"], errors="coerce").pct_change().values
        return out


def _inprocess_parallel(tasks, cfg, fingerprint, workers, chunk_size):
    """`run_parallel` yerine: ayni parcalama + `run_chunk`, ama bu surecte (stub patch'i gecerli)."""
    size = pp.resolve_chunk_size(chunk_size, len(tasks), workers)
    return [pp.run_chunk(c, cfg, fingerprint) for c in pp.split_chunks(tasks, size)]


def _stub_setup(tmp_path) -> PooledLoaderConfig:
    data_dir = os.path.join(str(tmp_path), "sd")
    os.makedirs(data_dir)
    names = ["AAA", "BAD", "BBB", "CCC", "TINY", "WARN", "ZZZ"]
    for i, s in enumerate(names):
        _write_csv(os.path.join(data_dir, f"{s}.csv"), 20 if s == "TINY" else 120, 10 + i,
                   bad_row=(s == "CCC"))
    return PooledLoaderConfig(data_dir=data_dir, universe_file=os.path.join(data_dir, "u.csv"),
                              min_rows=60, quarantine_jumps=True)


@pytest.mark.parametrize("chunk", [1, 2, 3, 100])
def test_merge_error_order_and_chunk_independence(tmp_path, monkeypatch, chunk):
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    monkeypatch.setattr(pl, "run_parallel", _inprocess_parallel)
    cfg = _stub_setup(tmp_path)
    seq = PooledPanelLoader(cfg)
    p_seq = seq.load()
    par = PooledPanelLoader(replace(cfg, n_jobs=3, parallel_chunk_size=chunk))
    p_par = par.load()
    _assert_same(seq, p_seq, par, p_par)
    assert par.report["BAD"].startswith("error:RuntimeError") or par.report["BAD"].startswith("error:sahte")
    assert len(par.report["BAD"]) <= len("error:") + 60
    assert par.report["TINY"].startswith("too_short")
    assert par.rejected_rows == {"CCC": 1}
    assert list(par.report) == sorted(par.report)            # sembol sirasi korunur
    assert "BAD" not in set(p_par["symbol"]) and "AAA" in set(p_par["symbol"])


def test_delisted_excluded_in_parallel_path(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    monkeypatch.setattr(pl, "run_parallel", _inprocess_parallel)
    cfg = _stub_setup(tmp_path)
    pd.DataFrame({"Symbol": ["AAA", "CCC"], "Sector": ["A", "B"],
                  "Delisted_Date": ["", "2019-06-03"]}).to_csv(
        cfg.universe_file, index=False, encoding="utf-8-sig")
    cfg = replace(cfg, include_delisted=False)
    seq = PooledPanelLoader(cfg)
    p_seq = seq.load()
    par = PooledPanelLoader(replace(cfg, n_jobs=2, parallel_chunk_size=2))
    p_par = par.load()
    _assert_same(seq, p_seq, par, p_par)
    assert par.report["CCC"] == "delisted_excluded"


def test_worker_warnings_are_returned_and_replayed(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    cfg = _stub_setup(tmp_path)
    paths = {s: os.path.join(cfg.data_dir, f"{s}.csv") for s in ("AAA", "WARN")}
    tasks = [pp.SymbolTask(s, "Unknown", None, p) for s, p in paths.items()]
    res = pp.run_chunk(tasks, cfg, None)
    assert any(name == "tsfl.test_worker" and "uyari sembol=WARN" in msg
               for name, _, msg in res.logs)
    assert [r.symbol for r in res.results] == ["AAA", "WARN"]
    with caplog.at_level(logging.WARNING, logger="tsfl.test_worker"):
        caplog.clear()
        pp.replay_logs([("tsfl.test_worker", logging.WARNING, "yeniden yayin")])
    assert "yeniden yayin" in caplog.text


def test_single_symbol_or_n_jobs_1_stays_sequential(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    called = []
    monkeypatch.setattr(pl, "run_parallel", lambda *a, **k: called.append(1))
    cfg = _stub_setup(tmp_path)
    PooledPanelLoader(cfg).load()                             # n_jobs=1
    loader = PooledPanelLoader(replace(cfg, n_jobs=4))
    loader._symbols = lambda: ["AAA"]                         # build_pooled_dataset(limit=1) gibi
    loader.load()
    assert called == []
