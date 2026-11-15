# -*- coding: utf-8 -*-
"""Kol-B sembol basi Parquet ozellik onbellegi + kod parmak izi testleri."""

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.data.pooled_loader as pl
import src.features.feature_cache as fc_mod
from src.data.pooled_loader import PooledLoaderConfig, PooledPanelLoader
from src.features.code_fingerprint import (
    PIPELINE_ROOTS, POOLED_ROOTS, code_fingerprint, module_closure)
from src.features.feature_cache import FeatureCache
from src.features.pooled_feature_cache import PooledFeatureCache, frame_content_hash

KW = {"feature_mode": "stationary_features", "enable_calendar_features": False}


def _raw(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10 + np.cumsum(rng.normal(size=n))
    return pd.DataFrame({
        "Date": pd.bdate_range("2021-01-01", periods=n),
        "Open": close, "High": close + 1, "Low": close - 1, "Close": close,
        "Volume": rng.integers(1000, 5000, n).astype("float64"),
    })


def _feat(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Date": raw["Date"], "Close": raw["Close"]})
    out["f_float"] = raw["Close"].pct_change() * 1.0000000001
    out["f_inf"] = np.where(np.arange(len(raw)) % 7 == 0, np.inf, 0.1)
    out["f_f32"] = out["f_float"].astype("float32")
    out["f_int"] = np.arange(len(raw), dtype="int64")
    out["f_bool"] = (np.arange(len(raw)) % 2 == 0)
    out["f_obj"] = np.where(np.arange(len(raw)) % 2 == 0, "a", "b")
    out["f_cat"] = pd.Categorical(out["f_obj"])
    return out.iloc[3:].reset_index(drop=True)


def _cache(tmp_path, fp: str = "fp1") -> PooledFeatureCache:
    return PooledFeatureCache(os.path.join(str(tmp_path), "fc"), fingerprint=fp)


def test_hit_returns_exact_frame_and_dtypes(tmp_path):
    cache, raw = _cache(tmp_path), _raw()
    calls = []

    def compute():
        calls.append(1)
        return _feat(raw)

    first = cache.get_or_compute("AAA", raw, compute=compute, **KW)
    second = cache.get_or_compute("AAA", raw, compute=compute, **KW)
    assert len(calls) == 1
    assert cache.stats() == {"hits": 1, "misses": 1, "errors": 0}
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    pd.testing.assert_frame_equal(_feat(raw), second, check_exact=True)
    assert (second.dtypes == _feat(raw).dtypes).all()
    assert second["Date"].dtype == "datetime64[ns]"
    assert second["f_float"].dtype == np.float64
    assert second["f_cat"].dtype.name == "category"
    assert second.index.equals(pd.RangeIndex(len(second)))
    second.loc[0, "f_float"] = 1.0       # donen cerceve yazilabilir olmali
    assert not [f for f in os.listdir(cache.cache_dir) if f.endswith(".tmp")]


def test_changed_input_value_is_a_miss(tmp_path):
    cache, raw = _cache(tmp_path), _raw()
    cache.get_or_compute("AAA", raw, compute=lambda: _feat(raw), **KW)
    raw2 = raw.copy()
    raw2.loc[10, "Close"] = raw2.loc[10, "Close"] + 1e-9
    calls = []
    cache.get_or_compute("AAA", raw2, compute=lambda: calls.append(1) or _feat(raw2), **KW)
    assert calls == [1] and cache.misses == 2 and cache.hits == 0
    # mod/takvim bayragi da anahtarda
    cache.get_or_compute("AAA", raw2, compute=lambda: calls.append(1) or _feat(raw2),
                         feature_mode="other", enable_calendar_features=False)
    assert len(calls) == 2


def test_changed_fingerprint_is_a_miss(tmp_path):
    raw = _raw()
    d = os.path.join(str(tmp_path), "fc")
    PooledFeatureCache(d, fingerprint="v1").get_or_compute(
        "AAA", raw, compute=lambda: _feat(raw), **KW)
    other = PooledFeatureCache(d, fingerprint="v2")
    calls = []
    other.get_or_compute("AAA", raw, compute=lambda: calls.append(1) or _feat(raw), **KW)
    assert calls == [1] and other.stats() == {"hits": 0, "misses": 1, "errors": 0}
    same = PooledFeatureCache(d, fingerprint="v2")     # yeniden yazilmis olmali
    same.get_or_compute("AAA", raw, compute=lambda: pytest.fail("hit bekleniyordu"), **KW)
    assert same.hits == 1


def test_corrupt_file_is_miss_and_rewritten(tmp_path, caplog):
    cache, raw = _cache(tmp_path), _raw()
    cache.get_or_compute("AAA", raw, compute=lambda: _feat(raw), **KW)
    path = os.path.join(cache.cache_dir, "AAA.parquet")
    with open(path, "wb") as fh:
        fh.write(b"not a parquet file")
    cache2 = PooledFeatureCache(cache.cache_dir, fingerprint="fp1")
    with caplog.at_level("WARNING"):
        out = cache2.get_or_compute("AAA", raw, compute=lambda: _feat(raw), **KW)
    pd.testing.assert_frame_equal(out, _feat(raw), check_exact=True)
    assert cache2.stats() == {"hits": 0, "misses": 1, "errors": 1}
    assert any("bozuk dosya" in r.message for r in caplog.records)
    cache3 = PooledFeatureCache(cache.cache_dir, fingerprint="fp1")
    cache3.get_or_compute("AAA", raw, compute=lambda: pytest.fail("yeniden yazilmamis"), **KW)
    assert cache3.hits == 1


def test_unsafe_symbol_never_touches_filesystem(tmp_path):
    cache, raw = _cache(tmp_path), _raw()
    out = cache.get_or_compute("../evil", raw, compute=lambda: _feat(raw), **KW)
    assert out is not None
    assert os.listdir(cache.cache_dir) == []
    assert not os.path.exists(os.path.join(str(tmp_path), "evil.parquet"))


def test_empty_or_none_result_not_cached(tmp_path):
    cache, raw = _cache(tmp_path), _raw()
    assert cache.get_or_compute("AAA", raw, compute=lambda: None, **KW) is None
    assert os.listdir(cache.cache_dir) == []


def test_content_hash_sensitivity():
    raw = _raw()
    h = frame_content_hash(raw)
    assert h == frame_content_hash(raw.copy())
    assert h != frame_content_hash(raw.rename(columns={"Open": "open"}))
    assert h != frame_content_hash(raw.astype({"Volume": "float32"}))
    r2 = raw.copy()
    r2.loc[0, "Volume"] = np.nan
    assert h != frame_content_hash(r2)


# ------------------------------------------------------------- fingerprint
def test_code_fingerprint_closure_and_stability():
    pooled = set(module_closure(POOLED_ROOTS))
    assert {"src/features/feature_pipeline.py", "src/features/correlation_pruning.py",
            "src/features/sector_mapping.py", "src/xai/feature_dictionary.py"} <= pooled
    assert pooled < set(module_closure(PIPELINE_ROOTS))
    assert "src/features/macro_pipeline.py" in module_closure(PIPELINE_ROOTS)
    assert code_fingerprint(POOLED_ROOTS) == code_fingerprint(POOLED_ROOTS)
    assert code_fingerprint(POOLED_ROOTS) != code_fingerprint(PIPELINE_ROOTS)
    assert len(code_fingerprint(POOLED_ROOTS)) == 64


def test_kol_a_key_includes_code_fingerprint(tmp_path, monkeypatch):
    csv = os.path.join(str(tmp_path), "x.csv")
    open(csv, "w").write("a\n1\n")
    cfg = SimpleNamespace(feature_mode="stationary_features")
    cache = FeatureCache(os.path.join(str(tmp_path), "fc"))
    k1 = cache.make_key(csv, cfg)
    assert k1 == cache.make_key(csv, cfg)
    monkeypatch.setattr(fc_mod, "code_fingerprint", lambda roots: "changed")
    assert cache.make_key(csv, cfg) != k1


# ----------------------------------------------------------- loader wiring
class _CountingFP:
    calls = 0

    def __init__(self, *a, **k) -> None:
        pass

    def engineer_features(self, raw, macro_df=None, symbol=None, **k):
        type(self).calls += 1
        out = pd.DataFrame({"Date": raw["Date"]})
        out["feat_a"] = pd.to_numeric(raw["Close"], errors="coerce").pct_change().values
        return out


def _write_csv(path: str, n: int, bump: float = 0.0) -> None:
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    close = np.linspace(10.0, 20.0, n) + bump
    pd.DataFrame({
        "Tarih": dates, "Açılış": close, "Yüksek": close, "Düşük": close,
        "Kapanış": close, "Düzeltilmiş_Kapanış": close,
        "Hacim": np.linspace(1000, 5000, n),
    }).to_csv(path, index=False, encoding="utf-8-sig")


def _loader(tmp_path, monkeypatch, cache_dir) -> PooledPanelLoader:
    monkeypatch.setattr(pl, "FeaturePipeline", _CountingFP)
    data_dir = os.path.join(str(tmp_path), "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in ("AAA", "BBB"):
        if not os.path.exists(os.path.join(data_dir, f"{name}.csv")):
            _write_csv(os.path.join(data_dir, f"{name}.csv"), 90)
    return PooledPanelLoader(PooledLoaderConfig(
        data_dir=data_dir, universe_file=os.path.join(data_dir, "u.csv"),
        min_rows=60, feature_cache_dir=cache_dir))


def test_loader_second_build_makes_zero_pipeline_calls(tmp_path, monkeypatch):
    cdir = os.path.join(str(tmp_path), "fc")
    _CountingFP.calls = 0
    p1 = _loader(tmp_path, monkeypatch, cdir)
    panel1 = p1.load()
    assert _CountingFP.calls == 2
    assert p1.feature_cache_stats == {"enabled": True, "hits": 0, "misses": 2, "errors": 0}
    p2 = _loader(tmp_path, monkeypatch, cdir)
    panel2 = p2.load()
    assert _CountingFP.calls == 2                       # ikinci yapim: 0 yeni cagri
    assert p2.feature_cache_stats == {"enabled": True, "hits": 2, "misses": 0, "errors": 0}
    pd.testing.assert_frame_equal(panel1, panel2, check_exact=True)
    # tek sembolun verisi degisince yalniz o sembol yeniden hesaplanir
    _write_csv(os.path.join(str(tmp_path), "data", "BBB.csv"), 90, bump=0.5)
    p3 = _loader(tmp_path, monkeypatch, cdir)
    p3.load()
    assert _CountingFP.calls == 3
    assert (p3.feature_cache_stats["hits"], p3.feature_cache_stats["misses"]) == (1, 1)


def test_loader_cache_disabled_by_default(tmp_path, monkeypatch):
    _CountingFP.calls = 0
    loader = _loader(tmp_path, monkeypatch, None)
    assert PooledLoaderConfig(data_dir="d", universe_file="u").feature_cache_dir is None
    loader.load()
    loader2 = _loader(tmp_path, monkeypatch, None)
    loader2.load()
    assert _CountingFP.calls == 4
    assert loader.feature_cache_stats == {"enabled": False, "hits": 0, "misses": 0, "errors": 0}
