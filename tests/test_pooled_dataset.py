# -*- coding: utf-8 -*-
"""Kol-B veri seti kurulumu (K4): etiketli egitim paneli + etiketsiz skorlama satirlari."""

import os

import numpy as np
import pandas as pd

import src.data.pooled_loader as pl
from src.data.pooled_dataset import build_pooled_dataset
from src.data.pooled_loader import PooledLoaderConfig


class _StubFP:
    def __init__(self, *a, **k) -> None:
        pass

    def engineer_features(self, raw, macro_df=None, symbol=None, **k):
        out = pd.DataFrame({"Date": raw["Date"]})
        close = pd.to_numeric(raw["Close"], errors="coerce")
        out["feat_a"] = close.pct_change().values
        return out


def _write(path, n, seed):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    pd.DataFrame({
        "Tarih": dates, "Açılış": close, "Yüksek": close, "Düşük": close,
        "Kapanış": close, "Düzeltilmiş_Kapanış": close,
        "Hacim": np.linspace(1000, 5000, n),
    }).to_csv(path, index=False, encoding="utf-8-sig")


def _cfg(tmp_path):
    d = os.path.join(str(tmp_path), "data")
    os.makedirs(d, exist_ok=True)
    for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        _write(os.path.join(d, f"{s}.csv"), 90, seed=i)
    pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC", "DDD"],
                  "Sector": ["Energy", "Energy", "Banks", "Banks"]}).to_csv(
        os.path.join(d, "bist_universe.csv"), index=False, encoding="utf-8-sig")
    return PooledLoaderConfig(data_dir=d, universe_file=os.path.join(d, "bist_universe.csv"),
                              target_horizon=5, min_rows=60)


def test_dataset_separates_labeled_training_from_unlabeled_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    ds = build_pooled_dataset(_cfg(tmp_path), min_names=2)
    h = 5
    # egitim paneli yalniz etiketli ve target_cs tasir
    assert ds.aug["target"].notna().all() and ds.aug["target_cs"].notna().all()
    # skorlama paneli: her sembol icin h ek (etiketsiz) satir
    extra = len(ds.full) - len(ds.aug)
    assert extra == 4 * h
    assert ds.full["Date"].max() == ds.latest_market_day > ds.aug["Date"].max()
    # en guncel gunde tum semboller var ve cs-ozellikleri dolu (hedefsiz hesaplandi)
    last = ds.full[ds.full["Date"] == ds.latest_market_day]
    assert set(last["symbol"]) == {"AAA", "BBB", "CCC", "DDD"}
    csr = [c for c in ds.feats if c.endswith("_csr")]
    assert csr and last[csr].notna().all().all()
    assert not any(c.startswith("target") for c in ds.feats)


def test_dataset_training_rows_match_labeled_loader_rows(tmp_path, monkeypatch):
    """Etiketli satir kumesi, eski (yalniz etiketli) yukleme ile ayni (symbol, Date) anahtarlari."""
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    cfg = _cfg(tmp_path)
    ds = build_pooled_dataset(cfg, min_names=2)
    old = pl.PooledPanelLoader(cfg).load()
    k_old = set(zip(old["symbol"], old["Date"]))
    k_new = set(zip(ds.aug["symbol"], ds.aug["Date"]))
    assert k_new == k_old


def test_dataset_limit_restricts_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    ds = build_pooled_dataset(_cfg(tmp_path), min_names=2, limit=3)
    assert ds.full["symbol"].nunique() == 3
