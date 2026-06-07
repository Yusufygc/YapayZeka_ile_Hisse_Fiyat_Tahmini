# -*- coding: utf-8 -*-
"""Structure + causality tests for the pooled panel loader (stubbed features)."""

import os

import numpy as np
import pandas as pd

import src.data.pooled_loader as pl
from src.data.pooled_loader import PooledLoaderConfig, PooledPanelLoader


class _StubFP:
    """Date + tek causal ozellik (feat_a) donduren sahte FeaturePipeline."""

    def __init__(self, *a, **k) -> None:
        pass

    def engineer_features(self, raw, macro_df=None, symbol=None, **k):
        out = pd.DataFrame({"Date": raw["Date"]})
        close = pd.to_numeric(raw["Close"], errors="coerce")
        out["feat_a"] = close.pct_change().values
        return out


def _write_csv(path: str, n: int) -> None:
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    close = np.linspace(10.0, 20.0, n)
    df = pd.DataFrame({
        "Tarih": dates, "Açılış": close, "Yüksek": close,
        "Düşük": close, "Kapanış": close, "Düzeltilmiş_Kapanış": close,
        "Hacim": np.linspace(1000, 5000, n),
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _setup(tmp_path) -> str:
    data_dir = os.path.join(str(tmp_path), "data")
    os.makedirs(data_dir, exist_ok=True)
    _write_csv(os.path.join(data_dir, "AAA.csv"), 90)
    _write_csv(os.path.join(data_dir, "BBB.csv"), 90)
    _write_csv(os.path.join(data_dir, "CCC.csv"), 70)   # daha kisa (delisted-vari)
    _write_csv(os.path.join(data_dir, "TINY.csv"), 20)  # min_rows altinda -> atlanir
    uni = pd.DataFrame({
        "Symbol": ["AAA", "BBB", "CCC", "TINY"],
        "Sector": ["Industrials", "Financial Services", "", "Energy"],
        "Sector_Index": ["", "XBANK", "", ""],
    })
    uni.to_csv(os.path.join(data_dir, "bist_universe.csv"), index=False, encoding="utf-8-sig")
    return data_dir


def _loader(tmp_path, monkeypatch, **over) -> PooledPanelLoader:
    monkeypatch.setattr(pl, "FeaturePipeline", _StubFP)
    data_dir = _setup(tmp_path)
    cfg = PooledLoaderConfig(
        data_dir=data_dir,
        universe_file=os.path.join(data_dir, "bist_universe.csv"),
        target_horizon=5, min_rows=60, **over,
    )
    return PooledPanelLoader(cfg)


def test_panel_schema_and_symbol_uniqueness(tmp_path, monkeypatch):
    panel = _loader(tmp_path, monkeypatch).load()
    for col in ["symbol", "Date", "feat_a", "target", "target_date",
                "sector", "liq_log", "vol", "symbol_id"]:
        assert col in panel.columns
    assert not panel.duplicated(subset=["symbol", "Date"]).any()


def test_tiny_symbol_skipped_min_rows(tmp_path, monkeypatch):
    loader = _loader(tmp_path, monkeypatch)
    panel = loader.load()
    assert "TINY" not in set(panel["symbol"])
    assert loader.report.get("TINY", "").startswith("too_short")


def test_target_is_h_day_forward_log_return(tmp_path, monkeypatch):
    panel = _loader(tmp_path, monkeypatch).load()
    aaa = panel[panel.symbol == "AAA"].sort_values("Date").reset_index(drop=True)
    # ham seriden beklenen
    n = 90
    close = np.linspace(10.0, 20.0, n)
    dates = pd.bdate_range("2020-01-01", periods=n)
    first = aaa.iloc[0]
    pos = int(np.where(dates == first["Date"])[0][0])
    expected = np.log(close[pos + 5] / close[pos])
    assert abs(first["target"] - expected) < 1e-9
    # target_date = t+5 tarihi
    assert pd.Timestamp(first["target_date"]) == dates[pos + 5]


def test_last_h_rows_dropped_no_nan_target(tmp_path, monkeypatch):
    panel = _loader(tmp_path, monkeypatch).load()
    assert not panel["target"].isna().any()
    aaa = panel[panel.symbol == "AAA"]
    # 90 satir, ilk pct_change NaN + son 5 target NaN -> en fazla 84 satir
    assert len(aaa) <= 84


def test_sector_merged_and_unknown_fallback(tmp_path, monkeypatch):
    panel = _loader(tmp_path, monkeypatch).load()
    assert set(panel.loc[panel.symbol == "AAA", "sector"]) == {"Industrials"}
    # CCC universe'de Sector bos -> Unknown
    assert set(panel.loc[panel.symbol == "CCC", "sector"]) == {"Unknown"}


def test_symbol_id_stable_sorted(tmp_path, monkeypatch):
    panel = _loader(tmp_path, monkeypatch).load()
    mapping = panel.drop_duplicates("symbol").set_index("symbol")["symbol_id"].to_dict()
    assert mapping == {s: i for i, s in enumerate(sorted(mapping))}
