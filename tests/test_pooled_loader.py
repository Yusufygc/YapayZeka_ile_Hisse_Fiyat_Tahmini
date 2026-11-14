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


def test_keep_unlabeled_retains_last_h_rows_with_nan_target(tmp_path, monkeypatch):
    """K4: skorlama icin hedefi gerceklesmemis son h satir panelde kalir."""
    lab = _loader(tmp_path, monkeypatch).load()
    unl = _loader(tmp_path, monkeypatch, keep_unlabeled=True).load()
    h = 5
    aaa_l = lab[lab.symbol == "AAA"]
    aaa_u = unl[unl.symbol == "AAA"].sort_values("Date")
    assert len(aaa_u) == len(aaa_l) + h
    tail = aaa_u.tail(h)
    assert tail["target"].isna().all() and tail["target_date"].isna().all()
    assert aaa_u.head(len(aaa_l))["target"].notna().all()
    # en guncel piyasa gunu etiketsiz panelde, etiketlide h satir geride
    assert unl["Date"].max() > lab["Date"].max()


def test_keep_unlabeled_default_is_unchanged(tmp_path, monkeypatch):
    panel = _loader(tmp_path, monkeypatch).load()
    assert panel["target"].notna().all()


def _set_universe(loader, **cols):
    """Evren dosyasini yeniden yaz: Status/Delisted_Date kolonlari ekle (K1 testleri)."""
    uni = pd.read_csv(loader.cfg.universe_file, encoding="utf-8-sig")
    for name, values in cols.items():
        uni[name] = values
    uni.to_csv(loader.cfg.universe_file, index=False, encoding="utf-8-sig")


def test_no_delisted_in_universe_warns_and_flags_status(tmp_path, monkeypatch, caplog):
    """K1: evrende hic delisting yoksa sessiz gecme; uyari + durum bayragi."""
    loader = _loader(tmp_path, monkeypatch)
    _set_universe(loader, Status=["Active"] * 4, Delisted_Date=[""] * 4)
    with caplog.at_level("WARNING", logger="src.data.pooled_loader"):
        loader.load()
    assert loader.survivorship_status == "no_delisted_in_universe"
    assert any("hayatta kalma" in r.getMessage() for r in caplog.records)


def test_delisted_history_kept_but_stale_rows_after_delisting_dropped(tmp_path, monkeypatch):
    loader = _loader(tmp_path, monkeypatch, include_delisted=True)
    cut = pd.bdate_range("2020-01-01", periods=70)[65]   # 66 satir >= min_rows
    _set_universe(loader, Status=["Active", "Active", "Delisted", "Active"],
                  Delisted_Date=["", "", cut.strftime("%Y-%m-%d"), ""])
    panel = loader.load()
    assert loader.survivorship_status == "delisted_handled"
    ccc = panel[panel.symbol == "CCC"]
    assert len(ccc) > 0                                 # gecmisi DAHIL
    assert ccc["Date"].max() <= cut                     # delisting sonrasi satir yok
    # delisting oncesi son h=5 gun hedefsiz kalir (terminal getiri modellenmez; bilinen sinir)
    assert ccc["Date"].max() < cut


def test_include_delisted_false_excludes_symbol(tmp_path, monkeypatch):
    loader = _loader(tmp_path, monkeypatch, include_delisted=False)
    _set_universe(loader, Status=["Active", "Active", "Delisted", "Active"],
                  Delisted_Date=["", "", "2020-03-02", ""])
    panel = loader.load()
    assert "CCC" not in set(panel["symbol"])
    assert loader.report["CCC"] == "delisted_excluded"
    assert {"AAA", "BBB"} <= set(panel["symbol"])


def test_status_delisted_without_date_excluded_only_when_flag_false(tmp_path, monkeypatch):
    kw = dict(Status=["Active", "Active", "Delisted", "Active"], Delisted_Date=[""] * 4)
    keep = _loader(tmp_path, monkeypatch, include_delisted=True)
    _set_universe(keep, **kw)
    full = keep.load()
    assert "CCC" in set(full["symbol"])                 # tarih yok -> kesme yapilmaz
    drop = _loader(tmp_path, monkeypatch, include_delisted=False)
    _set_universe(drop, **kw)
    assert "CCC" not in set(drop.load()["symbol"])


def _write_close_csv(path: str, close: np.ndarray) -> None:
    n = len(close)
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    pd.DataFrame({
        "Tarih": dates, "Açılış": close, "Yüksek": close, "Düşük": close,
        "Kapanış": close, "Düzeltilmiş_Kapanış": close,
        "Hacim": np.linspace(1000, 5000, n),
    }).to_csv(path, index=False, encoding="utf-8-sig")


def test_fallback_dir_fills_missing_symbols_but_never_overrides_primary(tmp_path, monkeypatch):
    """K7: yedek klasoru yalniz eksik sembolleri tamamlar; birincil dosya onceliklidir."""
    loader = _loader(tmp_path, monkeypatch)          # birincil: AAA, BBB, CCC, TINY
    primary = loader.cfg.data_dir
    backup = os.path.join(str(tmp_path), "yedek")
    os.makedirs(backup)
    _write_close_csv(os.path.join(backup, "BBB.csv"), np.linspace(100.0, 50.0, 90))  # farkli
    _write_close_csv(os.path.join(backup, "DDD.csv"), np.linspace(5.0, 9.0, 90))
    cfg = PooledLoaderConfig(**{**loader.cfg.__dict__, "fallback_dirs": (backup,)})
    fb = PooledPanelLoader(cfg)
    panel = fb.load()
    assert {"AAA", "BBB", "DDD"} <= set(panel["symbol"])
    assert fb.source_dir["BBB"] == primary and fb.source_dir["DDD"] == backup
    bbb_close = panel[panel.symbol == "BBB"]["Close"]
    assert bbb_close.max() <= 20.0                   # birincil BBB (10..20), yedek (100..50) degil


def test_jump_quarantine_removes_rows_around_scale_break_only(tmp_path, monkeypatch):
    """K8: bolunme benzeri olcek kirigi (x0.5) etrafindaki satirlar atilir, diger semboller etkilenmez."""
    def build(**over):
        loader = _loader(tmp_path, monkeypatch, **over)
        close = np.linspace(10.0, 20.0, 90)
        close[40:] *= 0.5                            # 40. gunde -%50 yapay dusus
        _write_close_csv(os.path.join(loader.cfg.data_dir, "AAA.csv"), close)
        return loader

    plain = build()
    p0 = plain.load()
    assert (p0[p0.symbol == "AAA"]["target"] < -0.5).any()    # artefakt panelde (test anlamli)

    q = build(quarantine_jumps=True, quarantine_rows=20)
    p1 = q.load()
    aaa = p1[p1.symbol == "AAA"]
    assert len(aaa) > 0 and (aaa["target"].abs() < 0.3).all()
    assert q.quarantined_days["AAA"] > 0
    # sicrama gunu ve sonraki quarantine_rows satir yok
    jump_day = pd.bdate_range("2020-01-01", periods=90)[40]
    assert not ((aaa["Date"] >= jump_day) &
                (aaa["Date"] < pd.bdate_range("2020-01-01", periods=90)[60])).any()
    # etkilenmeyen sembol ayni
    assert "BBB" not in q.quarantined_days
    assert len(p1[p1.symbol == "BBB"]) == len(p0[p0.symbol == "BBB"])
