# -*- coding: utf-8 -*-
"""P1 derin denetim (2026-11) — veri + model katmani duzeltmeleri.

1. Ham OHLCV seviyeleri tek-sembol modellerinin X'ine girmez.
2. WF embargo varsayilani `target_horizon + 5`; explicit embargo < h reddedilir;
   embargo yalnizca etiket ufkuna baglidir.
3. ExecutionConfig maliyet varsayilanlari BIST gercegine hizali (sifir degil).
4. DataUpdater: overlap'li indirme; split/temettu sonrasi tam yeniden indirme.
5. data_loader: Close=Adj_Close iken O/H/L ayni faktorle olceklenir.
6. LightGBM bagging gercekten aktif (subsample_freq).
7. (kaldirildi: LSTM Lite silindi)
8. Varsayilan uretim adaylari: XGBoost + Ridge Return + LightGBM Return.

Ag erisimi yok: indirici sahte provider ile enjekte edilir.
"""
from __future__ import annotations

import inspect
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.pipeline.config import DataConfig, ExecutionConfig, ValidationConfig
from src.pipeline.data_manager import DataManager
from src.utils.data_splitter import TimeSeriesSplitter, _resolve_wf_embargo_size


# --------------------------------------------------------------------------- #
#  Yardimcilar                                                                 #
# --------------------------------------------------------------------------- #
def _ohlcv_frame(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.sin(np.arange(n) / 3.0))
    return pd.DataFrame({
        "Date": dates,
        "Open": close * 0.99,
        "High": close * 1.01,
        "Low": close * 0.98,
        "Close": close,
        "Adj_Close": close,
        "Volume": np.linspace(1e6, 2e6, n),
        "Feat_A": np.sin(np.arange(n) / 5.0),
        "Feat_B": np.cos(np.arange(n) / 7.0),
    })


def _manager(tmp_path, *, val_cfg=None, target_horizon=1) -> DataManager:
    return DataManager(
        data_cfg=DataConfig(
            data_file=os.path.join(str(tmp_path), "TEST.csv"),
            use_macro=False,
            prune_correlated_features=False,
            target_horizon=target_horizon,
        ),
        val_cfg=val_cfg or ValidationConfig(),
        models_dir=os.path.join(str(tmp_path), "models"),
    )


# --------------------------------------------------------------------------- #
#  1) Ham OHLCV seviyeleri X'te yok                                            #
# --------------------------------------------------------------------------- #
RAW_LEVELS = {"Open", "High", "Low", "Volume", "Adj_Close", "Close", "Date"}


def test_prepare_tensors_excludes_raw_ohlcv_levels(tmp_path):
    manager = _manager(tmp_path)
    df = _ohlcv_frame(40)

    tensors = manager.prepare_tensors(df.iloc[:25].copy(), df.iloc[25:].copy())

    assert tensors["features"] == ["Feat_A", "Feat_B"]
    assert not RAW_LEVELS & set(tensors["features"])
    assert tensors["X_train"].shape[1] == 2


def test_model_feature_names_drop_raw_levels():
    from src.pipeline.data_services import model_feature_names

    names = ["Open", "High", "Low", "Volume", "Adj_Close", "Adj Close", "RSI", "Log_Return"]
    assert model_feature_names(names) == ["RSI", "Log_Return"]


# --------------------------------------------------------------------------- #
#  2) Embargo                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("h", [1, 5, 10])
def test_default_embargo_is_horizon_plus_five(h):
    assert _resolve_wf_embargo_size(None, target_horizon=h) == h + 5
    assert _resolve_wf_embargo_size(0, target_horizon=h) == h + 5


def test_default_embargo_no_longer_200():
    assert _resolve_wf_embargo_size(None) == 6  # h=1 -> 6, SMA200 tamponu degil


def test_explicit_embargo_below_horizon_raises():
    with pytest.raises(ValueError):
        _resolve_wf_embargo_size(3, target_horizon=5)
    assert _resolve_wf_embargo_size(5, target_horizon=5) == 5
    assert _resolve_wf_embargo_size(50, target_horizon=5) == 50


def test_split_data_uses_horizon_aware_default_embargo(tmp_path):
    val_cfg = ValidationConfig(
        wf_n_splits=2, wf_min_train_size=40, wf_test_size=10,
        wf_max_train_size=None, wf_embargo_size=None, final_holdout_size=0,
    )
    manager = _manager(tmp_path, val_cfg=val_cfg, target_horizon=5)
    manager.df = _ohlcv_frame(150)

    manager.split_data("walk_forward")

    assert manager.validation_config["wf_embargo_size"] == 10
    assert all(s["embargo_size"] == 10 for s in manager.wf_splits)
    assert all(len(s["embargo_context"]) == 10 for s in manager.wf_splits)


def test_split_data_rejects_explicit_embargo_below_horizon(tmp_path):
    val_cfg = ValidationConfig(
        wf_n_splits=2, wf_min_train_size=40, wf_test_size=10,
        wf_max_train_size=None, wf_embargo_size=3, final_holdout_size=0,
    )
    # DataManager kurulumda embargoyu gercek horizon ile cozer -> erken reddeder.
    with pytest.raises(ValueError, match="embargo"):
        manager = _manager(tmp_path, val_cfg=val_cfg, target_horizon=5)
        manager.df = _ohlcv_frame(150)
        manager.split_data("walk_forward")


# --------------------------------------------------------------------------- #
#  3) Maliyet varsayilanlari                                                   #
# --------------------------------------------------------------------------- #
def test_execution_config_cost_defaults_are_bist_realistic():
    from src.forecasting.bist_rules import BistMarketRules

    sig = inspect.signature(BistMarketRules.trend_threshold)
    comm = sig.parameters["commission_bps"].default
    slip = sig.parameters["slippage_bps"].default
    cfg = ExecutionConfig()
    # BSMV (%5, komisyon uzerinden) komisyona katlanir: 10 * 1.05
    assert cfg.commission_bps == pytest.approx(comm * 1.05)
    assert cfg.slippage_bps == pytest.approx(slip)
    assert cfg.commission_bps > 0 and cfg.slippage_bps > 0


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


# --------------------------------------------------------------------------- #
#  6) LightGBM bagging                                                         #
# --------------------------------------------------------------------------- #
def test_lightgbm_bagging_is_active():
    from src.models.gradient_boosting_model import LightGBMReturnModel

    model = LightGBMReturnModel()
    assert model.params["subsample"] < 1.0
    assert model.params.get("subsample_freq", 0) >= 1
    lgb_model = model._build_model()
    assert lgb_model.get_params()["subsample_freq"] >= 1


# --------------------------------------------------------------------------- #
#  8) Varsayilan uretim adaylari                                               #
# --------------------------------------------------------------------------- #
def test_default_candidate_set_is_production_safe():
    from src.pipeline.model_scope import default_candidate_models

    assert set(default_candidate_models()) == {"XGBoost", "Ridge Return", "LightGBM Return"}
    for removed in ("LSTM", "DLinear", "NLinear", "Prophet", "ARIMA"):
        assert removed not in default_candidate_models()
