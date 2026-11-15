# -*- coding: utf-8 -*-
"""Risk testleri icin sentetik pooled panel (yalniz sentetik veri)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_panel(n_sym: int = 24, n_days: int = 1100, seed: int = 0,
               drop: dict | None = None) -> pd.DataFrame:
    """Kalici (AR(1)) log-vol'lu sembol getirileri; kolonlar loader ile uyumlu:
    symbol, Date, Close, NATR_14, liq_log. `drop`={symbol: [tarih,...]} eksik gun."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_days)
    mkt = rng.normal(0, 0.006, n_days)
    rows = []
    for i in range(n_sym):
        lv = np.empty(n_days)
        lv[0] = np.log(0.02) + 0.3 * rng.normal()
        for t in range(1, n_days):
            lv[t] = np.log(0.02) + 0.97 * (lv[t - 1] - np.log(0.02)) + 0.12 * rng.normal()
        sig = np.exp(lv)
        r = sig * rng.standard_normal(n_days) + 0.6 * mkt
        close = 50.0 * np.exp(np.cumsum(r))
        natr = np.clip(1.4 * sig, 1e-4, None)
        liq = 14.0 + 0.4 * i + 0.01 * rng.normal(size=n_days)
        df = pd.DataFrame({"symbol": f"S{i:02d}", "Date": dates, "Close": close,
                           "NATR_14": natr, "liq_log": liq})
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    for sym, ds in (drop or {}).items():
        panel = panel[~((panel["symbol"] == sym) & panel["Date"].isin(pd.to_datetime(ds)))]
    return panel.reset_index(drop=True)
