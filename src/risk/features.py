# -*- coding: utf-8 -*-
"""Risk (oynaklik) tahmini ozellik matrisi + hedef — urun karari "A".

Per-sembol yon tahmini emekli; yerine TEK havuzlu modelle sembol basina
beklenen 5-gun oynaklik + kalibre %80 fiyat araligi sunulur. Bu modul pooled
paneli (src/data/pooled_loader.py ciktisi) HAR-X ozellik matrisine cevirir.

Girdi panel kolonlari: symbol, Date, Close, NATR_14, liq_log (loader'dan).
Fiyat TAKVIM-pozisyonuna (panelin tum tarihlerinin birlesimi) hizalanir; bir
sembolun eksik gunu (askiya alma, sifir-hacim, karantina) o gunun ve ertesi gunun
getirisini NaN yapar -> cok-gunluk getiri gizlice "gunluk" sayilmaz.

Tanimlar (r_t = log(C_t / C_{t-1}); rv_w = log(sqrt(mean_{son w gun} r^2) + 1e-4)):

  Ozellikler (HEPSI yalniz t ve oncesi — geriye pencere, nedensel):
    rv_1          log(|r_t| + 1e-4)                                  [HAR gunluk]
    rv_5          log RV, son 5 gun                                   [HAR haftalik]
    rv_21         log RV, son 21 gun                                  [HAR aylik; naive persistence]
    rv_63         log RV, son 63 gun                                  [HAR ceyreklik]
    log_natr_14   log(NATR_14 + 1e-4)  (ATR/Close; loader ozelligi)
    mkt_rv_21     rv_21'in o gunku sembol-arasi MEDYANI (piyasa oynaklik rejimi)
    rv_1_down     rv_1 eger r_t < 0 aksi 0  (asagi yonlu sok)
    liq_log       log1p(63g medyan ciro Close*Volume)  (loader; likidite)
    mkt_ret       o gunku esit-agirlikli sembol-ortalamasi log-getiri (piyasa getirisi)

  Hedef / etiket (YALNIZ t+1..t+h kullanir; t ve oncesine bagli DEGIL, bkz. not):
    fwd_rv2       mean(r_{t+1..t+h}^2)               QLIKE gerceklesen varyans vekili
    y             log(sqrt(fwd_rv2) + 1e-4)          model hedefi (log gunluk vol)
    fwd_ret       log(C_{t+h} / C_t)                 kapsama/konformal icin h-gun getirisi
    target_date   takvimde t+h tarihi                purge icin (pooled_cv `target_date`)

  Not: r_{t+1} = log(C_{t+1}/C_t) referans seviye olarak C_t'yi icerir; bu bir
  gelecek-getirisi tanimidir (t fiyatindan sonraki hareket), sizinti degil. y_t,
  t'den ONCEKI hicbir fiyata bagli degildir.

Model girdisi yalniz RISK_FEATURE_COLS; makro/BIST100 kolonu gerekmez (panel
`macro_df=None` ile kurulur; piyasa kolonlari panelden turetilir).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

RISK_HORIZON = 5
EPS = 1e-4                       # log(vol + EPS): sifir-getiri gunlerinde log(0) onler
RV_WINDOWS = (1, 5, 21, 63)
MIN_WINDOW_FRAC = 0.8            # w>5 pencerede gecerli gun orani (eksik gune tolerans)
MIN_MARKET_NAMES = 10            # mkt_* icin o gun en az sembol

RISK_FEATURE_COLS: tuple[str, ...] = (
    "rv_1", "rv_5", "rv_21", "rv_63", "log_natr_14",
    "mkt_rv_21", "rv_1_down", "liq_log", "mkt_ret",
)
NAIVE_COL = "rv_21"              # naive persistence tahmini (y'nin kendi birimiyle)
REQUIRED_PANEL_COLS = ("symbol", "Date", "Close", "NATR_14", "liq_log")
FRAME_COLS = ("symbol", "Date", "target_date", "y", "fwd_ret", "fwd_rv2", *RISK_FEATURE_COLS)


def _min_periods(w: int) -> int:
    return int(w) if w <= 5 else int(math.ceil(MIN_WINDOW_FRAC * w))


def _log_rms(r2: pd.DataFrame, w: int) -> pd.DataFrame:
    """log(sqrt(rolling mean r^2) + EPS); geriye pencere (t dahil, gelecek yok)."""
    ms = r2.rolling(int(w), min_periods=_min_periods(w)).mean()
    return np.log(np.sqrt(ms) + EPS)


def build_risk_frame(
    panel: pd.DataFrame,
    horizon: int = RISK_HORIZON,
    min_market_names: int = MIN_MARKET_NAMES,
) -> pd.DataFrame:
    """Pooled panelden risk ozellik/hedef cercevesi.

    Satirlar panelin (symbol, Date) satirlariyla birebir; hedef kolonlari son
    `horizon` takvim gununde NaN (etiketsiz — skorlama satirlari). Egitim icin
    `labeled_rows`, skorlama icin `scoring_rows` kullanilir.
    """
    missing = [c for c in REQUIRED_PANEL_COLS if c not in panel.columns]
    if missing:
        raise ValueError(f"risk paneli kolon eksik: {missing}")
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon >= 1 olmali")
    df = panel[list(REQUIRED_PANEL_COLS)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if df.duplicated(["symbol", "Date"]).any():
        raise ValueError("panelde tekrar eden (symbol, Date) satiri var")
    cal = pd.DatetimeIndex(np.sort(df["Date"].unique()))

    close = pd.to_numeric(df["Close"], errors="coerce").where(lambda s: s > 0)
    wide = (df.assign(Close=close).pivot(index="Date", columns="symbol", values="Close")
            .reindex(cal))
    logc = np.log(wide)
    r = logc.diff()                  # t-1 veya t eksikse NaN (cok-gunluk getiri yok)
    r2 = r * r

    feats_wide = {f"rv_{w}": _log_rms(r2, w) for w in RV_WINDOWS}
    feats_wide["rv_1_down"] = feats_wide["rv_1"].where(r < 0, 0.0).where(r.notna())
    mkt_rv21 = feats_wide["rv_21"].median(axis=1)
    mkt_ret = r.mean(axis=1).where(r.count(axis=1) >= int(min_market_names))
    mkt_rv21 = mkt_rv21.where(feats_wide["rv_21"].count(axis=1) >= int(min_market_names))

    # hedef: yalniz t+1..t+h (rolling t+h'de biter, h geri kaydirilir)
    fwd_rv2 = r2.rolling(h, min_periods=h).mean().shift(-h)
    fwd_ret = (logc.shift(-h) - logc).where(fwd_rv2.notna())

    ri = cal.get_indexer(df["Date"])
    ci = wide.columns.get_indexer(df["symbol"])

    def take(w: pd.DataFrame) -> np.ndarray:
        return w.to_numpy(dtype=float)[ri, ci]

    out = pd.DataFrame({"symbol": df["symbol"].to_numpy(), "Date": df["Date"].to_numpy()})
    tpos = ri + h
    tdates = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = tpos < len(cal)
    tdates[ok] = cal.to_numpy()[tpos[ok]]
    out["target_date"] = tdates
    f2 = take(fwd_rv2)
    out["y"] = np.log(np.sqrt(f2) + EPS)
    out["fwd_ret"] = take(fwd_ret)
    out["fwd_rv2"] = f2
    for w in RV_WINDOWS:
        out[f"rv_{w}"] = take(feats_wide[f"rv_{w}"])
    natr = pd.to_numeric(df["NATR_14"], errors="coerce").clip(lower=0)
    out["log_natr_14"] = np.log(natr.to_numpy() + EPS)
    out["mkt_rv_21"] = mkt_rv21.to_numpy(dtype=float)[ri]
    out["rv_1_down"] = take(feats_wide["rv_1_down"])
    out["liq_log"] = pd.to_numeric(df["liq_log"], errors="coerce").to_numpy()
    out["mkt_ret"] = mkt_ret.to_numpy(dtype=float)[ri]
    out = out.replace([np.inf, -np.inf], np.nan)
    return (out[list(FRAME_COLS)].sort_values(["Date", "symbol"]).reset_index(drop=True))


def labeled_rows(frame: pd.DataFrame, feature_cols: Sequence[str] = RISK_FEATURE_COLS) -> pd.DataFrame:
    """Egitim/OOS satirlari: ozellikler + hedef + h-gun getirisi hepsi sonlu."""
    need = [*feature_cols, "y", "fwd_ret", "fwd_rv2", "target_date"]
    return frame.dropna(subset=need).reset_index(drop=True)


def scoring_rows(
    frame: pd.DataFrame, as_of: pd.Timestamp,
    feature_cols: Sequence[str] = RISK_FEATURE_COLS,
) -> pd.DataFrame:
    """`as_of` gununun, tum ozellikleri sonlu olan sembol satirlari (hedef gerekmez)."""
    day = frame[frame["Date"] == pd.Timestamp(as_of)]
    return day.dropna(subset=list(feature_cols)).reset_index(drop=True)
