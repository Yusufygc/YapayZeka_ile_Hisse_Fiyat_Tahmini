# -*- coding: utf-8 -*-
"""Pooled (panel) loader for the E2 global model.

Tasarim: docs/wiki/e2-faz2-pooled-cv-design.md.

Tum hisse CSV'lerini tek uzun panele yukler:
    symbol, Date, <FeaturePipeline ozellikleri>, target, target_date,
    sector, symbol_id, liq_log, vol

- Ozellikler her sembol icin AYRI uretilir (capraz-sembol kontaminasyon yok),
  sonra birlestirilir.
- target = log(close[t+h]/close[t]); target_date = sembolun t+h tarihi (kesin
  purge icin -> pooled_cv). Son h satir (NaN target) dusulur.
- Kosullandirma: sector (universe, GICS), symbol_id (stabil), liq_log (trailing
  TL ciro medyani, causal), vol (trailing realized-vol, causal). Bucketleme/
  olcekleme CV/model asamasina birakilir (loader'da global quantile = lookahead).
- Survivorship: delisted sembollerin gecmisi DAHIL (include_delisted).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.feature_pipeline import FeaturePipeline  # tests monkeypatch eder

_COL_MAP = {
    "Tarih": "Date", "Açılış": "Open", "Yüksek": "High",
    "Düşük": "Low", "Düzeltilmiş_Kapanış": "Close", "Hacim": "Volume",
}
_EXCLUDE_STEMS = {"bist_universe", "advisory_history"}
_NON_FEATURE = {"Date", "Close", "Open", "High", "Low", "Volume",
                "Adj Close", "Düzeltilmiş_Kapanış", "Hacim"}


@dataclass(frozen=True)
class PooledLoaderConfig:
    data_dir: str
    universe_file: str
    target_horizon: int = 5
    feature_mode: str = "stationary_features"
    min_rows: int = 60            # bundan az ham satirli sembol atlanir
    include_delisted: bool = True
    liq_lookback: int = 63
    vol_lookback: int = 63
    min_usable_rows: int = 10     # ozellik+target sonrasi en az kullanilabilir satir


class PooledPanelLoader:
    def __init__(self, cfg: PooledLoaderConfig) -> None:
        self.cfg = cfg
        self.report: dict[str, str] = {}  # symbol -> skip reason / "ok"

    # ----------------------------------------------------------------- public
    def load(self) -> pd.DataFrame:
        sector_map = self._sector_map()
        frames: list[pd.DataFrame] = []
        for sym in self._symbols():
            try:
                f = self._engineer_symbol(sym, sector_map.get(sym, "Unknown"))
            except Exception as exc:  # tek sembol hatasi paneli durdurmasin
                self.report[sym] = f"error:{str(exc)[:60]}"
                continue
            if f is None or f.empty:
                continue
            frames.append(f)
            self.report[sym] = "ok"
        if not frames:
            return pd.DataFrame()
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["Date", "symbol"]).reset_index(drop=True)
        codes = {s: i for i, s in enumerate(sorted(panel["symbol"].unique()))}
        panel["symbol_id"] = panel["symbol"].map(codes).astype(int)
        return panel

    # ---------------------------------------------------------------- helpers
    def _symbols(self) -> list[str]:
        out = []
        for p in glob.glob(os.path.join(self.cfg.data_dir, "*.csv")):
            stem = os.path.basename(p)[:-4]
            if stem not in _EXCLUDE_STEMS:
                out.append(stem)
        return sorted(out)

    def _sector_map(self) -> dict[str, str]:
        if not os.path.exists(self.cfg.universe_file):
            return {}
        uni = pd.read_csv(self.cfg.universe_file, encoding="utf-8-sig")
        if "Symbol" not in uni.columns or "Sector" not in uni.columns:
            return {}
        out = {}
        for _, r in uni.iterrows():
            val = r.get("Sector")
            sec = "" if pd.isna(val) else str(val).strip()
            out[str(r["Symbol"]).strip().upper()] = sec or "Unknown"
        return out

    def _engineer_symbol(self, symbol: str, sector: str) -> pd.DataFrame | None:
        path = os.path.join(self.cfg.data_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            self.report[symbol] = "no_csv"
            return None
        raw = pd.read_csv(path, encoding="utf-8-sig").rename(columns=_COL_MAP)
        if "Date" not in raw.columns or "Close" not in raw.columns:
            self.report[symbol] = "bad_schema"
            return None
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
        if len(raw) < self.cfg.min_rows:
            self.report[symbol] = f"too_short:{len(raw)}"
            return None

        # Yalniz standart OHLCV'yi FeaturePipeline'a ver; eslenmemis ham kolonlar
        # (or. "Kapanış" = ham fiyat seviyesi) ozellik olarak sizmasin.
        std_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"]
                    if c in raw.columns]
        raw = raw[std_cols]

        # Ozellikler (per-symbol). Tests modul-seviyesi FeaturePipeline'i patch'ler.
        fp = FeaturePipeline(feature_mode=self.cfg.feature_mode,
                             enable_calendar_features=False)
        feat = fp.engineer_features(raw, macro_df=None, symbol=symbol)
        if feat is None or feat.empty or "Date" not in feat.columns:
            self.report[symbol] = "no_features"
            return None

        # Causal kosullandirma + target ham seriden (Date indexli).
        cond = self._causal_columns(raw)

        feat_cols = [c for c in feat.columns if c not in _NON_FEATURE]
        feat_clean = feat.drop(columns=["Close"], errors="ignore")
        merged = feat_clean.merge(cond, on="Date", how="inner")
        merged = merged.dropna(subset=feat_cols + ["target"]).reset_index(drop=True)
        if len(merged) < self.cfg.min_usable_rows:
            self.report[symbol] = f"few_usable:{len(merged)}"
            return None

        merged["symbol"] = symbol
        merged["sector"] = sector
        keep = ["symbol", "Date", "Close"] + feat_cols + [
            "target", "target_date", "sector", "liq_log", "vol"
        ]
        return merged[keep]

    def _causal_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        h = max(1, int(self.cfg.target_horizon))
        close = pd.to_numeric(raw["Close"], errors="coerce")
        vol_sh = pd.to_numeric(raw.get("Volume", pd.Series(np.nan, index=raw.index)), errors="coerce")
        # target: y[t] = log(close[t+h]/close[t]); target_date = t+h tarihi
        target = np.log(close.shift(-h) / close)
        target_date = raw["Date"].shift(-h)
        # liq_log: trailing TL ciro medyani (causal); vol: trailing realized-vol
        turnover = (close * vol_sh).clip(lower=0)
        liq_log = np.log1p(turnover.rolling(self.cfg.liq_lookback, min_periods=5).median())
        logret = np.log(close).diff()
        vol = logret.rolling(self.cfg.vol_lookback, min_periods=5).std()
        return pd.DataFrame({
            "Date": raw["Date"],
            "Close": close.values,
            "target": target.values,
            "target_date": target_date.values,
            "liq_log": liq_log.values,
            "vol": vol.values,
        })
