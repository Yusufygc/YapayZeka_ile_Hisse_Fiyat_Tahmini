# -*- coding: utf-8 -*-
"""
risk_free_rate.py - Dinamik TCMB faiz orani yardimcisi.

Sprint 1 (2026-05-25) — Plan v1.0 A1.1:
  Sabit %40 fallback kaldirildi. Macro cache veya environment yoksa
  fonksiyon ``None`` doner. Cagiran katman (financial_metrics,
  backtesting.metrics) ``None`` aldiginda Sharpe/Sortino degerini
  ``NaN`` olarak isaretler ve `risk_free_unavailable` uyarisini
  metric sozlugune ekler. Bu uyari ileride
  ``confidence.warnings`` zincirine bağlanir (Sprint 8'de).

Motivasyon:
  Sharpe ve Sortino hesaplamalarinda kullanilan risk-free rate (rf)
  sabit %40 olarak kodlanmisti. Macro cache yoksa metric sessizce
  yanlis cikiyordu. Advisory sistemi icin bu kabul edilemez:
  rf yoksa kullanici NaN gormeli ve uyari almali.

Cozum:
  - Oncelik 1: macro_pipeline'in cache ettigi INTEREST_RATE.csv'den
    en son gecerli TCMB faizini oku.
  - Oncelik 2: Environment degiskeni RISK_FREE_RATE_ANNUAL.
  - Oncelik 3 (deprecated/legacy): ``fallback`` parametresi (opt-in,
    default None).  Test ortamlarinda explicit 0.0 vermek icin
    saklandi; production kodu ``fallback=None`` cagirmalidir.

Kullanim:
    from src.utils.risk_free_rate import get_current_risk_free_rate
    rf = get_current_risk_free_rate(macro_cache_dir="data/macro")
    if rf is None:
        # Sharpe hesaplanamaz, NaN dondur + warning
        ...
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def get_current_risk_free_rate(
    macro_cache_dir: str = "data/macro",
    fallback: Optional[float] = None,
    project_root: Optional[str] = None,
) -> Optional[float]:
    """
    Guncel TCMB risk-free faiz oranini doner (yillik, ondalik) ya da ``None``.

    Oncelik sirasi:
      1. macro_cache_dir/INTEREST_RATE.csv — en son satir (decimal olarak)
      2. Environment degiskeni: RISK_FREE_RATE_ANNUAL
      3. fallback parametresi (default None — yani fail-loud)

    Sprint 1: Onceki sabit %40 fallback kaldirildi; macro cache yoksa
    fonksiyon None doner ve cagiran kod metric'i NaN olarak isaretler.

    Parameters
    ----------
    macro_cache_dir : str
        MacroPipeline'in faiz verisini cacheledigi dizin.
        Proje kokune gore goreli veya mutlak yol.
    fallback : float, optional
        Cache + environment okunamayinca kullanilacak deger.
        ``None`` (default) ise fonksiyon None doner ve cagiran katman
        bu durumu fail-loud islemelidir.
    project_root : str, optional
        Proje kok dizini. None ise bu dosyanin konumundan otomatik hesaplanir.

    Returns
    -------
    Optional[float]
        Yillik risk-free faiz orani (0.40 = %40) veya ``None`` (veri yok).
    """
    # Environment degiskeni kontrolu
    env_val = os.environ.get("RISK_FREE_RATE_ANNUAL")
    if env_val is not None:
        try:
            rate = float(env_val)
            if 0.0 < rate < 5.0:  # sanity check: %0-%500 arasi makul
                return rate
        except ValueError:
            pass

    # INTEREST_RATE.csv'den oku
    rate_from_cache = _read_rate_from_cache(macro_cache_dir, project_root)
    if rate_from_cache is not None:
        return rate_from_cache

    # Plan v1.0 Sprint 1 A1.1: fail-loud. fallback None ise None doner.
    return fallback


def get_risk_free_rate_series(
    dates: Sequence | pd.Index | pd.Series,
    macro_cache_dir: str = "data/macro",
    project_root: Optional[str] = None,
    lag_days: int = 1,
    annual: bool = False,
) -> pd.Series:
    """Point-in-time risk-free serisi (``dates``'e hizali, gunluk oran).

    ``get_current_risk_free_rate()`` tek (en son) orani tum gecmise uygular;
    bu, gecmis backtest'lerde look-ahead ve yanlis rf demektir. Bu fonksiyon
    INTEREST_RATE.csv'yi makro pipeline ile AYNI yayin kuralina gore hizalar:
    yayin tarihi = ay sonu + ``lag_days`` (default 1, bkz.
    ``src.features.macro_transforms.lag_monthly_features``) ve gunluk takvime
    ``merge_asof(direction="backward")`` ile tasinir.

    Oncelik: RISK_FREE_RATE_ANNUAL ortam degiskeni gecerliyse (get_current ile
    ayni sanity araligi) tum tarihlere sabit olarak uygulanir. Aksi halde CSV.
    Ilk yayin tarihinden onceki (veya cozulemeyen) tarihler NaN kalir;
    cagiran katman fallback'i secer.

    Returns:
        ``dates`` ile ayni uzunluk/sirada, index'i ``dates`` olan pd.Series.
        ``annual=False`` -> gunluk bilesik oran ((1+r)^(1/252)-1);
        ``annual=True`` -> yillik oran (0.40 = %40).
    """
    index = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates), errors="coerce"))
    annual_values = np.full(len(index), np.nan, dtype=float)

    env_rate = _read_rate_from_env()
    if env_rate is not None:
        annual_values[:] = env_rate
    else:
        monthly = _read_rate_frame(macro_cache_dir, project_root)
        if monthly is not None and len(index):
            from src.features.macro_transforms import lag_monthly_features

            released = lag_monthly_features(
                monthly[["Date", "Rate"]],
                lag_days=max(0, int(lag_days)),
                raw_date_column="Rate_Raw_Date",
            )[["Date", "Rate"]]
            released["Date"] = pd.to_datetime(released["Date"]).astype("datetime64[ns]")
            released = released.drop_duplicates("Date", keep="last").sort_values("Date", kind="stable")
            left = pd.DataFrame({
                "Date": index.astype("datetime64[ns]"),
                "_pos": np.arange(len(index)),
            })
            valid = left["Date"].notna()
            merged = pd.merge_asof(
                left[valid].sort_values("Date", kind="stable"),
                released,
                on="Date",
                direction="backward",
            )
            annual_values[merged["_pos"].to_numpy()] = merged["Rate"].to_numpy(dtype=float)

    values = annual_values if annual else np.where(
        np.isfinite(annual_values), (1.0 + annual_values) ** (1.0 / 252.0) - 1.0, np.nan
    )
    return pd.Series(values, index=index, dtype=float)


def _read_rate_from_env() -> Optional[float]:
    """RISK_FREE_RATE_ANNUAL ortam degiskeni (gecerliyse) ya da None."""
    env_val = os.environ.get("RISK_FREE_RATE_ANNUAL")
    if env_val is None:
        return None
    try:
        rate = float(env_val)
    except ValueError:
        return None
    return rate if 0.0 < rate < 5.0 else None


def _read_rate_frame(
    macro_cache_dir: str,
    project_root: Optional[str],
) -> Optional[pd.DataFrame]:
    """INTEREST_RATE.csv -> DataFrame[Date, Rate(ondalik)] ya da None."""
    cache_path = macro_cache_dir
    if not os.path.isabs(cache_path):
        root = project_root or _infer_project_root()
        cache_path = os.path.join(root, macro_cache_dir)
    rate_file = os.path.join(cache_path, "INTEREST_RATE.csv")
    if not os.path.exists(rate_file):
        return None
    try:
        df = pd.read_csv(rate_file, parse_dates=["Date"])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        import logging

        logging.getLogger(__name__).warning("INTEREST_RATE.csv okunamadi: %s", exc)
        return None
    col = next((c for c in df.columns if c != "Date"), None)
    if df.empty or col is None:
        return None
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"]),
        "Rate": pd.to_numeric(df[col], errors="coerce"),
    })
    out["Rate"] = np.where(out["Rate"] > 1.0, out["Rate"] / 100.0, out["Rate"])
    out = out[(out["Rate"] > 0.0) & (out["Rate"] < 5.0)]
    return out.sort_values("Date").reset_index(drop=True) if not out.empty else None


def _read_rate_from_cache(
    macro_cache_dir: str,
    project_root: Optional[str],
) -> Optional[float]:
    """
    INTEREST_RATE.csv'den en son aylık faiz degerini okur.

    Dosya formati (MacroPipeline tarafindan olusturulur):
      Date,INTEREST_RATE
      2024-01-01,0.4250
      2024-02-01,0.4500
      ...

    Deger zaten ondalik (0.45 = %45) olarak saklanir.
    """
    try:
        import pandas as pd

        # Yolu coz
        cache_path = macro_cache_dir
        if not os.path.isabs(cache_path):
            root = project_root or _infer_project_root()
            cache_path = os.path.join(root, macro_cache_dir)

        rate_file = os.path.join(cache_path, "INTEREST_RATE.csv")
        if not os.path.exists(rate_file):
            return None

        df = pd.read_csv(rate_file, parse_dates=["Date"])
        if df.empty:
            return None

        # Son satiri al
        last_row = df.sort_values("Date").iloc[-1]
        col = next((c for c in df.columns if c != "Date"), None)
        if col is None:
            return None

        raw_value = float(last_row[col])

        # Deger % birimiyle mi yoksa ondalik mi?
        # Cache ondalik (ornegin 0.42) ya da yuzde (ornegin 42.0) olabilir.
        # Eger > 1 ise yuzdelik olarak yorumla.
        if raw_value > 1.0:
            raw_value = raw_value / 100.0

        if 0.0 < raw_value < 5.0:  # sanity check
            return round(raw_value, 4)
        return None

    except Exception:
        return None


def _infer_project_root() -> str:
    """Bu dosyanin konumundan proje kokunu cikar (src/utils/risk_free_rate.py)."""
    this_file = os.path.abspath(__file__)
    # src/utils/risk_free_rate.py -> ../../ (proje koku)
    return os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
