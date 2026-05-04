# -*- coding: utf-8 -*-
"""
risk_free_rate.py - Dinamik TCMB faiz orani yardimcisi (Faz 3.6).

Motivasyon:
  Sharpe ve Sortino hesaplamalarinda kullanilan risk-free rate (rf)
  sabit %40 olarak kodlanmistir. TCMB faizi degistikce bu deger yanlislanir.

Cozum:
  - Oncelik 1: macro_pipeline'in cache ettigi INTEREST_RATE.csv dosyasindan
    en son gecerli TCMB faizini oku.
  - Oncelik 2: macro_pipeline'in EVDS'den cektigi son rate_level degerini kullan.
  - Oncelik 3 (fallback): yapilandirma dosyasindan veya environment degiskeninden.
  - Oncelik 4: sabit varsayilan (0.40 = %40).

Kullanim:
    from src.utils.risk_free_rate import get_current_risk_free_rate
    rf = get_current_risk_free_rate(macro_cache_dir="data/macro")
"""

from __future__ import annotations

import os
from typing import Optional


_DEFAULT_RISK_FREE_RATE = 0.40  # Fallback: %40 TCMB referans faizi


def get_current_risk_free_rate(
    macro_cache_dir: str = "data/macro",
    fallback: float = _DEFAULT_RISK_FREE_RATE,
    project_root: Optional[str] = None,
) -> float:
    """
    Guncel TCMB risk-free faiz oranini doner (yillik, ondalik).

    Oncelik sirasi:
      1. macro_cache_dir/INTEREST_RATE.csv — en son satir (decimal olarak)
      2. Environment degiskeni: RISK_FREE_RATE_ANNUAL
      3. fallback parametresi (varsayilan: 0.40)

    Parameters
    ----------
    macro_cache_dir : str
        MacroPipeline'in faiz verisini cacheledigii dizin.
        Proje kokune gore goreli veya mutlak yol.
    fallback : float
        Cache okunamayinca kullanilacak varsayilan deger (ondalik, orn. 0.40).
    project_root : str, optional
        Proje kok dizini. None ise bu dosyanin konumundan otomatik hesaplanir.

    Returns
    -------
    float
        Yillik risk-free faiz orani (0.40 = %40).
    """
    # Cevresel degisken kontrolu
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

    return fallback


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
        # MacroPipeline Rate_Level'i ondalik olarak saklar (ornegin 0.42).
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
