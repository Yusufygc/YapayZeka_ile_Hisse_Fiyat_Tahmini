# -*- coding: utf-8 -*-
"""
macro_forward_projection.py - Recursive forecast icin macro seriler ileri projection.

Sprint 5 (2026-05-25) Plan A5.2:
  Recursive future forecast'in 5+ gun horizon'da macro feature'lerin
  dondurulmus (frozen) olmasi sorununu kapatir. Her macro seri icin basit
  ARIMA(1,1,1) fit + horizon_days kadar forward forecast uretilir. ARIMA
  fail olursa son N gunluk dogrusal trend extrapolation (fallback).

Kullanim:
    projector = MacroForwardProjector()
    new_frame = projector.project_last_row(frame, target_date=ts)

Donus:
    frame'in son satirinda macro sutunlari projection sonuclariyla
    guncellenmis. Diger satirlar dokunulmaz.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# Macro feature whitelist (data/wiki uyumu). Bilinen sutun isimleri.
_KNOWN_MACRO_COLUMNS = {
    "USDTRY", "USDTRY_Return",
    "BIST100", "BIST100_Return",
    "VIX",
    "INTEREST_RATE",
    "CPI",
    "BRENT",
    "GOLD",
    "EURTRY",
    "BIST_BANK_Return", "BIST_HOLD_Return", "BIST_SANAYI_Return",
    "BIST_INSAAT_Return", "BIST_GIDAI_Return", "BIST_KIMYA_Return",
    "BIST_METALU_Return", "BIST_TICRT_Return", "BIST_ULASTRMA_Return",
    "BIST_TURIZM_Return", "BIST_BILMEMURENGEL_Return",
}


_ARIMA_HISTORY_WINDOW = 252  # son ~1 yil veriyle fit
_ARIMA_ORDER = (1, 1, 1)


class MacroForwardProjector:
    """Recursive forecast row'una macro projection uygular.

    `auto_columns=True` ise frame icindeki bilinen macro sutunlari otomatik
    secer; `columns` override ile manuel liste verilebilir.
    """

    def __init__(
        self,
        columns: Optional[Iterable[str]] = None,
        history_window: int = _ARIMA_HISTORY_WINDOW,
        arima_order: tuple[int, int, int] = _ARIMA_ORDER,
    ) -> None:
        self.columns = set(columns) if columns is not None else None
        self.history_window = int(history_window)
        self.arima_order = tuple(int(x) for x in arima_order)

    def _resolve_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in frame.columns]
        # auto: known macro columns intersected with frame.
        return [c for c in frame.columns if c in _KNOWN_MACRO_COLUMNS]

    @staticmethod
    def _fallback_trend(series: pd.Series) -> float:
        """ARIMA fail olursa: son 20 gunun ortalama degisim + son deger."""
        clean = series.dropna()
        if len(clean) < 2:
            return float(clean.iloc[-1]) if len(clean) else 0.0
        last = float(clean.iloc[-1])
        tail = clean.iloc[-min(20, len(clean)):]
        # Ortalama gunluk degisim
        avg_delta = float(tail.diff().dropna().mean()) if len(tail) >= 2 else 0.0
        return last + avg_delta

    def _arima_forecast_one(self, series: pd.Series) -> Optional[float]:
        """Tek macro seri icin 1 step forward forecast. ARIMA fail -> None."""
        clean = series.dropna()
        if len(clean) < 30:
            return None
        sample = clean.iloc[-self.history_window:]
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except Exception:
            return None
        try:
            fitted = ARIMA(sample.values, order=self.arima_order).fit()
            forecast = fitted.forecast(steps=1)
            value = float(np.asarray(forecast).ravel()[-1])
            if not np.isfinite(value):
                return None
            return value
        except Exception:
            return None

    def project_last_row(
        self,
        frame: pd.DataFrame,
        *,
        target_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Son satirin macro feature'lerini bir adim ileri projeksiyona alir."""
        if frame.empty:
            return frame
        out = frame.copy()
        cols = self._resolve_columns(out)
        if not cols:
            return out
        last_idx = out.index[-1]
        # Historic series: son satir HARIC (projection input).
        hist = out.iloc[:-1]
        for col in cols:
            series = hist[col] if col in hist.columns else None
            if series is None or series.dropna().empty:
                continue
            projected = self._arima_forecast_one(series)
            if projected is None:
                projected = self._fallback_trend(series)
            out.at[last_idx, col] = float(projected)
        return out
