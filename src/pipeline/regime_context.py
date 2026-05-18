# -*- coding: utf-8 -*-
"""Piyasa rejimi ve trend bağlamı hesabı (Adim 2.3).

compute_regime_context(): BIST100 endeks DataFrame'i ve hisse fiyat serisi
kullanarak piyasa rejimini ve hissenin göreli gücünü hesaplar.

Çıktılar:
  - market_regime: 'bull' | 'bear' | 'sideways' | 'uncertain'
  - relative_strength: 'outperforming' | 'inline' | 'underperforming'
  - alignment_with_forecast: 'aligned' | 'misaligned' | 'neutral'
  - regime_misalignment: bool (trend yönü ve tahmin yönü zıt ise True)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


_SMA_SHORT = 50
_SMA_LONG = 200
_SLOPE_LOOKBACK = 20
_RS_LOOKBACK = 60
_RS_OUTPERFORM_THRESHOLD = 0.02
_RS_UNDERPERFORM_THRESHOLD = -0.02


def _sma(prices: np.ndarray, window: int) -> Optional[float]:
    if len(prices) < window:
        return None
    return float(np.mean(prices[-window:]))


def _slope(prices: np.ndarray, window: int) -> Optional[float]:
    if len(prices) < window:
        return None
    y = prices[-window:]
    x = np.arange(window, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope / (float(np.mean(y)) + 1e-9)


def compute_market_regime(index_close: np.ndarray) -> str:
    """BIST100 fiyat serisinden piyasa rejimini hesapla.

    Returns
    -------
    'bull' | 'bear' | 'sideways' | 'uncertain'
    """
    if index_close is None or len(index_close) < _SMA_SHORT:
        return "uncertain"

    sma_short = _sma(index_close, _SMA_SHORT)
    sma_long = _sma(index_close, _SMA_LONG)
    slope = _slope(index_close, _SLOPE_LOOKBACK)

    current = float(index_close[-1])

    if sma_long is None:
        if sma_short is None:
            return "uncertain"
        above_sma50 = current > sma_short
        if above_sma50 and (slope is not None and slope > 0):
            return "bull"
        if not above_sma50 and (slope is not None and slope < 0):
            return "bear"
        return "uncertain"

    above_sma50 = current > sma_short
    above_sma200 = current > sma_long
    sma_cross_bull = sma_short > sma_long

    if above_sma50 and above_sma200 and sma_cross_bull and (slope is not None and slope > 0):
        return "bull"
    if not above_sma50 and not above_sma200 and not sma_cross_bull and (slope is not None and slope < 0):
        return "bear"
    if slope is not None and abs(slope) < 0.001:
        return "sideways"
    return "uncertain"


def compute_relative_strength(
    stock_close: np.ndarray,
    index_close: np.ndarray,
    lookback: int = _RS_LOOKBACK,
) -> str:
    """Hisse ile BIST100 arasındaki göreli gücü hesapla.

    Returns
    -------
    'outperforming' | 'inline' | 'underperforming'
    """
    if stock_close is None or index_close is None:
        return "inline"
    n_stock = len(stock_close)
    n_index = len(index_close)
    k = min(lookback, n_stock, n_index)
    if k < 5:
        return "inline"

    stock_ret = float(stock_close[-1]) / float(stock_close[-k]) - 1.0
    index_ret = float(index_close[-1]) / float(index_close[-k]) - 1.0
    diff = stock_ret - index_ret

    if diff > _RS_OUTPERFORM_THRESHOLD:
        return "outperforming"
    if diff < _RS_UNDERPERFORM_THRESHOLD:
        return "underperforming"
    return "inline"


def compute_regime_context(
    stock_close: np.ndarray,
    index_close: Optional[np.ndarray] = None,
    forecast_direction: Optional[float] = None,
) -> Dict[str, Any]:
    """Tam rejim bağlamı payload'u üret.

    Parameters
    ----------
    stock_close:
        Hisse kapanış fiyatları serisi (kronolojik).
    index_close:
        BIST100 endeks kapanış fiyatları (opsiyonel).
    forecast_direction:
        En son model tahmininin yönü: +1 yukarı, -1 aşağı, 0 belirsiz.

    Returns
    -------
    dict:
        market_regime, relative_strength, alignment_with_forecast,
        regime_misalignment (bool)
    """
    market_regime = "uncertain"
    if index_close is not None and len(index_close) >= _SMA_SHORT:
        market_regime = compute_market_regime(index_close)

    relative_strength = "inline"
    if index_close is not None and len(index_close) >= 5 and stock_close is not None and len(stock_close) >= 5:
        relative_strength = compute_relative_strength(stock_close, index_close)

    alignment = "neutral"
    regime_misalignment = False
    if forecast_direction is not None and forecast_direction != 0:
        forecast_up = forecast_direction > 0
        if market_regime == "bull" and forecast_up:
            alignment = "aligned"
        elif market_regime == "bear" and not forecast_up:
            alignment = "aligned"
        elif market_regime == "bull" and not forecast_up:
            alignment = "misaligned"
            regime_misalignment = True
        elif market_regime == "bear" and forecast_up:
            alignment = "misaligned"
            regime_misalignment = True

    return {
        "market_regime": market_regime,
        "relative_strength": relative_strength,
        "alignment_with_forecast": alignment,
        "regime_misalignment": regime_misalignment,
    }
