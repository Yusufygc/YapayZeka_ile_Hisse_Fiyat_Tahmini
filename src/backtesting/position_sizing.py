# -*- coding: utf-8 -*-
"""
position_sizing.py - Kelly Criterion pozisyon buyuklugu hesaplayici (Faz 4.5).

Motivasyon:
  Mevcut backtest: her sinyal icin sabit pozisyon (fully in / flat).
  Gercekci alternif: Kelly Criterion ile model guvene gore olcekleme.

  Kelly Formulü:  f* = (p * b - q) / b
    p = kazanma olasiligi (directional accuracy)
    q = 1 - p
    b = ortalama kazanc / ortalama kayip (odds ratio)

  Half-Kelly (f* / 2) kullanilir: tam Kelly psikolojik olarak cok agresif.
  max_fraction ile ust limit uygulanir (orn. 0.25 = max %25 portfoy).

Kullanim:
    from src.backtesting.position_sizing import kelly_fraction, kelly_position_sizes

    frac = kelly_fraction(win_prob=0.55, avg_win=0.012, avg_loss=0.008)
    # -> 0.1875 (portfoyin %18.75'i)

    sizes = kelly_position_sizes(signals, trade_wins, avg_win, avg_loss)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def kelly_fraction(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    max_fraction: float = 0.25,
    use_half_kelly: bool = True,
) -> float:
    """
    Kelly Criterion ile risk-adjusted pozisyon buyuklugu hesapla.

    Parameters
    ----------
    win_prob : float
        Kazanma olasiligi (modelin directional accuracy / 100).
        Ornek: directional_accuracy=55.0 ise win_prob=0.55
    avg_win : float
        Kazanan islemlerin ortalama getirisi (pozitif, orn. 0.012 = %1.2).
    avg_loss : float
        Kaybeden islemlerin ortalama getirisi (pozitif mutlak deger, orn. 0.008 = %0.8).
    max_fraction : float
        Maksimum portfoy yuzde limiti (orn. 0.25 = max %25). Guvenlik siniri.
    use_half_kelly : bool
        True ise half-Kelly (f* / 2) kullanilir — onerilen standart.

    Returns
    -------
    float
        Portfoyin kac yuzdesi bu pozisyona alinmali (0.0 ile max_fraction arasi).

    Notlar
    ------
    - Sonuc negatifse sifir doner (short pozisyonu simule etmez).
    - avg_loss=0 durumunda sifir doner (hesaplama guvensiz).
    - b (odds ratio): avg_win / avg_loss — kazancin kayba orani.
    """
    if avg_loss <= 0 or win_prob <= 0:
        return 0.0

    loss_prob = 1.0 - win_prob
    b = avg_win / avg_loss  # odds ratio

    # Kelly: f* = (p*b - q) / b  <==>  f* = p - q/b
    kelly = (win_prob * b - loss_prob) / b

    if kelly <= 0:
        return 0.0

    if use_half_kelly:
        kelly *= 0.5

    return float(min(kelly, max_fraction))


def kelly_position_sizes(
    signals: np.ndarray,
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    max_fraction: float = 0.25,
    use_half_kelly: bool = True,
) -> np.ndarray:
    """
    Sinyal dizisi icin Kelly-scaled pozisyon buyuklukleri uret.

    Parameters
    ----------
    signals : np.ndarray
        Binary sinyal dizisi (1 = long, 0 = flat).
    win_prob, avg_win, avg_loss, max_fraction, use_half_kelly
        kelly_fraction() ile ayni parametreler.

    Returns
    -------
    np.ndarray
        Her bar icin pozisyon buyuklugu (0.0 ile max_fraction arasi float).
        Orijinal sinyal=0 olan barlar her zaman 0.0 doner.
    """
    frac = kelly_fraction(win_prob, avg_win, avg_loss, max_fraction, use_half_kelly)
    signals = np.asarray(signals, dtype=float)
    return np.where(signals > 0, frac, 0.0)


def kelly_from_backtest_metrics(
    backtest_summary: dict,
    max_fraction: float = 0.25,
    use_half_kelly: bool = True,
) -> dict:
    """
    summarize_backtest() ciktisından dogrudan Kelly Criterion hesapla.

    Parameters
    ----------
    backtest_summary : dict
        summarize_backtest() tarafindan donen sozluk.
        Kullanilan anahtarlar: Win_Rate, Avg_Win, Avg_Loss.
    max_fraction : float
        Guvenlik limiti.
    use_half_kelly : bool
        True ise half-Kelly.

    Returns
    -------
    dict with keys:
        kelly_fraction          : float — hesaplanan pozisyon buyuklugu
        win_prob                : float — giris degerinden
        avg_win                 : float — giris degerinden
        avg_loss                : float — giris degerinden (mutlak)
        b_odds_ratio            : float — avg_win / avg_loss
        full_kelly              : float — half-Kelly kullanilmadigi durumda
        recommended_fraction    : float — kelly_fraction ile ayni
        notes                   : str
    """
    win_rate = float(backtest_summary.get("Win_Rate", 0.0))
    avg_win = float(backtest_summary.get("Avg_Win", 0.0))
    avg_loss = abs(float(backtest_summary.get("Avg_Loss", 0.0)))

    win_prob = win_rate / 100.0 if win_rate > 1.0 else win_rate

    frac = kelly_fraction(win_prob, avg_win, avg_loss, max_fraction, use_half_kelly)
    full_kelly = kelly_fraction(win_prob, avg_win, avg_loss, max_fraction=1.0, use_half_kelly=False)
    b = avg_win / avg_loss if avg_loss > 0 else 0.0

    notes = []
    if frac == 0.0:
        notes.append("Kelly negatif veya sifir: bu parametrelerle islem yapilmaz.")
    if frac >= max_fraction:
        notes.append(f"Kelly limiti {max_fraction:.0%} ile sinirlandirildi.")
    if use_half_kelly:
        notes.append("Half-Kelly kullanildi (volatility ve model belirsizligi icin onerilen).")

    return {
        "kelly_fraction": round(frac, 6),
        "win_prob": round(win_prob, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "b_odds_ratio": round(b, 4),
        "full_kelly": round(full_kelly, 6),
        "recommended_fraction": round(frac, 6),
        "notes": " | ".join(notes) if notes else "Kelly hesaplamaları geçerli.",
    }
