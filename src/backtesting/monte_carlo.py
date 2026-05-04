# -*- coding: utf-8 -*-
"""
monte_carlo.py - Backtest bootstrap dogrulama (Faz 4.4).

Motivasyon:
  Tek bir backtest sonucu istatistiksel olarak zayiftir.
  Ayni sinyal stratejisi rastgele giriş zamanlamasiyla da calisir mi?
  bootstrap_backtest() sinyal vektorunu N kez shuffle ederek:
    - Sharpe dagiliimi uretir
    - p-value hesaplar (gercek Sharpe'in random baseline'i ne siklikla gecip gecmedigi)
    - %5 / %95 guven araliklerini verir

Kullanim:
    from src.backtesting.monte_carlo import bootstrap_backtest
    result = bootstrap_backtest(signals, returns, n_simulations=1000)
    print(result["p_value"], result["sharpe_percentile"])
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def bootstrap_backtest(
    signals: np.ndarray,
    returns: np.ndarray,
    n_simulations: int = 1000,
    seed: int = 42,
    risk_free_annual: float | None = None,
    commission_daily_equiv: float = 0.0,
) -> Dict[str, Any]:
    """
    Sinyal vektorunu N kez shuffle ederek beklenen getiri dagilimini uretir.

    Parameters
    ----------
    signals : np.ndarray of int/float
        Her barda sinyal degeri (1 = long, 0 = flat, -1 = short).
        Sekil: (T,)
    returns : np.ndarray of float
        Gercek bar getirileri (log return veya basit return).
        Sekil: (T,)
    n_simulations : int
        Monte Carlo simulasyon sayisi (varsayilan: 1000).
    seed : int
        Tekrarlanabilirlik icin rastgele tohum.
    risk_free_annual : float or None
        Yillik risksiz faiz orani. None ise macro cache'ten okunur (fallback 0.40).
    commission_daily_equiv : float
        Islem basina maliyet (getiri birimi cinsinden, opsiyonel).

    Returns
    -------
    dict with keys:
        real_sharpe          : float  — gercek sinyalden hesaplanan Sharpe
        real_net_return      : float  — gercek sinyalden hesaplanan net getiri
        sim_sharpe_mean      : float  — simule edilmis Sharpe ortalamasi
        sim_sharpe_std       : float  — simule edilmis Sharpe standart sapmasi
        sim_sharpe_p5        : float  — %5 persentil
        sim_sharpe_p95       : float  — %95 persentil
        p_value              : float  — gercek Sharpe'in random baseline'i astigi oran (1-sided)
        sharpe_percentile    : float  — gercek Sharpe'in simule edilmis dagilimdaki persentili
        significant_at_05    : bool   — p_value < 0.05 mi?
        n_simulations        : int    — kullanilan simulasyon sayisi
        sim_net_return_mean  : float  — simule edilmis ortalama net getiri
    """
    if risk_free_annual is None:
        try:
            from src.utils.risk_free_rate import get_current_risk_free_rate
            risk_free_annual = get_current_risk_free_rate()
        except ImportError:
            risk_free_annual = 0.40

    signals = np.asarray(signals, dtype=float).ravel()
    returns = np.asarray(returns, dtype=float).ravel()
    n = min(len(signals), len(returns))
    signals = signals[:n]
    returns = returns[:n]

    real_sharpe, real_net_return = _compute_strategy_stats(
        signals, returns, risk_free_annual, commission_daily_equiv
    )

    rng = np.random.default_rng(seed)
    sim_sharpes = np.empty(n_simulations, dtype=float)
    sim_net_returns = np.empty(n_simulations, dtype=float)

    for i in range(n_simulations):
        shuffled = rng.permutation(signals)
        sim_sharpes[i], sim_net_returns[i] = _compute_strategy_stats(
            shuffled, returns, risk_free_annual, commission_daily_equiv
        )

    # p-value: kac simulasyon gercek Sharpe'tan buyuk?
    p_value = float(np.mean(sim_sharpes >= real_sharpe))
    sharpe_percentile = float(np.mean(sim_sharpes < real_sharpe) * 100.0)

    return {
        "real_sharpe": round(real_sharpe, 6),
        "real_net_return": round(real_net_return, 6),
        "sim_sharpe_mean": round(float(np.mean(sim_sharpes)), 6),
        "sim_sharpe_std": round(float(np.std(sim_sharpes)), 6),
        "sim_sharpe_p5": round(float(np.percentile(sim_sharpes, 5)), 6),
        "sim_sharpe_p95": round(float(np.percentile(sim_sharpes, 95)), 6),
        "p_value": round(p_value, 6),
        "sharpe_percentile": round(sharpe_percentile, 2),
        "significant_at_05": bool(p_value < 0.05),
        "n_simulations": n_simulations,
        "sim_net_return_mean": round(float(np.mean(sim_net_returns)), 6),
    }


def _compute_strategy_stats(
    signals: np.ndarray,
    returns: np.ndarray,
    risk_free_annual: float,
    commission_daily_equiv: float,
) -> tuple[float, float]:
    """Sinyal + getiri dizisinden Sharpe ve net getiri hesapla."""
    strategy_returns = signals * returns
    if commission_daily_equiv > 0.0:
        # Islem maliyeti: pozisyon degisimlerinde uygula
        trade_events = np.abs(np.diff(np.concatenate(([0.0], signals)))) > 0
        strategy_returns = strategy_returns - (trade_events.astype(float) * commission_daily_equiv)

    net_return = float(np.sum(strategy_returns))
    daily_rf = float((1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0)
    excess = strategy_returns - daily_rf
    std = float(np.std(excess))
    sharpe = float(np.mean(excess) * np.sqrt(252) / std) if std > 1e-10 else 0.0
    return sharpe, net_return
