# -*- coding: utf-8 -*-
"""
scoring.py - Composite skor ve Diebold-Mariano anlamlilik testi.

Composite_Score YALNIZCA gosterim + es-puan kirici (tie-break) icindir. Model
secimi once benchmark'a karsi anlamlilik (Diebold-Mariano) ve RMSE_vs_benchmark
<= 1 kapisindan gecmek zorundadir (bkz. metrics_reporter._select_best_model).

Formul (toplam agirlik = 1.0)::

    rmse_skill      = 50 + clip((1 - RMSE_vs_benchmark) * 1000, -50, 50)
                      (%1 RMSE iyilesmesi = +10 puan; 1.0 civarinda surekli;
                       oran <= 0.95 -> 100, oran >= 1.05 -> 0)
    direction_skill = 50 + clip((Dir_Acc - 50) * 5, -50, 50)
                      (yazi-tura = %50'ye karsi; benchmark'a gore DEGIL)
    sharpe_skill    = 50 * (tanh(Sharpe_excess_vs_buy_hold / 1.5) + 1)
                      (NaN -> 50 notr)

    composite = 0.5 * rmse_skill + 0.3 * direction_skill + 0.2 * sharpe_skill
                - min(15, 0.15 * Neutral_Rate)

    Kapaklar: RMSE_vs_benchmark > 1 -> <= 49 ; Eligible_For_Leader False -> <= 49 ;
              RMSE_vs_benchmark NaN/eksik -> <= 49 (benchmark yok => lider olamaz).

Hit_Rate ve DirAcc_vs_benchmark formulden CIKARILDI: ikisi de Dir_Acc ile ayni
sinyalin tekrari (eski formulde ayni sinyal %60 agirlikla sayiliyordu) ve
benchmark Naive Zero Return iken DirAcc_vs_benchmark ~ Dir_Acc sabit bonusa
doyuyordu. Alanlar bilgi amacli satirlarda kalir.

Eksik/NaN Dir_Acc 0 sayilmaz; nötr (50) alinir ve ``Composite_Incomplete=True``
isaretlenir. ``compute_composite_score`` yalniz float doner; bayrak icin
``compute_composite_details`` kullanilir.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

import numpy as np

DM_ALPHA = 0.10  # secim icin tek-tarafli anlamlilik esigi

W_RMSE = 0.5
W_DIRECTION = 0.3
W_SHARPE = 0.2
CAP_SCORE = 49.0


def _finite_float(value: Any) -> float:
    """Sayiya cevir; cevrilemiyor/NaN/inf ise NaN dondur."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def compute_composite_details(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Composite skoru ve eksik-veri bayragini dondurur."""
    rmse_ratio = _finite_float(metrics.get("RMSE_vs_benchmark"))
    dir_acc = _finite_float(metrics.get("Dir_Acc"))
    sharpe_excess = _finite_float(metrics.get("Sharpe_excess_vs_buy_hold"))
    neutral_rate = _finite_float(metrics.get("Neutral_Rate"))
    if math.isnan(neutral_rate):
        neutral_rate = 0.0

    incomplete = False

    if math.isnan(rmse_ratio):
        rmse_skill = 50.0
        incomplete = True
    else:
        rmse_skill = 50.0 + min(50.0, max(-50.0, (1.0 - rmse_ratio) * 1000.0))

    if math.isnan(dir_acc):
        direction_skill = 50.0
        incomplete = True
    else:
        direction_skill = 50.0 + min(50.0, max(-50.0, (dir_acc - 50.0) * 5.0))

    if math.isnan(sharpe_excess):
        sharpe_skill = 50.0  # rf yoksa rutin durum; eksik sayilmaz
    else:
        sharpe_skill = (math.tanh(sharpe_excess / 1.5) + 1.0) * 50.0

    composite = W_RMSE * rmse_skill + W_DIRECTION * direction_skill + W_SHARPE * sharpe_skill
    composite -= min(15.0, neutral_rate * 0.15)

    if math.isnan(rmse_ratio) or rmse_ratio > 1.0:
        composite = min(composite, CAP_SCORE)
    if metrics.get("Eligible_For_Leader") is False:
        composite = min(composite, CAP_SCORE)

    return {
        "Composite_Score": round(max(0.0, composite), 4),
        "Composite_Incomplete": incomplete,
    }


def compute_composite_score(metrics: Mapping[str, Any]) -> float:
    """Composite skor (0-100). Formul icin modul docstring'ine bakin."""
    return float(compute_composite_details(metrics)["Composite_Score"])


def diebold_mariano(
    e_model: Any,
    e_bench: Any,
    h: int = 1,
    lag: int | None = None,
) -> Tuple[float, float]:
    """Diebold-Mariano testi (kare hata kaybi, HLN duzeltmesi), tek tarafli.

    H0: model ve benchmark esit dogruluk; H1: model daha iyi.
    ``d_t = e_bench_t^2 - e_model_t^2`` (pozitif = model daha iyi). Uzun donem
    varyans Newey-West (Bartlett agirlikli) ile ``lag`` (varsayilan ``h-1``)
    gecikmesine kadar; HLN kucuk orneklem duzeltmesi ve t(T-1) referansi.

    Returns:
        (stat, p_value_one_sided). Veri yetersizse (n < 3) ``(nan, nan)``;
        iki hata serisi ozdesse ``(0.0, 1.0)``.
    """
    em = np.asarray(e_model, dtype=float).ravel()
    eb = np.asarray(e_bench, dtype=float).ravel()
    n = min(em.size, eb.size)
    em, eb = em[:n], eb[:n]
    mask = np.isfinite(em) & np.isfinite(eb)
    em, eb = em[mask], eb[mask]
    t_len = int(em.size)
    h = max(1, int(h))
    if t_len < 3:
        return float("nan"), float("nan")

    d = eb ** 2 - em ** 2
    d_bar = float(np.mean(d))
    centered = d - d_bar
    max_lag = (h - 1) if lag is None else max(0, int(lag))
    max_lag = min(max_lag, t_len - 1)

    lrv = float(np.dot(centered, centered) / t_len)
    for k in range(1, max_lag + 1):
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / t_len)
        lrv += 2.0 * (1.0 - k / (max_lag + 1.0)) * gamma_k

    scale = float(np.mean(eb ** 2) + np.mean(em ** 2))
    if lrv <= 1e-18 * max(scale, 1e-300) ** 2 or lrv <= 0.0:
        if abs(d_bar) <= 1e-12 * max(scale, 1e-300):
            return 0.0, 1.0
        return (math.inf, 0.0) if d_bar > 0 else (-math.inf, 1.0)

    hln_arg = (t_len + 1 - 2 * h + h * (h - 1) / t_len) / t_len
    if hln_arg <= 0.0:
        return float("nan"), float("nan")
    stat = d_bar / math.sqrt(lrv / t_len) * math.sqrt(hln_arg)

    from scipy import stats as _stats

    p_value = float(_stats.t.sf(stat, df=t_len - 1))
    return float(stat), p_value
