# -*- coding: utf-8 -*-
"""
Sprint 1 (2026-05-25) Plan A1.2 + A1.4 testleri.

A1.2: METRICS_REPORT_COLUMNS sirasi advisory-oriented:
  - Dir_Acc/Hit_Rate Composite_Score'dan once gelir
  - Net_Return / BuyHold_Return en dipte (yan referans olarak)
  - Sharpe-tipi metric'ler ortada, RMSE_vs_benchmark Composite'in hemen yaninda

A1.4: compute_composite_score formulu Dir_Acc + Hit_Rate'i one cikarir.
  - Yuksek Dir_Acc + Hit_Rate -> daha yuksek composite
  - Sharpe NaN ise crash etmez, notr puan alir
"""

from __future__ import annotations

import math
import pytest

from src.evaluation.evaluator import METRICS_REPORT_COLUMNS
from src.database.stock_model_db import compute_composite_score


def test_dir_acc_appears_before_composite_score():
    """A1.2: Dir_Acc, advisory icin one cikar -> Composite_Score'dan once."""
    cols = METRICS_REPORT_COLUMNS
    assert cols.index("Dir_Acc") < cols.index("Composite_Score")
    assert cols.index("Hit_Rate") < cols.index("Composite_Score")


def test_buy_hold_return_after_composite():
    """A1.2: BuyHold_Return Composite_Score'dan SONRA (dipnot)."""
    cols = METRICS_REPORT_COLUMNS
    # BuyHold_Return + Net_Return dipnota tasindi.
    assert "BuyHold_Return" in cols
    assert "Net_Return" in cols
    assert cols.index("BuyHold_Return") > cols.index("Composite_Score")
    assert cols.index("Net_Return") > cols.index("Composite_Score")


def test_risk_free_warning_columns_present():
    """A1.1 + A1.2: Risk_Free_Unavailable + Sharpe_Warning kolonlari raporda."""
    assert "Risk_Free_Unavailable" in METRICS_REPORT_COLUMNS
    assert "Sharpe_Warning" in METRICS_REPORT_COLUMNS


def test_calmar_and_deflated_sharpe_present_in_report_order():
    """A1.2: Calmar + Deflated_Sharpe Sharpe'dan once gosterilir."""
    cols = METRICS_REPORT_COLUMNS
    assert "Calmar" in cols
    assert "Deflated_Sharpe" in cols
    assert cols.index("Calmar") < cols.index("Sharpe")
    assert cols.index("Deflated_Sharpe") < cols.index("Sharpe")


def test_composite_score_rewards_higher_dir_acc():
    """A1.4: Daha yuksek Dir_Acc -> daha yuksek composite (diger esit)."""
    base = {
        "RMSE_vs_benchmark": 0.95,
        "DirAcc_vs_benchmark": 0.0,
        "Sharpe_excess_vs_buy_hold": 0.0,
        "Neutral_Rate": 0.0,
        "Hit_Rate": 50.0,
        "Eligible_For_Leader": True,
    }
    low_dir = compute_composite_score(dict(base, Dir_Acc=45.0))
    high_dir = compute_composite_score(dict(base, Dir_Acc=70.0))
    assert high_dir > low_dir


def test_composite_score_rewards_higher_hit_rate():
    """A1.4: Daha yuksek Hit_Rate -> daha yuksek composite."""
    base = {
        "RMSE_vs_benchmark": 0.95,
        "DirAcc_vs_benchmark": 0.0,
        "Sharpe_excess_vs_buy_hold": 0.0,
        "Neutral_Rate": 0.0,
        "Dir_Acc": 55.0,
        "Eligible_For_Leader": True,
    }
    low_hit = compute_composite_score(dict(base, Hit_Rate=40.0))
    high_hit = compute_composite_score(dict(base, Hit_Rate=70.0))
    assert high_hit > low_hit


def test_composite_score_nan_sharpe_does_not_crash():
    """A1.4: Sharpe NaN ise composite hesaplanabilir (notr puan)."""
    metrics = {
        "RMSE_vs_benchmark": 0.90,
        "DirAcc_vs_benchmark": 5.0,
        "Sharpe_excess_vs_buy_hold": float("nan"),
        "Neutral_Rate": 0.0,
        "Dir_Acc": 60.0,
        "Hit_Rate": 55.0,
        "Eligible_For_Leader": True,
    }
    score = compute_composite_score(metrics)
    assert math.isfinite(score)
    assert score > 0.0


def test_composite_score_ineligible_capped():
    """Eligible_For_Leader=False -> skor <=49 (geriye uyum)."""
    metrics = {
        "RMSE_vs_benchmark": 0.5,
        "DirAcc_vs_benchmark": 10.0,
        "Sharpe_excess_vs_buy_hold": 1.0,
        "Neutral_Rate": 0.0,
        "Dir_Acc": 80.0,
        "Hit_Rate": 70.0,
        "Eligible_For_Leader": False,
    }
    score = compute_composite_score(metrics)
    assert score <= 49.0


def test_composite_score_ignores_net_return():
    """A1.4: Net_Return formule dahil degildir; degeri composite'i etkilemez."""
    base = {
        "RMSE_vs_benchmark": 0.9,
        "DirAcc_vs_benchmark": 5.0,
        "Sharpe_excess_vs_buy_hold": 0.5,
        "Neutral_Rate": 0.0,
        "Dir_Acc": 55.0,
        "Hit_Rate": 55.0,
        "Eligible_For_Leader": True,
    }
    low_nr = compute_composite_score(dict(base, Net_Return=-0.5))
    high_nr = compute_composite_score(dict(base, Net_Return=2.0))
    assert low_nr == high_nr
