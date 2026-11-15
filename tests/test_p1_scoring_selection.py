# -*- coding: utf-8 -*-
"""P1 skorlama + secim duzeltmeleri.

- composite RMSE orani < 1 tarafinda monoton (benchmark'i yenmek odullendirilir)
- benchmark Dir_Acc ~ 0 iken sabit bonus yok (DirAcc_vs_benchmark formulden cikti)
- Diebold-Mariano (HLN, kare hata kaybi) testi
- secim: DM p >= 0.10 olan model composite'i en yuksek olsa da secilmez
- NaN RMSE orani -> composite <= 49
- stock_model_db.compute_composite_score geriye uyumlu re-export
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.scoring import (
    DM_ALPHA,
    compute_composite_details,
    compute_composite_score,
    diebold_mariano,
)
from src.pipeline.metrics_reporter import _MetricsReporterMixin
from src.pipeline.selection_guard import (
    compute_eligibility,
    evaluate_best_model_eligibility,
)


def _row(**overrides):
    base = {
        "RMSE_vs_benchmark": 0.95,
        "DirAcc_vs_benchmark": 0.0,
        "Dir_Acc": 55.0,
        "Hit_Rate": 50.0,
        "Sharpe_excess_vs_buy_hold": 0.0,
        "Neutral_Rate": 0.0,
        "Eligible_For_Leader": True,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #

def test_composite_monotone_in_rmse_ratio_below_one():
    s95 = compute_composite_score(_row(RMSE_vs_benchmark=0.95))
    s99 = compute_composite_score(_row(RMSE_vs_benchmark=0.99))
    s100 = compute_composite_score(_row(RMSE_vs_benchmark=1.0))
    assert s95 > s99 > s100


def test_composite_rmse_term_is_continuous_up_to_one_then_capped():
    s_below = compute_composite_score(_row(RMSE_vs_benchmark=0.999))
    s_one = compute_composite_score(_row(RMSE_vs_benchmark=1.0))
    # 0.001 oran farki = 1 rmse puani * 0.5 agirlik = 0.5 puan; ucurum yok.
    assert s_below - s_one == pytest.approx(0.5, abs=1e-3)
    # ratio > 1 kapagi (<=49) bilincli olarak korunur.
    assert compute_composite_score(_row(RMSE_vs_benchmark=1.001, Dir_Acc=90.0)) <= 49.0


def test_composite_not_saturated_when_benchmark_diracc_is_zero():
    """Naive Zero Return benchmark: DirAcc_vs_benchmark = Dir_Acc - 0 sabit bonus vermemeli."""
    low = compute_composite_score(_row(Dir_Acc=48.0, DirAcc_vs_benchmark=48.0))
    mid = compute_composite_score(_row(Dir_Acc=55.0, DirAcc_vs_benchmark=55.0))
    high = compute_composite_score(_row(Dir_Acc=60.0, DirAcc_vs_benchmark=60.0))
    assert low < mid < high
    # DirAcc_vs_benchmark formulde yok: degeri skoru etkilemez.
    assert compute_composite_score(_row(DirAcc_vs_benchmark=0.0)) == compute_composite_score(
        _row(DirAcc_vs_benchmark=40.0)
    )


def test_composite_ignores_hit_rate_collinear_term():
    assert compute_composite_score(_row(Hit_Rate=10.0)) == compute_composite_score(_row(Hit_Rate=90.0))


def test_composite_formula_exact_values():
    # rmse 0.98 -> 70 ; Dir_Acc 54 -> 70 ; sharpe 0 -> 50
    expected = 0.5 * 70.0 + 0.3 * 70.0 + 0.2 * 50.0
    got = compute_composite_score(_row(RMSE_vs_benchmark=0.98, Dir_Acc=54.0))
    assert got == pytest.approx(expected, abs=1e-3)


def test_composite_nan_rmse_ratio_is_capped_and_incomplete():
    details = compute_composite_details(_row(RMSE_vs_benchmark=float("nan"), Dir_Acc=80.0))
    assert details["Composite_Score"] <= 49.0
    assert details["Composite_Incomplete"] is True
    assert compute_composite_score(_row(RMSE_vs_benchmark=float("nan"))) <= 49.0


def test_composite_missing_diracc_is_neutral_not_zero_and_flagged():
    row = _row()
    del row["Dir_Acc"]
    details = compute_composite_details(row)
    neutral = compute_composite_score(_row(Dir_Acc=50.0))
    assert details["Composite_Score"] == pytest.approx(neutral)
    assert details["Composite_Incomplete"] is True
    assert compute_composite_details(_row())["Composite_Incomplete"] is False


def test_composite_caps_and_nan_sharpe():
    assert compute_composite_score(_row(RMSE_vs_benchmark=1.2, Dir_Acc=90.0)) <= 49.0
    assert compute_composite_score(_row(Eligible_For_Leader=False, Dir_Acc=90.0)) <= 49.0
    assert math.isfinite(compute_composite_score(_row(Sharpe_excess_vs_buy_hold=float("nan"))))


def test_stock_model_db_reexport_is_same_function():
    from src.database import stock_model_db
    from src.evaluation import scoring

    assert stock_model_db.compute_composite_score is scoring.compute_composite_score


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #

def test_dm_model_strictly_better_is_significant():
    rng = np.random.default_rng(0)
    e_bench = rng.normal(0.0, 1.0, 300)
    e_model = 0.7 * e_bench  # kare hata her adimda daha kucuk
    stat, p = diebold_mariano(e_model, e_bench, h=1)
    assert stat > 0
    assert p < 0.05


def test_dm_equal_accuracy_not_significant():
    rng = np.random.default_rng(1)
    e_bench = rng.normal(0.0, 1.0, 400)
    e_model = rng.normal(0.0, 1.0, 400)
    _, p = diebold_mariano(e_model, e_bench, h=1)
    assert p > 0.3


def test_dm_model_worse_has_large_one_sided_p():
    rng = np.random.default_rng(2)
    e_bench = rng.normal(0.0, 1.0, 300)
    e_model = 1.5 * e_bench
    _, p = diebold_mariano(e_model, e_bench, h=1)
    assert p > 0.95


def test_dm_degenerate_inputs_return_nan_or_neutral():
    stat, p = diebold_mariano(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert math.isnan(p)  # n < 3
    e = np.linspace(-1, 1, 20)
    stat, p = diebold_mariano(e, e)
    assert p == pytest.approx(1.0)  # ozdes hata: kanit yok


def test_dm_multi_step_lag_changes_variance_but_stays_valid():
    rng = np.random.default_rng(3)
    e_bench = rng.normal(0.0, 1.0, 300)
    e_model = 0.8 * e_bench
    _, p1 = diebold_mariano(e_model, e_bench, h=1)
    _, p5 = diebold_mariano(e_model, e_bench, h=5)
    assert 0.0 <= p1 <= 1.0 and 0.0 <= p5 <= 1.0


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def _cand(composite, rmse_ratio, dm_p=None, rmse=1.0):
    row = {
        "Composite_Score": composite,
        "RMSE": rmse,
        "RMSE_vs_benchmark": rmse_ratio,
        "Candidate_For_Selection": True,
    }
    if dm_p is not None:
        row["DM_PValue_vs_benchmark"] = dm_p
    return row


def test_selection_skips_best_composite_when_dm_not_significant():
    metrics = {
        "LSTM": _cand(80.0, 0.90, dm_p=0.50),
        "GRU": _cand(60.0, 0.95, dm_p=0.04),
    }
    assert _MetricsReporterMixin._select_best_model(metrics) == "GRU"


def test_selection_no_significant_candidate_returns_none_and_flags():
    metrics = {
        "LSTM": _cand(80.0, 0.90, dm_p=0.50),
        "GRU": _cand(60.0, 1.05, dm_p=0.01),
        "TFT": _cand(70.0, float("nan")),
    }
    assert _MetricsReporterMixin._select_best_model(metrics) is None
    assert all(r["Selection_Status"] == "no_significant_candidate" for r in metrics.values())


def test_selection_without_dm_still_requires_finite_ratio_le_one():
    metrics = {
        "A": _cand(90.0, float("nan")),
        "B": _cand(70.0, 1.2),
        "C": _cand(50.0, 0.99),
    }
    assert _MetricsReporterMixin._select_best_model(metrics) == "C"


def test_selection_among_significant_picks_max_composite_then_rmse():
    metrics = {
        "A": _cand(70.0, 0.9, dm_p=0.05, rmse=2.0),
        "B": _cand(70.0, 0.9, dm_p=0.02, rmse=1.0),
        "C": _cand(65.0, 0.9, dm_p=0.01, rmse=0.5),
    }
    assert _MetricsReporterMixin._select_best_model(metrics) == "B"
    assert DM_ALPHA == pytest.approx(0.10)


def test_attach_composite_scores_adds_dm_pvalue_from_predictions():
    rng = np.random.default_rng(4)
    y = rng.normal(0.0, 0.02, 250)
    preds = {
        "Naive Zero Return": np.zeros_like(y),
        "Good Model": y * 0.8 + rng.normal(0.0, 0.002, 250),
    }

    def _m(p):
        err = y - p
        rmse = float(np.sqrt(np.mean(err ** 2)))
        return {"RMSE": rmse, "MAE": float(np.mean(np.abs(err))), "Dir_Acc": 60.0,
                "Hit_Rate": 55.0, "Sharpe": 0.0, "BuyHold_Sharpe": 0.0, "Neutral_Rate": 0.0}

    metrics = {name: _m(p) for name, p in preds.items()}
    scored = _MetricsReporterMixin._attach_composite_scores(metrics, predictions=preds, y_true=y)
    assert scored["Good Model"]["DM_PValue_vs_benchmark"] < 0.05
    assert math.isnan(scored["Naive Zero Return"]["DM_PValue_vs_benchmark"])
    assert "Composite_Incomplete" in scored["Good Model"]


def test_attach_composite_scores_without_predictions_has_no_dm_column():
    metrics = {
        "Naive Zero Return": {"RMSE": 1.0, "Dir_Acc": 50.0},
        "M": {"RMSE": 0.9, "Dir_Acc": 55.0},
    }
    scored = _MetricsReporterMixin._attach_composite_scores(metrics)
    assert "DM_PValue_vs_benchmark" not in scored["M"]


# --------------------------------------------------------------------------- #
# Selection guard
# --------------------------------------------------------------------------- #

def test_guard_dm_pvalue_above_threshold_fails_benchmark():
    status, reason = compute_eligibility(
        model_name="LSTM", is_production_candidate=True, is_baseline=False,
        total_trade_count=20, rmse_vs_benchmark=0.9, dm_p_value=0.5,
    )
    assert status == "benchmark_failed"
    assert "Diebold" in reason or "DM" in reason


def test_guard_dm_pvalue_significant_or_missing_passes():
    for p in (0.03, None, float("nan")):
        status, _ = compute_eligibility(
            model_name="LSTM", is_production_candidate=True, is_baseline=False,
            total_trade_count=20, rmse_vs_benchmark=0.9, dm_p_value=p,
        )
        assert status == "eligible"


def test_evaluate_best_model_eligibility_reads_dm_from_row():
    row = {
        "model_name": "LSTM", "is_production_candidate": 1, "Trade_Count": 20,
        "RMSE_vs_benchmark": 0.9, "DM_PValue_vs_benchmark": 0.4,
    }
    assert evaluate_best_model_eligibility(row)[0] == "benchmark_failed"
    row["DM_PValue_vs_benchmark"] = 0.01
    assert evaluate_best_model_eligibility(row)[0] == "eligible"
    row.pop("DM_PValue_vs_benchmark")
    assert evaluate_best_model_eligibility(row)[0] == "eligible"
