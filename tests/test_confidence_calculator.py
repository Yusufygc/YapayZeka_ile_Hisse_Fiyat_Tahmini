# -*- coding: utf-8 -*-
"""confidence_calculator birim testleri — 12+ senaryo matrisi."""
import pytest

from src.pipeline.confidence_calculator import compute_confidence


def _c(**kw):
    return compute_confidence(**kw)


# ─── Hard block testleri ────────────────────────────────────────────────────

def test_no_candidate_is_low():
    r = _c(eligibility_status="no_candidate")
    assert r.label == "low"
    assert "no_candidate" in r.reasons[0]


def test_naive_low_trades_is_low():
    r = _c(eligibility_status="naive_low_trades")
    assert r.label == "low"


def test_insufficient_trades_eligibility_is_low():
    r = _c(eligibility_status="insufficient_trades")
    assert r.label == "low"


def test_stale_data_is_low():
    r = _c(data_freshness="stale_data")
    assert r.label == "low"
    assert len(r.warnings) > 0


def test_dir_acc_below_50_is_low():
    r = _c(directional_accuracy=48.0)
    assert r.label == "low"


def test_rmse_vs_benchmark_above_1_is_low():
    r = _c(directional_accuracy=55.0, rmse_vs_benchmark=1.05)
    assert r.label == "low"


def test_psi_high_is_low():
    r = _c(directional_accuracy=56.0, psi_high=True)
    assert r.label == "low"
    assert any("PSI" in w or "distribution" in w.lower() for w in r.warnings)


def test_corporate_action_anomaly_is_low():
    r = _c(directional_accuracy=56.0, corporate_action_anomaly=True)
    assert r.label == "low"


def test_model_degraded_is_low():
    r = _c(directional_accuracy=56.0, model_status="degraded")
    assert r.label == "low"


# ─── Soft degradation → medium cap testleri ─────────────────────────────────

def test_insufficient_trades_signal_diagnosis_caps_medium():
    r = _c(directional_accuracy=56.0, rmse_vs_benchmark=0.9, signal_diagnosis="insufficient_trades")
    assert r.label == "medium"
    assert any("insufficient_trades" in reason for reason in r.reasons)


def test_underperform_buyhold_caps_medium():
    r = _c(directional_accuracy=60.0, rmse_vs_benchmark=0.8, signal_diagnosis="underperform_buyhold")
    assert r.label == "medium"


def test_stability_degraded_prevents_high():
    r = _c(directional_accuracy=60.0, rmse_vs_benchmark=0.8, stability_score=-0.5)
    assert r.label == "medium"
    assert any("stability" in reason.lower() for reason in r.reasons)


def test_rolling_ratio_below_05_prevents_high():
    r = _c(directional_accuracy=60.0, rmse_vs_benchmark=0.8, rolling_positive_window_ratio=0.4)
    assert r.label == "medium"


def test_ensemble_low_caps_medium():
    r = _c(directional_accuracy=60.0, rmse_vs_benchmark=0.8, ensemble_direction_agreement=0.3)
    assert r.label == "medium"


def test_regime_misalignment_caps_medium():
    r = _c(directional_accuracy=60.0, rmse_vs_benchmark=0.8, regime_misalignment=True)
    assert r.label == "medium"


# ─── medium — no soft degradation ───────────────────────────────────────────

def test_medium_when_no_soft_and_dir_acc_between_50_55():
    r = _c(directional_accuracy=53.0, rmse_vs_benchmark=0.9)
    assert r.label == "medium"


# ─── high koşulları ─────────────────────────────────────────────────────────

def test_high_when_all_conditions_met():
    r = _c(
        directional_accuracy=57.0,
        rmse_vs_benchmark=0.85,
        stability_score=0.7,
        ensemble_direction_agreement=0.8,
    )
    assert r.label == "high"
    assert r.reasons == []


def test_high_without_optional_faz2_fields():
    r = _c(directional_accuracy=57.0, rmse_vs_benchmark=0.85, stability_score=0.7)
    assert r.label == "high"


def test_high_requires_dir_acc_above_55():
    r = _c(directional_accuracy=54.9, rmse_vs_benchmark=0.85, stability_score=0.7)
    assert r.label == "medium"


def test_high_requires_stability_above_threshold():
    r = _c(directional_accuracy=57.0, rmse_vs_benchmark=0.85, stability_score=0.1)
    # 0.1 < 0.5 threshold → no high
    assert r.label == "medium"


# ─── Bileşik senaryolar ──────────────────────────────────────────────────────

def test_multiple_soft_degradations_still_medium():
    r = _c(
        directional_accuracy=56.0,
        rmse_vs_benchmark=0.9,
        signal_diagnosis="gate_too_strict,model_signal_weak",
        stability_score=-0.2,
    )
    assert r.label == "medium"
    assert len(r.reasons) >= 2


def test_ok_signal_diagnosis_treated_as_no_labels():
    r = _c(directional_accuracy=57.0, rmse_vs_benchmark=0.85, stability_score=0.7, signal_diagnosis="ok")
    assert r.label == "high"
