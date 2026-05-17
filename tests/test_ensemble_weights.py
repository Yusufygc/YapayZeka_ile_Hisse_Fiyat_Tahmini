# -*- coding: utf-8 -*-
"""Faz 5 Katman 1 — EnsembleModel performans-tabanlı ağırlık optimizerları."""
from __future__ import annotations

import numpy as np

from src.models.ensemble import EnsembleModel


def test_optimize_by_dsr_positive_only():
    weights = EnsembleModel.optimize_by_dsr({
        "A": 1.5,
        "B": 0.5,
        "C": -0.3,
        "D": 0.0,
    })
    # Negatif/sıfır dışlanır.
    assert weights["C"] == 0.0
    assert weights["D"] == 0.0
    # Pozitifler toplamı 1.
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    # A > B (DSR oranıyla).
    assert weights["A"] > weights["B"]


def test_optimize_by_dsr_all_negative_falls_back_to_equal():
    weights = EnsembleModel.optimize_by_dsr({"A": -1.0, "B": -0.5})
    assert abs(weights["A"] - 0.5) < 1e-6
    assert abs(weights["B"] - 0.5) < 1e-6


def test_optimize_by_profit_factor_excludes_below_one():
    weights = EnsembleModel.optimize_by_profit_factor({
        "A": 2.0,
        "B": 1.5,
        "C": 0.9,
        "D": 1.0,
    })
    assert weights["C"] == 0.0
    assert weights["D"] == 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    # PF=2 → 1.0 katkı, PF=1.5 → 0.5 katkı; A ağırlığı 2x B olmalı.
    # round(6) rounding ile +/- 1e-6 toleransa kadar.
    assert abs(weights["A"] - 2 * weights["B"]) < 1e-5


def test_optimize_by_profit_factor_all_losers_falls_back_to_equal():
    weights = EnsembleModel.optimize_by_profit_factor({"A": 0.7, "B": 0.5})
    assert abs(weights["A"] - 0.5) < 1e-6
    assert abs(weights["B"] - 0.5) < 1e-6


def test_optimize_by_sharpe_uses_directional_pnl():
    """Tahmin yönü gerçeği takip eden model daha yüksek ağırlık almalı."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(scale=0.01, size=100)
    # A: y_true ile aynı işaret → pozitif Sharpe
    # B: y_true ile zıt işaret → negatif Sharpe → ağırlık 0
    # C: gürültü → ~0 Sharpe
    predictions = {
        "A": y_true.copy(),
        "B": -y_true.copy(),
        "C": rng.normal(scale=0.01, size=100),
    }
    weights = EnsembleModel.optimize_by_sharpe(y_true, predictions)
    assert weights["A"] > weights["C"]
    assert weights["B"] == 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_optimize_by_sharpe_handles_short_series():
    """Tek noktalık seri Sharpe hesaplayamaz → ağırlık 0 ya da eşit fallback."""
    y_true = np.array([0.01])
    predictions = {"A": np.array([0.01]), "B": np.array([0.005])}
    weights = EnsembleModel.optimize_by_sharpe(y_true, predictions)
    # Tüm Sharpe 0 → eşit fallback.
    assert abs(weights["A"] - 0.5) < 1e-6
    assert abs(weights["B"] - 0.5) < 1e-6


def test_positive_normalize_floor_zero():
    weights = EnsembleModel._positive_normalize({"A": 2.0, "B": 1.0, "C": -1.0}, floor=0.0)
    assert weights["C"] == 0.0
    assert abs(weights["A"] - 2 / 3) < 1e-6
    assert abs(weights["B"] - 1 / 3) < 1e-6


def test_positive_normalize_floor_one():
    """floor=1 → PF semantics; values <= 1 düşer."""
    weights = EnsembleModel._positive_normalize({"A": 3.0, "B": 1.0, "C": 2.0}, floor=1.0)
    # A: 2.0, B: 0, C: 1.0 → total 3 → A=2/3, C=1/3
    assert weights["B"] == 0.0
    assert abs(weights["A"] - 2 / 3) < 1e-6
    assert abs(weights["C"] - 1 / 3) < 1e-6


# ───────────── Faz 5 Katman 2 — Risk-parity (inverse-volatility) ─────────


def test_optimize_by_risk_parity_inverse_volatility():
    """w_i ∝ 1/σ_i: düşük volatilite yüksek ağırlık."""
    weights = EnsembleModel.optimize_by_risk_parity({"A": 0.01, "B": 0.02, "C": 0.04})
    # 1/0.01=100, 1/0.02=50, 1/0.04=25 → total 175 → 100/175, 50/175, 25/175
    assert abs(weights["A"] - 100 / 175) < 1e-5
    assert abs(weights["B"] - 50 / 175) < 1e-5
    assert abs(weights["C"] - 25 / 175) < 1e-5
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_optimize_by_risk_parity_drops_zero_vol():
    """σ ≤ 0 olan model dışlanır."""
    weights = EnsembleModel.optimize_by_risk_parity({"A": 0.02, "B": 0.0, "C": -0.01})
    assert weights["B"] == 0.0
    assert weights["C"] == 0.0
    assert abs(weights["A"] - 1.0) < 1e-6


def test_optimize_by_risk_parity_all_zero_falls_back_to_equal():
    weights = EnsembleModel.optimize_by_risk_parity({"A": 0.0, "B": 0.0})
    assert abs(weights["A"] - 0.5) < 1e-6
    assert abs(weights["B"] - 0.5) < 1e-6


def test_compute_pnl_volatilities():
    """PnL_i = sign(pred_i) * y_true; std(ddof=1) döner."""
    rng = np.random.default_rng(1)
    y_true = rng.normal(scale=0.02, size=100)
    # A: y_true ile aynı yön → PnL = |y_true|
    # B: rastgele tahmin → PnL volatil
    predictions = {
        "A": y_true.copy(),
        "B": rng.normal(scale=0.02, size=100),
    }
    vols = EnsembleModel.compute_pnl_volatilities(y_true, predictions)
    assert vols["A"] > 0.0
    assert vols["B"] > 0.0
    # Doğrulama: A'nın PnL'i |y_true| → ddof=1 std
    expected_a = float(np.abs(y_true).std(ddof=1))
    assert abs(vols["A"] - expected_a) < 1e-12


def test_compute_pnl_volatilities_handles_short_series():
    """Tek noktalık seri → σ = 0."""
    y_true = np.array([0.01])
    predictions = {"A": np.array([0.01])}
    vols = EnsembleModel.compute_pnl_volatilities(y_true, predictions)
    assert vols["A"] == 0.0


# ───────────── Faz 5 Katman 5 — Cash signal / veto gate ──────────────


def test_directional_agreement_full_consensus():
    """Tüm modeller aynı yön → agreement = 1.0."""
    preds = {
        "A": np.array([0.01, 0.02, -0.01]),
        "B": np.array([0.05, 0.03, -0.02]),
        "C": np.array([0.02, 0.01, -0.005]),
    }
    agreement = EnsembleModel.compute_directional_agreement(preds)
    assert np.allclose(agreement, 1.0)


def test_directional_agreement_split():
    """3 model: 2 pozitif, 1 negatif → agreement = 2/3."""
    preds = {
        "A": np.array([0.01]),
        "B": np.array([0.02]),
        "C": np.array([-0.01]),
    }
    agreement = EnsembleModel.compute_directional_agreement(preds)
    assert abs(agreement[0] - 2 / 3) < 1e-9


def test_directional_agreement_empty():
    assert len(EnsembleModel.compute_directional_agreement({})) == 0


def test_cash_gate_magnitude_zeros_small_predictions():
    """|pred| < magnitude_threshold → 0."""
    target = np.array([0.001, 0.05, -0.0005, 0.02])
    out = EnsembleModel.apply_cash_gate(target, None, magnitude_threshold=0.01)
    assert out[0] == 0.0
    assert out[1] == 0.05
    assert out[2] == 0.0
    assert out[3] == 0.02


def test_cash_gate_agreement_zeros_low_consensus():
    """Agreement < threshold → ensemble target sıfırlanır."""
    base = {
        "A": np.array([0.01, 0.02]),
        "B": np.array([0.01, -0.02]),
        "C": np.array([-0.01, -0.02]),
    }
    # t=0: 2 pos, 1 neg → 2/3. t=1: 1 pos, 2 neg → 2/3.
    target = np.array([0.005, 0.003])
    out = EnsembleModel.apply_cash_gate(target, base, agreement_threshold=0.7)
    # 2/3 ≈ 0.667 < 0.7 → ikisi de 0
    assert out[0] == 0.0
    assert out[1] == 0.0
    # threshold 0.6 → 2/3 ≥ 0.6 → değişmez
    out2 = EnsembleModel.apply_cash_gate(target, base, agreement_threshold=0.6)
    assert out2[0] == 0.005
    assert out2[1] == 0.003


def test_cash_gate_no_op_when_thresholds_zero():
    target = np.array([0.001, 0.05])
    base = {"A": np.array([0.01]), "B": np.array([-0.01])}
    out = EnsembleModel.apply_cash_gate(target, base, magnitude_threshold=0.0, agreement_threshold=0.0)
    assert np.allclose(out, target)


def test_cash_gate_combined_magnitude_and_agreement():
    """İki gate birlikte: küçük + düşük agreement → 0."""
    base = {
        "A": np.array([0.01, 0.02]),
        "B": np.array([-0.01, 0.02]),  # t=0 split, t=1 consensus
    }
    target = np.array([0.05, 0.005])
    out = EnsembleModel.apply_cash_gate(
        target, base, magnitude_threshold=0.01, agreement_threshold=0.6
    )
    # t=0: |0.05|>=0.01 ok, agreement = 1/2 = 0.5 < 0.6 → 0
    # t=1: |0.005|<0.01 → 0 (magnitude gate)
    assert out[0] == 0.0
    assert out[1] == 0.0


# ───────────── Faz 5 Katman 4 — Ridge meta-stacker ────────────────────


def test_ridge_stacker_recovers_dominant_model():
    """Bir model y_true'yu mükemmel takip → ağırlığının çoğunluğunu almalı."""
    rng = np.random.default_rng(3)
    y = rng.normal(scale=0.02, size=200)
    predictions = {
        "Perfect": y.copy(),
        "Noise": rng.normal(scale=0.02, size=200),
        "Random": rng.normal(scale=0.05, size=200),
    }
    weights = EnsembleModel.optimize_by_ridge_stacker(y, predictions, alpha=0.01)
    assert abs(sum(weights.values()) - 1.0) < 1e-5
    assert weights["Perfect"] > weights["Noise"]
    assert weights["Perfect"] > weights["Random"]
    assert weights["Perfect"] > 0.5


def test_ridge_stacker_non_negative_clip():
    """non_negative=True: tüm ağırlıklar >= 0."""
    rng = np.random.default_rng(4)
    y = rng.normal(scale=0.02, size=150)
    predictions = {
        "A": -y + rng.normal(scale=0.01, size=150),  # zıt → negatif coef
        "B": y + rng.normal(scale=0.01, size=150),
    }
    weights = EnsembleModel.optimize_by_ridge_stacker(y, predictions, alpha=0.1, non_negative=True)
    assert all(w >= 0.0 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-5
    assert weights["A"] == 0.0  # zıt yön → kırpıldı
    assert weights["B"] > 0.0


def test_ridge_stacker_short_series_falls_back_to_equal():
    """Sample < n_models+1 → eşit ağırlık fallback."""
    y = np.array([0.01, 0.02])
    predictions = {"A": np.array([0.01, 0.015]), "B": np.array([0.02, 0.018]), "C": np.array([0.015, 0.017])}
    weights = EnsembleModel.optimize_by_ridge_stacker(y, predictions)
    # 3 model + intercept-off: min_len=2 < 3+1 = fallback
    for w in weights.values():
        assert abs(w - 1 / 3) < 1e-5


def test_ridge_stacker_all_negative_fallback():
    """Tüm coef'ler kırpıldıktan sonra 0 → eşit fallback."""
    rng = np.random.default_rng(5)
    y = rng.normal(scale=0.02, size=100)
    # Tüm tahminler zıt yön
    predictions = {
        "A": -y + rng.normal(scale=0.001, size=100),
        "B": -y + rng.normal(scale=0.001, size=100),
    }
    weights = EnsembleModel.optimize_by_ridge_stacker(y, predictions, alpha=0.01, non_negative=True)
    for w in weights.values():
        assert abs(w - 0.5) < 1e-5


def test_ridge_stacker_high_alpha_approaches_equal():
    """Çok yüksek alpha → coef'ler küçülür/eşitlenir → ağırlıklar yakınsar."""
    rng = np.random.default_rng(6)
    y = rng.normal(scale=0.02, size=200)
    predictions = {
        "A": y + rng.normal(scale=0.005, size=200),
        "B": y + rng.normal(scale=0.01, size=200),
    }
    w_low = EnsembleModel.optimize_by_ridge_stacker(y, predictions, alpha=0.001)
    w_high = EnsembleModel.optimize_by_ridge_stacker(y, predictions, alpha=1e6)
    spread_low = abs(w_low["A"] - w_low["B"])
    spread_high = abs(w_high["A"] - w_high["B"])
    assert spread_high < spread_low


# ───────────── Faz 5 Katman 3 — Hierarchical (category-gated) ─────────


def test_hierarchical_balanced_categories():
    """2 kategori, her birinde 2 model → her model 0.25."""
    predictions = {
        "T1": np.zeros(10),
        "T2": np.zeros(10),
        "L1": np.zeros(10),
        "L2": np.zeros(10),
    }
    categories = {"T1": "tree", "T2": "tree", "L1": "linear_shrinkage", "L2": "linear_shrinkage"}
    weights = EnsembleModel.optimize_hierarchical_by_category(predictions, categories)
    for name in predictions:
        assert abs(weights[name] - 0.25) < 1e-6
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_hierarchical_unbalanced_categories():
    """tree=3 model, linear=1 model → tree üyeleri 1/6, linear üyesi 1/2."""
    predictions = {f"T{i}": np.zeros(10) for i in range(3)}
    predictions["L1"] = np.zeros(10)
    categories = {"T0": "tree", "T1": "tree", "T2": "tree", "L1": "linear"}
    weights = EnsembleModel.optimize_hierarchical_by_category(predictions, categories)
    for k in ("T0", "T1", "T2"):
        assert abs(weights[k] - (0.5 / 3)) < 1e-5
    assert abs(weights["L1"] - 0.5) < 1e-6
    # round(6) ile 3*0.166667 = 0.500001 — toplam 1.000001
    assert abs(sum(weights.values()) - 1.0) < 1e-5


def test_hierarchical_unknown_falls_back_to_own_group():
    """Kategori eşlenmemiş model 'unknown' grubuna girer."""
    predictions = {"A": np.zeros(10), "B": np.zeros(10)}
    weights = EnsembleModel.optimize_hierarchical_by_category(predictions, {})
    # Tek 'unknown' kategori → her ikisi 0.5.
    assert abs(weights["A"] - 0.5) < 1e-6
    assert abs(weights["B"] - 0.5) < 1e-6


def test_hierarchical_empty_returns_empty():
    assert EnsembleModel.optimize_hierarchical_by_category({}, {}) == {}


def test_risk_parity_end_to_end_via_pnl():
    """compute_pnl_volatilities + optimize_by_risk_parity zinciri."""
    rng = np.random.default_rng(2)
    y_true = rng.normal(scale=0.02, size=200)
    # A: aynı yön, küçük gürültü → düşük PnL vol (tutarlı)
    # B: zıt yön + büyük gürültü → yüksek PnL vol
    predictions = {
        "A": y_true + rng.normal(scale=0.005, size=200),
        "B": -y_true + rng.normal(scale=0.03, size=200),
    }
    vols = EnsembleModel.compute_pnl_volatilities(y_true, predictions)
    weights = EnsembleModel.optimize_by_risk_parity(vols)
    # Düşük vol = yüksek ağırlık olmalı; vols sıralaması ile weights ters orantılı
    assert (vols["A"] < vols["B"]) == (weights["A"] > weights["B"])
    assert abs(sum(weights.values()) - 1.0) < 1e-6
