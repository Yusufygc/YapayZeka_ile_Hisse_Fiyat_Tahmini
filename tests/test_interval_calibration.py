# -*- coding: utf-8 -*-
"""interval_calibration B2 (residual band) + C (conformal) birim testleri."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.forecasting.interval_calibration import (
    adaptive_conformal_update,
    compute_conformal_calibration,
    compute_residual_calibration,
    conformal_band,
    residual_band,
    resolve_active_calibration,
    sigma_for_regime,
    z_for_level,
)


def _fold_records(residuals, regimes=None):
    """y_pred=0 alınca residual = y_true. Tek fold record üretir."""
    n = len(residuals)
    rec = {
        "y_true_target": list(residuals),
        "y_pred_target": [0.0] * n,
    }
    if regimes is not None:
        rec["market_regime"] = list(regimes)
    return [rec]


# --- z / level ----------------------------------------------------------

def test_z_for_level_known_values():
    assert z_for_level(0.8) == pytest.approx(1.2816, abs=1e-3)
    assert z_for_level(0.9) == pytest.approx(1.6449, abs=1e-3)
    assert z_for_level(0.95) == pytest.approx(1.9600, abs=1e-3)


def test_z_for_level_invalid():
    with pytest.raises(ValueError):
        z_for_level(0.0)
    with pytest.raises(ValueError):
        z_for_level(1.0)


# --- B2 residual --------------------------------------------------------

def test_residual_sigma_matches_numpy():
    residuals = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
    calib = compute_residual_calibration(_fold_records(residuals), levels=(0.8,))
    assert calib is not None
    assert calib["method"] == "residual_b2"
    assert calib["sigma"] == pytest.approx(float(np.std(residuals, ddof=1)))
    assert calib["n_samples"] == len(residuals)


def test_residual_band_symmetry_and_sqrt_h():
    sigma = 2.0
    lower1, upper1 = residual_band(10.0, sigma, horizon_index=1, level=0.9)
    # simetri: merkez ortalaması p50
    assert (lower1 + upper1) / 2.0 == pytest.approx(10.0)
    half1 = upper1 - 10.0
    # √h ölçeği: h=4 -> 2x genişlik
    _, upper4 = residual_band(10.0, sigma, horizon_index=4, level=0.9)
    half4 = upper4 - 10.0
    assert half4 == pytest.approx(half1 * math.sqrt(4))


def test_residual_band_invalid_sigma():
    with pytest.raises(ValueError):
        residual_band(1.0, -0.5, horizon_index=1, level=0.8)


def test_residual_per_regime_sigma():
    # iki rejim, farklı dağılım
    residuals = [1.0, -1.0, 1.0, -1.0, 5.0, -5.0, 5.0, -5.0]
    regimes = ["calm", "calm", "calm", "calm", "wild", "wild", "wild", "wild"]
    calib = compute_residual_calibration(_fold_records(residuals, regimes), per_regime=True)
    assert "calm" in calib["sigma_by_regime"]
    assert "wild" in calib["sigma_by_regime"]
    assert calib["sigma_by_regime"]["wild"] > calib["sigma_by_regime"]["calm"]
    assert sigma_for_regime(calib, "wild") == calib["sigma_by_regime"]["wild"]
    # bilinmeyen rejim -> global σ
    assert sigma_for_regime(calib, "unknown") == calib["sigma"]


def test_residual_empty_returns_none():
    assert compute_residual_calibration([]) is None
    assert compute_residual_calibration(_fold_records([])) is None


# --- C conformal --------------------------------------------------------

def test_conformal_qhat_empirical_quantile():
    # |residual| skorları bilinen; q̂ = ceil((n+1)*level)/n quantile (higher)
    residuals = [0.0, 1.0, -2.0, 3.0, -4.0]  # |.| = 0,1,2,3,4
    calib = compute_conformal_calibration(_fold_records(residuals), level=0.9)
    assert calib is not None
    assert calib["method"] == "conformal"
    n = 5
    rank = min(math.ceil((n + 1) * 0.9) / n, 1.0)
    expected = float(np.quantile([0, 1, 2, 3, 4], rank, method="higher"))
    assert calib["q_hat"] == pytest.approx(expected)
    assert calib["n_calib"] == n


def test_conformal_coverage_meets_nominal_on_synthetic():
    # Normal residual; conformal band nominal kapsamayı sağlamalı (split conformal garanti).
    rng = np.random.default_rng(7)
    resid = rng.normal(0.0, 1.0, size=2000)
    calib = compute_conformal_calibration(_fold_records(resid), level=0.9)
    q_hat = calib["q_hat"]
    # bağımsız test örneği
    test_resid = rng.normal(0.0, 1.0, size=5000)
    covered = np.mean(np.abs(test_resid) <= q_hat)
    assert covered >= 0.88  # ~0.90 nominal, örneklem toleransı


def test_conformal_band_and_invalid():
    lower, upper = conformal_band(5.0, 2.0)
    assert lower == pytest.approx(3.0)
    assert upper == pytest.approx(7.0)
    with pytest.raises(ValueError):
        conformal_band(1.0, -1.0)


def test_resolve_active_calibration_switch():
    calib = {
        "method": "residual_b2",
        "sigma": 0.05,
        "conformal": {"method": "conformal", "q_hat": 0.07, "level": 0.9},
    }
    # B2 tercih -> top-level residual
    assert resolve_active_calibration(calib, "residual_b2")["method"] == "residual_b2"
    # conformal tercih -> gömülü conformal
    active = resolve_active_calibration(calib, "conformal")
    assert active["method"] == "conformal"
    assert active["q_hat"] == 0.07
    # conformal yoksa B2'ye düşer
    only_b2 = {"method": "residual_b2", "sigma": 0.05}
    assert resolve_active_calibration(only_b2, "conformal")["method"] == "residual_b2"
    # None -> None
    assert resolve_active_calibration(None, "conformal") is None


def test_adaptive_conformal_widens_when_undercovered():
    # kapsama hedefin altında -> band genişler (q̂ artar)
    q_new = adaptive_conformal_update(2.0, recent_coverage=0.80, target_level=0.9, gamma=0.5)
    assert q_new > 2.0
    # kapsama hedefin üstünde -> daralır
    q_new2 = adaptive_conformal_update(2.0, recent_coverage=0.98, target_level=0.9, gamma=0.5)
    assert q_new2 < 2.0
    # asla negatif
    assert adaptive_conformal_update(0.1, recent_coverage=0.0, target_level=0.9, gamma=10.0) >= 0.0
