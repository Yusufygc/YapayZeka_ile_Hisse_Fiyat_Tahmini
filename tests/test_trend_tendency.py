# -*- coding: utf-8 -*-
"""Peer rank -> mutlak trend egilimi tests — E2 Faz 7."""

import numpy as np

from src.serving.trend_tendency import (
    TrendCalibration,
    TrendTendency,
    trend_from_peer,
)


def test_high_percentile_is_yukari():
    t = trend_from_peer(95.0, universe_size=100)
    assert t.label == "yukarı"
    assert t.prob_up is not None and t.prob_up > 0.5
    assert t.expected_return is not None and t.expected_return > 0


def test_low_percentile_is_asagi():
    t = trend_from_peer(5.0, universe_size=100)
    assert t.label == "aşağı"
    assert t.prob_up < 0.5
    assert t.expected_return < 0


def test_mid_percentile_is_yatay():
    t = trend_from_peer(50.0, universe_size=100)
    assert t.label == "yatay"


def test_thin_universe_is_belirsiz():
    t = trend_from_peer(95.0, universe_size=5)  # < min_names 15
    assert t.label == "belirsiz"
    assert t.prob_up is None and t.expected_return is None


def test_nan_percentile_is_belirsiz():
    t = trend_from_peer(float("nan"), universe_size=100)
    assert t.label == "belirsiz"


def test_none_inputs_safe():
    t = trend_from_peer(None, None)
    assert t.label == "belirsiz"


def test_calibration_monotone_across_quintiles():
    """Q1->Q5 percentil arttikca prob_up ve expected_return monoton artar."""
    pcts = [10.0, 30.0, 50.0, 70.0, 90.0]
    pups = [trend_from_peer(p, 100).prob_up for p in pcts]
    exps = [trend_from_peer(p, 100).expected_return for p in pcts]
    assert pups == sorted(pups)
    assert exps == sorted(exps)
    assert all(np.isfinite(x) for x in pups + exps)


def test_quintile_boundaries_map_correctly():
    # 0-20 Q1, 20-40 Q2, 40-60 Q3, 60-80 Q4, 80-100 Q5
    cfg = TrendCalibration()
    assert trend_from_peer(0.0, 100, cfg).prob_up == cfg.quintile_prob_up[0]
    assert trend_from_peer(99.9, 100, cfg).prob_up == cfg.quintile_prob_up[4]
    assert trend_from_peer(50.0, 100, cfg).prob_up == cfg.quintile_prob_up[2]


def test_custom_calibration_thresholds():
    cfg = TrendCalibration(lo_pct=20.0, hi_pct=80.0)
    assert trend_from_peer(75.0, 100, cfg).label == "yatay"  # 70<75<80
    assert trend_from_peer(85.0, 100, cfg).label == "yukarı"


def test_result_type_and_reasons():
    t = trend_from_peer(90.0, 100)
    assert isinstance(t, TrendTendency)
    assert t.reasons and isinstance(t.reasons, list)
    assert t.basis
