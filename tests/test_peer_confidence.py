# -*- coding: utf-8 -*-
"""Peer-signal confidence tests — E2 Faz 5."""

from src.serving.confidence import peer_confidence


def test_strong_segment_high():
    c = peer_confidence(1.35, "outperform")
    assert c.label == "high"
    assert any("güçlü" in r for r in c.reasons)


def test_medium_segment_medium():
    assert peer_confidence(0.66, "inline").label == "medium"


def test_weak_segment_low():
    assert peer_confidence(0.30, "underperform").label == "low"


def test_not_tradable_warns_but_keeps_signal_label():
    # kalibrasyon 2026-11-16: en az likit segmentte sinyal en guclu; likidite ayri uyari
    c = peer_confidence(1.50, "outperform", tradable=False)
    assert c.label == "high"
    assert any("Likidite düşük" in w for w in c.warnings)


def test_stale_forces_low():
    assert peer_confidence(1.50, "outperform", stale=True).label == "low"


def test_unknown_label_or_thin_universe_low():
    assert peer_confidence(1.50, "unknown").label == "low"
    assert peer_confidence(1.50, "outperform", universe_ok=False).label == "low"


def test_missing_segment_ic_low():
    assert peer_confidence(None, "outperform").label == "low"
    assert peer_confidence(float("nan"), "outperform").label == "low"


def test_relative_thresholds_split_universe_into_terciles():
    from src.serving.confidence import relative_thresholds
    vals = [0.1 * i for i in range(1, 10)]          # 0.1 .. 0.9
    thr = relative_thresholds(vals)
    labels = [peer_confidence(v, "inline", thr=thr, n_eff=100).label for v in vals]
    assert labels == ["low"] * 3 + ["medium"] * 3 + ["high"] * 3


def test_relative_thresholds_floor_and_fallback():
    from src.serving.confidence import ConfidenceThresholds, relative_thresholds
    thr = relative_thresholds([-0.3, -0.2, -0.1])   # hepsi negatif -> floor 0: hepsi low
    assert all(peer_confidence(v, "inline", thr=thr).label == "low" for v in (-0.3, -0.2, -0.1))
    assert relative_thresholds([1.0, float("nan")]) == ConfidenceThresholds()
