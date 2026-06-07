# -*- coding: utf-8 -*-
"""Peer-signal confidence tests — E2 Faz 5."""

from src.serving.confidence import peer_confidence


def test_strong_segment_high():
    c = peer_confidence(1.35, "outperform")
    assert c.label == "high"
    assert any("guclu" in r for r in c.reasons)


def test_medium_segment_medium():
    assert peer_confidence(0.66, "inline").label == "medium"


def test_weak_segment_low():
    assert peer_confidence(0.30, "underperform").label == "low"


def test_not_tradable_forces_low_even_with_strong_signal():
    c = peer_confidence(1.50, "outperform", tradable=False)
    assert c.label == "low"
    assert any("kapisi" in w for w in c.warnings)


def test_stale_forces_low():
    assert peer_confidence(1.50, "outperform", stale=True).label == "low"


def test_unknown_label_or_thin_universe_low():
    assert peer_confidence(1.50, "unknown").label == "low"
    assert peer_confidence(1.50, "outperform", universe_ok=False).label == "low"


def test_missing_segment_ic_low():
    assert peer_confidence(None, "outperform").label == "low"
    assert peer_confidence(float("nan"), "outperform").label == "low"


def test_tradability_gate_precedes_signal():
    """Sert kapi (tradable=False) sinyal gucunden ONCE: en az likit (guclu IC)
    ama islem yapilamayan hisse -> low. Faz 6 urun gerginligi."""
    c = peer_confidence(1.35, "outperform", tradable=False)
    assert c.label == "low"
