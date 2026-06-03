# -*- coding: utf-8 -*-
"""Peer-signal confidence for serving — E2 Faz 5.

Faz 6 bulgusu: sinyal segment-bagimli ve sinyalin en guclu oldugu yerde
(en az likit) islem yapilabilirlik en zayif. Bu yuzden guven SADECE IC degil:
    confidence = f(segment_ICIR)  AND-gated by  tradability + freshness
Sert kapilar her zaman bastirir: islem yapilamayan / bayat / ince-evren ->
ne kadar guclu IC olursa olsun `low`.

Cikti mevcut API `ConfidenceBlock` ile uyumlu: label low/medium/high +
reasons + warnings.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceThresholds:
    icir_high: float = 1.0     # bu ICIR'in ustunde sinyal guclu
    icir_medium: float = 0.5   # bu ICIR'in ustunde sinyal orta


@dataclass(frozen=True)
class PeerConfidence:
    label: str           # low | medium | high
    reasons: list[str]
    warnings: list[str]


def peer_confidence(
    segment_icir: float | None,
    peer_label: str,
    *,
    tradable: bool = True,
    stale: bool = False,
    universe_ok: bool = True,
    thr: ConfidenceThresholds | None = None,
) -> PeerConfidence:
    """Segment ICIR + sert kapilardan guven uret.

    segment_icir : sembolun segmentinin (likidite/vol/sektor) OOS ICIR'i; None
                   veya NaN -> bilinmiyor (low).
    peer_label   : rank_to_peer_scores etiketi ('unknown' -> low).
    tradable     : likidite/islem kapisi gectiyse True.
    stale        : veri bayatsa True (sert dusurme).
    universe_ok  : skorlama gunu evren yeterince genis miydi.
    """
    thr = thr or ConfidenceThresholds()
    reasons: list[str] = []
    warnings: list[str] = []

    # --- sert kapilar (her zaman low) ---
    if not tradable:
        warnings.append("Likidite/islem kapisi: islem yapilabilirlik dusuk.")
        return PeerConfidence("low", reasons, warnings)
    if stale:
        warnings.append("Veri bayat: guncel fiyat yok.")
        return PeerConfidence("low", reasons, warnings)
    if not universe_ok or peer_label == "unknown":
        warnings.append("Cross-sectional evren yetersiz: goreli skor anlamsiz.")
        return PeerConfidence("low", reasons, warnings)

    # --- sinyal gucu (segment ICIR) ---
    if segment_icir is None or segment_icir != segment_icir:  # None/NaN
        warnings.append("Segment IC bilinmiyor.")
        return PeerConfidence("low", reasons, warnings)

    if segment_icir >= thr.icir_high:
        reasons.append(f"Segment cross-sectional sinyali guclu (ICIR {segment_icir:.2f}).")
        label = "high"
    elif segment_icir >= thr.icir_medium:
        reasons.append(f"Segment cross-sectional sinyali orta (ICIR {segment_icir:.2f}).")
        label = "medium"
    else:
        reasons.append(f"Segment cross-sectional sinyali zayif (ICIR {segment_icir:.2f}).")
        label = "low"
    return PeerConfidence(label, reasons, warnings)
