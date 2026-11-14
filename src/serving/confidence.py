# -*- coding: utf-8 -*-
"""Peer-signal confidence for serving — E2 Faz 5.

Faz 6 bulgusu: sinyal segment-bagimli ve sinyalin en guclu oldugu yerde
(en az likit) islem yapilabilirlik en zayif. Bu yuzden guven SADECE IC degil:
    confidence = f(segment_ICIR)  AND-gated by  tradability + freshness
Sert kapilar her zaman bastirir: islem yapilamayan / bayat / ince-evren ->
ne kadar guclu IC olursa olsun `low`.

ICIR esikleri (1.0 / 0.5) ve n_eff esigi (50) HEURISTIKTIR, kalibre edilmis
istatistiksel esikler degildir. Esiklerin uygulanacagi ICIR ORTUSMEYEN
(`icir_nonoverlap`) segment ICIR'i olmalidir (ortusmeli gunluk ICIR SE'yi dusuk
gosterir); en iyi segmenti secmek winner's curse yaratir -> n_eff kapisi
("high" icin n_eff >= 50 ortusmeyen tarih). Serving segmenti NEDENSEL
(`attach_date_segments`/`latest_date_segments`) atanmali; statik
`symbol_segments` (tam-orneklem medyan, look-ahead) guvende kullanilmamali.

Cikti mevcut API `ConfidenceBlock` ile uyumlu: label low/medium/high +
reasons + warnings.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceThresholds:
    # HEURISTIK esikler (kalibre degil); ortusmeyen segment ICIR'ine uygulanmali.
    icir_high: float = 1.0     # bu ICIR'in ustunde sinyal guclu
    icir_medium: float = 0.5   # bu ICIR'in ustunde sinyal orta
    min_n_eff_high: int = 50   # "high" icin gereken min ortusmeyen tarih sayisi


def select_segment_icir(stats: dict | None) -> tuple[float | None, int | None]:
    """Segment istatistik satirindan (icir, n_eff) sec; ortusmeyen tercih edilir.

    `icir_nonoverlap` + `n_nonoverlap` varsa onlar; yoksa eski `icir` (ortusmeli,
    n_eff=None -> n_eff kapisi uygulanamaz). stats None -> (None, None).
    """
    if not stats:
        return None, None
    v, n = stats.get("icir_nonoverlap"), stats.get("n_nonoverlap")
    if v is not None and v == v:
        return float(v), (int(n) if n is not None and n == n else None)
    v = stats.get("icir")
    return (float(v) if v is not None and v == v else None), None


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
    n_eff: int | None = None,
) -> PeerConfidence:
    """Segment ICIR + sert kapilardan guven uret.

    segment_icir : sembolun segmentinin (likidite/vol/sektor) OOS ICIR'i; None
                   veya NaN -> bilinmiyor (low).
    peer_label   : rank_to_peer_scores etiketi ('unknown' -> low).
    tradable     : likidite/islem kapisi gectiyse True.
    stale        : veri bayatsa True (sert dusurme).
    universe_ok  : skorlama gunu evren yeterince genis miydi.
    n_eff        : segment ICIR'inin dayandigi ORTUSMEYEN tarih sayisi. Verilirse
                   ve < thr.min_n_eff_high ise "high" verilmez ("medium"e iner).
                   None = bilinmiyor (eski cagrilar; kapi uygulanmaz).
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
        if n_eff is not None and n_eff < thr.min_n_eff_high:
            reasons.append(
                f"Segment cross-sectional sinyali orta (ICIR {segment_icir:.2f}; "
                f"yuksek icin yetersiz ornek).")
            warnings.append(
                f"n_eff={n_eff} < {thr.min_n_eff_high}: ortusmeyen ornek az, "
                f"'high' verilmedi.")
            label = "medium"
        else:
            reasons.append(f"Segment cross-sectional sinyali guclu (ICIR {segment_icir:.2f}).")
            label = "high"
    elif segment_icir >= thr.icir_medium:
        reasons.append(f"Segment cross-sectional sinyali orta (ICIR {segment_icir:.2f}).")
        label = "medium"
    else:
        reasons.append(f"Segment cross-sectional sinyali zayif (ICIR {segment_icir:.2f}).")
        label = "low"
    return PeerConfidence(label, reasons, warnings)
