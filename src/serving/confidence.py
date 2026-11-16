# -*- coding: utf-8 -*-
"""Peer-signal confidence for serving — E2 Faz 5.

Faz 6 bulgusu: sinyal segment-bagimli ve sinyalin en guclu oldugu yerde
(en az likit) islem yapilabilirlik en zayif.
    confidence = f(segment_ICIR)  AND-gated by  freshness + evren
Sert kapilar her zaman bastirir: bayat / ince-evren -> `low`.

2026-11-16 kalibrasyon olcumu (`tools/calibration_report.py`, docs/wiki/validation-and-backtesting.md):
sabit 1.0/0.5 esikleri satirlarin %95'ini `medium` yapiyordu ve etiket ayirt edici degildi;
surekli segment ICIR'i ise sonraki IC'yi siraliyordu. Bu yuzden serving esikleri
`relative_thresholds` ile skorlama gununun evreni icinde TERTILDIR (ust 1/3 high); `low` icin
ek olarak mutlak taban (ICIR < 0.5) gerekir.
Likidite kapisi artik etiketi dusurmez: en az likit segmentte sinyal en guclu cikti
(IC 0.159); bunun yerine ayri "islem zor" uyarisi eklenir.

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

import numpy as np


@dataclass(frozen=True)
class ConfidenceThresholds:
    # HEURISTIK esikler (kalibre degil); ortusmeyen segment ICIR'ine uygulanmali.
    icir_high: float = 1.0     # bu ICIR'in ustunde sinyal guclu
    icir_medium: float = 0.5   # bu ICIR'in ustunde sinyal orta
    min_n_eff_high: int = 50   # "high" icin gereken min ortusmeyen tarih sayisi


def relative_thresholds(
    icirs, base: ConfidenceThresholds | None = None, floor: float = 0.0,
) -> ConfidenceThresholds:
    """Evren ici tertil esikleri: segment ICIR'i ust 1/3 -> high, orta 1/3 -> medium.

    Alt tertil yalniz ICIR'i AYRICA `base.icir_medium`'un (mutlak taban, 0.5) altindaysa `low`:
    medium esigi = min(alt tertil siniri, base.icir_medium). 2026-11-16 run 6: saf tertil likit
    hisselerin cogunu (THYAO, EREGL, KCHOL; ICIR 0.50-0.76) `low` yapiyordu; bu dilimin IC'si
    zayif ama anlamli (0.067), "guvenilmez" degil. `floor` mutlak alt sinir: ICIR < floor
    hep `low`. 3'ten az sonlu deger -> `base` (sabit esikler).
    """
    base = base or ConfidenceThresholds()
    v = np.asarray([float(x) for x in icirs if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 3:
        return base
    lo, hi = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    lo = float(np.nextafter(lo, np.inf))   # alt tertil siniri dahil low (esit degerlerde)
    medium = min(lo, base.icir_medium)      # low icin mutlak taban: tertil + ICIR < 0.5
    return ConfidenceThresholds(icir_high=float(max(hi, floor)), icir_medium=float(max(medium, floor)),
                                min_n_eff_high=base.min_n_eff_high)


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

    # --- likidite: etiketi dusurmez, ayri uyari (kalibrasyon 2026-11-16) ---
    if not tradable:
        warnings.append("Likidite düşük: işlem yapmak zor olabilir (fiyat kayması, "
                        "emir derinliği); sinyalin gücünden bağımsız bir uyarıdır.")
    # --- sert kapilar (her zaman low) ---
    if stale:
        warnings.append("Veri bayat: güncel fiyat yok.")
        return PeerConfidence("low", reasons, warnings)
    if not universe_ok or peer_label == "unknown":
        warnings.append("Cross-sectional evren yetersiz: göreli skor anlamsız.")
        return PeerConfidence("low", reasons, warnings)

    # --- sinyal gucu (segment ICIR) ---
    if segment_icir is None or segment_icir != segment_icir:  # None/NaN
        warnings.append("Segment IC bilinmiyor.")
        return PeerConfidence("low", reasons, warnings)

    if segment_icir >= thr.icir_high:
        if n_eff is not None and n_eff < thr.min_n_eff_high:
            reasons.append(
                f"Segment cross-sectional sinyali orta (ICIR {segment_icir:.2f}; "
                f"yüksek için yetersiz örnek).")
            warnings.append(
                f"n_eff={n_eff} < {thr.min_n_eff_high}: örtüşmeyen örnek az, "
                f"'high' verilmedi.")
            label = "medium"
        else:
            reasons.append(f"Segment cross-sectional sinyali güçlü (ICIR {segment_icir:.2f}).")
            label = "high"
    elif segment_icir >= thr.icir_medium:
        reasons.append(f"Segment cross-sectional sinyali orta (ICIR {segment_icir:.2f}).")
        label = "medium"
    else:
        reasons.append(f"Segment cross-sectional sinyali zayıf (ICIR {segment_icir:.2f}).")
        label = "low"
    return PeerConfidence(label, reasons, warnings)
