# -*- coding: utf-8 -*-
"""Aralik (band) hesabi icin normal z yardimcisi.

Kol-A forward interval kalibrasyonu (residual band / conformal) emekli edildi;
geriye yalnizca Kol-B serving'in fiyat bandi icin kullandigi ``z_for_level``
kaldi. Band ``p50 * exp(+/- z * sigma)`` gibi parametrik (normallik varsayimli)
hesaplanir; sonlu-orneklem kapsama garantisi YOKTUR.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Dict

# Standart iki-tarafli normal z degerleri (level -> z). Ara degerler NormalDist
# ile hesaplanir; bu tablo yalnizca hiz/okunabilirlik icindir.
_COMMON_Z: Dict[float, float] = {
    0.80: 1.2815515594965686,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
}


def _validate_level(level: float) -> float:
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError(f"level 0<level<1 olmali, alindi: {level}")
    return level


def z_for_level(level: float) -> float:
    """Iki-tarafli guven seviyesi icin normal z degeri doner (orn 0.9 -> 1.645)."""
    level = _validate_level(level)
    if level in _COMMON_Z:
        return _COMMON_Z[level]
    return float(NormalDist().inv_cdf((1.0 + level) / 2.0))
