# -*- coding: utf-8 -*-
"""interval_calibration.z_for_level birim testleri (geriye kalan tek yardimci)."""
from __future__ import annotations

import pytest

from src.forecasting.interval_calibration import z_for_level


def test_z_for_level_known_values():
    assert z_for_level(0.8) == pytest.approx(1.2816, abs=1e-3)
    assert z_for_level(0.9) == pytest.approx(1.6449, abs=1e-3)
    assert z_for_level(0.95) == pytest.approx(1.9600, abs=1e-3)


def test_z_for_level_interpolated_value():
    assert z_for_level(0.85) == pytest.approx(1.4395, abs=1e-3)


def test_z_for_level_invalid():
    with pytest.raises(ValueError):
        z_for_level(0.0)
    with pytest.raises(ValueError):
        z_for_level(1.0)
