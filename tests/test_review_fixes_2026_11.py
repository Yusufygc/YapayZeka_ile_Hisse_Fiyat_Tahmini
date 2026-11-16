# -*- coding: utf-8 -*-
"""Kod inceleme bulgulari (2026-11-16) icin regresyon testleri (Kol-B/serving/veri)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
#  7. Gecelik cikis: terk edilen cagri varsa os._exit                         #
# --------------------------------------------------------------------------- #
def test_call_with_deadline_counts_abandoned_calls():
    import threading

    from src.data import market_providers as mp

    mp.reset_abandoned_calls()
    release = threading.Event()
    with pytest.raises(mp.ProviderTimeout):
        mp.call_with_deadline(lambda: release.wait(5.0), 0.05, "unit")
    release.set()
    assert mp.abandoned_call_count() == 1
    mp.reset_abandoned_calls()


def test_exit_after_run_forces_exit_only_when_calls_abandoned(monkeypatch):
    import logging

    from src.serving import nightly

    shutdowns: list[bool] = []
    monkeypatch.setattr(logging, "shutdown", lambda: shutdowns.append(True))

    assert nightly.should_force_exit(0) is False
    assert nightly.should_force_exit(2) is True

    # Terk edilen cagri yok -> normal SystemExit(rc), os._exit cagrilmaz.
    called: list[int] = []
    with pytest.raises(SystemExit) as excinfo:
        nightly.exit_after_run(3, exit_fn=called.append, abandoned_calls=0)
    assert excinfo.value.code == 3 and called == []

    # Terk edilen cagri var -> exit_fn(rc) cagrilir, SystemExit firlatilmaz.
    nightly.exit_after_run(5, exit_fn=called.append, abandoned_calls=1)
    assert called == [5] and shutdowns == [True]


# --------------------------------------------------------------------------- #
#  8. PowerShell: $ErrorActionPreference=Stop altinda Write-Error + exit      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["nightly_serving.ps1", "register_nightly_task.ps1"])
def test_ps1_error_paths_do_not_use_write_error_before_exit(name):
    import re
    from pathlib import Path

    raw = (Path(__file__).resolve().parents[1] / "scripts" / name).read_bytes()
    assert all(b < 128 for b in raw), "betik ASCII kalmali"
    text = raw.decode("ascii")
    # Write-Error, Stop tercihinde sonlandirici hata atar; `exit N` hic calismaz.
    assert not re.search(r"Write-Error[^\r\n]*\r?\n\s*exit\s+\d", text)
