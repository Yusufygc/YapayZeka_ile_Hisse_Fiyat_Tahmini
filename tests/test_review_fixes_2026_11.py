# -*- coding: utf-8 -*-
"""Kod inceleme bulgulari (2026-11-16) icin regresyon testleri (Kol-B/serving/veri)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import FeaturePipeline


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


# --------------------------------------------------------------------------- #
#  10. market_momentum_60d piyasa geneli (makro takviminde, tarihe gore)      #
# --------------------------------------------------------------------------- #
def _momentum_stock(dates, seed):
    rng = np.random.default_rng(seed)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    return pd.DataFrame({"Date": dates, "Close": close})


def test_market_momentum_is_identical_across_stocks_with_missing_days():
    n = 150
    dates = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(3)
    macro = pd.DataFrame(
        {"Date": dates, "BIST100_Return": rng.normal(0.0005, 0.012, n)}
    )
    full = _momentum_stock(dates, 1)
    gappy_dates = dates.delete([20, 45, 46, 80, 100, 101, 102])  # eksik islem gunleri
    gappy = _momentum_stock(gappy_dates, 2)

    fp = FeaturePipeline(lag_feature_count=0)
    m_full = fp._merge_macro(full, macro).set_index("Date")
    m_gappy = fp._merge_macro(gappy, macro).set_index("Date")

    common = m_gappy.index.intersection(m_full.index)
    a = m_full.loc[common, "market_momentum_60d"].dropna()
    b = m_gappy.loc[a.index, "market_momentum_60d"]
    assert len(a) > 30
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-12)

    # Beklenen: BIST100 getirilerinin makro takvimindeki 60g kumulatifi.
    expected = np.expm1(np.log1p(macro.set_index("Date")["BIST100_Return"]).rolling(60).sum())
    np.testing.assert_allclose(
        a.to_numpy(), expected.loc[a.index].to_numpy(), rtol=0, atol=1e-12
    )
    # relative_to_market_60d = momentum_60d - market_momentum_60d korunur.
    rel = m_gappy["relative_to_market_60d"] - (
        m_gappy["momentum_60d"] - m_gappy["market_momentum_60d"]
    )
    assert rel.dropna().abs().max() < 1e-12


def test_recompute_momentum_keeps_market_history_and_refreshes_last_row():
    n = 150
    dates = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(5)
    macro = pd.DataFrame({"Date": dates, "BIST100_Return": rng.normal(0.0, 0.01, n)})
    gappy = _momentum_stock(dates.delete([30, 31, 70]), 4)
    fp = FeaturePipeline(lag_feature_count=0)
    merged = fp._merge_macro(gappy, macro).dropna().reset_index(drop=True)
    history = merged["market_momentum_60d"].copy()

    stale = merged.copy()
    stale.loc[stale.index[-1], "BIST100_Return"] = 0.0
    out = fp.recompute_momentum_features(stale)
    pd.testing.assert_series_equal(
        out["market_momentum_60d"].iloc[:-1], history.iloc[:-1], check_names=False
    )
    assert out["relative_to_market_60d"].iloc[-1] == pytest.approx(
        out["momentum_60d"].iloc[-1] - out["market_momentum_60d"].iloc[-1]
    )
