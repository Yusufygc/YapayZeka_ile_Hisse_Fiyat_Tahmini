# -*- coding: utf-8 -*-
"""Stage 1/6 code-review fix'leri icin regresyon testleri (2026-05-31).

Kapsam:
  - compute_psi ham OHLCV/fiyat sutunlarini disliyor (psi_high yanlis tetikleme).
  - prune_correlated_features fit_df ile korelasyonu egitim diliminde hesaplar
    (feature-secim leakage onlemi); dusurme tum frame'e uygulanir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.quality import compute_psi
from src.features.correlation_pruning import prune_correlated_features


def test_compute_psi_excludes_raw_price_columns():
    n = 300
    rng = np.random.default_rng(0)
    # Ham Close: trend (non-stationary) -> train/holdout dagilimi cok farkli.
    train = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "Close": np.linspace(10.0, 50.0, n),          # trend
        "Volume": np.linspace(1e6, 5e6, n),           # trend
        "log_return": rng.normal(0, 0.01, n),         # durağan
    })
    holdout = pd.DataFrame({
        "Date": pd.date_range("2025-03-01", periods=n, freq="B"),
        "Close": np.linspace(60.0, 120.0, n),         # cok farkli seviye
        "Volume": np.linspace(6e6, 9e6, n),
        "log_return": rng.normal(0, 0.01, n),
    })
    psi = compute_psi(train, holdout)
    # Ham fiyat/hacim PSI'a girmemeli (trend yaniltici yuksek deger verir).
    assert "Close" not in psi
    assert "Volume" not in psi
    # Durağan feature degerlendirilir.
    assert "log_return" in psi


def test_prune_correlated_fit_df_uses_training_slice_only():
    n = 200
    base = np.linspace(0.0, 1.0, n)
    # train diliminde A ve B mukemmel korele; holdout tail'de B bozuluyor.
    a = base.copy()
    b = base.copy()
    b[-40:] = base[-40:][::-1]  # son 40 satirda korelasyon kirilir
    df = pd.DataFrame({"A": a, "B": b, "C": np.sin(base * 6)})

    fit_df = df.iloc[:-40]  # holdout tail'i corr fit'inden cikar
    pruned_df, kept, report = prune_correlated_features(
        df, ["A", "B", "C"], threshold=0.95, fit_df=fit_df
    )
    # Egitim diliminde A~B mukemmel korele -> biri dusurulur.
    assert ("A" in kept) != ("B" in kept)
    assert "C" in kept
    assert report["dropped_features"]


def test_prune_correlated_full_frame_default_backcompat():
    # fit_df=None -> eski davranis (tum frame).
    n = 120
    base = np.linspace(0.0, 1.0, n)
    df = pd.DataFrame({"A": base, "B": base * 2.0 + 1.0, "C": np.cos(base * 4)})
    _, kept, report = prune_correlated_features(df, ["A", "B", "C"], threshold=0.95)
    assert ("A" in kept) != ("B" in kept)
    assert "C" in kept
