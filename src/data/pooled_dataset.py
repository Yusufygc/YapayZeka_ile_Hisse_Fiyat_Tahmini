# -*- coding: utf-8 -*-
"""Kol-B egitim/OOS/skorlama veri seti kurulumu (tek yer).

`tools/e2_faz5_nightly_scoring.py` ve `tools/e2_kolb_validity_report.py` ayni akisi
paylasir: yukle (hedefi olmayan son h satir DAHIL) -> etiketli alt kume + cross-sectional
hedef -> cross-sectional ozellikler (tarih-ici, hedefsiz) -> egitim/OOS paneli.

Skorlama (`full`) en guncel PIYASA gununu icerir; egitim/OOS (`aug`) yalniz etiketlidir.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from src.data.cross_sectional import (
    add_cross_sectional_features,
    add_cross_sectional_target,
)
from src.data.pooled_loader import PooledLoaderConfig, PooledPanelLoader
from src.models.global_pooled_model import build_pooled_features


@dataclass
class PooledDataset:
    full: pd.DataFrame        # ozellik + cs-ozellik, etiketsiz satirlar dahil (skorlama)
    labeled: pd.DataFrame     # etiketli yukleyici paneli + target_cs (segment/likidite referansi)
    aug: pd.DataFrame         # egitim/OOS paneli (etiketli + target_cs + cs-ozellikler)
    feats: list[str]
    cat_idx: list[int]
    latest_market_day: pd.Timestamp
    loader_report: dict
    # survivorship_status, fallback kullanan semboller (K7), karantina gun sayilari (K8)
    loader_meta: dict


def build_pooled_dataset(
    loader_cfg: PooledLoaderConfig, min_names: int = 15, limit: int = 0,
) -> PooledDataset:
    """Paneli kur. `limit>0` -> ilk N sembol (smoke/test)."""
    loader = PooledPanelLoader(replace(loader_cfg, keep_unlabeled=True))
    if limit > 0:
        syms = loader._symbols()[:limit]
        loader._symbols = lambda: syms
    full = loader.load()
    if full.empty:
        raise ValueError("pooled panel bos (veri/evren/min_rows kontrol edin)")
    labeled = add_cross_sectional_target(full[full["target"].notna()], min_names=min_names)

    _, f0, _ = build_pooled_features(full)
    full, _ = add_cross_sectional_features(
        full, [c for c in f0 if c not in ("symbol_id", "sector_code")])
    full_aug, feats, cat_idx = build_pooled_features(full)
    assert not any(c.startswith("target") for c in feats), "target leak!"
    aug = full_aug.merge(
        labeled[["symbol", "Date", "target_cs"]], on=["symbol", "Date"], how="inner",
    ).reset_index(drop=True)
    return PooledDataset(
        full=full_aug, labeled=labeled, aug=aug, feats=feats, cat_idx=cat_idx,
        latest_market_day=pd.Timestamp(full["Date"].max()), loader_report=dict(loader.report),
        loader_meta={
            "survivorship_status": loader.survivorship_status,
            "fallback_symbols": sorted(
                s for s, d in loader.source_dir.items() if d != loader_cfg.data_dir),
            "quarantined_days": dict(loader.quarantined_days),
        },
    )
