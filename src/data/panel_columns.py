# -*- coding: utf-8 -*-
"""Pooled panel kolon siniflandirmasi — TEK tanim (K9).

Eskiden uc yerde (pooled_loader, pooled_oos, global_pooled_model) ayri/farkli
kume vardi; `pooled_oos` varsayilan yolunda ham duzeltilmis fiyat seviyesi (`Close`)
ozellik olarak sizabiliyordu. Yeni kolon eklemek artik tek dosya.
"""
from __future__ import annotations

# FeaturePipeline CIKTISINDA ozellik OLMAYAN ham OHLCV kolonlari (yukleyici icin).
PIPELINE_RAW_COLS: frozenset[str] = frozenset({
    "Date", "Close", "Open", "High", "Low", "Volume",
    "Adj Close", "Düzeltilmiş_Kapanış", "Hacim",
})

# PANEL'de (yukleyici ciktisi + CV/OOS alanlari) ozellik OLMAYAN kolonlar.
PANEL_NON_FEATURE_COLS: frozenset[str] = frozenset({
    "symbol", "Date", "Close", "target_date", "sector", "symbol_id",
    "liq_log", "vol", "sector_code",
})


def is_non_feature(col: str) -> bool:
    """Kolon model girdisi OLMAMALI mi? `target*` varyantlari ASLA ozellik degildir (sizinti)."""
    return col in PANEL_NON_FEATURE_COLS or col == "target" or col.startswith("target_")
