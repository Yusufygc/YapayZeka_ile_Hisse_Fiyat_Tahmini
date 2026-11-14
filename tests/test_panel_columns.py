# -*- coding: utf-8 -*-
"""K9: panel kolon siniflandirmasi tek yerde; uc tuketici ayni tanimi kullanir."""

import numpy as np
import pandas as pd

import src.models.global_pooled_model as gpm
import src.validation.pooled_oos as oos
from src.data.panel_columns import PANEL_NON_FEATURE_COLS, is_non_feature


def test_target_variants_and_panel_columns_are_never_features():
    for col in ["target", "target_cs", "target_rank", "target_date", "symbol", "Date",
                "Close", "sector", "sector_code", "symbol_id", "liq_log", "vol"]:
        assert is_non_feature(col), col
    for col in ["feat_a", "RSI_14", "SMA_14_rel", "feat_a_csr"]:
        assert not is_non_feature(col), col


def test_all_consumers_share_the_single_definition():
    assert oos._is_non_feature is is_non_feature
    assert gpm._is_non_feature is is_non_feature


def test_auto_feature_cols_excludes_raw_close_level():
    """Eskiden pooled_oos varsayilan yolu 'Close' (ham fiyat seviyesi) kolonunu ozellik sayiyordu."""
    panel = pd.DataFrame({
        "symbol": ["A"], "Date": [pd.Timestamp("2021-01-04")], "Close": [10.0],
        "feat_a": [0.1], "target": [0.0], "target_cs": [0.0], "target_date": [pd.Timestamp("2021-01-11")],
        "liq_log": [1.0], "vol": [0.1], "sector_code": [0], "symbol_id": [0],
    })
    assert oos._auto_feature_cols(panel) == ["feat_a"]
    _, feats, cat = gpm.build_pooled_features(panel.assign(sector="Energy"))
    assert "Close" not in feats and "Close" in PANEL_NON_FEATURE_COLS
    assert feats[-2:] == ["symbol_id", "sector_code"] and len(cat) == 2
    assert not any(c.startswith("target") for c in feats)
    _ = np  # noqa: F841
