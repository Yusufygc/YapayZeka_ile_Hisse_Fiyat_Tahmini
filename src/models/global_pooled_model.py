# -*- coding: utf-8 -*-
"""Global pooled (conditioned) model for the E2 epic — Faz 3.

Tum hisselerin havuzlanmis satirlari uzerinde TEK model egitilir; sembol
ayrimi `conditioning` ozellikleriyle yapilir (sector, symbol_id kategorik;
liq_log, vol sayisal). Serving per-symbol: bir sembolun satirlari verilince
model o sembole ozgu kosullarla tahmin uretir.

Tasarim: docs/wiki/e2-faz2-pooled-cv-design.md (Faz 3 bolumu).

Mimari karar — LightGBM (native API):
  - Kategorik `sector_code`/`symbol_id` dogal destek (one-hot patlamasi yok,
    ~600 sembol yuksek-kardinaliteyi tasiyabilir).
  - Olcekleme gereksiz, eksik-deger toleransli, panelde hizli.
  - Champion/challenger uyumlu: periyodik batch retrain, sorgu-ani egitim yok.
  - NN symbol embedding (opsiyonel) Faz 4 alani; Faz 3 = guclu tablo baseline.

`build_pooled_features` paneli kosullandirma ozellikleriyle zenginlestirir ve
(feature_cols, cat_indices) dondurur; bunlar dogrudan `evaluate_per_symbol`
(pooled_oos) harness'ine beslenir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.data.panel_columns import is_non_feature
from src.data.pooled_matrix import as_float32_matrix, as_float32_vector

# panel kolon siniflandirmasi: tek tanim src/data/panel_columns.py (target* ASLA ozellik degil).
_is_non_feature = is_non_feature

# kosullandirma sirasi sabit: once sayisal, sonra kategorik (cat_indices sonda).
_NUMERIC_COND = ["liq_log", "vol"]
_CATEGORICAL_COND = ["symbol_id", "sector_code"]


def build_pooled_features(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Paneli kosullandirma ozellikleriyle zenginlestir.

    Returns
    -------
    (panel_aug, feature_cols, cat_indices)
        panel_aug   : `sector_code` (stabil int) eklenmis kopya
        feature_cols: base ozellikler + [liq_log, vol, symbol_id, sector_code]
        cat_indices : feature_cols icindeki kategorik kolon POZISYONLARI
    """
    df = panel.copy()
    # sector -> stabil int kod (sirali benzersiz). symbol_id loader'da zaten int.
    if "sector" in df.columns:
        cats = sorted(str(s) for s in df["sector"].fillna("Unknown").unique())
        code = {s: i for i, s in enumerate(cats)}
        df["sector_code"] = df["sector"].fillna("Unknown").astype(str).map(code).astype(int)
    else:
        df["sector_code"] = 0
    if "symbol_id" not in df.columns:
        codes = {s: i for i, s in enumerate(sorted(df["symbol"].unique()))}
        df["symbol_id"] = df["symbol"].map(codes).astype(int)

    base_feats = [
        c for c in df.columns
        if not _is_non_feature(c) and pd.api.types.is_numeric_dtype(df[c])
    ]
    num_cond = [c for c in _NUMERIC_COND if c in df.columns]
    cat_cond = [c for c in _CATEGORICAL_COND if c in df.columns]
    feature_cols = base_feats + num_cond + cat_cond
    cat_indices = [feature_cols.index(c) for c in cat_cond]
    return df, feature_cols, cat_indices


@dataclass
class GlobalPooledConfig:
    num_boost_round: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_data_in_leaf: int = 100      # panel buyuk; asiri-uyumu engelle
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    max_cat_threshold: int = 64
    seed: int = 42
    cat_indices: Sequence[int] = field(default_factory=tuple)
    # X icindeki `symbol_id` kolon pozisyonu (build_pooled_features'in feature_cols
    # sirasindan). `use_symbol_id=False` iken bu kolon fit/predict'te dusurulur.
    symbol_id_index: int | None = None


class GlobalPooledModel:
    """sklearn-vari pooled LightGBM (fit/predict). pooled_oos harness uyumlu.

    use_symbol_id (varsayilan True = uretim davranisi DEGISMEZ): `symbol_id`
    yuksek-kardinalite kategorik oldugundan model sembol-basina ortalamayi
    ezberleyebilir (cross-sectional siralamayi sisirebilir). False ise
    `cfg.symbol_id_index` kolonu fit/predict'ten dusurulur (ablasyon icin).
    Not: False iken booster'in ozellik sayisi X'ten 1 eksiktir; XAI'ya X verirken
    ayni kolonu dusurun. Oneri: uretimi degistirmeden once use_symbol_id=False ile
    ortusmeyen ICIR karsilastirmasi yapin (tools/ ablasyonu).
    """

    def __init__(self, cfg: GlobalPooledConfig | None = None,
                 use_symbol_id: bool = True) -> None:
        self.cfg = cfg or GlobalPooledConfig()
        self.use_symbol_id = bool(use_symbol_id)
        if not self.use_symbol_id and self.cfg.symbol_id_index is None:
            raise ValueError(
                "use_symbol_id=False icin cfg.symbol_id_index (kolon pozisyonu) gerekli.")
        self.booster = None

    def _prepare(self, X) -> tuple[np.ndarray, list[int]]:
        """Float32 matris + (gerekirse symbol_id dusurulmus) kategorik indeksler."""
        X = as_float32_matrix(X)
        cat = list(self.cfg.cat_indices)
        if self.use_symbol_id:
            return X, cat
        drop = int(self.cfg.symbol_id_index)
        X = np.delete(X, drop, axis=1)
        cat = [c - 1 if c > drop else c for c in cat if c != drop]
        return X, cat

    def _params(self) -> dict:
        c = self.cfg
        return {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": c.learning_rate,
            "num_leaves": c.num_leaves,
            "min_data_in_leaf": c.min_data_in_leaf,
            "feature_fraction": c.feature_fraction,
            "bagging_fraction": c.bagging_fraction,
            "bagging_freq": c.bagging_freq,
            "lambda_l2": c.lambda_l2,
            "max_cat_threshold": c.max_cat_threshold,
            "seed": c.seed,
            "bagging_seed": c.seed,
            "feature_fraction_seed": c.seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,           # belirlenebilirlik (test) icin
            "verbosity": -1,
        }

    def fit(self, X, y):
        import lightgbm as lgb

        X, cat = self._prepare(X)
        y = as_float32_vector(y)
        dset = lgb.Dataset(
            X, label=y,
            categorical_feature=cat if cat else "auto",
            free_raw_data=False,
        )
        self.booster = lgb.train(
            self._params(), dset,
            num_boost_round=int(self.cfg.num_boost_round),
        )
        return self

    def predict(self, X):
        if self.booster is None:
            raise RuntimeError("GlobalPooledModel egitilmedi.")
        return np.asarray(self.booster.predict(self._prepare(X)[0]), dtype=float)


def make_global_model_factory(
    cat_indices: Sequence[int],
    cfg: GlobalPooledConfig | None = None,
    use_symbol_id: bool = True,
    symbol_id_index: int | None = None,
) -> Callable[[], GlobalPooledModel]:
    """pooled_oos.evaluate_per_symbol icin taze-model fabrikasi dondurur.

    use_symbol_id=False icin `symbol_id_index` (veya cfg.symbol_id_index) gerekir.
    """
    base = cfg or GlobalPooledConfig()

    def _factory() -> GlobalPooledModel:
        extra = {"cat_indices": tuple(cat_indices)}
        if symbol_id_index is not None:
            extra["symbol_id_index"] = int(symbol_id_index)
        c = GlobalPooledConfig(**{**base.__dict__, **extra})
        return GlobalPooledModel(c, use_symbol_id=use_symbol_id)

    return _factory
