# -*- coding: utf-8 -*-
"""Pooled ENSEMBLE — LightGBM + cok-seed DEEP-MLP (E2 Faz 9 → serving).

PoC bulgusu (tools/e2_poc_deep_ensemble.py, full evren): LGB ICIR 1.553, MLP
3-seed avg 1.645, ENSEMBLE 1.665-1.670 (LGB'ye karsi IC +%18). Ensemble her iki
bileseni gecer cunku hatalari decorrelated. Bu modul ensemble'i serving-uyumlu
tek model nesnesi olarak paketler.

Blend = tarih-ici pct-rank agirlikli ortalama:
    score = w_lgb * pctrank(lgb_pred) + (1 - w_lgb) * pctrank(mlp_avg_pred)
pct-rank `predict`'e verilen X SATIRLARI uzerinden hesaplanir. Serving'de
`score_latest_universe` predict'i TEK tarihin evren satirlariyla cagirir ->
tarih-ici cross-sectional rank dogru. (Cok-tarihli toplu predict bu nedenle
desteklenmez; serving akisi tek-kesit cagirir.)

MLP nondeterminizmini cok-seed (varsayilan 3) ortalamasi sondurur. Determinizm
icin segment ICIR/confidence kalibrasyonu LGB-only kalir (bkz. nightly tool);
ensemble yalniz final skorlamada kullanilir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.pooled_matrix import as_float32_matrix, as_float32_vector
from src.models.global_pooled_model import GlobalPooledConfig, GlobalPooledModel
from src.models.torch_mlp_model import TorchMLPConfig, TorchMLPModel


@dataclass
class EnsemblePooledConfig:
    blend_weight_lgb: float = 0.5          # 50/50 (PoC: en kararli, %IC>0 en yuksek)
    mlp_seeds: Sequence[int] = (42, 7, 123)
    lgb: GlobalPooledConfig = field(default_factory=GlobalPooledConfig)
    mlp: TorchMLPConfig = field(default_factory=TorchMLPConfig)
    cat_indices: Sequence[int] = field(default_factory=tuple)
    cat_cardinalities: Sequence[int] = field(default_factory=tuple)


def _pct_rank(values: np.ndarray) -> np.ndarray:
    """Tarih-ici yuzde-rank [0,1] (scipy'siz). NaN -> 0.5 notr."""
    s = pd.Series(values, dtype=float)
    r = s.rank(method="average", pct=True)
    return r.fillna(0.5).to_numpy()


class EnsemblePooledModel:
    """LGB + cok-seed MLP, tarih-ici rank-blend. sklearn-vari fit/predict.

    `predict` TEK cross-section (bir tarihin evreni) bekler — serving sozlesmesi.
    """

    def __init__(self, cfg: EnsemblePooledConfig | None = None) -> None:
        self.cfg = cfg or EnsemblePooledConfig()
        w = float(self.cfg.blend_weight_lgb)
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"blend_weight_lgb [0,1] olmali, geldi: {w}")
        if not self.cfg.mlp_seeds:
            raise ValueError("en az bir mlp_seed gerekli")
        self.lgb: GlobalPooledModel | None = None
        self.mlps: list[TorchMLPModel] = []

    def fit(self, X, y):
        X = as_float32_matrix(X)
        y = as_float32_vector(y)

        lgb_cfg = GlobalPooledConfig(**{**self.cfg.lgb.__dict__,
                                        "cat_indices": tuple(self.cfg.cat_indices)})
        self.lgb = GlobalPooledModel(lgb_cfg).fit(X, y)

        self.mlps = []
        for sd in self.cfg.mlp_seeds:
            mlp_cfg = TorchMLPConfig(**{**self.cfg.mlp.__dict__,
                                        "seed": int(sd),
                                        "cat_indices": tuple(self.cfg.cat_indices),
                                        "cat_cardinalities": tuple(self.cfg.cat_cardinalities)})
            self.mlps.append(TorchMLPModel(mlp_cfg).fit(X, y))
        return self

    def predict(self, X):
        if self.lgb is None or not self.mlps:
            raise RuntimeError("EnsemblePooledModel egitilmedi (once fit cagir).")
        raw_shape = np.shape(X)
        if np.ndim(X) != 2:
            raise ValueError(f"EnsemblePooledModel.predict 2D X bekler, geldi: {raw_shape}")
        X = as_float32_matrix(X)
        if len(X) < 2:
            raise ValueError(
                "EnsemblePooledModel.predict tek tarihli cross-section icin en az "
                "2 satir bekler; tek sembol rank-blend anlamsizdir."
            )
        lgb_pred = np.asarray(self.lgb.predict(X), dtype=float).ravel()
        mlp_stack = np.vstack([np.asarray(m.predict(X), dtype=float).ravel() for m in self.mlps])
        mlp_pred = mlp_stack.mean(axis=0)

        w = float(self.cfg.blend_weight_lgb)
        return w * _pct_rank(lgb_pred) + (1.0 - w) * _pct_rank(mlp_pred)
