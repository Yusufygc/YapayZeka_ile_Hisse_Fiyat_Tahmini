# -*- coding: utf-8 -*-
"""Pooled DEEP model — embedding'li feedforward MLP (E2 Faz 9 → serving).

Pooled cross-sectional gorevde LightGBM'e ikinci (decorrelated) bacak. PoC
(`tools/e2_poc_deep_*`) bu mimarinin LGB'yi seed-bazinda gectigini ve ensemble'in
ikisini de gectigini gosterdi (IC +%18). Bu modul o MLP'yi production'a tasir.

Tasarim:
  - sklearn-vari `fit(X, y)` / `predict(X)` -> `pooled_oos` harness + serving
    `score_latest_universe` ile birebir uyumlu (GlobalPooledModel ile ayni protokol).
  - Kategorik kolonlar (symbol_id, sector_code) feature matrisinde POZISYON ile
    verilir (`cat_indices`); her biri ogrenilen embedding'e gider (LGB native
    categorical destegine adil karsilik; ham int magnitude DEGIL).
  - Sayisal kolonlar fit icinde TRAIN-ONLY standardize edilir (leakage yok).
  - Deterministik seed (torch + numpy); cok-seed ortalamasi nondeterminizmi
    ensemble katmaninda sondurur.

torch importu metod-ici (lazy): modul importu torch gerektirmez.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class TorchMLPConfig:
    hidden: Sequence[int] = (256, 128, 64)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 15
    batch_size: int = 8192
    predict_batch_size: int | None = None
    dataloader_num_workers: int = 0
    seed: int = 42
    max_embedding_dim: int = 50
    cat_indices: Sequence[int] = field(default_factory=tuple)
    cat_cardinalities: Sequence[int] = field(default_factory=tuple)


class TorchMLPModel:
    """Embedding'li feedforward MLP; sklearn-vari fit/predict.

    Parameters
    ----------
    cfg : TorchMLPConfig
        `cat_indices` ve `cat_cardinalities` ayni uzunlukta olmali. cardinalities
        global (CV-oncesi) vocab boyutudur -> leakage degil (yalniz embedding tablo
        boyutu).
    """

    def __init__(self, cfg: TorchMLPConfig | None = None) -> None:
        self.cfg = cfg or TorchMLPConfig()
        if len(self.cfg.cat_indices) != len(self.cfg.cat_cardinalities):
            raise ValueError(
                "cat_indices ve cat_cardinalities ayni uzunlukta olmali: "
                f"{len(self.cfg.cat_indices)} != {len(self.cfg.cat_cardinalities)}")
        self.cat_idx = list(self.cfg.cat_indices)
        self.cardinalities = [int(c) for c in self.cfg.cat_cardinalities]
        self.net = None
        self.mu = None
        self.sd = None
        self.num_idx: list[int] | None = None

    # --- ag insasi ---
    def _build(self, n_numeric: int):
        import torch.nn as nn

        emb_dims = [min(self.cfg.max_embedding_dim, (c + 1) // 2) for c in self.cardinalities]
        embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(self.cardinalities, emb_dims)])
        in_dim = n_numeric + sum(emb_dims)
        layers: list = []
        prev = in_dim
        for hid in self.cfg.hidden:
            layers += [nn.Linear(prev, hid), nn.ReLU(), nn.BatchNorm1d(hid),
                       nn.Dropout(self.cfg.dropout)]
            prev = hid
        layers += [nn.Linear(prev, 1)]
        return _MLPNet(embs, layers, self.cat_idx, self.num_idx)

    # --- batch yardimcilari ---
    def _as_float32_matrix(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X 2 boyutlu olmali, geldi: {X_arr.shape}")
        return np.ascontiguousarray(np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0))

    def _as_float32_target(self, y) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        return np.ascontiguousarray(np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0))

    def _standardize_batch(self, xb):
        import torch

        out = xb.clone()
        if self.num_idx:
            mu = torch.as_tensor(self.mu, dtype=out.dtype, device=out.device)
            sd = torch.as_tensor(self.sd, dtype=out.dtype, device=out.device)
            out[:, self.num_idx] = (out[:, self.num_idx] - mu) / sd
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, y):
        import torch

        from torch.utils.data import DataLoader, TensorDataset

        X = self._as_float32_matrix(X)
        y = self._as_float32_target(y)
        if len(X) != len(y):
            raise ValueError(f"X/y uzunluk uyumsuz: {len(X)} != {len(y)}")
        n_features = X.shape[1]
        for j in self.cat_idx:
            if j < 0 or j >= n_features:
                raise ValueError(f"cat_index {j} feature araliginda degil [0,{n_features})")

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.num_idx = [j for j in range(n_features) if j not in self.cat_idx]

        num = X[:, self.num_idx]
        self.mu = num.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.sd = num.std(axis=0, dtype=np.float64).astype(np.float32)
        self.sd[self.sd == 0] = 1.0

        self.net = self._build(len(self.num_idx))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        loss_fn = torch.nn.MSELoss()
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        bs = max(1, int(self.cfg.batch_size))
        generator = torch.Generator()
        generator.manual_seed(int(self.cfg.seed))
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=max(0, int(self.cfg.dataloader_num_workers)),
            generator=generator,
        )
        self.net.train()
        for _ in range(int(self.cfg.epochs)):
            for xb, yb in loader:
                if len(yb) < 2:  # BatchNorm tek-ornekte patlar
                    continue
                opt.zero_grad()
                loss = loss_fn(self.net(self._standardize_batch(xb)), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self.net is None:
            raise RuntimeError("TorchMLPModel egitilmedi (once fit cagir).")
        X = self._as_float32_matrix(X)
        bs = self.cfg.predict_batch_size if self.cfg.predict_batch_size is not None else self.cfg.batch_size
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X)),
            batch_size=max(1, int(bs)),
            shuffle=False,
            num_workers=max(0, int(self.cfg.dataloader_num_workers)),
        )
        outputs = []
        self.net.eval()
        with torch.no_grad():
            for (xb,) in loader:
                outputs.append(self.net(self._standardize_batch(xb)).numpy())
        if not outputs:
            return np.asarray([], dtype=float)
        return np.asarray(np.concatenate(outputs), dtype=float).ravel()


def _make_net_class():
    import torch
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self, embs, mlp_layers, cat_idx, num_idx):
            super().__init__()
            self.embs = embs
            self.mlp = nn.Sequential(*mlp_layers)
            self.cat_idx = cat_idx
            self.num_idx = num_idx

        def forward(self, x):
            num = x[:, self.num_idx]
            parts = [num]
            for j, emb in enumerate(self.embs):
                idx = x[:, self.cat_idx[j]].long().clamp(0, emb.num_embeddings - 1)
                parts.append(emb(idx))
            return self.mlp(torch.cat(parts, dim=1)).squeeze(-1)

    return _Net


def _MLPNet(embs, mlp_layers, cat_idx, num_idx):
    """Lazy torch nn.Module fabrikasi (modul importu torch gerektirmesin diye)."""
    return _make_net_class()(embs, mlp_layers, cat_idx, num_idx)


def make_mlp_factory(
    cat_indices: Sequence[int],
    cat_cardinalities: Sequence[int],
    cfg: TorchMLPConfig | None = None,
) -> Callable[[], TorchMLPModel]:
    """pooled_oos.evaluate_per_symbol icin taze-MLP fabrikasi."""
    base = cfg or TorchMLPConfig()

    def _factory() -> TorchMLPModel:
        c = TorchMLPConfig(**{**base.__dict__,
                              "cat_indices": tuple(cat_indices),
                              "cat_cardinalities": tuple(cat_cardinalities)})
        return TorchMLPModel(c)

    return _factory
