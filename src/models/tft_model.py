# -*- coding: utf-8 -*-
"""
tft_model.py — Temporal Fusion Transformer (Pure PyTorch Implementation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Makaledeki (Lim et al., 2021) orijinal TFT mimarisini uygular:

  • Variable Selection Networks (VSN)   — hangi özelliğin önemli olduğunu öğrenir
  • Gated Residual Networks    (GRN)   — seçici bilgi akışı
  • Interpretable Multi-Head Attention — zaman adımları arası uzun vadeli bağımlılıklar
  • Quantile Loss (P10, P50, P90)      — belirsizlik tahminleri

Arayüz:
  BaseModel'i devralır → pipeline geri kalanı hiç değişmez.
  train(X_np, y_np)  : numpy girdi, arka planda PyTorch DataLoader'a çevrilir
  predict(X_np)      : P50 (medyan) tahminini numpy olarak döndürür
  predict_quantiles  : P10 / P50 / P90 üçlüsünü döndürür (belirsizlik analizi)
  save / load        : torch.save / torch.load ile model ağırlıkları + config
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple

from .base_model import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Quantile Loss
# ─────────────────────────────────────────────────────────────────────────────
class QuantileLoss(nn.Module):
    """
    Pinball / Quantile Loss.
    q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)
    Üç kuantil için toplam kayıp döndürür.
    """
    def __init__(self, quantiles: List[float] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds  : (batch, num_quantiles)
        # target : (batch,) veya (batch, 1)
        target = target.view(-1)          # (batch,) — her kuantil için aynı hedef
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target - preds[:, i]    # (batch,) - (batch,) ✓
            losses.append(torch.max(q * err, (q - 1) * err))
        return torch.stack(losses, dim=1).sum(dim=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Gated Linear Unit (GLU)
# ─────────────────────────────────────────────────────────────────────────────
class GLU(nn.Module):
    """Gated Linear Unit: x₁ ⊙ σ(x₂)"""
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        x1, x2 = out.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


# ─────────────────────────────────────────────────────────────────────────────
# Gated Residual Network (GRN)
# ─────────────────────────────────────────────────────────────────────────────
class GRN(nn.Module):
    """
    Gated Residual Network.
    input_dim → hidden_dim → output_dim  (+ residual + LayerNorm)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        # Projeksiyon (input_dim ≠ output_dim ise)
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        h = self.fc_out(h)
        return self.norm(h + residual)


# ─────────────────────────────────────────────────────────────────────────────
# Variable Selection Network (VSN)
# ─────────────────────────────────────────────────────────────────────────────
class VariableSelectionNetwork(nn.Module):
    """
    Her değişken için bağımsız GRN + ağırlık GRN → softmax seçim.
    Girdi  : (batch, [time,] num_vars)
    Çıktı  : (batch, [time,] d_model) — ağırlıklı birleşim
             (batch, [time,] num_vars) — seçim ağırlıkları (yorumlanabilirlik)
    """
    def __init__(self, num_vars: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.var_grns = nn.ModuleList([
            GRN(1, d_model, d_model, dropout) for _ in range(num_vars)
        ])
        self.weight_grn = GRN(num_vars, d_model, num_vars, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, time, num_vars)  veya  (batch, num_vars)
        # --- Per-variable transform ---
        per_var = torch.stack(
            [grn(x[..., i : i + 1]) for i, grn in enumerate(self.var_grns)],
            dim=-2,
        )
        # per_var: (batch, [time,] num_vars, d_model)

        # --- Seçim ağırlıkları ---
        weights = torch.softmax(self.weight_grn(x), dim=-1)
        # weights: (batch, [time,] num_vars)

        # --- Ağırlıklı topla ---
        output = (weights.unsqueeze(-1) * per_var).sum(dim=-2)
        # output: (batch, [time,] d_model)
        return output, weights


# ─────────────────────────────────────────────────────────────────────────────
# Interpretable Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────
class InterpretableMultiHeadAttention(nn.Module):
    """
    TFT'nin yorumlanabilir dikkat mekanizması.
    Her kafa ayrı V projeksiyon yerine paylaşımlı V kullanır.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model, num_heads'e tam bölünmeli"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_head)   # paylaşımlı V
        self.W_o = nn.Linear(self.d_head, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = query.shape

        Q = self.W_q(query).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(key).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(value)  # (B, T, d_head)  — paylaşımlı

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Tüm kafaların dikkat ağırlıklarının ortalaması → yorumlanabilirlik
        attn_mean = attn.mean(dim=1)   # (B, T, T)

        # V tekrar: (B, H, T, d_head) için V'yi genişlet
        V_exp = V.unsqueeze(1).expand(B, self.num_heads, T, self.d_head)
        context = torch.matmul(attn, V_exp)                           # (B, H, T, d_head)
        context = context.mean(dim=1)                                  # (B, T, d_head)
        output = self.W_o(context)                                     # (B, T, d_model)

        return output, attn_mean


# ─────────────────────────────────────────────────────────────────────────────
# TFT Ağı (PyTorch Module)
# ─────────────────────────────────────────────────────────────────────────────
class _TFTNetwork(nn.Module):
    """
    İç PyTorch modülü. TFTModel sınıfı tarafından sarmalanır.

    Mimari:
      Input (B, T, F)
        → VSN              — özellik seçimi
        → LSTM Encoder     — yerel zaman bağımlılıkları
        → GRN              — encoder çıktısı dönüşümü
        → IMHA             — uzun vadeli bağımlılıklar
        → GRN + Add & Norm
        → Dense(num_q)     — son zaman adımından kuantil tahminleri
    """
    def __init__(
        self,
        num_features: int,
        d_model:      int   = 64,
        num_heads:    int   = 4,
        lstm_layers:  int   = 2,
        dropout:      float = 0.1,
        quantiles:    List[float] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model      = d_model
        self.quantiles    = list(quantiles)

        # 1. Variable Selection
        self.vsn = VariableSelectionNetwork(num_features, d_model, dropout)

        # 2. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = d_model,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # 3. Post-sequence GRN
        self.post_seq_grn = GRN(d_model, d_model, d_model, dropout)

        # 4. Interpretable Multi-Head Attention
        self.attn = InterpretableMultiHeadAttention(d_model, num_heads, dropout)

        # 5. Post-attention GRN + Add & Norm
        self.post_attn_grn = GRN(d_model, d_model, d_model, dropout)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # 6. Çıkış katmanı
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, len(quantiles)),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, time_steps, num_features)
        Döndürür:
          preds        : (batch, num_quantiles)   — son adım tahmini
          var_weights  : (batch, time_steps, num_features) — VSN ağırlıkları
        """
        # 1. VSN
        vsn_out, var_weights = self.vsn(x)            # (B, T, d_model)

        # 2. LSTM
        lstm_out, _ = self.lstm(vsn_out)              # (B, T, d_model)

        # 3. Post-seq GRN
        enc = self.post_seq_grn(lstm_out)             # (B, T, d_model)

        # 4. Self-Attention
        attn_out, _ = self.attn(enc, enc, enc)        # (B, T, d_model)

        # 5. Artık bağlantı + Norm
        out = self.post_attn_norm(
            self.post_attn_grn(attn_out) + enc
        )                                              # (B, T, d_model)

        # 6. Son zaman adımı → kuantil çıktıları
        last = out[:, -1, :]                          # (B, d_model)
        preds = self.output_fc(last)                  # (B, num_quantiles)

        return preds, var_weights


# ─────────────────────────────────────────────────────────────────────────────
# TFTModel — BaseModel Uyumlu Sarmalayıcı
# ─────────────────────────────────────────────────────────────────────────────
class TFTModel(BaseModel):
    """
    PyTorch tabanlı Temporal Fusion Transformer.

    BaseModel arayüzünü korur:
      train(X_np, y_np)      — numpy array, dahili olarak DataLoader'a dönüştürülür
      predict(X_np)          — P50 (medyan) tahminini numpy olarak döndürür
      predict_quantiles(X_np) — (P10, P50, P90) üçlüsünü döndürür
      save(path)  /  load(path) — torch.save / torch.load

    Args:
        d_model      : Gizli katman boyutu (default 64)
        num_heads    : Dikkat kafası sayısı (default 4)
        lstm_layers  : LSTM yığın derinliği (default 2)
        dropout      : Dropout oranı (default 0.1)
        epochs       : Maksimum eğitim turu (default 80)
        batch_size   : Mini-batch boyutu (default 32)
        learning_rate: Adam öğrenme hızı (default 1e-3)
        patience     : Early stopping sabrı (default 15)
        quantiles    : Tahmin kuantilleri (default [0.1, 0.5, 0.9])
    """

    def __init__(
        self,
        d_model:       int   = 64,
        num_heads:     int   = 4,
        lstm_layers:   int   = 2,
        dropout:       float = 0.3,    # 0.1 → 0.3: train/val gap 33× idi, regularize
        epochs:        int   = 80,
        batch_size:    int   = 32,
        learning_rate: float = 1e-3,
        patience:      int   = 15,
        weight_decay:  float = 1e-4,   # L2 regularizasyon — overfitting'e karşı
        quantiles:     List[float] = None,
    ):
        self.d_model       = d_model
        self.num_heads     = num_heads
        self.lstm_layers   = lstm_layers
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.patience      = patience
        self.weight_decay  = weight_decay
        self.quantiles     = quantiles or [0.1, 0.5, 0.9]

        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network: Optional[_TFTNetwork] = None
        self._num_features: Optional[int]   = None

    @staticmethod
    def _chronological_validation_split(
        X: np.ndarray,
        y: np.ndarray,
        validation_ratio: float = 0.1,
        min_val_samples: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Zaman serisinde shuffle etmeden son bölümden validation ayırır.
        """
        n_samples = len(X)
        if n_samples < 4:
            raise ValueError("TFT eğitimi için yeterli sequence yok.")

        n_val = max(1, int(n_samples * validation_ratio))
        n_val = min(max(min_val_samples, n_val), max(1, n_samples - 1))
        n_train = n_samples - n_val

        if n_train <= 0:
            raise ValueError("Chronological validation split sonrası train örneği kalmadı.")

        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    # ── Dahili Yardımcılar ────────────────────────────────────────────────────
    def _numpy_to_loader(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """Numpy dizilerini PyTorch DataLoader'a dönüştürür."""
        X_t = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_t = torch.tensor(y.ravel(), dtype=torch.float32)
            ds = TensorDataset(X_t, y_t)
        else:
            ds = TensorDataset(X_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _build_network(self, num_features: int) -> _TFTNetwork:
        return _TFTNetwork(
            num_features = num_features,
            d_model      = self.d_model,
            num_heads    = self.num_heads,
            lstm_layers  = self.lstm_layers,
            dropout      = self.dropout,
            quantiles    = self.quantiles,
        ).to(self.device)

    # ── BaseModel Arayüzü ────────────────────────────────────────────────────
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        TFT modelini eğitir.

        X_train : (samples, time_steps, num_features)
        y_train : (samples,)  veya (samples, 1)  — ölçeklenmiş hedef
        """
        if X_train.ndim != 3:
            raise ValueError(
                f"TFT girdi tensörü 3-boyutlu olmalıdır (samples, time_steps, features), "
                f"alınan: {X_train.ndim}D shape={X_train.shape}"
            )

        num_features = X_train.shape[2]
        self._num_features = num_features
        self.network = self._build_network(num_features)

        criterion = QuantileLoss(self.quantiles)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,   # L2 regularizasyon
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        X_tr, y_tr, X_val, y_val = self._chronological_validation_split(X_train, y_train)

        train_loader = self._numpy_to_loader(X_tr, y_tr, shuffle=True)
        val_loader   = self._numpy_to_loader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        best_weights  = None
        patience_cnt  = 0

        print(f"\n  [TFT-PyTorch] Eğitim başlıyor | "
              f"device={self.device} | features={num_features} | "
              f"d_model={self.d_model} | heads={self.num_heads} | "
              f"train={len(X_tr)} | val={len(X_val)}")

        for epoch in range(1, self.epochs + 1):
            # --- Eğitim ---
            self.network.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                preds, _ = self.network(X_b)
                loss = criterion(preds, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # --- Doğrulama ---
            self.network.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    preds, _ = self.network(X_b)
                    val_loss += criterion(preds, y_b).item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:4d}/{self.epochs} | "
                    f"train={train_loss:.5f}  val={val_loss:.5f}  lr={lr_now:.2e}"
                )

            # --- Early Stopping ---
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights  = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"  [Early Stop] Epoch {epoch} — val_loss artmıyor, en iyi ağırlıklar yüklendi.")
                    break

        # En iyi ağırlıkları geri yükle
        if best_weights is not None:
            self.network.load_state_dict(best_weights)
        self.network.to(self.device)

        print(f"  [OK] TFT-PyTorch eğitildi. En iyi val_loss={best_val_loss:.5f}")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        P50 (medyan) tahminini numpy dizisi olarak döndürür.
        Pipeline geri kalanıyla tam uyumludur.
        """
        q_preds = self.predict_quantiles(X_test)
        p50_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        return q_preds[:, p50_idx]

    def predict_quantiles(self, X_test: np.ndarray) -> np.ndarray:
        """
        Tüm kuantil tahminlerini döndürür.

        Returns:
            np.ndarray  shape (samples, num_quantiles)
            Sütun sırası: P10, P50, P90  (varsayılan kuantil listesine göre)
        """
        if self.network is None:
            raise RuntimeError("Model henüz eğitilmedi. Önce train() çağırın.")

        loader = self._numpy_to_loader(X_test, shuffle=False)
        self.network.eval()
        all_preds = []

        with torch.no_grad():
            for (X_b,) in loader:
                X_b = X_b.to(self.device)
                preds, _ = self.network(X_b)
                all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def get_variable_importances(self, X_sample: np.ndarray) -> np.ndarray:
        """
        VSN seçim ağırlıklarını döndürür — özellik önem skoru.

        Returns:
            np.ndarray  shape (time_steps, num_features)
            Değerler 0-1 aralığında (toplam 1'e normalize).
        """
        if self.network is None:
            raise RuntimeError("Model henüz eğitilmedi.")
        self.network.eval()
        X_t = torch.tensor(X_sample[:1], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, weights = self.network(X_t)
        return weights.squeeze(0).cpu().numpy()   # (time_steps, num_features)

    # ── Kaydet / Yükle ────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """
        Model ağırlıklarını ve konfigürasyonu kaydeder.
        .pt uzantısı kullanılır (.keras yerine).
        """
        if self.network is None:
            raise RuntimeError("Kaydedilecek model yok.")

        # path .keras olarak geldiyse .pt'ye çevir
        if path.endswith(".keras"):
            path = path.replace(".keras", ".pt")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        config = {
            "d_model":       self.d_model,
            "num_heads":     self.num_heads,
            "lstm_layers":   self.lstm_layers,
            "dropout":       self.dropout,
            "epochs":        self.epochs,
            "batch_size":    self.batch_size,
            "learning_rate": self.learning_rate,
            "patience":      self.patience,
            "weight_decay":  self.weight_decay,
            "quantiles":     self.quantiles,
            "num_features":  self._num_features,
        }

        torch.save(
            {
                "config":       config,
                "state_dict":   self.network.state_dict(),
            },
            path,
        )
        print(f"  [OK] TFT-PyTorch kaydedildi → {path}")

    def load(self, path: str) -> None:
        """Kaydedilmiş TFT modelini geri yükler."""
        if path.endswith(".keras"):
            path = path.replace(".keras", ".pt")

        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint["config"]

        self.d_model       = config["d_model"]
        self.num_heads     = config["num_heads"]
        self.lstm_layers   = config["lstm_layers"]
        self.dropout       = config["dropout"]
        self.weight_decay  = config.get("weight_decay", 1e-4)  # eski kayıtlarla uyumluluk
        self.quantiles     = config["quantiles"]
        self._num_features = config["num_features"]

        self.network = self._build_network(self._num_features)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.to(self.device)
        self.network.eval()

        print(f"  [OK] TFT-PyTorch yüklendi ← {path}")
