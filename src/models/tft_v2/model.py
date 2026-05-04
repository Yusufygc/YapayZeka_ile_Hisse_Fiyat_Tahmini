# -*- coding: utf-8 -*-
"""
tft_v2/model.py — TFT Ağı ve BaseModel Sarmalayıcı
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
_TFTNetwork  : PyTorch modülü
TFTModel     : BaseModel uyumlu sarmalayıcı

Değişiklikler (v1 → v2):
  [A4] output_fc → SortedQuantileOutput  (P10 ≤ P50 ≤ P90 garantisi)
  [A4] QuantileLoss → SortedQuantileLoss
  [A5] predict_quantiles(return_attention=True) desteği
  [A1] StaticCovariateEncoder — static_input_size > 0 ise aktif
       c_h / c_c → LSTM hidden/cell init
       c_s       → VSN selection context
       c_e       → encoder enrichment (post-seq GRN context)
  static=None ile v1 davranışı korunur — pipeline kırılmaz.

Sonraki adımlar:
  [A2] past_lstm / future_lstm ayrımı
  [A3] predict_multihorizon() + MultiHorizonHead
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel
from src.models.tft_v2.blocks import (
    GRN,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)
from src.models.tft_v2.encoders import StaticCovariateEncoder
from src.models.tft_v2.output_heads import (
    MultiHorizonHead,
    SortedQuantileLoss,
    SortedQuantileOutput,
)


# ─────────────────────────────────────────────────────────────────────────────
# _TFTNetwork — iç PyTorch modülü
# ─────────────────────────────────────────────────────────────────────────────
class _TFTNetwork(nn.Module):
    """
    TFT ağı (v2).

    Mimari:
      [A1] static (B, S)  → StaticCovariateEncoder → c_s, c_e, c_h, c_c
      Input (B, T, F)
        → VSN [c_s]              — özellik seçimi
        → past_lstm [c_h/c_c]   — [A2] encoder: T adım geçmişi kodlar
        → post_seq_grn [c_e]     — encoder belleği zenginleştir
        → prediction_query       — [A2] öğrenilebilir tahmin tokeni (1 adım)
        → future_lstm [(h_n,c_n)] — [A2] decoder: encoder finalinden başlar
        → cross-attention         — [A2] decoder future_out, encoder mem sorgular
        → GRN + Add & Norm
        → MultiHorizonHead [A3/A4] — H ufuk × (P10 ≤ P50 ≤ P90) garantisi

    static=None → tüm bağlam sıfır, v1 davranışı korunur.
    horizons=[1] (default) → backward compat, tek adım tahmin.
    horizons=[1,5,10,21]  → predict_multihorizon() ile çoklu ufuk.
    """

    def __init__(
        self,
        num_features:      int,
        d_model:           int         = 64,
        num_heads:         int         = 4,
        lstm_layers:       int         = 2,
        dropout:           float       = 0.1,
        quantiles:         List[float] = (0.1, 0.5, 0.9),
        static_input_size: int         = 0,
        horizons:          List[int]   = (1,),
    ) -> None:
        super().__init__()
        self.num_features      = num_features
        self.d_model           = d_model
        self.quantiles         = list(quantiles)
        self.lstm_layers       = lstm_layers
        self.static_input_size = static_input_size
        self.horizons          = list(horizons)
        self.num_horizons      = len(horizons)

        # [A1] Static Covariate Encoder
        self.static_encoder: Optional[StaticCovariateEncoder] = (
            StaticCovariateEncoder(static_input_size, d_model, dropout)
            if static_input_size > 0 else None
        )

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            num_vars    = num_features,
            d_model     = d_model,
            dropout     = dropout,
            context_dim = d_model if static_input_size > 0 else None,
        )

        # 2. [A2] Past Encoder LSTM — geçmiş gözlemleri kodlar
        self.past_lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = d_model,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # 3. Post-sequence GRN — encoder belleğini zenginleştirir
        self.post_seq_grn = GRN(
            input_dim   = d_model,
            hidden_dim  = d_model,
            output_dim  = d_model,
            dropout     = dropout,
            context_dim = d_model if static_input_size > 0 else None,
        )

        # [A2/A3] Öğrenilebilir tahmin tokenleri — H ufuk için H adet token
        # horizons=(1,) → (1, 1, d_model), backward compat
        # horizons=(1,5,10,21) → (1, 4, d_model)
        self.prediction_query = nn.Parameter(
            torch.zeros(1, self.num_horizons, d_model)
        )
        nn.init.xavier_uniform_(self.prediction_query.view(self.num_horizons, d_model))

        # [A2] Future Decoder LSTM — encoder finalinden başlayarak tahmin üretir
        self.future_lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = d_model,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # 4. [A2] Cross-Attention — decoder future_out, encoder belleği sorgular
        self.attn = InterpretableMultiHeadAttention(d_model, num_heads, dropout)

        # 5. Post-attention GRN + Add & Norm
        self.post_attn_grn  = GRN(d_model, d_model, d_model, dropout)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # 6. [A3] MultiHorizonHead — H ufuk × 3 kuantil
        self.output_head = MultiHorizonHead(d_model, self.horizons)

    def forward(
        self,
        x:                   torch.Tensor,
        static:              Optional[torch.Tensor] = None,
        return_attention:    bool                   = False,
        return_all_horizons: bool                   = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x                   : (batch, T, F)
            static              : (batch, S) | None
            return_attention    : attn_weights döndür
            return_all_horizons : True → tüm ufukları döndür (B, H, 3)
                                  False → yalnızca ilk ufku döndür (B, 3)  [backward compat]

        Döndürür:
            preds        : (batch, 3) [H=1 veya return_all_horizons=False]
                           (batch, H, 3) [return_all_horizons=True]
            var_weights  : (batch, T, F)       — VSN seçim ağırlıkları
            attn_weights : (batch, H, T) | (batch, 1, T) | None
        """
        B = x.size(0)
        H = self.num_horizons

        # ── [A1] Static encoding ──────────────────────────────────────────────
        if self.static_encoder is not None and static is not None:
            c_s, c_e, c_h, c_c = self.static_encoder(static)
        else:
            c_s = c_e = c_h = c_c = None

        # ── 1. VSN ───────────────────────────────────────────────────────────
        vsn_out, var_weights = self.vsn(x, context=c_s)        # (B, T, d_model)

        # ── 2. [A2] Past Encoder LSTM ────────────────────────────────────────
        if c_h is not None and c_c is not None:
            h_0 = c_h.unsqueeze(0).expand(self.lstm_layers, B, self.d_model).contiguous()
            c_0 = c_c.unsqueeze(0).expand(self.lstm_layers, B, self.d_model).contiguous()
            past_out, (h_n, c_n) = self.past_lstm(vsn_out, (h_0, c_0))
        else:
            past_out, (h_n, c_n) = self.past_lstm(vsn_out)     # (B, T, d_model)

        # ── 3. Post-seq GRN — encoder belleği ────────────────────────────────
        enc_mem = self.post_seq_grn(past_out, context=c_e)     # (B, T, d_model)

        # ── [A2/A3] Future Decoder LSTM ──────────────────────────────────────
        # H öğrenilebilir token: (1, H, d_model) → (B, H, d_model)
        query_in   = self.prediction_query.expand(B, H, self.d_model)
        future_out, _ = self.future_lstm(query_in, (h_n, c_n)) # (B, H, d_model)

        # ── 4. [A2/A3] Cross-Attention: future_out × enc_mem ─────────────────
        # query=(B,H,d), key/value=(B,T,d) → attn_out=(B,H,d), attn=(B,H,T)
        attn_out, attn_weights = self.attn(
            future_out, enc_mem, enc_mem, return_weights=return_attention
        )

        # ── 5. Artık bağlantı + Norm (decoder residual) ──────────────────────
        out = self.post_attn_norm(
            self.post_attn_grn(attn_out) + future_out
        )                                                       # (B, H, d_model)

        # ── 6. [A3] MultiHorizonHead — (B, H, 3) ─────────────────────────────
        all_preds = self.output_head(out)                       # (B, H, 3)

        if return_all_horizons:
            return all_preds, var_weights, attn_weights
        else:
            # Backward compat: yalnızca ilk ufuk  (B, 3)
            attn_first = attn_weights[:, 0:1, :] if attn_weights is not None else None
            return all_preds[:, 0, :], var_weights, attn_first


# ─────────────────────────────────────────────────────────────────────────────
# TFTModel — BaseModel uyumlu sarmalayıcı
# ─────────────────────────────────────────────────────────────────────────────
class TFTModel(BaseModel):
    """
    PyTorch tabanlı Temporal Fusion Transformer (v2).

    v1 ile tam geriye dönük uyumluluk:
      - train(X_np, y_np)         → değişmedi
      - predict(X_np)             → P50 döndürür, değişmedi
      - predict_quantiles(X_np)   → (N, 3); return_attention=True ile attn da gelir
      - save(path) / load(path)   → .pt formatı, değişmedi

    Yeni özellikler (v2):
      - [A4] P10 ≤ P50 ≤ P90 garantili çıktı; SortedQuantileLoss
      - [A5] predict_quantiles(return_attention=True) → attention heatmap
      - [A1] static_input_size > 0 → StaticCovariateEncoder aktif
             train(X, y, static_features=S) ile statik bilgi enjekte edilir

    Static özellik formatı (örnek, 4 sütun):
        sector_id        : 0-10 arası kategorik (normalize edilmiş)
        market_cap_cat   : 0=küçük, 1=orta, 2=büyük (normalize edilmiş)
        index_membership : 0/1/2 (normalize edilmiş)
        listing_age_norm : [0,1] normalize yıl
        → static_features: np.ndarray  (N_samples, num_static)

    Args:
        static_input_size : Statik özellik sayısı; 0 = static encoder kapalı (default 0)
        [diğerleri aşağıda]
    """

    def __init__(
        self,
        d_model:           int         = 64,
        num_heads:         int         = 4,
        lstm_layers:       int         = 2,
        dropout:           float       = 0.3,
        epochs:            int         = 80,
        batch_size:        int         = 32,
        learning_rate:     float       = 1e-3,
        patience:          int         = 15,
        weight_decay:      float       = 1e-4,
        quantiles:         List[float] = None,
        crossing_penalty:  float       = 0.5,
        static_input_size: int         = 0,
        horizons:          List[int]   = None,
        validation_ratio:  float       = 0.1,
        min_val_samples:   int         = 32,
        lr_patience:       int         = 5,
    ) -> None:
        self.d_model           = d_model
        self.num_heads         = num_heads
        self.lstm_layers       = lstm_layers
        self.dropout           = dropout
        self.epochs            = epochs
        self.batch_size        = batch_size
        self.learning_rate     = learning_rate
        self.patience          = patience
        self.weight_decay      = weight_decay
        self.quantiles         = quantiles or [0.1, 0.5, 0.9]
        self.crossing_penalty  = crossing_penalty
        self.static_input_size = static_input_size
        self.horizons          = horizons or [1]
        self.validation_ratio  = validation_ratio
        self.min_val_samples   = min_val_samples
        self.lr_patience       = lr_patience

        self.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network: Optional[_TFTNetwork] = None
        self._num_features: Optional[int]   = None

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _chronological_validation_split(
        self, X: np.ndarray, y: np.ndarray,
        static: Optional[np.ndarray] = None,
    ) -> Tuple:
        n = len(X)
        if n < 4:
            raise ValueError("TFT eğitimi için yeterli sequence yok.")
        n_val   = min(max(self.min_val_samples, int(n * self.validation_ratio)), max(1, n - 1))
        n_train = n - n_val
        if n_train <= 0:
            raise ValueError("Chronological split sonrası train örneği kalmadı.")
        s_tr = static[:n_train] if static is not None else None
        s_val = static[n_train:] if static is not None else None
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:], s_tr, s_val

    def _numpy_to_loader(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray]      = None,
        static: Optional[np.ndarray] = None,
        shuffle: bool                = False,
    ) -> DataLoader:
        """X (3D), y (1D, opsiyonel), static (2D, opsiyonel) → DataLoader."""
        tensors = [torch.tensor(X, dtype=torch.float32)]
        if y is not None:
            tensors.append(torch.tensor(y.ravel(), dtype=torch.float32))
        if static is not None:
            tensors.append(torch.tensor(static, dtype=torch.float32))
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _build_network(self, num_features: int) -> _TFTNetwork:
        return _TFTNetwork(
            num_features      = num_features,
            d_model           = self.d_model,
            num_heads         = self.num_heads,
            lstm_layers       = self.lstm_layers,
            dropout           = self.dropout,
            quantiles         = self.quantiles,
            static_input_size = self.static_input_size,
            horizons          = self.horizons,
        ).to(self.device)

    def _unpack_batch(
        self,
        batch: tuple,
        has_static: bool,
        has_y: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """DataLoader batch'ini (X, y?, static?) olarak ayrıştırır."""
        # Loader sırası: X, [y], [static]
        idx   = 0
        X_b   = batch[idx].to(self.device); idx += 1
        y_b   = batch[idx].to(self.device) if has_y else None
        if has_y: idx += 1
        s_b   = batch[idx].to(self.device) if has_static else None
        return X_b, y_b, s_b

    # ── BaseModel Arayüzü ────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        static_features: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        TFT v2 modelini eğitir.

        Args:
            X_train         : (N, time_steps, num_features)   — 3D zorunlu
            y_train         : (N,) veya (N, 1)                — ölçeklenmiş log-return
            static_features : (N, num_static) | None           — [A1] opsiyonel statik
        """
        if X_train.ndim != 3:
            raise ValueError(
                f"TFT girdi tensörü 3-boyutlu olmalı (N, T, F), "
                f"alınan: {X_train.ndim}D shape={X_train.shape}"
            )

        # [A1] static_input_size tutarlılık kontrolü
        if static_features is not None:
            if self.static_input_size == 0:
                raise ValueError(
                    "static_features verildi ama static_input_size=0. "
                    "TFTModel(static_input_size=N) ile oluşturun."
                )
            if static_features.shape[0] != X_train.shape[0]:
                raise ValueError(
                    f"static_features satır sayısı ({static_features.shape[0]}) "
                    f"X_train ile eşleşmiyor ({X_train.shape[0]})."
                )

        num_features       = X_train.shape[2]
        self._num_features = num_features
        self.network       = self._build_network(num_features)

        criterion = SortedQuantileLoss(self.quantiles, self.crossing_penalty)
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.lr_patience, min_lr=1e-6
        )

        X_tr, y_tr, X_val, y_val, s_tr, s_val = self._chronological_validation_split(
            X_train, y_train, static_features
        )
        has_static   = static_features is not None
        train_loader = self._numpy_to_loader(X_tr,  y_tr,  s_tr,  shuffle=True)
        val_loader   = self._numpy_to_loader(X_val, y_val, s_val, shuffle=False)

        best_val_loss = float("inf")
        best_weights  = None
        patience_cnt  = 0

        H          = len(self.horizons)
        multi_mode = H > 1          # True → tüm ufuklarda loss, False → tek ufuk

        static_tag   = f" | static={self.static_input_size}" if has_static else ""
        horizons_tag = f" | horizons={self.horizons}" if multi_mode else ""
        print(
            f"\n  [TFT-v2] Eğitim başlıyor | device={self.device} | "
            f"features={num_features}{static_tag}{horizons_tag} | d_model={self.d_model} | "
            f"heads={self.num_heads} | train={len(X_tr)} | val={len(X_val)}"
        )

        def _compute_loss(X_b, y_b, s_b):
            """Multi-horizon ise her ufuk için pinball loss toplar."""
            if multi_mode:
                all_preds, _, _ = self.network(
                    X_b, static=s_b, return_all_horizons=True
                )                                       # (B, H, 3)
                total = sum(
                    criterion(all_preds[:, h, :], y_b) for h in range(H)
                ) / H
                return total
            else:
                preds, _, _ = self.network(X_b, static=s_b)   # (B, 3)
                return criterion(preds, y_b)

        for epoch in range(1, self.epochs + 1):
            # — Eğitim —
            self.network.train()
            train_loss = 0.0
            for batch in train_loader:
                X_b, y_b, s_b = self._unpack_batch(batch, has_static, has_y=True)
                optimizer.zero_grad()
                loss = _compute_loss(X_b, y_b, s_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # — Doğrulama —
            self.network.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    X_b, y_b, s_b = self._unpack_batch(batch, has_static, has_y=True)
                    val_loss += _compute_loss(X_b, y_b, s_b).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:4d}/{self.epochs} | "
                    f"train={train_loss:.5f}  val={val_loss:.5f}  lr={lr_now:.2e}"
                )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights  = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"  [Early Stop] Epoch {epoch} — en iyi ağırlıklar geri yüklendi.")
                    break

        if best_weights:
            self.network.load_state_dict(best_weights)
        self.network.to(self.device)
        print(f"  [OK] TFT-v2 eğitildi. En iyi val_loss={best_val_loss:.5f}")

    def predict(
        self,
        X_test: np.ndarray,
        static_features: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """P50 (medyan) döndürür. v1 API ile birebir uyumlu."""
        q_preds = self.predict_quantiles(X_test, static_features=static_features)
        p50_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        return q_preds[:, p50_idx]


    def predict_quantiles(
        self,
        X_test: np.ndarray,
        static_features: Optional[np.ndarray] = None,
        return_attention: bool = False,
    ) -> np.ndarray:
        """
        Kuantil tahminleri. Donus: (N,3) [P10,P50,P90].
        return_attention=True ise (q_preds, attn) tuple.
        """
        if self.network is None:
            raise RuntimeError("Model henuz egitilmedi.")

        has_static = static_features is not None
        loader = self._numpy_to_loader(X_test, static=static_features, shuffle=False)
        self.network.eval()
        all_preds, all_attn = [], []

        with torch.no_grad():
            for batch in loader:
                X_b, _, s_b = self._unpack_batch(batch, has_static, has_y=False)
                preds, _, attn = self.network(X_b, static=s_b, return_attention=return_attention)
                all_preds.append(preds.cpu().numpy())
                if return_attention and attn is not None:
                    all_attn.append(attn.cpu().numpy())

        q_preds = np.concatenate(all_preds, axis=0)
        if return_attention:
            return q_preds, (np.concatenate(all_attn, axis=0) if all_attn else None)
        return q_preds

    def predict_multihorizon(
        self,
        X_test: np.ndarray,
        static_features: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        [A3] Tüm ufuklar için kuantil tahminleri döndürür.

        Args:
            X_test          : (N, time_steps, num_features)
            static_features : (N, num_static) | None

        Returns:
            Dict keyed by "h{k}" (e.g. "h1", "h5", "h10", "h21")
            Her değer: (N, 3) — [P10, P50, P90]

        Örnek:
            results = model.predict_multihorizon(X)
            p50_1day  = results["h1"][:, 1]   # P50, 1 gün
            p50_5day  = results["h5"][:, 1]   # P50, 5 gün
        """
        if self.network is None:
            raise RuntimeError("Model henüz eğitilmedi.")
        if len(self.horizons) == 1:
            # Tek ufuklu model — h1 anahtarıyla döndür
            key = f"h{self.horizons[0]}"
            return {key: self.predict_quantiles(X_test, static_features)}

        has_static = static_features is not None
        loader = self._numpy_to_loader(X_test, static=static_features, shuffle=False)
        self.network.eval()
        all_batches: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                X_b, _, s_b = self._unpack_batch(batch, has_static, has_y=False)
                all_preds, _, _ = self.network(
                    X_b, static=s_b, return_all_horizons=True
                )                                               # (B, H, 3)
                all_batches.append(all_preds.cpu().numpy())

        stacked = np.concatenate(all_batches, axis=0)          # (N, H, 3)
        return {
            f"h{h}": stacked[:, i, :]
            for i, h in enumerate(self.horizons)
        }

    def get_variable_importances(
        self,
        X_sample: np.ndarray,
        static_sample: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """VSN secim agirliklarini dondurur: (time_steps, num_features)."""
        if self.network is None:
            raise RuntimeError("Model henuz egitilmedi.")
        self.network.eval()
        X_t = torch.tensor(X_sample[:1], dtype=torch.float32).to(self.device)
        s_t = (torch.tensor(static_sample[:1], dtype=torch.float32).to(self.device)
               if static_sample is not None else None)
        with torch.no_grad():
            _, weights, _ = self.network(X_t, static=s_t)
        return weights.squeeze(0).cpu().numpy()

    def get_attention_heatmap(
        self,
        X_test: np.ndarray,
        static_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        [A5] Çapraz dikkat (cross-attention) ısı haritasını döndürür.

        _TFTNetwork.forward(return_attention=True, return_all_horizons=True)
        çağrısından elde edilen (B, H, T) tensörünü birleştirir.

        Args:
            X_test          : (N, time_steps, num_features)
            static_features : (N, num_static) | None

        Returns:
            attn : (N, H, T) float32 numpy dizisi
                   N = örnek sayısı
                   H = ufuk sayısı  (horizons=[1] → H=1)
                   T = time_steps
                   Değerler softmax normalise edilmiş dikkat ağırlıklarıdır
                   (her H için T üzerinde yaklaşık toplam ≈ 1).
        """
        if self.network is None:
            raise RuntimeError("Model henüz eğitilmedi.")

        has_static = static_features is not None
        loader     = self._numpy_to_loader(X_test, static=static_features, shuffle=False)
        self.network.eval()
        all_attn: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                X_b, _, s_b = self._unpack_batch(batch, has_static, has_y=False)
                _, _, attn  = self.network(
                    X_b,
                    static              = s_b,
                    return_attention    = True,
                    return_all_horizons = True,
                )                               # attn: (B, H, T)
                if attn is not None:
                    all_attn.append(attn.cpu().numpy())

        if not all_attn:
            # Dikkat döndürülmediyse (örn. num_heads=0 kenar durumu) sıfır dizi
            T = X_test.shape[1]
            H = len(self.horizons)
            return np.zeros((len(X_test), H, T), dtype=np.float32)

        return np.concatenate(all_attn, axis=0)    # (N, H, T)

    # ── Kaydet / Yukle ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self.network is None:
            raise RuntimeError("Kaydedilecek model yok.")
        if path.endswith(".keras"):
            path = path.replace(".keras", ".pt")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        config = {
            "d_model":           self.d_model,
            "num_heads":         self.num_heads,
            "lstm_layers":       self.lstm_layers,
            "dropout":           self.dropout,
            "epochs":            self.epochs,
            "batch_size":        self.batch_size,
            "learning_rate":     self.learning_rate,
            "patience":          self.patience,
            "weight_decay":      self.weight_decay,
            "quantiles":         self.quantiles,
            "crossing_penalty":  self.crossing_penalty,
            "static_input_size": self.static_input_size,
            "horizons":          self.horizons,
            "num_features":      self._num_features,
            "validation_ratio":  self.validation_ratio,
            "min_val_samples":   self.min_val_samples,
            "lr_patience":       self.lr_patience,
            "version":           "v2",
        }
        torch.save({"config": config, "state_dict": self.network.state_dict()}, path)
        print("  [OK] TFT-v2 kaydedildi -> " + path)

    def load(self, path: str) -> None:
        if path.endswith(".keras"):
            path = path.replace(".keras", ".pt")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]
        self.d_model           = cfg["d_model"]
        self.num_heads         = cfg["num_heads"]
        self.lstm_layers       = cfg["lstm_layers"]
        self.dropout           = cfg["dropout"]
        self.epochs            = cfg.get("epochs", self.epochs)
        self.batch_size        = cfg.get("batch_size", self.batch_size)
        self.learning_rate     = cfg.get("learning_rate", self.learning_rate)
        self.patience          = cfg.get("patience", self.patience)
        self.weight_decay      = cfg.get("weight_decay", 1e-4)
        self.quantiles         = cfg["quantiles"]
        self.crossing_penalty  = cfg.get("crossing_penalty", 0.5)
        self.static_input_size = cfg.get("static_input_size", 0)
        self.horizons          = cfg.get("horizons", [1])
        self._num_features     = cfg["num_features"]
        self.validation_ratio  = cfg.get("validation_ratio", 0.1)
        self.min_val_samples   = cfg.get("min_val_samples", 32)
        self.lr_patience       = cfg.get("lr_patience", 5)
        self.network = self._build_network(self._num_features)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.to(self.device)
        self.network.eval()
        print("  [OK] TFT-v2 yuklendi <- " + path)
