# -*- coding: utf-8 -*-
"""
tft_v2/blocks.py — Temel yapı taşları
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Temporal Fusion Transformer'ın yeniden kullanılabilir PyTorch bileşenleri:
  - GLU    : Gated Linear Unit
  - GRN    : Gated Residual Network
  - VSN    : Variable Selection Network
  - IMHA   : Interpretable Multi-Head Attention

Bu dosyadaki sınıflar pipeline'dan bağımsız — sadece torch.nn kullanır.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# GLU — Gated Linear Unit
# ─────────────────────────────────────────────────────────────────────────────
class GLU(nn.Module):
    """
    Gated Linear Unit: x₁ ⊙ σ(x₂)
    Bilgi akışını kapılar — gereksiz aktivasyon gürültüsünü bastırır.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        x1, x2 = out.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


# ─────────────────────────────────────────────────────────────────────────────
# GRN — Gated Residual Network
# ─────────────────────────────────────────────────────────────────────────────
class GRN(nn.Module):
    """
    Gated Residual Network (Lim et al., 2021 — Eq. 2-4).

    input_dim → hidden_dim → output_dim  (+residual skip + LayerNorm)
    Kapı mekanizması gereksiz transformasyon bloklarını atlar.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc_ctx  = nn.Linear(context_dim, hidden_dim) if context_dim else None
        self.elu     = nn.ELU()
        self.fc2     = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu     = GLU(hidden_dim)
        self.fc_out  = nn.Linear(hidden_dim, output_dim)
        self.norm    = nn.LayerNorm(output_dim)
        self.proj    = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = self.proj(x)
        h = self.fc1(x)
        if context is not None and self.fc_ctx is not None:
            ctx = self.fc_ctx(context)
            while ctx.ndim < h.ndim:
                ctx = ctx.unsqueeze(-2)
            h = h + ctx
        h = self.elu(h)
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        h = self.fc_out(h)
        return self.norm(h + residual)


# ─────────────────────────────────────────────────────────────────────────────
# VSN — Variable Selection Network
# ─────────────────────────────────────────────────────────────────────────────
class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (Lim et al., 2021 — Eq. 5-6).

    Her değişken için bağımsız GRN + ağırlık GRN → softmax seçim.

    Girdi  : (batch, [time,] num_vars)
    Çıktı  : (batch, [time,] d_model)   — ağırlıklı birleşim
             (batch, [time,] num_vars)  — seçim ağırlıkları (yorumlanabilirlik)
    """

    def __init__(
        self,
        num_vars: int,
        d_model: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.var_grns = nn.ModuleList([
            GRN(1, d_model, d_model, dropout) for _ in range(num_vars)
        ])
        self.weight_grn = GRN(
            input_dim=num_vars,
            hidden_dim=d_model,
            output_dim=num_vars,
            dropout=dropout,
            context_dim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, time, num_vars)  veya  (batch, num_vars)
        per_var = torch.stack(
            [grn(x[..., i: i + 1]) for i, grn in enumerate(self.var_grns)],
            dim=-2,
        )
        # per_var: (batch, [time,] num_vars, d_model)

        weights = torch.softmax(self.weight_grn(x, context), dim=-1)
        # weights: (batch, [time,] num_vars)

        output = (weights.unsqueeze(-1) * per_var).sum(dim=-2)
        # output: (batch, [time,] d_model)
        return output, weights


# ─────────────────────────────────────────────────────────────────────────────
# IMHA — Interpretable Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────
class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention (Lim et al., 2021 — Eq. 11-13).

    Her kafa ayrı V projeksiyon yerine paylaşımlı V kullanır — bu sayede
    kafa ortalaması alınarak yorumlanabilir tek bir attention haritası elde edilir.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model, num_heads'e tam bölünmeli"
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_head)   # paylaşımlı V
        self.W_o = nn.Linear(self.d_head, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale   = self.d_head ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Döndürür:
            output       : (batch, T_q, d_model)
            attn_weights : (batch, T_q, T_k)  — heads ortalaması
                           return_weights=False ise None döner
        """
        B, T_q, _ = query.shape
        T_k       = key.size(1)   # cross-attention: T_k != T_q olabilir [A2]

        Q = self.W_q(query).view(B, T_q, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(key).view(B, T_k, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(value)   # (B, T_k, d_head) — paylaşımlı V

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, H, T_q, T_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_mean = attn.mean(dim=1) if return_weights else None   # (B, T_q, T_k)

        V_exp   = V.unsqueeze(1).expand(B, self.num_heads, T_k, self.d_head)
        context = torch.matmul(attn, V_exp).mean(dim=1)             # (B, T_q, d_head)
        output  = self.W_o(context)                                  # (B, T_q, d_model)

        return output, attn_mean
