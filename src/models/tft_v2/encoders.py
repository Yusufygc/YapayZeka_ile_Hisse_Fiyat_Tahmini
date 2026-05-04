# -*- coding: utf-8 -*-
"""
tft_v2/encoders.py — Static Covariate Encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A1 — Statik Değişken Kodlayıcı

TFT'nin orijinal tasarımında (Lim et al., 2021 — Eq. 7-10) statik özellikler
(sektör, piyasa değeri kategorisi, endeks üyeliği vb.) zaman içinde değişmez.
Bu bilgi dört bağlam vektörüne dönüştürülerek şu noktalara enjekte edilir:

    c_s  →  VSN context        (hangi değişkenin seçileceğini etkiler)
    c_e  →  encoder context    (LSTM encoder çıktısını zenginleştirir)
    c_h  →  LSTM hidden init   (geçmiş encoder başlangıç durumu)
    c_c  →  LSTM cell init     (geçmiş encoder hücre durumu)

Kullanım:
    encoder = StaticCovariateEncoder(num_static=4, d_model=64)
    c_s, c_e, c_h, c_c = encoder(static_tensor)  # (batch, 4) → 4 × (batch, d_model)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.tft_v2.blocks import GRN, VariableSelectionNetwork


class StaticCovariateEncoder(nn.Module):
    """
    Statik değişkenleri dört bağlam vektörüne dönüştürür.

    Mimari (Lim et al., 2021 — Figure 3):
        static_input (batch, num_static)
          → StaticVSN           — hangi statik özelliğin önemli olduğunu öğrenir
          → GRN_cs              → c_s  (VSN selection context)
          → GRN_ce              → c_e  (encoder enrichment context)
          → GRN_ch              → c_h  (LSTM hidden init)
          → GRN_cc              → c_c  (LSTM cell init)

    Args:
        num_static : Statik özellik sayısı (örn. 4: sector, market_cap, index, age)
        d_model    : Model gizli boyutu
        dropout    : Dropout oranı
    """

    def __init__(
        self,
        num_static: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_static < 1:
            raise ValueError(f"num_static ≥ 1 olmalı, alınan: {num_static}")

        # Statik değişken seçimi — hangi statik özellik önemli?
        self.static_vsn = VariableSelectionNetwork(
            num_vars    = num_static,
            d_model     = d_model,
            dropout     = dropout,
            context_dim = None,      # statik için üst-bağlam yok
        )

        # Dört bağlam vektörü — her biri ayrı GRN ile öğrenilir
        self.grn_cs = GRN(d_model, d_model, d_model, dropout)   # VSN context
        self.grn_ce = GRN(d_model, d_model, d_model, dropout)   # encoder context
        self.grn_ch = GRN(d_model, d_model, d_model, dropout)   # LSTM hidden
        self.grn_cc = GRN(d_model, d_model, d_model, dropout)   # LSTM cell

    def forward(
        self,
        static: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            static : (batch, num_static)  — normalize edilmiş statik özellikler

        Returns:
            c_s : (batch, d_model)  — VSN selection context
            c_e : (batch, d_model)  — encoder enrichment context
            c_h : (batch, d_model)  — LSTM hidden state başlangıcı
            c_c : (batch, d_model)  — LSTM cell state başlangıcı
        """
        # VSN: statik değişken seçimi ve birleştirme
        h, _vsn_weights = self.static_vsn(static)   # (batch, d_model)

        c_s = self.grn_cs(h)
        c_e = self.grn_ce(h)
        c_h = self.grn_ch(h)
        c_c = self.grn_cc(h)

        return c_s, c_e, c_h, c_c
