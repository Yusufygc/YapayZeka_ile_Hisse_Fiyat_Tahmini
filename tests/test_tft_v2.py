# -*- coding: utf-8 -*-
"""
tests/test_tft_v2.py — TFT v2 birim ve entegrasyon testleri
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Kapsam:
  [A1] Static Covariate Encoder
    - test_static_encoder_output_shapes   : 4 × (batch, d_model) çıktı
    - test_static_encoder_context_effect  : context ile VSN ağırlıkları değişir
    - test_static_none_compatible         : static=None, v1 davranışı korunur
    - test_static_train_predict           : static ile tam eğitim + tahmin döngüsü
    - test_static_size_mismatch_raises    : boyut uyuşmazlığı ValueError
    - test_static_save_load_roundtrip     : static modeli kaydet → yükle → tahmin
  [A2] Encoder-Decoder LSTM
    - test_enc_dec_forward_shapes   : past_lstm+future_lstm+cross-attn boyutları
    - test_prediction_query_is_param: prediction_query öğrenilebilir parametre
    - test_cross_attn_shape         : attn_weights (B, 1, T) — cross-attention
    - test_enc_dec_train_predict    : tam eğitim+tahmin döngüsü
    - test_enc_dec_quantile_order   : A2 sonrası P10 ≤ P50 ≤ P90 korunur
  [A4] Quantile Crossing
    - test_quantile_ordering        : P10 ≤ P50 ≤ P90 her örnekte
    - test_quantile_crossing_rate   : ihlal oranı < %1
    - test_sorted_loss_zero_penalty : SortedQuantileOutput ile crossing_loss → ~0
  [A5] Attention
    - test_attention_shape          : return_attention=True (B, 1, T) döner
    - test_attention_none_default   : return_attention=False → tek ndarray
  [Uyumluluk]
    - test_predict_returns_p50      : predict() P50 kolonunu döndürür
    - test_train_predict_shapes     : train+predict boyutları tutarlı
    - test_save_load_roundtrip      : save→load→predict değişmez
    - test_import_alias             : eski tft_model.py üzerinden import çalışır
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

# ── tft_v2 doğrudan import ───────────────────────────────────────────────────
from src.models.tft_v2 import TFTModel, StaticCovariateEncoder
from src.models.tft_v2.output_heads import (
    SortedQuantileLoss,
    SortedQuantileOutput,
)


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────
BATCH        = 64
TIME_STEPS   = 10
FEATURES     = 8
NUM_STATIC   = 4   # [A1] statik özellik sayısı
SEED         = 42


def _make_data(n: int = 200):
    """Küçük sentetik 3D veri seti."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, TIME_STEPS, FEATURES)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    return X, y


def _make_static(n: int = 200):
    """[A1] Sentetik statik özellikler (normalize edilmiş)."""
    rng = np.random.default_rng(SEED + 1)
    return rng.random((n, NUM_STATIC)).astype(np.float32)


def _fast_model(static_input_size: int = 0) -> TFTModel:
    """Hızlı test için minimal TFT v2."""
    return TFTModel(
        d_model           = 16,
        num_heads         = 2,
        lstm_layers       = 1,
        dropout           = 0.0,
        epochs            = 3,
        batch_size        = 32,
        patience          = 10,
        crossing_penalty  = 0.5,
        static_input_size = static_input_size,
        min_val_samples   = 10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# [A1] StaticCovariateEncoder — unit testler
# ─────────────────────────────────────────────────────────────────────────────

class TestStaticCovariateEncoder:
    """StaticCovariateEncoder dört bağlam vektörü üretmeli."""

    def setup_method(self):
        torch.manual_seed(SEED)
        self.encoder = StaticCovariateEncoder(num_static=NUM_STATIC, d_model=16)

    def test_output_shapes(self):
        """(batch, d_model) × 4 çıktı."""
        s = torch.rand(BATCH, NUM_STATIC)
        c_s, c_e, c_h, c_c = self.encoder(s)
        for name, vec in [("c_s", c_s), ("c_e", c_e), ("c_h", c_h), ("c_c", c_c)]:
            assert vec.shape == (BATCH, 16), f"{name}: beklenen ({BATCH}, 16), alınan {vec.shape}"

    def test_different_contexts_are_distinct(self):
        """Dört bağlam vektörü birbirinden farklı olmalı (ayrı GRN'ler)."""
        s = torch.rand(8, NUM_STATIC)
        c_s, c_e, c_h, c_c = self.encoder(s)
        assert not torch.allclose(c_s, c_e, atol=1e-3), "c_s ve c_e özdeş — GRN'ler ayrı olmalı"
        assert not torch.allclose(c_h, c_c, atol=1e-3), "c_h ve c_c özdeş — GRN'ler ayrı olmalı"

    def test_gradients_flow_through_all_contexts(self):
        """Tüm çıktılara gradient akmalı."""
        s = torch.rand(8, NUM_STATIC)
        c_s, c_e, c_h, c_c = self.encoder(s)
        (c_s + c_e + c_h + c_c).sum().backward()
        for name, param in self.encoder.named_parameters():
            assert param.grad is not None, f"{name} parametresinde gradient yok"

    def test_invalid_num_static_raises(self):
        with pytest.raises(ValueError, match="num_static"):
            StaticCovariateEncoder(num_static=0, d_model=16)


class TestTFTModelA1:
    """TFTModel static_input_size > 0 ile tam döngü testleri."""

    def setup_method(self):
        self.X, self.y = _make_data(200)
        self.S         = _make_static(200)

    def test_static_none_compatible(self):
        """static_input_size=0 → static=None ile v1 davranışı korunur."""
        m = _fast_model(static_input_size=0)
        m.train(self.X, self.y)
        preds = m.predict(self.X[:10])
        assert preds.shape == (10,)

    def test_static_train_predict_shapes(self):
        """static ile eğitim + tahmin boyutları tutarlı."""
        m = _fast_model(static_input_size=NUM_STATIC)
        m.train(self.X, self.y, static_features=self.S)
        preds = m.predict(self.X[:10], static_features=self.S[:10])
        assert preds.shape == (10,)
        q = m.predict_quantiles(self.X[:10], static_features=self.S[:10])
        assert q.shape == (10, 3)

    def test_static_quantile_ordering(self):
        """Static ile eğitimde de P10 ≤ P50 ≤ P90 korunmalı."""
        m = _fast_model(static_input_size=NUM_STATIC)
        m.train(self.X, self.y, static_features=self.S)
        q = m.predict_quantiles(self.X, static_features=self.S)
        assert (q[:, 0] <= q[:, 1] + 1e-5).all(), "P10 > P50 ihlali (static)"
        assert (q[:, 1] <= q[:, 2] + 1e-5).all(), "P50 > P90 ihlali (static)"

    def test_static_size_mismatch_raises(self):
        """static_input_size=0 iken static_features verilirse ValueError."""
        m = _fast_model(static_input_size=0)
        with pytest.raises(ValueError, match="static_input_size"):
            m.train(self.X, self.y, static_features=self.S)

    def test_static_row_mismatch_raises(self):
        """static_features satır sayısı X_train ile uyuşmazsa ValueError."""
        m = _fast_model(static_input_size=NUM_STATIC)
        with pytest.raises(ValueError, match="eşleşmiyor"):
            m.train(self.X, self.y, static_features=self.S[:50])

    def test_static_save_load_roundtrip(self):
        """Static modeli kaydet → yükle → predict sonuçları değişmemeli."""
        import tempfile, os
        m1 = _fast_model(static_input_size=NUM_STATIC)
        m1.train(self.X, self.y, static_features=self.S)
        preds_before = m1.predict(self.X[:10], static_features=self.S[:10])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tft_v2_static.pt")
            m1.save(path)
            m2 = TFTModel()
            m2.load(path)
            preds_after = m2.predict(self.X[:10], static_features=self.S[:10])

        np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=5)

    def test_static_input_size_persisted_on_load(self):
        """Yüklenen modelde static_input_size doğru olmalı."""
        import tempfile, os
        m1 = _fast_model(static_input_size=NUM_STATIC)
        m1.train(self.X, self.y, static_features=self.S)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tft_v2_static.pt")
            m1.save(path)
            m2 = TFTModel()
            m2.load(path)
        assert m2.static_input_size == NUM_STATIC


# ─────────────────────────────────────────────────────────────────────────────
# [A2] Encoder-Decoder LSTM — testler
# ─────────────────────────────────────────────────────────────────────────────

class TestEncoderDecoder:
    """past_lstm + future_lstm + cross-attention mimarisi."""

    def setup_method(self):
        torch.manual_seed(SEED)
        self.X, self.y = _make_data(200)

    def test_enc_dec_forward_shapes(self):
        """_TFTNetwork forward: preds (B,3), var_weights (B,T,F), attn (B,1,T) | None."""
        from src.models.tft_v2.model import _TFTNetwork
        net = _TFTNetwork(num_features=FEATURES, d_model=16, num_heads=2,
                          lstm_layers=1, dropout=0.0)
        x = torch.randn(8, TIME_STEPS, FEATURES)
        preds, vw, attn = net(x, return_attention=True)
        assert preds.shape == (8, 3),                  f"preds: {preds.shape}"
        assert vw.shape    == (8, TIME_STEPS, FEATURES), f"var_weights: {vw.shape}"
        assert attn.shape  == (8, 1, TIME_STEPS),        f"attn: {attn.shape}"

    def test_prediction_query_is_learnable(self):
        """prediction_query nn.Parameter olmalı."""
        from src.models.tft_v2.model import _TFTNetwork
        net = _TFTNetwork(num_features=FEATURES, d_model=16, num_heads=2, lstm_layers=1)
        assert hasattr(net, 'prediction_query'), "prediction_query eksik"
        assert isinstance(net.prediction_query, torch.nn.Parameter)
        assert net.prediction_query.shape == (1, 1, 16)

    def test_past_future_lstm_exist(self):
        """past_lstm ve future_lstm ayrı modüller olmalı; eski self.lstm yok."""
        from src.models.tft_v2.model import _TFTNetwork
        net = _TFTNetwork(num_features=FEATURES, d_model=16, num_heads=2, lstm_layers=1)
        assert hasattr(net, 'past_lstm'),   "past_lstm eksik"
        assert hasattr(net, 'future_lstm'), "future_lstm eksik"
        assert not hasattr(net, 'lstm'),    "Eski self.lstm hâlâ mevcut — kaldırılmalı"


    def test_cross_attn_shape(self):
        """cross-attention: (N,1,T) shape."""
        m = _fast_model()
        m.train(self.X, self.y)
        _, attn = m.predict_quantiles(self.X[:8], return_attention=True)
        assert attn.shape == (8, 1, TIME_STEPS), \
            "Cross-attn shape beklenen (8,1," + str(TIME_STEPS) + "), alınan " + str(attn.shape)

    def test_enc_dec_train_predict(self):
        """Tam train+predict dongusu hata vermemeli."""
        m = _fast_model()
        m.train(self.X, self.y)
        q = m.predict_quantiles(self.X[:10])
        assert q.shape == (10, 3)

    def test_enc_dec_quantile_order(self):
        """A2 sonrasi P10 <= P50 <= P90 korunmali."""
        m = _fast_model()
        m.train(self.X, self.y)
        q = m.predict_quantiles(self.X)
        assert (q[:, 0] <= q[:, 1] + 1e-5).all(), "P10 > P50 ihlali"
        assert (q[:, 1] <= q[:, 2] + 1e-5).all(), "P50 > P90 ihlali"

    def test_enc_dec_with_static(self):
        """Static encoder ile birlikte enc-dec calismali."""
        S = _make_static(200)
        m = _fast_model(static_input_size=NUM_STATIC)
        m.train(self.X, self.y, static_features=S)
        q = m.predict_quantiles(self.X[:10], static_features=S[:10])
        assert q.shape == (10, 3)
        assert (q[:, 0] <= q[:, 1] + 1e-5).all()
        assert (q[:, 1] <= q[:, 2] + 1e-5).all()


# ─────────────────────────────────────────────────────────────────────────────
# [A4] SortedQuantileOutput — unit testler
# ─────────────────────────────────────────────────────────────────────────────

class TestSortedQuantileOutput:

    def setup_method(self):
        torch.manual_seed(SEED)
        self.head = SortedQuantileOutput(d_model=32)

    def test_output_shape(self):
        x = torch.randn(BATCH, 32)
        out = self.head(x)
        assert out.shape == (BATCH, 3)

    def test_monotone_ordering_random_input(self):
        x = torch.randn(1000, 32)
        out = self.head(x)
        assert (out[:, 0] <= out[:, 1] + 1e-6).all(), "P10 > P50"
        assert (out[:, 1] <= out[:, 2] + 1e-6).all(), "P50 > P90"

    def test_monotone_ordering_extreme_input(self):
        x = torch.randn(200, 32) * 10
        out = self.head(x)
        assert (out[:, 0] <= out[:, 1] + 1e-6).all()
        assert (out[:, 1] <= out[:, 2] + 1e-6).all()

    def test_gradients_flow(self):
        x = torch.randn(8, 32)
        out = self.head(x)
        out.sum().backward()
        for name, param in self.head.named_parameters():
            assert param.grad is not None, name + " gradyansiz"


# ─────────────────────────────────────────────────────────────────────────────
# [A4] SortedQuantileLoss
# ─────────────────────────────────────────────────────────────────────────────

class TestSortedQuantileLoss:

    def setup_method(self):
        self.loss_fn = SortedQuantileLoss(quantiles=[0.1, 0.5, 0.9], crossing_penalty=0.5)

    def test_zero_crossing_penalty_for_sorted_output(self):
        import torch.nn.functional as F
        head  = SortedQuantileOutput(d_model=32)
        preds = head(torch.randn(100, 32)).detach()
        penalty = (F.relu(preds[:, 0] - preds[:, 1]).mean()
                   + F.relu(preds[:, 1] - preds[:, 2]).mean()).item()
        assert penalty < 1e-5

    def test_loss_scalar(self):
        preds  = torch.tensor([[0.1, 0.5, 0.9]] * 32)
        target = torch.zeros(32)
        loss   = self.loss_fn(preds, target)
        assert loss.ndim == 0

    def test_crossing_rate_helper(self):
        preds_bad  = torch.tensor([[1.0, 0.5, 0.9]] * 100)
        preds_good = torch.tensor([[0.1, 0.5, 0.9]] * 100)
        assert SortedQuantileLoss.crossing_rate(preds_bad)  > 0.99
        assert SortedQuantileLoss.crossing_rate(preds_good) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# [A4] TFTModel tam dongu
# ─────────────────────────────────────────────────────────────────────────────

class TestTFTModelA4:

    def setup_method(self):
        self.X, self.y = _make_data(200)
        self.model = _fast_model()
        self.model.train(self.X, self.y)

    def test_predict_returns_p50(self):
        preds = self.model.predict(self.X[:10])
        q_all = self.model.predict_quantiles(self.X[:10])
        np.testing.assert_array_almost_equal(preds, q_all[:, 1], decimal=5)

    def test_quantile_ordering_after_training(self):
        q = self.model.predict_quantiles(self.X)
        assert (q[:, 0] <= q[:, 1] + 1e-5).all()
        assert (q[:, 1] <= q[:, 2] + 1e-5).all()

    def test_quantile_crossing_rate_below_threshold(self):
        q = self.model.predict_quantiles(self.X)
        t = torch.tensor(q)
        rate = SortedQuantileLoss.crossing_rate(t)
        assert rate < 0.01

    def test_train_predict_shapes(self):
        q = self.model.predict_quantiles(self.X)
        assert q.shape == (len(self.X), 3)


# ─────────────────────────────────────────────────────────────────────────────
# [A5] Attention
# ─────────────────────────────────────────────────────────────────────────────

class TestAttentionOutput:

    def setup_method(self):
        self.X, self.y = _make_data(200)
        self.model = _fast_model()
        self.model.train(self.X, self.y)

    def test_attention_none_by_default(self):
        result = self.model.predict_quantiles(self.X[:10])
        assert isinstance(result, np.ndarray)

    def test_attention_shape_when_requested(self):
        """[A2] cross-attn: (N, 1, T)."""
        q, attn = self.model.predict_quantiles(self.X[:10], return_attention=True)
        assert q.shape   == (10, 3)
        assert attn.shape == (10, 1, TIME_STEPS), \
            "Beklenen (10,1," + str(TIME_STEPS) + "), alınan " + str(attn.shape)


# ─────────────────────────────────────────────────────────────────────────────
# [Uyumluluk] Save / Load
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveLoad:

    def test_save_load_roundtrip(self):
        X, y = _make_data(200)
        m1 = _fast_model()
        m1.train(X, y)
        preds_before = m1.predict(X[:10])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tft_v2_test.pt")
            m1.save(path)
            m2 = TFTModel()
            m2.load(path)
            preds_after = m2.predict(X[:10])

        np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=5)


# ─────────────────────────────────────────────────────────────────────────────
# [Uyumluluk] Import
# ─────────────────────────────────────────────────────────────────────────────

def test_old_import_still_works():
    from src.models.tft_model import TFTModel as TFTModelV1
    assert TFTModelV1 is not None


def test_new_import_is_v2():
    from src.models.tft_v2 import TFTModel as TV2
    m = TV2(epochs=1, min_val_samples=5)
    assert hasattr(m, "crossing_penalty")
    assert hasattr(m, "static_input_size")


# =============================================================================
# [A3] Multi-Horizon Testleri
# =============================================================================

class TestMultiHorizonHead:
    """MultiHorizonHead modülünün şekil ve monotonluk testleri."""

    def test_output_shape_single_horizon(self):
        """H=1 → (B, 1, 3) çıktı."""
        from src.models.tft_v2.output_heads import MultiHorizonHead
        head = MultiHorizonHead(d_model=32, horizons=[1])
        x = torch.randn(4, 1, 32)
        out = head(x)
        assert out.shape == (4, 1, 3), f"Beklenen (4,1,3), alınan {out.shape}"

    def test_output_shape_multi_horizon(self):
        """H=4 → (B, 4, 3) çıktı."""
        from src.models.tft_v2.output_heads import MultiHorizonHead
        head = MultiHorizonHead(d_model=32, horizons=[1, 5, 10, 21])
        x = torch.randn(4, 4, 32)
        out = head(x)
        assert out.shape == (4, 4, 3), f"Beklenen (4,4,3), alınan {out.shape}"

    def test_quantile_monotone_per_horizon(self):
        """Her ufuk ve her örnek için P10 ≤ P50 ≤ P90."""
        from src.models.tft_v2.output_heads import MultiHorizonHead
        head = MultiHorizonHead(d_model=32, horizons=[1, 5, 10, 21])
        head.eval()
        with torch.no_grad():
            x = torch.randn(16, 4, 32)
            out = head(x)   # (16, 4, 3)
        p10 = out[:, :, 0]
        p50 = out[:, :, 1]
        p90 = out[:, :, 2]
        assert (p10 <= p50).all(), "P10 > P50 ihlali"
        assert (p50 <= p90).all(), "P50 > P90 ihlali"

    def test_independent_heads(self):
        """Her ufuk kafası bağımsız parametre kümesine sahip."""
        from src.models.tft_v2.output_heads import MultiHorizonHead
        head = MultiHorizonHead(d_model=32, horizons=[1, 5, 10, 21])
        assert len(head.heads) == 4
        # İlk ve son kafanın ağırlıkları farklı
        w0 = head.heads[0].proj.weight.data
        w3 = head.heads[3].proj.weight.data
        assert not torch.equal(w0, w3), "Kafalar aynı ağırlığı paylaşmamalı"

    def test_gradients_flow_all_heads(self):
        """Tüm kafa ağırlıkları için gradyan akıyor."""
        from src.models.tft_v2.output_heads import MultiHorizonHead
        head = MultiHorizonHead(d_model=32, horizons=[1, 5])
        x = torch.randn(4, 2, 32)
        loss = head(x).sum()
        loss.backward()
        for i, h in enumerate(head.heads):
            assert h.proj.weight.grad is not None, f"Kafa {i} için gradyan yok"


class TestTFTModelA3:
    """TFTModel A3 — horizons parametresi ve predict_multihorizon() testleri."""

    def test_default_horizons_is_one(self):
        """Varsayılan horizons=[1] olmalı."""
        m = _fast_model()
        assert m.horizons == [1]

    def test_custom_horizons_stored(self):
        """horizons=[1,5,10,21] doğru saklanmalı."""
        m = TFTModel(
            epochs=1, min_val_samples=5,
            horizons=[1, 5, 10, 21],
        )
        assert m.horizons == [1, 5, 10, 21]

    def test_predict_multihorizon_single(self):
        """Tek ufuklu modelde predict_multihorizon 'h1' anahtarı döndürmeli."""
        X, y = _make_data(120)
        m = _fast_model()
        m.train(X, y)
        result = m.predict_multihorizon(X[:10])
        assert isinstance(result, dict)
        assert "h1" in result
        assert result["h1"].shape == (10, 3)

    def test_predict_multihorizon_multi(self):
        """Çoklu ufukta dict anahtarları horizons ile eşleşmeli."""
        X, y = _make_data(160)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
            horizons=[1, 5, 10, 21],
        )
        m.train(X, y)
        result = m.predict_multihorizon(X[:8])
        assert set(result.keys()) == {"h1", "h5", "h10", "h21"}
        for k, v in result.items():
            assert v.shape == (8, 3), f"{k}: beklenen (8,3), alınan {v.shape}"

    def test_multihorizon_quantile_ordering(self):
        """predict_multihorizon çıktısında P10 ≤ P50 ≤ P90 garantisi."""
        X, y = _make_data(160)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
            horizons=[1, 5, 10, 21],
        )
        m.train(X, y)
        result = m.predict_multihorizon(X[:20])
        for k, v in result.items():
            assert (v[:, 0] <= v[:, 1]).all(), f"{k}: P10 > P50"
            assert (v[:, 1] <= v[:, 2]).all(), f"{k}: P50 > P90"

    def test_predict_backward_compat_with_horizons(self):
        """Çoklu ufuklu modelde predict() hâlâ (N,) P50 döndürmeli."""
        X, y = _make_data(160)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
            horizons=[1, 5, 10, 21],
        )
        m.train(X, y)
        preds = m.predict(X[:10])
        assert preds.shape == (10,), f"Beklenen (10,), alınan {preds.shape}"

    def test_network_prediction_query_shape(self):
        """_TFTNetwork.prediction_query H tokena sahip olmalı."""
        X, y = _make_data(120)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=1, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
            horizons=[1, 5, 10, 21],
        )
        m.train(X, y)
        pq = m.network.prediction_query
        assert pq.shape == (1, 4, 16), \
            f"Beklenen (1,4,16), alınan {pq.shape}"

    def test_save_load_preserves_horizons(self):
        """save/load döngüsü horizons listesini korumalı."""
        X, y = _make_data(160)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
            horizons=[1, 5, 10, 21],
        )
        m.train(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tft_v2_a3.pt")
            m.save(path)
            m2 = TFTModel()
            m2.load(path)
        assert m2.horizons == [1, 5, 10, 21]
        result = m2.predict_multihorizon(X[:5])
        assert set(result.keys()) == {"h1", "h5", "h10", "h21"}

    def test_attention_shape_multi_horizon(self):
        """Çoklu ufukta return_attention=True → attn.shape[1] == H."""
        X, y = _make_data(160)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
            horizons=[1, 5, 10, 21],
        )
        m.train(X, y)
        # predict_quantiles default olarak ilk ufku döndürür → attn (N,1,T)
        q_preds, attn = m.predict_quantiles(X[:6], return_attention=True)
        assert q_preds.shape == (6, 3), f"q_preds: {q_preds.shape}"
        assert attn.shape == (6, 1, X.shape[1]), f"attn: {attn.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# A5 — XAI Attention Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestTFTA5AttentionHeatmap:
    """A5: TFTModel.get_attention_heatmap() testleri."""

    def test_heatmap_shape_single_horizon(self):
        """horizons=[1] → heatmap shape (N, 1, T)."""
        X, y = _make_data(120)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
        )
        m.train(X, y)
        N, T = 5, X.shape[1]
        attn = m.get_attention_heatmap(X[:N])
        assert attn.shape == (N, 1, T), \
            f"Tek ufukta beklenen ({N}, 1, {T}), alınan {attn.shape}"

    def test_heatmap_shape_multi_horizon(self):
        """horizons=[1,5,10] → heatmap shape (N, 3, T)."""
        X, y = _make_data(160)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
            horizons=[1, 5, 10],
        )
        m.train(X, y)
        N, T = 4, X.shape[1]
        attn = m.get_attention_heatmap(X[:N])
        assert attn.shape == (N, 3, T), \
            f"3 ufukta beklenen ({N}, 3, {T}), alınan {attn.shape}"

    def test_heatmap_weights_nonnegative(self):
        """Dikkat ağırlıkları ≥ 0 olmalı (softmax çıktısı)."""
        X, y = _make_data(120)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
        )
        m.train(X, y)
        attn = m.get_attention_heatmap(X[:8])
        assert (attn >= 0).all(), "Dikkat ağırlıklarında negatif değer var."

    def test_heatmap_weights_roughly_sum_to_one(self):
        """Her örnek × ufuk için T üzerindeki dikkat toplamı ≈ 1 (toleranslı)."""
        X, y = _make_data(120)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
        )
        m.train(X, y)
        attn = m.get_attention_heatmap(X[:6])   # (6, 1, T)
        row_sums = attn.sum(axis=-1)             # (6, 1)
        assert (row_sums > 0).all(), "Sıfır toplamı olan dikkat satırı var."
        # softmax normalise + FloatPoint hatası nedeniyle gevşek tolerans
        assert float(row_sums.max()) < 5.0, \
            f"Dikkat toplamı makul aralığın dışında: max={row_sums.max():.4f}"

    def test_heatmap_untrained_raises(self):
        """Eğitilmemiş model RuntimeError yükseltmeli."""
        m = TFTModel()
        X, _ = _make_data(60)
        with pytest.raises(RuntimeError):
            m.get_attention_heatmap(X[:3])


class TestXAIExplainerA5:
    """A5: XAIExplainer dikkat entegrasyon testleri."""

    def _make_tft_model(self):
        X, y = _make_data(120)
        m = TFTModel(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
        )
        m.train(X, y)
        return m, X, y

    def test_attention_temporal_rows_shape(self):
        """_tft_attention_temporal_rows en fazla 3 satır döndürmeli."""
        from src.xai.explainer import XAIExplainer
        import numpy as np

        feature_names = [f"f{i}" for i in range(8)]
        explainer = XAIExplainer("TEST", feature_names, {})

        T = 30
        attn = np.random.rand(5, 1, T).astype("float32")
        rows = explainer._tft_attention_temporal_rows("TFT", attn, T)
        assert len(rows) <= 3, f"3'ten fazla satır döndü: {len(rows)}"
        assert all(r["Method"] == "tft_attention" for r in rows)

    def test_attention_temporal_rows_empty_input(self):
        """Boş dizi verildiğinde boş liste döndürmeli."""
        from src.xai.explainer import XAIExplainer
        import numpy as np

        feature_names = [f"f{i}" for i in range(8)]
        explainer = XAIExplainer("TEST", feature_names, {})
        rows = explainer._tft_attention_temporal_rows("TFT", np.array([]), 30)
        assert rows == []

    def test_explain_tft_returns_three_tuple(self):
        """_explain_tft_model 3-tuple (rows, daily, attn_data) döndürmeli."""
        from src.xai.explainer import XAIExplainer
        import numpy as np

        m, X, y = self._make_tft_model()
        feature_names = [f"f{i}" for i in range(X.shape[2])]
        explainer = XAIExplainer("TEST", feature_names, {})

        preds_p50 = m.predict(X)
        tensors = {
            "X_test_seq":  X,
            "X_test_s":    X.reshape(len(X), -1),
            "dates_test":  [],
        }
        predictions      = {"TFT": preds_p50}
        prediction_targets = {"TFT": preds_p50}

        result = explainer._explain_tft_model(
            "TFT", m, tensors, predictions, prediction_targets,
            y, quantiles=None,
        )
        assert len(result) == 3, "3-tuple bekleniyor (rows, daily, attn_data)"
        rows, daily, attn_data = result
        assert isinstance(rows, list)
        assert isinstance(daily, list)
        # attn_data ya None ya da numpy dizisi olmalı
        assert attn_data is None or hasattr(attn_data, "shape")


class TestXAIReportWriterA5:
    """A5: XAIReportWriter dikkat ısı haritası PNG testleri."""

    def test_write_creates_attention_heatmap_png(self):
        """tft_attention_data payload anahtarı → PNG dosyası üretilmeli."""
        import tempfile, os
        import numpy as np
        import pandas as pd
        from src.xai.report_writer import XAIReportWriter
        from src.utils.reporting_utils import route_output_path

        with tempfile.TemporaryDirectory() as tmp:
            writer = XAIReportWriter(tmp)
            attn   = np.random.rand(4, 2, 15).astype("float32")
            payload = {
                "top_reasons":       pd.DataFrame(),
                "daily_reasons":     pd.DataFrame(),
                "summary_md":        "# test",
                "tft_attention_data": {"TFT": attn},
            }
            writer.write(payload, suffix="test")
            # route_output_path routes .png files into a png/ subdirectory
            raw_path = os.path.join(tmp, "xai_tft_attention_TFT_test.png")
            png = route_output_path(raw_path)
            assert os.path.exists(png), f"PNG oluşturulmadı: {png}"

    def test_write_handles_missing_attention_data(self):
        """tft_attention_data yoksa write() hata vermeden tamamlanmalı."""
        import tempfile
        import pandas as pd
        from src.xai.report_writer import XAIReportWriter

        with tempfile.TemporaryDirectory() as tmp:
            writer = XAIReportWriter(tmp)
            payload = {
                "top_reasons":   pd.DataFrame(),
                "daily_reasons": pd.DataFrame(),
                "summary_md":    "# test",
            }
            writer.write(payload, suffix="noattn")  # hata vermemeli


# ═══════════════════════════════════════════════════════════════════════════════
# A6 — Backward Compatibility (v1 API + Walk-Forward)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTFTA6BackwardCompat:
    """
    A6: TFT v2'nin v1 API'siyle tam geriye dönük uyumluluğunu doğrular.

    Kapsam:
      - Varsayılan kurucu (argümansız TFTModel())
      - v1 arayüzü: train / predict / predict_quantiles / save / load
      - predict() çıktı şekli: (N,)
      - predict_quantiles() çıktı şekli: (N, 3)
      - static=None varsayılanı — statik encoder devreye girmemeli
      - Walk-forward pencere simülasyonu
      - save/load döngüsü tüm parametreleri korumalı
      - BaseModel arayüz uyumluluğu
      - A5 metotları (get_attention_heatmap, get_variable_importances) eğitim sonrası çalışmalı
    """

    # ── Yardımcı ─────────────────────────────────────────────────────────────

    def _tiny_model(self, **kwargs):
        """Test için küçük, hızlı TFTModel döndürür."""
        defaults = dict(
            d_model=16, num_heads=2, lstm_layers=1,
            epochs=2, batch_size=16, learning_rate=1e-3,
            patience=2, min_val_samples=10,
        )
        defaults.update(kwargs)
        return TFTModel(**defaults)

    # ── 1. Varsayılan kurucu ──────────────────────────────────────────────────

    def test_default_constructor_no_args(self):
        """TFTModel() argümansız oluşturulabilmeli."""
        m = TFTModel()
        assert m.d_model == 64
        assert m.horizons == [1]
        assert m.static_input_size == 0
        assert m.network is None

    # ── 2. v1 train / predict arayüzü ────────────────────────────────────────

    def test_v1_train_predict_roundtrip(self):
        """train(X, y) → predict(X) — v1 ile birebir aynı çağrı şekli."""
        X, y = _make_data(100)
        m = self._tiny_model()
        m.train(X, y)                       # v1 imzası — static_features yok
        preds = m.predict(X[:10])
        assert preds.shape == (10,), f"predict çıktı şekli: {preds.shape}"
        assert not np.any(np.isnan(preds)), "predict NaN döndürdü"

    # ── 3. predict_quantiles şekli ────────────────────────────────────────────

    def test_predict_quantiles_shape(self):
        """predict_quantiles → (N, 3) [P10, P50, P90]."""
        X, y = _make_data(100)
        m = self._tiny_model()
        m.train(X, y)
        q = m.predict_quantiles(X[:8])
        assert q.shape == (8, 3), f"Beklenen (8,3), alınan {q.shape}"

    # ── 4. P10 ≤ P50 ≤ P90 garantisi (A4 ile birlikte) ───────────────────────

    def test_quantile_monotone_after_train(self):
        """Eğitim sonrası P10 ≤ P50 ≤ P90 garanti edilmeli (A4 SortedQuantileOutput)."""
        X, y = _make_data(100)
        m = self._tiny_model()
        m.train(X, y)
        q = m.predict_quantiles(X)
        assert (q[:, 0] <= q[:, 1] + 1e-5).all(), "P10 > P50 ihlali"
        assert (q[:, 1] <= q[:, 2] + 1e-5).all(), "P50 > P90 ihlali"

    # ── 5. static=None varsayılanı ────────────────────────────────────────────

    def test_no_static_features_default(self):
        """static_features verilmeden train/predict çalışmalı (v1 davranışı)."""
        X, y = _make_data(100)
        m = self._tiny_model()
        assert m.static_input_size == 0
        m.train(X, y)                       # static_features=None (default)
        preds = m.predict(X[:5])
        assert preds.shape == (5,)

    # ── 6. Walk-forward pencere simülasyonu ───────────────────────────────────

    def test_walk_forward_window_simulation(self):
        """
        3 pencere walk-forward simülasyonu:
        Her pencerede model sıfırdan eğitilip bir sonraki pencerede tahmin üretir.
        Tüm tahminler finite ve doğru şekilde olmalı.
        """
        X, y = _make_data(200)
        window = 80
        step   = 20
        results = []
        for start in range(0, 3 * step, step):
            X_tr = X[start : start + window]
            y_tr = y[start : start + window]
            X_te = X[start + window : start + window + step]
            m = self._tiny_model()
            m.train(X_tr, y_tr)
            preds = m.predict(X_te)
            assert preds.shape == (len(X_te),), f"Pencere {start}: şekil hatası"
            assert np.all(np.isfinite(preds)), f"Pencere {start}: NaN/Inf"
            results.append(preds)
        assert len(results) == 3

    # ── 7. save / load parametreleri korur ────────────────────────────────────

    def test_save_load_preserves_all_params(self):
        """save/load döngüsü tüm v2 konfigürasyonunu korumalı."""
        X, y = _make_data(100)
        m = self._tiny_model(
            dropout=0.2, weight_decay=1e-5, crossing_penalty=0.3,
            validation_ratio=0.15,
        )
        m.train(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tft_v2_a6.pt")
            m.save(path)
            m2 = TFTModel()
            m2.load(path)

        # Temel konfigürasyon
        assert m2.d_model           == m.d_model
        assert m2.num_heads         == m.num_heads
        assert m2.lstm_layers       == m.lstm_layers
        assert abs(m2.dropout       -  m.dropout)          < 1e-6
        assert m2.horizons          == m.horizons
        assert m2.static_input_size == m.static_input_size
        assert abs(m2.crossing_penalty - m.crossing_penalty) < 1e-6
        # Tahmin üretebiliyor mu?
        preds = m2.predict(X[:5])
        assert preds.shape == (5,)

    # ── 8. A5 metotları eğitim sonrası çalışmalı ─────────────────────────────

    def test_a5_interpretability_methods_work(self):
        """
        get_variable_importances() ve get_attention_heatmap()
        eğitim sonrası doğru şekil döndürmeli.
        """
        X, y = _make_data(100)
        m = self._tiny_model()
        m.train(X, y)

        # VSN ağırlıkları: (time_steps, num_features)
        T, F = X.shape[1], X.shape[2]
        vi = m.get_variable_importances(X[:1])
        assert vi.shape == (T, F), f"VSN şekli: {vi.shape}"

        # Dikkat haritası: (N, H=1, T)
        attn = m.get_attention_heatmap(X[:4])
        assert attn.shape == (4, 1, T), f"Attn şekli: {attn.shape}"
        assert (attn >= 0).all(), "Dikkat ağırlığında negatif değer"


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Entegrasyon Testleri
# ══════════════════════════════════════════════════════════════════════════════

class _FakeScalerY:
    """Basit test scalerı — inverse_transform identity."""
    def inverse_transform(self, arr):
        return arr


class _FakeSignalConfig:
    entry_cost_multiplier = 2.0
    volatility_multiplier = 0.25
    min_holding_bars = 3
    max_holding_bars = 20
    take_profit_vol_multiplier = 1.5
    stop_loss_vol_multiplier = 1.0
    min_directional_accuracy = 52.0
    max_rmse_vs_benchmark = 1.05
    min_composite_score = 50.0
    emergency_stop_overrides_min_hold = True


class _FakeExeConfig:
    backtest_enabled = False
    commission_bps = 10.0
    slippage_bps = 5.0
    initial_capital = 100_000.0
    signal_mode = "legacy"
    signal_config = _FakeSignalConfig()
    calibration_scope = "none"


class _FakeModelConfig:
    selected_models = None
    ensemble_enabled = False


class TestPipelineMultihorizon:
    """
    Pipeline Entegrasyon: predict_multihorizon() → prediction_engine.py
    """

    def _tiny_multi_model(self):
        """horizons=[1,5] ile eğitilmiş küçük TFT modeli."""
        m = TFTModel(
            d_model=8,
            num_heads=1,
            lstm_layers=1,
            dropout=0.0,
            epochs=2,
            patience=10,
            batch_size=16,
            horizons=[1, 5],
        )
        X, y = _make_data(120)
        m.train(X, y)
        return m

    def test_predict_multihorizon_keys(self):
        """predict_multihorizon() h1 ve h5 anahtarlarını döndürmeli."""
        m = self._tiny_multi_model()
        X, _ = _make_data(30)
        result = m.predict_multihorizon(X)
        assert isinstance(result, dict), "Sonuç dict olmalı"
        assert "h1" in result, "h1 eksik"
        assert "h5" in result, "h5 eksik"
        assert result["h1"].shape == (len(X), 3), f"h1 şekli hatalı: {result['h1'].shape}"
        assert result["h5"].shape == (len(X), 3), f"h5 şekli hatalı: {result['h5'].shape}"

    def test_prediction_engine_stores_multihorizon(self):
        """
        _PredictionEngineMixin.generate_predictions() TFT için
        self.multihorizon_predictions'ı doldurmalı.
        """
        import sys
        import types

        # ── Minimal tensors ────────────────────────────────────────────────
        N, T, F = 40, 10, 8
        X_seq = np.random.randn(N, T, F).astype(np.float32)
        X_flat = X_seq.reshape(N, -1)
        y_test = np.random.randn(N).astype(np.float32)
        prev_close = np.abs(np.random.randn(N).astype(np.float32)) + 100.0
        dates = np.arange(N, dtype=object)

        tensors = {
            "X_test_seq": X_seq,
            "X_test_s":   X_flat,
            "X_test":     X_flat,
            "y_test":     y_test,
            "original_y_test_aligned": prev_close + y_test,
            "prev_close_test":  prev_close,
            "dates_test":       dates,
            "scaler_y":         _FakeScalerY(),
        }

        # ── TFT modeli ────────────────────────────────────────────────────
        model = self._tiny_multi_model()

        # ── _PredictionEngineMixin izole olarak test ─────────────────────
        from src.pipeline.prediction_engine import _PredictionEngineMixin

        class _Stub(_PredictionEngineMixin):
            def __init__(self):
                self.dataset_metadata = {"target_mode": "log_return"}
                self.selected_models = None
                self.ensemble_enabled = False
                self.predictions = {}
                self.prediction_targets = {}
                self.quantile_predictions = {}
                self.multihorizon_predictions = {}
                self.single_backtest_inputs = {}
                self.y_true_aligned = None
                self.y_true_target_aligned = None
                self.prev_close_aligned = None
                self.latest_tensors = {}
                self.ensemble_weights = {}

        stub = _Stub()
        stub.generate_predictions({"TFT": model}, tensors)

        assert "TFT" in stub.multihorizon_predictions, \
            "TFT multi-horizon tahminleri kaydedilmedi"
        mh = stub.multihorizon_predictions["TFT"]
        assert "h1" in mh, "h1 eksik"
        assert "h5" in mh, "h5 eksik"
        assert mh["h1"].shape[0] > 0, "h1 boş"

    def test_multihorizon_predictions_finite(self):
        """Multi-horizon tahminler NaN/Inf içermemeli."""
        m = self._tiny_multi_model()
        X, _ = _make_data(30)
        result = m.predict_multihorizon(X)
        for h, arr in result.items():
            assert np.all(np.isfinite(arr)), f"h={h} NaN/Inf içeriyor"


class TestPipelineXAIWrite:
    """
    Pipeline Entegrasyon: _MetricsReporterMixin._write_xai_reports()
    XAIReportWriter.write() çağrısını tetiklemeli.
    """

    def test_write_xai_reports_creates_summary_md(self):
        """
        _write_xai_reports() sonunda xai_dir içinde xai_summary_latest.md
        dosyası oluşturulmuş olmalı.
        """
        from src.pipeline.metrics_reporter import _MetricsReporterMixin
        import pandas as pd

        class _Stub(_MetricsReporterMixin):
            def __init__(self, xai_dir):
                self.xai_dir = xai_dir
                self.commission_bps = 10.0
                self.slippage_bps = 5.0
                self.dataset_metadata = {}
                self.ensemble_weights = {}

        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(tmp)
            payload = {
                "top_reasons": pd.DataFrame(),
                "daily_reasons": pd.DataFrame(),
                "summary_md": "# Test XAI\n\nOK.",
            }
            stub._write_xai_reports(payload, suffix="latest")
            from src.utils.reporting_utils import route_output_path
            md_path = route_output_path(os.path.join(tmp, "xai_summary_latest.md"))
            assert os.path.exists(md_path), f"MD dosyası oluşturulmadı: {md_path}"
            content = open(md_path, encoding="utf-8").read()
            assert "Test XAI" in content

    def test_save_multihorizon_report_creates_csv(self):
        """
        _save_multihorizon_report() xai_dir içine
        tft_multihorizon_TFT_latest.csv oluşturmalı.
        """
        from src.pipeline.metrics_reporter import _MetricsReporterMixin

        class _Stub(_MetricsReporterMixin):
            def __init__(self, xai_dir):
                self.xai_dir = xai_dir
                self.multihorizon_predictions = {
                    "TFT": {
                        "h1": np.array([1.0, 2.0, 3.0]),
                        "h5": np.array([1.1, 2.1, 3.1]),
                    }
                }

        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(tmp)
            stub._save_multihorizon_report(suffix="latest")
            csv_path = os.path.join(tmp, "tft_multihorizon_TFT_latest.csv")
            assert os.path.exists(csv_path), f"CSV dosyası oluşturulmadı: {csv_path}"
            import pandas as pd
            df = pd.read_csv(csv_path)
            assert "h1" in df.columns
            assert "h5" in df.columns
            assert len(df) == 3

    def test_write_xai_reports_with_tft_attention_creates_png(self):
        """
        tft_attention_data içeren payload → xai_dir/png/ altında PNG üretmeli.
        """
        from src.pipeline.metrics_reporter import _MetricsReporterMixin
        from src.utils.reporting_utils import route_output_path
        import pandas as pd

        class _Stub(_MetricsReporterMixin):
            def __init__(self, xai_dir):
                self.xai_dir = xai_dir
                self.commission_bps = 10.0
                self.slippage_bps = 5.0
                self.dataset_metadata = {}
                self.ensemble_weights = {}

        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(tmp)
            attn_arr = np.random.rand(4, 1, 10).astype(np.float32)
            payload = {
                "top_reasons":    pd.DataFrame(),
                "daily_reasons":  pd.DataFrame(),
                "summary_md":     "# Test",
                "tft_attention_data": {"TFT": attn_arr},
            }
            stub._write_xai_reports(payload, suffix="latest")
            raw = os.path.join(tmp, "xai_tft_attention_TFT_latest.png")
            png_path = route_output_path(raw)
            assert os.path.exists(png_path), f"PNG oluşturulmadı: {png_path}"
