# -*- coding: utf-8 -*-
"""
test_owner_forward_contract.py - E1 owner-forward epigi KARAKTERIZASYON golden'lari.

Bu dosya, owner-forward "magic" -> EvaluationContext/EvaluationState DI gecisi
(`docs/wiki/e1-owner-forward-epic.md`) boyunca DEGISMEZ referanstir. Testler
mekanizmayi (owner-forward `__getattr__`/`__setattr__`) DEGIL, public gozlemlenen
davranisi kilitler; boylece servisler DI'ya cevrilirken bu golden'lar yesil kalir.

Kapsam:
  1. EvaluationManager kurulus state golden'i (baslangic degerleri).
  2. manager <-> manager.state alias kontrati (ayni nesne, paylasilan mutable state).
  3. Servis-kompozisyonu uzerinden delege edilen saf hesaplamalar
     (_target_to_price / _weighted_average / _base_predictions_for_ensemble).
  4. Determinizm: ayni girdi ayni cikti.

Forward forecast ucundan-uca golden'i AYRICA `tests/test_forecasting.py`
(run_symbol band-clip + persistence) ve `tests/test_forecast_workflows.py`
icinde tutulur; burada duplike edilmez (Faz 5 referansi).

Calistirma: python -m pytest tests/test_owner_forward_contract.py -v
"""

import os
import tempfile

import numpy as np
import pytest

from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.evaluation_manager import EvaluationManager
from src.experiments.experiment_tracker import ExperimentTracker


def _make_manager(tmpdir: str, *, target_mode: str = "log_return") -> EvaluationManager:
    models_dir = os.path.join(tmpdir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return EvaluationManager(
        stock_symbol="TEST",
        outputs_dir=tmpdir,
        models_dir=models_dir,
        tracker=ExperimentTracker(os.path.join(tmpdir, "exp")),
        feature_names=["f1", "f2", "f3"],
        dataset_hash="abc123",
        dataset_metadata={"target_mode": target_mode, "signal_threshold_config": {}},
        exe_cfg=ExecutionConfig(),
        model_cfg=ModelConfig(ensemble_enabled=True),
    )


# --------------------------------------------------------------------------- #
#  1. Kurulus state golden                                                     #
# --------------------------------------------------------------------------- #

# Servisler tarafindan mutasyona ugrayan, kurulusta bos/varsayilan olmasi gereken
# public mutable state yuzeyi. Faz 1 bunlari EvaluationState'e tasir; degerler
# AYNI kalmali (property forward ile gozlemlenir).
EMPTY_DICT_STATE = (
    "predictions",
    "prediction_targets",
    "quantile_predictions",
    "single_backtest_inputs",
    "latest_tensors",
    "latest_backtest_results",
    "latest_backtest_metrics",
    "latest_model_metrics",
    "ensemble_weights",
    "ensemble_weight_scope",
    "signal_threshold_calibration_summary",
)

NONE_STATE = (
    "y_true_aligned",
    "y_true_target_aligned",
    "prev_close_aligned",
)


def test_initial_mutable_state_golden():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = _make_manager(tmpdir)
        for attr in EMPTY_DICT_STATE:
            assert getattr(m, attr) == {}, f"{attr} kurulusta bos dict olmali"
        for attr in NONE_STATE:
            assert getattr(m, attr) is None, f"{attr} kurulusta None olmali"


def test_initial_signal_state_golden():
    with tempfile.TemporaryDirectory() as tmpdir:
        exe = ExecutionConfig()
        m = _make_manager(tmpdir)
        assert m.signal_threshold_source == "default_config"
        assert m.signal_config is exe.signal_config or m.signal_config == exe.signal_config
        assert m.default_signal_config == m.signal_config
        assert m.xai_dir == os.path.join(tmpdir, "xai")


# --------------------------------------------------------------------------- #
#  2. manager <-> state alias kontrati                                          #
# --------------------------------------------------------------------------- #

# Servisler owner.predictions'a yazar, workflow'lar ayni nesneyi okur. Refactor
# sonrasi manager.predictions -> manager.state.predictions property'sine dusse de
# AYNI nesne kimligi korunmali (paylasilan mutable state kontrati).
STATE_ALIASED_ATTRS = (
    "predictions",
    "prediction_targets",
    "quantile_predictions",
    "single_backtest_inputs",
    "latest_tensors",
    "latest_backtest_results",
    "latest_backtest_metrics",
    "latest_model_metrics",
    "ensemble_weights",
)


def test_manager_state_alias_identity():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = _make_manager(tmpdir)
        for attr in STATE_ALIASED_ATTRS:
            assert getattr(m, attr) is getattr(m.state, attr), (
                f"manager.{attr} ile manager.state.{attr} ayni nesne olmali"
            )


def test_shared_state_mutation_visible_both_directions():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = _make_manager(tmpdir)
        # state uzerinden yaz -> manager'dan oku
        m.state.predictions["A"] = np.array([1.0, 2.0])
        assert "A" in m.predictions
        # manager uzerinden yaz -> state'ten oku
        m.ensemble_weights["X"] = {"A": 1.0}
        assert m.state.ensemble_weights["X"] == {"A": 1.0}


# --------------------------------------------------------------------------- #
#  3. Servis-kompozisyonu uzerinden saf hesaplama golden'lari                  #
# --------------------------------------------------------------------------- #


def test_target_to_price_return_mode_golden():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = _make_manager(tmpdir, target_mode="return")
        prices = m._target_to_price(np.array([0.10, -0.05]), np.array([100.0, 200.0]))
        np.testing.assert_allclose(prices, np.array([110.0, 190.0]))


def test_target_to_price_price_mode_golden():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = _make_manager(tmpdir, target_mode="price")
        prices = m._target_to_price(np.array([123.0, 456.0]), np.array([1.0, 1.0]))
        np.testing.assert_allclose(prices, np.array([123.0, 456.0]))


def test_target_to_price_log_return_mode_golden():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = _make_manager(tmpdir, target_mode="log_return")
        prices = m._target_to_price(np.array([0.0, np.log(2.0)]), np.array([100.0, 50.0]))
        np.testing.assert_allclose(prices, np.array([100.0, 100.0]), rtol=1e-9)


def test_target_to_price_invalid_mode_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = _make_manager(tmpdir, target_mode="bogus")
        with pytest.raises(ValueError, match="Desteklenmeyen target_mode"):
            m._target_to_price(np.array([1.0]), np.array([1.0]))


def test_weighted_average_golden():
    preds = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([3.0, 2.0, 1.0])}
    out = EvaluationManager._weighted_average(preds, {"A": 0.75, "B": 0.25})
    np.testing.assert_allclose(out, np.array([1.5, 2.0, 2.5]))


def test_weighted_average_zero_weights_falls_back_to_uniform():
    preds = {"A": np.array([2.0, 4.0]), "B": np.array([4.0, 8.0])}
    out = EvaluationManager._weighted_average(preds, {"A": 0.0, "B": 0.0})
    np.testing.assert_allclose(out, np.array([3.0, 6.0]))


def test_base_predictions_for_ensemble_filters_ensembles_and_empty():
    preds = {
        "Ridge Return": np.array([1.0, 2.0]),
        "Ensemble Equal Weight": np.array([1.5, 1.5]),
        "Empty Model": np.array([]),
        "XGBoost": np.array([3.0, 4.0]),
    }
    base = EvaluationManager._base_predictions_for_ensemble(preds)
    assert set(base.keys()) == {"Ridge Return", "XGBoost"}


# --------------------------------------------------------------------------- #
#  4. Determinizm                                                              #
# --------------------------------------------------------------------------- #


def test_weighted_average_deterministic():
    preds = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([2.0, 1.0, 0.0])}
    w = {"A": 0.6, "B": 0.4}
    out1 = EvaluationManager._weighted_average(preds, w)
    out2 = EvaluationManager._weighted_average(preds, w)
    np.testing.assert_array_equal(out1, out2)
