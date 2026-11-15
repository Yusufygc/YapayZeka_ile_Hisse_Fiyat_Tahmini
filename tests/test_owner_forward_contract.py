# -*- coding: utf-8 -*-
"""
test_owner_forward_contract.py - EvaluationManager davranis golden'lari.

Manager -> servis/workflow kompozisyonu boyunca DEGISMEZ referanstir. Alias/forward
mekanizmasini degil, public gozlemlenen davranisi kilitler:
  1. Kurulus: EvaluationContext (config/identity) ve EvaluationState (mutable) golden'i.
  2. ExecutionConfig/ModelConfig -> EvaluationContext eslemesi (from_config).
  3. Servis/workflow'lar manager ile AYNI context/state nesnelerini paylasir.
  4. generate_predictions yolu state'i dolduran gercek cikti golden'i.
  5. Servis uzerinden saf hesaplamalar (_target_to_price / _weighted_average / ...).
  6. Determinizm.

Walk-forward / final-holdout run yollari `tests/test_p0_holdout_gate.py` ve
`tests/test_p1_dm_wiring.py` icinde; forward forecast ucundan-uca golden'i
`tests/test_forecasting.py` / `tests/test_forecast_workflows.py` icinde tutulur.

Calistirma: python -m pytest tests/test_owner_forward_contract.py -v
"""

import os
import tempfile

import numpy as np
import pytest

from _eval_helpers import make_eval_manager
from src.backtesting.signals import SignalConfig
from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.evaluation_manager import EvaluationManager
from src.pipeline.evaluation_services import (
    EvaluationContext,
    EvaluationState,
    PredictionService,
)


# --------------------------------------------------------------------------- #
#  1. Kurulus golden                                                           #
# --------------------------------------------------------------------------- #

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
        m = make_eval_manager(tmpdir)
        for attr in EMPTY_DICT_STATE:
            assert getattr(m.state, attr) == {}, f"state.{attr} kurulusta bos dict olmali"
        for attr in NONE_STATE:
            assert getattr(m.state, attr) is None, f"state.{attr} kurulusta None olmali"


def test_initial_signal_state_golden():
    with tempfile.TemporaryDirectory() as tmpdir:
        exe = ExecutionConfig()
        m = make_eval_manager(tmpdir)
        assert m.state.signal_threshold_source == "default_config"
        assert m.state.signal_config == exe.signal_config
        assert m.context.default_signal_config == m.state.signal_config
        assert m.context.xai_dir == os.path.join(tmpdir, "xai")
        # Kurulus, esik metadata snapshot'ini dataset_metadata'ya yazar.
        snapshot = m.context.dataset_metadata["signal_threshold_config"]
        assert snapshot["source"] == "default_config"


# --------------------------------------------------------------------------- #
#  2. Config -> EvaluationContext eslemesi                                     #
# --------------------------------------------------------------------------- #


def test_context_from_config_maps_execution_and_model_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        signal_cfg = SignalConfig(quality_gate_mode="soft")
        exe = ExecutionConfig(
            backtest_enabled=False,
            initial_capital=5000.0,
            commission_bps=7.0,
            slippage_bps=3.0,
            signal_mode="professional",
            signal_config=signal_cfg,
            calibration_scope="wf_train",
            signal_calibration_max_trials=12,
            signal_calibration_profile="research",
            signal_calibration_sampler="prefix",
            signal_calibration_seed=9,
            write_trade_logs=True,
            write_xai_tables=False,
            write_markdown_reports=False,
            min_signal_evaluation_folds=5,
            signal_calibration_train_ratio=0.6,
        )
        model = ModelConfig(ensemble_enabled=False, selected_models=["Ridge Return", "XGBoost"])
        stock_db = object()
        m = EvaluationManager(
            "TEST", tmpdir, os.path.join(tmpdir, "models"), ["f1"], "h", {"target_mode": "price"},
            exe, model, stock_db,
        )
        ctx = m.context
        assert ctx.exe_cfg is exe and ctx.model_cfg is model and ctx.stock_db is stock_db
        assert (ctx.stock_symbol, ctx.dataset_hash, ctx.feature_names) == ("TEST", "h", ["f1"])
        assert ctx.backtest_enabled is False
        assert (ctx.initial_capital, ctx.commission_bps, ctx.slippage_bps) == (5000.0, 7.0, 3.0)
        assert ctx.signal_mode == "professional"
        assert ctx.default_signal_config is signal_cfg and m.state.signal_config is signal_cfg
        assert ctx.signal_calibration_max_trials == 12
        assert (ctx.signal_calibration_profile, ctx.signal_calibration_sampler) == ("research", "prefix")
        assert ctx.signal_calibration_seed == 9
        assert ctx.write_trade_logs is True
        assert ctx.write_xai_tables is False and ctx.write_markdown_reports is False
        assert ctx.min_signal_evaluation_folds == 5
        assert ctx.signal_calibration_train_ratio == pytest.approx(0.6)
        assert ctx.ensemble_enabled is False
        assert ctx.selected_models == {"Ridge Return", "XGBoost"}


def test_context_from_config_selected_models_none_when_unset():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = make_eval_manager(tmpdir, ensemble_enabled=True)
        assert m.context.selected_models is None
        assert m.context.ensemble_enabled is True


# --------------------------------------------------------------------------- #
#  3. Servis / workflow paylasimi                                              #
# --------------------------------------------------------------------------- #


def test_services_and_workflows_share_manager_context_and_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = make_eval_manager(tmpdir)
        services = (
            m.prediction_service,
            m.backtest_service,
            m.signal_calibration_service,
            m.metrics_reporting_service,
        )
        workflows = (m.single_split_workflow, m.walk_forward_workflow, m.final_holdout_workflow)
        for obj in (*services, *workflows):
            assert obj.ctx is m.context
            assert obj.state is m.state

        # Servis state'e yazar -> workflow ayni nesneyi okur.
        m.prediction_service.state.predictions["A"] = np.array([1.0, 2.0])
        assert "A" in m.walk_forward_workflow.state.predictions


# --------------------------------------------------------------------------- #
#  4. generate_predictions run yolu                                            #
# --------------------------------------------------------------------------- #


class _FixedModel:
    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, X, **kwargs):
        return self._preds[-len(X):]


class _IdentityScaler:
    def inverse_transform(self, values):
        return np.asarray(values, dtype=float)


def test_generate_predictions_populates_state_golden():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = make_eval_manager(tmpdir, target_mode="return")
        n = 3
        tensors = {
            "X_test_s": np.zeros((n, 2)),
            "scaler_y": _IdentityScaler(),
            "y_test": np.array([0.10, -0.05, 0.02]),
            "prev_close_test": np.array([100.0, 200.0, 50.0]),
            "dates_test": np.array(["2025-01-02", "2025-01-03", "2025-01-06"], dtype="datetime64[ns]"),
            "original_y_test_aligned": np.array([110.0, 190.0, 51.0]),
        }
        m.generate_predictions({"Ridge Return": _FixedModel([0.10, -0.05, 0.02])}, tensors)

        np.testing.assert_allclose(m.state.predictions["Ridge Return"], np.array([110.0, 190.0, 51.0]))
        np.testing.assert_allclose(m.state.prediction_targets["Ridge Return"], np.array([0.10, -0.05, 0.02]))
        np.testing.assert_allclose(m.state.y_true_aligned, np.array([110.0, 190.0, 51.0]))
        np.testing.assert_allclose(m.state.prev_close_aligned, np.array([100.0, 200.0, 50.0]))
        assert m.state.latest_tensors is tensors


# --------------------------------------------------------------------------- #
#  5. Servis uzerinden saf hesaplama golden'lari                               #
# --------------------------------------------------------------------------- #


def _prediction_service(target_mode: str) -> PredictionService:
    return PredictionService(
        EvaluationContext(dataset_metadata={"target_mode": target_mode}), EvaluationState()
    )


def test_target_to_price_return_mode_golden():
    prices = _prediction_service("return")._target_to_price(np.array([0.10, -0.05]), np.array([100.0, 200.0]))
    np.testing.assert_allclose(prices, np.array([110.0, 190.0]))


def test_target_to_price_price_mode_golden():
    prices = _prediction_service("price")._target_to_price(np.array([123.0, 456.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(prices, np.array([123.0, 456.0]))


def test_target_to_price_log_return_mode_golden():
    prices = _prediction_service("log_return")._target_to_price(
        np.array([0.0, np.log(2.0)]), np.array([100.0, 50.0])
    )
    np.testing.assert_allclose(prices, np.array([100.0, 100.0]), rtol=1e-9)


def test_target_to_price_invalid_mode_raises():
    with pytest.raises(ValueError, match="Desteklenmeyen target_mode"):
        _prediction_service("bogus")._target_to_price(np.array([1.0]), np.array([1.0]))


def test_weighted_average_golden():
    preds = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([3.0, 2.0, 1.0])}
    out = PredictionService._weighted_average(preds, {"A": 0.75, "B": 0.25})
    np.testing.assert_allclose(out, np.array([1.5, 2.0, 2.5]))


def test_weighted_average_zero_weights_falls_back_to_uniform():
    preds = {"A": np.array([2.0, 4.0]), "B": np.array([4.0, 8.0])}
    out = PredictionService._weighted_average(preds, {"A": 0.0, "B": 0.0})
    np.testing.assert_allclose(out, np.array([3.0, 6.0]))


def test_base_predictions_for_ensemble_filters_ensembles_and_empty():
    preds = {
        "Ridge Return": np.array([1.0, 2.0]),
        "Ensemble Equal Weight": np.array([1.5, 1.5]),
        "Empty Model": np.array([]),
        "XGBoost": np.array([3.0, 4.0]),
    }
    base = PredictionService._base_predictions_for_ensemble(preds)
    assert set(base.keys()) == {"Ridge Return", "XGBoost"}


# --------------------------------------------------------------------------- #
#  6. Determinizm                                                              #
# --------------------------------------------------------------------------- #


def test_weighted_average_deterministic():
    preds = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([2.0, 1.0, 0.0])}
    w = {"A": 0.6, "B": 0.4}
    out1 = PredictionService._weighted_average(preds, w)
    out2 = PredictionService._weighted_average(preds, w)
    np.testing.assert_array_equal(out1, out2)
