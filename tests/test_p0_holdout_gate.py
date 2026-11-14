# -*- coding: utf-8 -*-
"""P0-6a/6b/7 regression tests: gate look-ahead, self-benchmark, selection set labelling.

* 6a: quality gate inputs of the holdout / WF-evaluation backtests must come from data
  strictly BEFORE the traded window (walk-forward metrics / calibration-fold metrics).
* 6b: a model must never be its own benchmark; NaN relative metrics cannot lead.
* 7 : WF evaluation-fold backtest is labelled selection-contaminated; the deflated-Sharpe
  trial count is the FINAL (post adaptive expansion) number of trials x models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtesting.signals import SignalConfig
from src.evaluation.evaluator import enrich_with_benchmark_metrics, select_benchmark_model
from src.pipeline import evaluation_workflows as ew
from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.evaluation_manager import EvaluationManager

MODEL = "Custom Model"


class _StubModel:
    """Predicts the true target + tiny noise so holdout Dir_Acc is high (~70+)."""

    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, X, **kwargs):
        return self._preds[-len(X):]

    def save(self, path):  # pragma: no cover - trivial
        return None


class _TargetPassthroughModel(_StubModel):
    pass


def _make_manager(tmp_path, **exe_overrides) -> EvaluationManager:
    exe = ExecutionConfig(signal_config=SignalConfig(quality_gate_mode="soft"), **exe_overrides)
    manager = EvaluationManager(
        "TEST",
        str(tmp_path / "out"),
        str(tmp_path / "models"),
        MagicMock(),
        ["f1"],
        "hash",
        {"target_mode": "log_return", "candidate_models": [MODEL], "run_id": "r"},
        exe,
        ModelConfig(),
        None,
    )
    (tmp_path / "out").mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)
    return manager


def _holdout_tensors(n=40, with_train=True):
    rng = np.random.default_rng(7)
    y_target = rng.normal(0.0, 0.01, n)
    prev_close = np.full(n, 100.0)
    y_price = prev_close * np.exp(y_target)
    tensors = {
        "X_test": np.zeros((n, 1)),
        "y_test": y_target,
        "prev_close_test": prev_close,
        "dates_test": pd.date_range("2025-01-01", periods=n, freq="B").to_numpy(),
        "original_y_test_aligned": y_price,
        "market_regime_test": np.zeros(n),
    }
    if with_train:
        tensors["X_train"] = np.zeros((60, 1))
        tensors["y_train"] = rng.normal(0.001, 0.01, 60)
    return tensors, y_target


def _wf_metrics(dir_acc=40.0):
    return {
        "Dir_Acc": dir_acc,
        "RMSE": 0.02,
        "MAE": 0.015,
        "MAPE": 1.0,
        "Sharpe": 0.1,
        "Hit_Rate": 45.0,
        "Neutral_Rate": 0.0,
        "BuyHold_Sharpe": 0.0,
        "RMSE_vs_benchmark": 0.97,
        "Composite_Score": 55.0,
    }


def _run_holdout(manager, tensors, y_target, wf_metrics):
    captured = {}

    def spy(backtest_inputs, suffix, model_metrics_by_model=None, **kwargs):
        captured["suffix"] = suffix
        captured["metrics"] = model_metrics_by_model
        captured["kwargs"] = kwargs
        return {}

    workflow = manager.final_holdout_workflow
    workflow._run_backtests = spy  # instance-level spy on the gate entry point
    if wf_metrics is not None:
        manager.state.latest_model_metrics["wf"] = {MODEL: wf_metrics}
    # Perfect-direction predictions -> holdout Dir_Acc far above the WF value (40).
    model = _StubModel(y_target + 1e-4)
    with patch.object(ew, "plot_comparison"), patch.object(ew, "export_final_holdout_result"):
        result = workflow.run(MODEL, model, tensors)
    return result, captured


# --------------------------------------------------------------------------- #
#  P0-6a: look-ahead gate                                                      #
# --------------------------------------------------------------------------- #

def test_holdout_gate_receives_walk_forward_metrics_not_holdout_metrics(tmp_path):
    manager = _make_manager(tmp_path)
    tensors, y_target = _holdout_tensors()
    result, captured = _run_holdout(manager, tensors, y_target, _wf_metrics(dir_acc=40.0))

    holdout_dir_acc = result["metrics"][MODEL]["Dir_Acc"]
    assert holdout_dir_acc > 60.0, "fixture sanity: holdout Dir_Acc must differ from WF"
    gate_input = captured["metrics"][MODEL]
    assert gate_input["Dir_Acc"] == 40.0, (
        f"gate saw Dir_Acc={gate_input['Dir_Acc']} (holdout-computed), expected WF value 40"
    )
    assert captured["kwargs"].get("gate_metrics_source") == "walk_forward"
    assert result["metrics"][MODEL]["Gate_Metrics_Source"] == "walk_forward"
    assert result["metrics"][MODEL]["Reporting_Role"] == "final_holdout"
    assert result["metrics"][MODEL]["Selection_Contaminated"] is False


def test_holdout_gate_is_disabled_when_no_walk_forward_metrics(tmp_path):
    manager = _make_manager(tmp_path)
    tensors, y_target = _holdout_tensors()
    result, captured = _run_holdout(manager, tensors, y_target, None)

    assert not captured["metrics"], "no pre-window metrics -> gate must not be fed holdout metrics"
    assert captured["kwargs"].get("gate_metrics_source") == "none"
    assert result["metrics"][MODEL]["Gate_Metrics_Source"] == "none"


def _wf_fixture(n_folds=10, rows_per_fold=6):
    folds = list(range(n_folds))
    models = [MODEL, "Naive Zero Return"]
    fold_rows = {m: [] for m in models}
    for fold in folds:
        calibration = fold < 7  # ratio 0.70 of 10 folds
        for m in models:
            if m == MODEL:
                dir_acc, rmse = (40.0, 0.020) if calibration else (75.0, 0.009)
            else:
                dir_acc, rmse = 50.0, 0.010
            fold_rows[m].append({
                "Fold": fold, "RMSE": rmse, "MAE": rmse * 0.8, "MAPE": 1.0,
                "Dir_Acc": dir_acc, "Sharpe": 0.1, "Hit_Rate": 50.0,
                "Neutral_Rate": 0.0, "BuyHold_Sharpe": 0.0,
            })
    wf_results = {}
    for m in models:
        rows = fold_rows[m]
        wf_results[m] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "Fold"}
    inputs = {}
    for m in models:
        n = len(folds) * rows_per_fold
        inputs[m] = {
            "fold_ids": np.repeat(folds, rows_per_fold),
            "dates": pd.date_range("2024-01-01", periods=n, freq="B").to_numpy(),
            "prediction_dates": pd.date_range("2024-01-01", periods=n, freq="B").to_numpy(),
            "y_true_price": np.full(n, 101.0),
            "pred_price": np.full(n, 102.0),
            "prev_close": np.full(n, 100.0),
            "market_regime": np.zeros(n),
        }
    y_true = np.full(len(folds) * rows_per_fold, 101.0)
    preds = {m: np.full(len(y_true), 102.0) for m in models}
    return wf_results, preds, y_true, inputs, fold_rows


def _run_wf(manager, wf_fixture):
    wf_results, preds, y_true, inputs, fold_rows = wf_fixture
    captured = {}

    def spy(backtest_inputs, suffix, model_metrics_by_model=None, **kwargs):
        captured["suffix"] = suffix
        captured["metrics"] = model_metrics_by_model
        captured["kwargs"] = kwargs
        captured["inputs"] = backtest_inputs
        return {}

    workflow = manager.walk_forward_workflow
    workflow._run_backtests = spy
    workflow._get_xai_walk_forward = lambda *a, **k: None
    workflow._plot_walk_forward_predictions = lambda *a, **k: None
    workflow._add_walk_forward_ensembles = lambda *a, **k: None
    with patch.object(ew, "export_walk_forward_results"):
        out = workflow.run(wf_results, preds, y_true, wf_backtest_inputs=inputs, wf_fold_metrics=fold_rows)
    return out, captured


def test_wf_evaluation_fold_gate_uses_calibration_fold_metrics_only(tmp_path):
    manager = _make_manager(tmp_path, enable_signal_execution_calibration=False)
    out, captured = _run_wf(manager, _wf_fixture())

    assert captured["suffix"] == "wf"
    # Evaluation folds (7..9) were traded; their Dir_Acc (75) must not reach the gate.
    assert set(np.unique(captured["inputs"][MODEL]["fold_ids"])) == {7, 8, 9}
    gate_input = captured["metrics"][MODEL]
    assert gate_input["Dir_Acc"] == pytest.approx(40.0), (
        f"gate saw Dir_Acc={gate_input['Dir_Acc']}; expected calibration-fold mean 40"
    )
    assert captured["kwargs"].get("gate_metrics_source") == "calibration_folds"


def test_wf_rows_are_labelled_selection_contaminated_confirmation(tmp_path):
    manager = _make_manager(tmp_path, enable_signal_execution_calibration=False)
    out, captured = _run_wf(manager, _wf_fixture())

    assert captured["kwargs"].get("reporting_role") == "confirmation"
    assert captured["kwargs"].get("selection_contaminated") is True
    row = out["metrics"][MODEL]
    assert row["Reporting_Role"] == "confirmation"
    assert row["Selection_Contaminated"] is True
    assert row["Gate_Metrics_Source"] == "calibration_folds"


def test_wf_gate_disabled_when_no_calibration_folds(tmp_path):
    manager = _make_manager(tmp_path, enable_signal_execution_calibration=False)
    wf = list(_wf_fixture(n_folds=2))
    wf_results, preds, y_true, inputs, fold_rows = wf
    out, captured = _run_wf(manager, tuple(wf))

    assert not captured["metrics"]
    assert captured["kwargs"].get("gate_metrics_source") == "none"
    assert out["metrics"][MODEL]["Gate_Metrics_Source"] == "none"


# --------------------------------------------------------------------------- #
#  P0-6b: self-benchmark                                                       #
# --------------------------------------------------------------------------- #

def _plain_metrics(rmse=0.02, dir_acc=60.0):
    return {
        "RMSE": rmse, "MAE": rmse * 0.8, "MAPE": 1.0, "Dir_Acc": dir_acc, "Sharpe": 0.5,
        "Hit_Rate": 55.0, "Neutral_Rate": 0.0, "BuyHold_Sharpe": 0.0,
    }


def test_select_benchmark_model_never_returns_non_benchmark_model():
    assert select_benchmark_model({"My Model": _plain_metrics()}) is None
    assert select_benchmark_model({"A": _plain_metrics(), "B": _plain_metrics()}) is None
    assert select_benchmark_model({"My Model": _plain_metrics(), "Naive Zero Return": _plain_metrics()}) == "Naive Zero Return"


def test_single_model_has_nan_relative_metrics_and_is_not_leader_eligible():
    enriched = enrich_with_benchmark_metrics({"My Model": _plain_metrics()})["My Model"]
    assert np.isnan(enriched["RMSE_vs_benchmark"])
    assert np.isnan(enriched["DirAcc_vs_benchmark"])
    assert enriched["Eligible_For_Leader"] is False
    assert enriched["Beats_Benchmark_RMSE"] is False


def test_single_model_composite_score_is_capped_below_leader_threshold(tmp_path):
    manager = _make_manager(tmp_path)
    scored = manager.metrics_reporting_service._attach_composite_scores(
        {"My Model": _plain_metrics(rmse=0.001, dir_acc=95.0)}
    )["My Model"]
    assert scored["Eligible_For_Leader"] is False
    assert scored["Composite_Score"] <= 49.0


def test_holdout_computes_benchmark_on_holdout_so_model_is_not_its_own_benchmark(tmp_path):
    manager = _make_manager(tmp_path)
    tensors, y_target = _holdout_tensors(with_train=True)
    result, _ = _run_holdout(manager, tensors, y_target, _wf_metrics())

    row = result["metrics"][MODEL]
    assert list(result["metrics"]) == [MODEL], "benchmark rows must not leak into holdout output"
    assert np.isfinite(row["RMSE_vs_benchmark"])
    assert row["RMSE_vs_benchmark"] != 1.0
    # near-perfect predictions beat the naive benchmark on the holdout
    assert row["RMSE_vs_benchmark"] < 1.0
    assert row["Eligible_For_Leader"] is True


def test_holdout_without_benchmark_inputs_yields_nan_and_ineligible(tmp_path):
    manager = _make_manager(tmp_path)
    tensors, y_target = _holdout_tensors(with_train=False)  # naive baselines cannot be trained
    result, _ = _run_holdout(manager, tensors, y_target, _wf_metrics())

    row = result["metrics"][MODEL]
    assert np.isnan(row["RMSE_vs_benchmark"])
    assert row["Eligible_For_Leader"] is False
    assert row["Composite_Score"] <= 49.0


# --------------------------------------------------------------------------- #
#  P0-7: trial counting                                                        #
# --------------------------------------------------------------------------- #

def test_trial_count_for_metrics_reflects_final_trials_after_adaptive_expansion(tmp_path):
    manager = _make_manager(tmp_path, signal_calibration_require_oos_confirmation=True)
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    payload = {
        "dates": dates,
        "prediction_dates": dates - pd.Timedelta(days=1),
        "y_true_price": np.full(8, 101.0),
        "pred_price": np.full(8, 104.0),
        "prev_close": np.full(8, 100.0),
        "market_regime": np.zeros(8),
    }
    weak_summary = {
        "Model": MODEL, "Net_Return": 0.05, "BuyHold_Return": 0.01, "Max_Drawdown": -0.02,
        "Trade_Count": 6, "Sharpe": -0.2, "Calmar": 2.5, "Beats_BuyHold_NetReturn": True,
    }
    seen = []

    def spy_summarize(result, initial_capital=None, trial_count=1, **kwargs):
        seen.append(int(trial_count))
        return dict(weak_summary)

    with patch("src.pipeline.signal_calibrator.run_backtest", return_value={}), \
         patch("src.pipeline.signal_calibrator.summarize_backtest", side_effect=spy_summarize):
        result = manager.signal_calibration_service._calibrate_walk_forward_signal_parameters(
            wf_backtest_inputs={MODEL: payload},
            wf_evaluation_backtest_inputs={MODEL: payload},
            model_metrics_by_model={},
        )

    executed = manager.signal_threshold_calibration_summary["executed_trials"]
    assert manager.signal_threshold_calibration_summary["adaptive_expanded"] is True
    assert executed > 64, "fixture sanity: adaptive expansion must have added trials"
    assert result["trial_count_for_metrics"] == executed * 1
    assert manager.signal_threshold_calibration_summary["trial_count_for_metrics"] == executed * 1
    # OOS-confirmation backtests (run after expansion) must use the final count.
    assert seen[-1] == executed * 1, f"OOS confirmation used stale trial_count={seen[-1]}"


def test_wf_backtest_trial_count_is_models_times_calibration_trials(tmp_path):
    manager = _make_manager(tmp_path)
    fixture = _wf_fixture()

    def fake_calibration(**kwargs):
        return {"trial_count_for_metrics": 128, "calibration_df": pd.DataFrame()}

    manager.walk_forward_workflow._calibrate_walk_forward_signal_parameters = fake_calibration
    manager.walk_forward_workflow._calibrate_signal_quality_thresholds = lambda *a, **k: None
    out, captured = _run_wf(manager, fixture)

    assert captured["kwargs"].get("trial_count") == 128


def test_selection_guard_rejects_nan_benchmark_ratio():
    from src.pipeline.selection_guard import compute_eligibility

    status, _ = compute_eligibility(
        model_name="XGBoost",
        is_production_candidate=True,
        is_baseline=False,
        total_trade_count=50,
        rmse_vs_benchmark=float("nan"),
    )
    assert status == "benchmark_failed"
