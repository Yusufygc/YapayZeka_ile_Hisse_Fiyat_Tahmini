# -*- coding: utf-8 -*-
"""P1 Diebold-Mariano wiring: DM p-values reach the WF / holdout rows and the DB guard.

* WF aggregate path: hedef-uzayi tahminleri (``wf_backtest_inputs[m]["pred_target"]``)
  secili benchmark'a karsi DM testine sokulur; ``DM_PValue_vs_benchmark`` satira yazilir.
* Final holdout: naive benchmark tahminleri de DM'ye girer.
* DB: ``update_production_best_model`` eligibility satiri DM p-degerini tasir.
"""

from __future__ import annotations

import math
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtesting.signals import SignalConfig
from src.pipeline import evaluation_workflows as ew
from src.pipeline.config import ExecutionConfig, ModelConfig
from src.pipeline.evaluation_manager import EvaluationManager
from src.pipeline.selection_guard import evaluate_best_model_eligibility

BENCH = "Naive Zero Return"
GOOD = "Good Model"
SAME = "Same As Benchmark"


class _StubModel:
    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, X, **kwargs):
        return self._preds[-len(X):]

    def save(self, path):  # pragma: no cover - trivial
        return None


def _make_manager(tmp_path, candidates, stock_db=None) -> EvaluationManager:
    exe = ExecutionConfig(
        signal_config=SignalConfig(quality_gate_mode="soft"),
        enable_signal_execution_calibration=False,
    )
    manager = EvaluationManager(
        "TEST",
        str(tmp_path / "out"),
        str(tmp_path / "models"),
        ["f1"],
        "hash",
        {"target_mode": "log_return", "candidate_models": list(candidates), "run_id": "r"},
        exe,
        ModelConfig(),
        stock_db,
    )
    (tmp_path / "out").mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)
    return manager


def _row(rmse, dir_acc, sharpe=0.2):
    return {
        "RMSE": rmse, "MAE": rmse * 0.8, "MAPE": 1.0, "Dir_Acc": dir_acc,
        "Sharpe": sharpe, "Hit_Rate": dir_acc, "Neutral_Rate": 0.0, "BuyHold_Sharpe": 0.0,
    }


def _wf_fixture(n=240, n_folds=4):
    """Sabit genlikli hedef: benchmark=0 tahmini. GOOD cok iyi; SAME'in hata kaybi
    farki d = [+1,+1,+1,-3]*c tekrar eder (ortalama tam 0 -> DM p = 0.5), yani RMSE'si
    benchmark ile ayni ama istatistiksel olarak benchmark'tan iyi degildir.
    """
    rng = np.random.default_rng(3)
    y = 0.02 * rng.choice([-1.0, 1.0], size=n)
    pattern = np.tile([1.0, 1.0, 1.0, -1.0], n // 4)
    preds = {
        BENCH: np.zeros(n),
        GOOD: y + rng.normal(0.0, 0.002, n),
        SAME: y * pattern,
    }
    rows_per_fold = n // n_folds
    fold_ids = np.repeat(np.arange(n_folds), rows_per_fold)
    inputs = {
        name: {
            "fold_ids": fold_ids,
            "dates": pd.date_range("2024-01-01", periods=n, freq="B").to_numpy(),
            "prediction_dates": pd.date_range("2024-01-01", periods=n, freq="B").to_numpy(),
            "y_true_price": np.full(n, 101.0),
            "pred_price": np.full(n, 102.0),
            "prev_close": np.full(n, 100.0),
            "market_regime": np.zeros(n),
            "pred_target": p,
            "y_true_target": y,
        }
        for name, p in preds.items()
    }
    # Fold-ortalamasi RMSE: SAME ratio 0.995 (RMSE kapisini gecer), Dir_Acc yuksek.
    wf_results = {
        BENCH: _row(0.0200, 50.0),
        GOOD: _row(0.0030, 80.0, sharpe=1.0),
        SAME: _row(0.0199, 90.0, sharpe=1.0),
    }
    price_preds = {name: np.full(n, 102.0) for name in preds}
    return wf_results, price_preds, np.full(n, 101.0), inputs


def _run_wf(manager, wf_results, preds, y_true, inputs):
    workflow = manager.walk_forward_workflow
    workflow._run_backtests = lambda *a, **k: {}
    workflow._get_xai_walk_forward = lambda *a, **k: None
    workflow._plot_walk_forward_predictions = lambda *a, **k: None
    workflow._add_walk_forward_ensembles = lambda *a, **k: None
    workflow._run_signal_calibration_steps = lambda *a, **k: {}  # DM ile ilgisiz, yavas
    with patch.object(ew, "export_walk_forward_results"):
        return workflow.run(wf_results, preds, y_true, wf_backtest_inputs=inputs, wf_fold_metrics={})


# --------------------------------------------------------------------------- #
#  Walk-forward aggregate path                                                 #
# --------------------------------------------------------------------------- #

def test_wf_rows_carry_dm_pvalue_good_significant_same_not(tmp_path):
    manager = _make_manager(tmp_path, [GOOD, SAME])
    out = _run_wf(manager, *_wf_fixture())
    metrics = out["metrics"]

    assert metrics[GOOD]["DM_PValue_vs_benchmark"] < 0.10
    assert metrics[SAME]["DM_PValue_vs_benchmark"] > 0.10
    assert math.isnan(metrics[BENCH]["DM_PValue_vs_benchmark"])


def test_wf_same_as_benchmark_is_not_selected_when_only_candidate(tmp_path):
    manager = _make_manager(tmp_path, [SAME])
    out = _run_wf(manager, *_wf_fixture())

    assert out["metrics"][SAME]["RMSE_vs_benchmark"] <= 1.0  # RMSE kapisi tek basina gecer
    assert out["best_model_name"] is None
    assert out["metrics"][SAME]["Selection_Status"] == "no_significant_candidate"


def test_wf_good_model_is_selected_and_same_is_benchmark_failed(tmp_path):
    manager = _make_manager(tmp_path, [GOOD, SAME])
    out = _run_wf(manager, *_wf_fixture())
    metrics = out["metrics"]

    assert out["best_model_name"] == GOOD
    for name, expected in ((GOOD, "eligible"), (SAME, "benchmark_failed")):
        row = {
            "model_name": name,
            "is_production_candidate": 1,
            "Trade_Count": 30,
            "RMSE_vs_benchmark": metrics[name]["RMSE_vs_benchmark"],
            "DM_PValue_vs_benchmark": metrics[name].get("DM_PValue_vs_benchmark"),
        }
        assert evaluate_best_model_eligibility(row)[0] == expected, name


def test_wf_without_target_predictions_leaves_dm_absent(tmp_path):
    manager = _make_manager(tmp_path, [GOOD, SAME])
    wf_results, preds, y_true, inputs = _wf_fixture()
    for payload in inputs.values():
        payload.pop("pred_target")
    out = _run_wf(manager, wf_results, preds, y_true, inputs)

    for row in out["metrics"].values():
        assert "DM_PValue_vs_benchmark" not in row or math.isnan(row["DM_PValue_vs_benchmark"])


def test_wf_ensemble_without_pred_target_is_skipped_not_crashing(tmp_path):
    manager = _make_manager(tmp_path, [GOOD])
    wf_results, preds, y_true, inputs = _wf_fixture()
    wf_results["Ensemble Inverse RMSE"] = _row(0.0040, 70.0)
    preds["Ensemble Inverse RMSE"] = np.full(len(y_true), 102.0)
    inputs["Ensemble Inverse RMSE"] = {**inputs[GOOD], "pred_target": None}
    out = _run_wf(manager, wf_results, preds, y_true, inputs)

    assert math.isnan(out["metrics"]["Ensemble Inverse RMSE"]["DM_PValue_vs_benchmark"])
    assert out["metrics"][GOOD]["DM_PValue_vs_benchmark"] < 0.10


def test_wf_shorter_ensemble_is_tail_aligned_pairwise(tmp_path):
    manager = _make_manager(tmp_path, [GOOD])
    wf_results, preds, y_true, inputs = _wf_fixture()
    k = 160
    ens = "Ensemble Inverse RMSE"
    wf_results[ens] = _row(0.0040, 70.0)
    preds[ens] = np.full(k, 102.0)
    inputs[ens] = {**inputs[GOOD], "pred_target": inputs[GOOD]["pred_target"][-k:],
                   "y_true_target": inputs[GOOD]["y_true_target"][-k:]}
    out = _run_wf(manager, wf_results, preds, y_true, inputs)

    assert out["metrics"][ens]["DM_PValue_vs_benchmark"] < 0.10


# --------------------------------------------------------------------------- #
#  Final holdout path                                                          #
# --------------------------------------------------------------------------- #

def _holdout_tensors(n=60):
    rng = np.random.default_rng(7)
    y_target = rng.normal(0.0, 0.01, n)
    prev_close = np.full(n, 100.0)
    return {
        "X_test": np.zeros((n, 1)),
        "y_test": y_target,
        "prev_close_test": prev_close,
        "dates_test": pd.date_range("2025-01-01", periods=n, freq="B").to_numpy(),
        "original_y_test_aligned": prev_close * np.exp(y_target),
        "market_regime_test": np.zeros(n),
        "X_train": np.zeros((60, 1)),
        "y_train": rng.normal(0.001, 0.01, 60),
    }, y_target


def test_holdout_row_carries_dm_pvalue_against_naive_benchmark(tmp_path):
    manager = _make_manager(tmp_path, ["Custom Model"])
    tensors, y_target = _holdout_tensors()
    workflow = manager.final_holdout_workflow
    workflow._run_backtests = lambda *a, **k: {}
    model = _StubModel(y_target + 1e-4)  # near-perfect
    with patch.object(ew, "plot_comparison"), patch.object(ew, "export_final_holdout_result"):
        result = workflow.run("Custom Model", model, tensors)

    row = result["metrics"]["Custom Model"]
    assert "DM_PValue_vs_benchmark" in row
    assert row["DM_PValue_vs_benchmark"] < 0.10


# --------------------------------------------------------------------------- #
#  DB: best-model eligibility row carries DM p                                 #
# --------------------------------------------------------------------------- #

def _log_holdout(db, model, dm_p):
    metrics = {
        "MAE": 1.0, "RMSE": 2.0, "Dir_Acc": 60.0, "Hit_Rate": 55.0, "Sharpe": 0.5,
        "Trade_Count": 30, "RMSE_vs_benchmark": 0.9, "Composite_Score": 70.0,
    }
    if dm_p is not None:
        metrics["DM_PValue_vs_benchmark"] = dm_p
    db.log_experiment(
        stock_symbol="TEST", model_name=model, metrics=metrics,
        validation_mode="final_holdout", is_production_candidate=True, run_id="r1",
    )


@pytest.fixture
def db_path():
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    yield path
    try:
        os.remove(path)
    except OSError as exc:  # pragma: no cover - Windows file lock
        print(f"cleanup skipped: {exc}")


@pytest.mark.parametrize(
    "dm_p, status",
    [(0.45, "benchmark_failed"), (0.02, "eligible"), (None, "eligible"), (float("nan"), "eligible")],
)
def test_best_model_eligibility_uses_dm_pvalue(db_path, dm_p, status):
    from src.database.stock_model_db import StockModelDB

    db = StockModelDB(db_path)
    _log_holdout(db, "XGBoost", dm_p)
    best = db.get_best_model("TEST")
    assert best is not None
    assert best["eligibility_status"] == status


def test_dm_pvalue_persisted_and_used_on_rebuild(tmp_path):
    """DM p-degeri experiments tablosuna yazilir; DB yeniden init edildiginde
    best_models rebuild'i bu degerle uygunlugu yeniden hesaplar."""
    from src.database.repositories.best_model import BestModelRepository
    from src.database.stock_model_db import StockModelDB

    db_path = str(tmp_path / "stock_models.db")
    db = StockModelDB(db_path)
    exp_id = db.log_experiment(
        stock_symbol="TEST",
        model_name="XGBoost",
        metrics={
            "RMSE": 1.0, "Dir_Acc": 55.0, "RMSE_vs_benchmark": 0.98,
            "Trade_Count": 40, "DM_PValue_vs_benchmark": 0.45,
        },
        validation_mode="final_holdout",
        dataset_metadata={},
        is_production_candidate=True,
        run_id="r1",
    )
    with db._connect() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
    assert row["dm_pvalue_vs_benchmark"] == pytest.approx(0.45)
    status, _ = BestModelRepository.eligibility_from_experiment_row(row)
    assert status == "benchmark_failed"
