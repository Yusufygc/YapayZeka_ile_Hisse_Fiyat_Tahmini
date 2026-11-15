# -*- coding: utf-8 -*-
"""P0-2: walk-forward ensemble agirliklari yalnizca ONCEKI fold'lardan fit edilmeli.

Eski davranis: agirliklar tum OOS satirlarinda (y_true dahil) fit edilip ayni
satirlarda metrik hesaplaniyordu -> in-sample ensemble metrikleri.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.ensemble import EnsembleModel
from src.pipeline.evaluation_services import EvaluationContext, EvaluationState, PredictionService

MODELS = ("XGBoost", "Ridge Return", "LightGBM Return")
# Tek ensemble varyanti kaldi; diger varyantlar model_scope.REMOVED_MODELS altinda.
WEIGHTED = ("Ensemble Inverse RMSE",)
N_PER_FOLD = 30
FOLDS = (1, 2, 3)


def _build(perturb_fold=None, with_fold_ids=True, models=MODELS):
    """3 fold x 30 satir sentetik WF cikti seti. ``perturb_fold`` y_true'yu degistirir."""
    rng = np.random.default_rng(7)
    n = N_PER_FOLD * len(FOLDS)
    fold_ids = np.repeat(np.asarray(FOLDS), N_PER_FOLD)
    prev_close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    y_true_target = rng.normal(0, 0.01, n)
    noise_scale = dict(zip(models, (0.004, 0.01, 0.02)))
    pred_target = {
        m: y_true_target + rng.normal(0, s, n) + (0.001 if i == 0 else 0.0)
        for i, (m, s) in enumerate(noise_scale.items())
    }
    if perturb_fold is not None:
        mask = fold_ids == perturb_fold
        y_true_target = y_true_target.copy()
        y_true_target[mask] = -y_true_target[mask] * 3.0 + 0.02
    y_true_price = prev_close * np.exp(y_true_target)

    wf_predictions, wf_inputs, wf_results = {}, {}, {}
    for m in models:
        pred_price = prev_close * np.exp(pred_target[m])
        wf_predictions[m] = pred_price
        payload = {
            "y_true_price": y_true_price,
            "pred_price": pred_price,
            "prev_close": prev_close,
            "pred_target": pred_target[m],
            "y_true_target": y_true_target,
        }
        if with_fold_ids:
            payload["fold_ids"] = fold_ids
        wf_inputs[m] = payload
        wf_results[m] = {}
    return wf_results, wf_predictions, y_true_price, wf_inputs, fold_ids, pred_target


def _run(perturb_fold=None, with_fold_ids=True, models=MODELS):
    ctx = EvaluationContext(
        ensemble_enabled=True,
        dataset_metadata={"target_mode": "log_return"},
    )
    state = EvaluationState()
    service = PredictionService(ctx, state)
    results, preds, y_true, inputs, fold_ids, pred_target = _build(perturb_fold, with_fold_ids, models)
    service._add_walk_forward_ensembles(results, preds, y_true, inputs)
    return service, ctx, results, preds, inputs, fold_ids, pred_target


def _fold_slice(arr, fold_ids, fold):
    return np.asarray(arr)[fold_ids == fold]


@pytest.mark.parametrize("name", WEIGHTED)
def test_future_fold_y_true_does_not_change_any_ensemble_prediction(name):
    """Fold 3 y_true'su fold 3'un kendi agirliklarini da fold 1-2'yi de etkilememeli."""
    _, _, _, base, _, fold_ids, _ = _run(perturb_fold=None)
    _, _, _, pert, _, _, _ = _run(perturb_fold=3)
    np.testing.assert_allclose(pert[name], base[name], rtol=0, atol=1e-12)


def test_earlier_fold_y_true_does_change_later_fold_only():
    """Fold 2 y_true'su fold 3 agirliklarini degistirir; fold 1-2 ayni kalir (expanding)."""
    name = "Ensemble Inverse RMSE"
    _, _, _, base, _, fold_ids, _ = _run()
    _, _, _, pert, _, _, _ = _run(perturb_fold=2)
    for f in (1, 2):
        np.testing.assert_allclose(
            _fold_slice(pert[name], fold_ids, f), _fold_slice(base[name], fold_ids, f), atol=1e-12
        )
    assert not np.allclose(_fold_slice(pert[name], fold_ids, 3), _fold_slice(base[name], fold_ids, 3))


def _equal_blend(name, preds, pred_target):
    """Beklenen esit-agirlik blend."""
    return EnsembleModel().combine({m: preds[m] for m in MODELS})


@pytest.mark.parametrize("name", WEIGHTED)
def test_first_fold_is_equal_weight_blend(name):
    _, _, _, preds, inputs, fold_ids, pred_target = _run()
    equal = _equal_blend(name, preds, pred_target)
    np.testing.assert_allclose(
        _fold_slice(preds[name], fold_ids, 1), _fold_slice(equal, fold_ids, 1), rtol=1e-9, atol=1e-9
    )


def test_fold2_weights_use_only_fold1_rows():
    """Fold 2 inverse-RMSE blend'i, yalnizca fold 1'den fit edilen agirliklarla birebir."""
    _, _, _, preds, _, fold_ids, _ = _run()
    base = {m: preds[m] for m in MODELS}
    _, _, y_true, _, _, _ = _build()
    f1, f2 = fold_ids == 1, fold_ids == 2
    w = EnsembleModel.optimize_inverse_rmse(y_true[f1], {m: base[m][f1] for m in MODELS})
    expected = EnsembleModel(w).combine({m: base[m][f2] for m in MODELS})
    np.testing.assert_allclose(_fold_slice(preds["Ensemble Inverse RMSE"], fold_ids, 2), expected, atol=1e-9)


def test_stored_final_weights_equal_all_folds_fit():
    service, _, _, preds, _, _, pred_target = _run()
    _, _, y_true, inputs, _, _ = _build()
    base = {m: preds[m] for m in MODELS}
    expected = EnsembleModel.optimize_inverse_rmse(y_true, base)
    assert service.state.ensemble_weights["Ensemble Inverse RMSE"] == expected
    assert set(service.state.ensemble_weights) == {"Ensemble Inverse RMSE"}


def test_weight_scope_recorded_as_expanding_prior_folds():
    service, ctx, *_ = _run()
    for name in service._ENSEMBLE_NAMES:
        assert service.state.ensemble_weight_scope[name] == "expanding_prior_folds"
    assert ctx.dataset_metadata["wf_ensemble_weight_scope"] == "expanding_prior_folds"


def test_missing_fold_ids_falls_back_to_equal_weights_not_in_sample():
    _, ctx, _, preds, _, _, pred_target = _run(with_fold_ids=False)
    for name in WEIGHTED:
        np.testing.assert_allclose(
            preds[name], _equal_blend(name, preds, pred_target), rtol=1e-9, atol=1e-9
        )
    assert ctx.dataset_metadata["wf_ensemble_weight_scope"] != "expanding_prior_folds"


def test_ensemble_metrics_and_payloads_cover_all_oos_rows():
    _, _, results, preds, inputs, _, _ = _run()
    n = N_PER_FOLD * len(FOLDS)
    for name in WEIGHTED:
        assert name in results and results[name]
        assert len(preds[name]) == n
        assert len(inputs[name]["pred_price"]) == n
        assert len(inputs[name]["fold_ids"]) == n


def test_only_inverse_rmse_ensemble_is_produced():
    _, _, results, preds, inputs, _, _ = _run()
    produced = {name for name in preds if name.startswith("Ensemble ")}
    assert produced == {"Ensemble Inverse RMSE"}
    assert {name for name in results if name.startswith("Ensemble ")} == produced

