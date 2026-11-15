# -*- coding: utf-8 -*-
"""Single-split `generate_predictions`: naive modeller scaler_y ile cift donusturulmemeli.

Naive modeller ham hedefle (X_train / y_train) egitilir ve orijinal olcekte tahmin
uretir; tree/linear modeller olcekli hedefle egitilir ve `scaler_y.inverse_transform`
gerektirir. Eski kod butun modellere inverse_transform uyguluyordu (naive icin hatali).
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models.linear_model import RidgeReturnModel
from src.models.naive_model import NaiveDriftModel, NaiveLastValueModel, NaiveZeroReturnModel
from _eval_helpers import make_eval_services


def _tensors(n_train=40, n_test=12):
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(n_train, 3))
    y_train = 0.02 + 0.01 * rng.normal(size=n_train)  # ham hedef: mean ~0.02, std ~0.01
    X_test = rng.normal(size=(n_test, 3))
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    y_test = 0.02 + 0.01 * rng.normal(size=n_test)
    prev_close = 100.0 + np.arange(n_test, dtype=float)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_train_s": scaler_X.transform(X_train),
        "y_train_s": scaler_y.transform(y_train.reshape(-1, 1)).ravel(),
        "X_test": X_test,
        "X_test_s": scaler_X.transform(X_test),
        "y_test": y_test,
        "scaler_y": scaler_y,
        "prev_close_test": prev_close,
        "original_y_test_aligned": prev_close * np.exp(y_test),
        "dates_test": np.arange(n_test),
    }


def test_naive_predictions_are_not_double_transformed():
    tensors = _tensors()
    zero, last, drift = NaiveZeroReturnModel(), NaiveLastValueModel(), NaiveDriftModel()
    for model in (zero, last, drift):
        model.train(tensors["X_train"], tensors["y_train"])
    ridge = RidgeReturnModel()
    ridge.train(tensors["X_train_s"], tensors["y_train_s"])

    svc = make_eval_services(ctx={"dataset_metadata": {"target_mode": "log_return"}})
    svc.prediction.generate_predictions(
        {"Naive Zero Return": zero, "Naive Last Value": last, "Naive Drift": drift, "Ridge Return": ridge},
        tensors,
    )
    targets = svc.state.prediction_targets

    np.testing.assert_allclose(targets["Naive Zero Return"], 0.0, atol=1e-12)
    np.testing.assert_allclose(targets["Naive Last Value"], tensors["y_train"][-1], atol=1e-12)
    np.testing.assert_allclose(targets["Naive Drift"], tensors["y_train"].mean(), atol=1e-12)
    # Naive Zero Return fiyati = prev_close (sifir getiri).
    np.testing.assert_allclose(svc.state.predictions["Naive Zero Return"], tensors["prev_close_test"])
    # Tree/linear yol: scaler_y ile geri cevrilmis olmali.
    expected = tensors["scaler_y"].inverse_transform(
        ridge.predict(tensors["X_test_s"]).reshape(-1, 1)
    ).ravel()
    np.testing.assert_allclose(targets["Ridge Return"], expected, atol=1e-12)


def test_single_split_matches_final_holdout_prediction_helper():
    """generate_predictions ve _predict_single_model ayni hedef-uzayi tahminini verir."""
    tensors = _tensors()
    zero = NaiveZeroReturnModel()
    zero.train(tensors["X_train"], tensors["y_train"])
    svc = make_eval_services(ctx={"dataset_metadata": {"target_mode": "log_return"}})
    svc.prediction.generate_predictions({"Naive Zero Return": zero}, tensors)
    holdout = svc.prediction._predict_single_model("Naive Zero Return", zero, tensors)
    np.testing.assert_allclose(svc.state.prediction_targets["Naive Zero Return"], holdout[1])
