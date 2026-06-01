# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.forecasting.workflows import LatestTargetPredictionWorkflow
from src.pipeline.evaluation_services import (
    EvaluationContext,
    EvaluationState,
    PredictionService,
)


class _DateAwareModel:
    def __init__(self):
        self.seen_dates = None

    def predict(self, x, dates_test=None):
        if dates_test is None:
            raise TypeError("dates_test is required")
        self.seen_dates = list(dates_test)
        return np.full(len(x), 0.05)


def _prediction_ctx_state():
    # Faz 3: PredictionService artık owner-forward yerine (ctx, state) DI alır.
    ctx = EvaluationContext(dataset_metadata={"target_mode": "price"}, ensemble_enabled=False)
    state = EvaluationState()
    return ctx, state


def _tensors():
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    return {
        "X_test": np.ones((3, 2)),
        "prev_close_test": np.array([10.0, 11.0, 12.0]),
        "dates_test": dates,
        "dates_prediction": dates,
        "original_y_test_aligned": np.array([10.0, 11.0, 12.0]),
        "y_test": np.array([10.0, 11.0, 12.0]),
    }


def test_prophet_hybrid_single_model_prediction_receives_dates_test():
    ctx, state = _prediction_ctx_state()
    service = PredictionService(ctx, state)
    model = _DateAwareModel()

    service._predict_single_model("Prophet-ML/DL Hybrid", model, _tensors())

    assert model.seen_dates == list(_tensors()["dates_test"])


def test_prophet_hybrid_batch_prediction_receives_dates_test():
    ctx, state = _prediction_ctx_state()
    service = PredictionService(ctx, state)
    model = _DateAwareModel()

    service.generate_predictions({"Prophet-ML/DL Hybrid": model}, _tensors())

    assert model.seen_dates == list(_tensors()["dates_test"])
    assert "Prophet-ML/DL Hybrid" in state.predictions


def test_latest_target_prediction_uses_date_aware_path_for_prophet_hybrid():
    workflow = LatestTargetPredictionWorkflow(SimpleNamespace())
    model = _DateAwareModel()
    last_date = pd.Timestamp("2026-05-22")

    prediction = workflow.predict(
        "Prophet-ML/DL Hybrid",
        model,
        {"latest_X": np.ones((1, 2)), "last_observed_date": last_date},
    )

    assert prediction == 0.05
    assert model.seen_dates == [last_date]
