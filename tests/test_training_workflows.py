# -*- coding: utf-8 -*-
"""Service-boundary tests for ModelTrainer Phase 3 decomposition."""

import os

import numpy as np

from src.experiments.experiment_tracker import ExperimentTracker
from src.pipeline.model_trainer import ModelTrainer
from src.pipeline.training_workflows import (
    FinalHoldoutTrainingWorkflow,
    SingleSplitTrainingWorkflow,
    WalkForwardTrainingWorkflow,
)


def _trainer(tmp_path, *, selected_models=None) -> ModelTrainer:
    return ModelTrainer(
        stock_symbol="TEST",
        tracker=ExperimentTracker(os.path.join(str(tmp_path), "exp")),
        feature_names=["f1", "f2"],
        selected_models=selected_models,
        dataset_hash="hash",
        dataset_metadata={"target_mode": "log_return"},
    )


def test_model_trainer_composes_training_workflows(tmp_path):
    trainer = _trainer(tmp_path)

    assert isinstance(trainer.single_split_training_workflow, SingleSplitTrainingWorkflow)
    assert isinstance(trainer.walk_forward_training_workflow, WalkForwardTrainingWorkflow)
    assert isinstance(trainer.final_holdout_training_workflow, FinalHoldoutTrainingWorkflow)


def test_train_public_methods_delegate_to_workflows(tmp_path):
    trainer = _trainer(tmp_path)
    calls = []

    class WorkflowStub:
        def __init__(self, label):
            self.label = label

        def run(self, *args, **kwargs):
            calls.append((self.label, args, kwargs))
            return {"workflow": self.label}

    trainer.single_split_training_workflow = WorkflowStub("single")
    trainer.walk_forward_training_workflow = WorkflowStub("wf")
    trainer.final_holdout_training_workflow = WorkflowStub("final")

    assert trainer.train_single_split({"X_train": np.ones((2, 1))}) == {"workflow": "single"}
    assert trainer.train_walk_forward([], object()) == {"workflow": "wf"}
    assert trainer.train_final_holdout_model("Linear Regression", object()) == {"workflow": "final"}

    assert [call[0] for call in calls] == ["single", "wf", "final"]


def test_selected_model_skip_policy_stays_on_facade_owner(tmp_path):
    trainer = _trainer(tmp_path, selected_models=["Random Forest"])

    assert trainer._skip("XGBoost") is True
    assert trainer._skip("Random Forest") is False
    assert trainer._skip("Naive Last Value") is False
