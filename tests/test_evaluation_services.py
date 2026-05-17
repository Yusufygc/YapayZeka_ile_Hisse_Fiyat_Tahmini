import numpy as np

from src.pipeline.backtest_runner import _BacktestRunnerMixin
from src.pipeline.evaluation_manager import EvaluationManager
from src.pipeline.evaluation_services import (
    BacktestService,
    MetricsReportingService,
    PredictionService,
    SignalCalibrationService,
)
from src.pipeline.evaluation_workflows import (
    FinalHoldoutEvaluationWorkflow,
    SingleSplitEvaluationWorkflow,
    WalkForwardEvaluationWorkflow,
)
from src.pipeline.metrics_reporter import _MetricsReporterMixin
from src.pipeline.prediction_engine import _PredictionEngineMixin
from src.pipeline.signal_calibrator import _SignalCalibratorMixin


def test_evaluation_manager_uses_service_composition_not_mixin_inheritance():
    assert not issubclass(EvaluationManager, _PredictionEngineMixin)
    assert not issubclass(EvaluationManager, _BacktestRunnerMixin)
    assert not issubclass(EvaluationManager, _SignalCalibratorMixin)
    assert not issubclass(EvaluationManager, _MetricsReporterMixin)

    manager = EvaluationManager.__new__(EvaluationManager)
    manager._init_services()

    assert isinstance(manager.prediction_service, PredictionService)
    assert isinstance(manager.backtest_service, BacktestService)
    assert isinstance(manager.signal_calibration_service, SignalCalibrationService)
    assert isinstance(manager.metrics_reporting_service, MetricsReportingService)
    assert isinstance(manager.single_split_workflow, SingleSplitEvaluationWorkflow)
    assert isinstance(manager.walk_forward_workflow, WalkForwardEvaluationWorkflow)
    assert isinstance(manager.final_holdout_workflow, FinalHoldoutEvaluationWorkflow)


def test_lazy_prediction_service_delegates_to_manager_state():
    manager = EvaluationManager.__new__(EvaluationManager)
    manager.dataset_metadata = {"target_mode": "return"}

    prices = manager._target_to_price(np.array([0.10, -0.05]), np.array([100.0, 200.0]))

    np.testing.assert_allclose(prices, np.array([110.0, 190.0]))
    assert isinstance(manager.prediction_service, PredictionService)


def test_evaluate_public_methods_delegate_to_workflows():
    manager = EvaluationManager.__new__(EvaluationManager)
    manager._init_services()

    calls = []

    class WorkflowStub:
        def __init__(self, label):
            self.label = label

        def run(self, *args, **kwargs):
            calls.append((self.label, args, kwargs))
            return {"workflow": self.label}

    manager.single_split_workflow = WorkflowStub("single")
    manager.walk_forward_workflow = WorkflowStub("wf")
    manager.final_holdout_workflow = WorkflowStub("final")

    assert manager.evaluate_single_split({"M": object()}) == {"workflow": "single"}
    assert manager.evaluate_walk_forward({"M": {}}, {"M": np.array([1.0])}, np.array([1.0])) == {"workflow": "wf"}
    assert manager.evaluate_final_holdout("M", object(), {"X_test": np.array([1.0])}) == {"workflow": "final"}

    assert [call[0] for call in calls] == ["single", "wf", "final"]
