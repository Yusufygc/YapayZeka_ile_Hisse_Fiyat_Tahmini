import numpy as np

from src.pipeline.backtest_runner import _BacktestRunnerMixin
from src.pipeline.evaluation_manager import EvaluationManager
from src.pipeline.evaluation_services import (
    BacktestService,
    MetricsReportingService,
    PredictionService,
    SignalCalibrationService,
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


def test_lazy_prediction_service_delegates_to_manager_state():
    manager = EvaluationManager.__new__(EvaluationManager)
    manager.dataset_metadata = {"target_mode": "return"}

    prices = manager._target_to_price(np.array([0.10, -0.05]), np.array([100.0, 200.0]))

    np.testing.assert_allclose(prices, np.array([110.0, 190.0]))
    assert isinstance(manager.prediction_service, PredictionService)
