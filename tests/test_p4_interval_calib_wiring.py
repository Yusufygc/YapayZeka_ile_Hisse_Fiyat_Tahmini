"""Final holdout sidecar'i interval kalibrasyonunu paylasilan state'teki WF girdisinden uretir."""
import numpy as np

from src.pipeline.evaluation_workflows import _build_interval_calibration


class _State:
    def __init__(self, inputs):
        self.wf_backtest_inputs = inputs


class _Workflow:
    """Workflow owner: wf_backtest_inputs ozelligi YOK, yalniz state uzerinden erisilir."""

    def __init__(self, inputs):
        self.state = _State(inputs)


def test_interval_calibration_reads_wf_inputs_from_shared_state():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 0.02, 300)
    y_pred = y_true + rng.normal(0, 0.01, 300)
    owner = _Workflow({"XGBoost": {"y_true_target": y_true, "pred_target": y_pred}})
    calib = _build_interval_calibration(owner, "XGBoost")
    assert calib is not None
    assert "conformal" in calib


def test_interval_calibration_none_without_wf_inputs():
    assert _build_interval_calibration(_Workflow({}), "XGBoost") is None


def test_manager_stashes_wf_inputs_on_state(tmp_path):
    from tests._eval_helpers import make_eval_manager

    manager = make_eval_manager(str(tmp_path))
    manager.walk_forward_workflow.run = lambda *a, **k: None
    inputs = {"XGBoost": {"y_true_target": [0.1], "pred_target": [0.1]}}
    manager.evaluate_walk_forward({}, {}, None, wf_backtest_inputs=inputs)
    assert manager.state.wf_backtest_inputs is inputs
    assert manager.final_holdout_workflow.state is manager.state
