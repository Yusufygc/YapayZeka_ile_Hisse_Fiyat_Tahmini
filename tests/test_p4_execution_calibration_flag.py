"""enable_signal_execution_calibration ExecutionConfig'ten okunur (varsayilan False)."""
from src.pipeline.config import ExecutionConfig
from tests._eval_helpers import make_eval_manager


def _run_steps(manager, calls):
    wf = manager.walk_forward_workflow
    wf._calibrate_signal_quality_thresholds = lambda *_a, **_k: None
    wf._calibrate_walk_forward_signal_parameters = lambda **kw: calls.append(kw) or {"ran": True}
    return wf._run_signal_calibration_steps(
        {"XGBoost": [{"Fold": 1}]},
        {"XGBoost": {"pred_target": [0.1]}},
        {},
        {"status": "ok"},
        {},
    )


def test_context_maps_flag_from_execution_config(tmp_path):
    off = make_eval_manager(str(tmp_path / "a"))
    on = make_eval_manager(str(tmp_path / "b"), exe_cfg=ExecutionConfig(enable_signal_execution_calibration=True))
    assert off.context.enable_signal_execution_calibration is False
    assert on.context.enable_signal_execution_calibration is True


def test_execution_calibration_skipped_when_disabled(tmp_path):
    calls = []
    manager = make_eval_manager(str(tmp_path))
    assert _run_steps(manager, calls) == {}
    assert calls == []
    summary = manager.walk_forward_workflow.signal_threshold_calibration_summary
    assert summary["execution_calibration_status"] == "disabled_by_config"


def test_execution_calibration_runs_when_enabled(tmp_path):
    calls = []
    manager = make_eval_manager(
        str(tmp_path), exe_cfg=ExecutionConfig(enable_signal_execution_calibration=True)
    )
    assert _run_steps(manager, calls) == {"ran": True}
    assert len(calls) == 1
