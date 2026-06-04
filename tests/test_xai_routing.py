import numpy as np

from src.pipeline.evaluation_services import EvaluationContext, EvaluationState
from src.pipeline.metrics_reporter import _MetricsReporterMixin


class _StubReporter(_MetricsReporterMixin):
    def __init__(self):
        # Faz 3.4 (E1 DI): mixin artik self.ctx (READ-ONLY config) / self.state
        # (mutable runtime) okur; stub bunlari gercek dataclass'larla kurar.
        self.ctx = EvaluationContext(
            stock_symbol="TEST",
            feature_names=["Close"],
            dataset_metadata={},
            xai_dir="unused",
        )
        self.state = EvaluationState(
            predictions={"Model": np.array([1.0, 2.0])},
            prediction_targets={"Model": np.array([0.01, 0.02])},
            y_true_aligned=np.array([1.1, 2.1]),
            quantile_predictions={},
            latest_backtest_results={"wf": {"Model": {"ok": True}}},
        )
        self.writes = []

    def _write_xai_reports(self, payload, suffix: str) -> None:
        self.writes.append((payload, suffix))


def test_single_split_xai_uses_latest_suffix(monkeypatch):
    import src.pipeline.metrics_reporter as metrics_reporter

    class _Explainer:
        def __init__(self, *args, **kwargs):
            pass

        def explain_single_split(self, **kwargs):
            return {"summary_md": "# Single"}

    monkeypatch.setattr(metrics_reporter, "XAIExplainer", _Explainer)
    reporter = _StubReporter()

    payload = reporter._get_xai_single_split(trained_models={}, tensors={})

    assert payload == {"summary_md": "# Single"}
    assert reporter.writes == [({"summary_md": "# Single"}, "latest")]


def test_single_split_xai_warning_names_single_split(monkeypatch, capsys):
    import src.pipeline.metrics_reporter as metrics_reporter

    class _Explainer:
        def __init__(self, *args, **kwargs):
            pass

        def explain_single_split(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(metrics_reporter, "XAIExplainer", _Explainer)
    reporter = _StubReporter()

    assert reporter._get_xai_single_split(trained_models={}, tensors={}) is None

    output = capsys.readouterr().out
    assert "Single split XAI" in output
    assert "Walk-forward XAI" not in output


def test_walk_forward_xai_keeps_wf_suffix(monkeypatch):
    import src.pipeline.metrics_reporter as metrics_reporter

    class _Explainer:
        def __init__(self, *args, **kwargs):
            pass

        def explain_walk_forward(self, **kwargs):
            return {"summary_md": "# WF"}

    monkeypatch.setattr(metrics_reporter, "XAIExplainer", _Explainer)
    reporter = _StubReporter()

    payload = reporter._get_xai_walk_forward(
        wf_predictions={"Model": np.array([1.0])},
        wf_y_true=np.array([1.1]),
        wf_backtest_inputs={},
    )

    assert payload == {"summary_md": "# WF"}
    assert reporter.writes == [({"summary_md": "# WF"}, "wf")]
