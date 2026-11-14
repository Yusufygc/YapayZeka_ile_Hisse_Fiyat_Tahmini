# -*- coding: utf-8 -*-
"""P0-1 / P0-8 regresyon testleri.

P0-1: Walk-forward `tune_on_fit=True` yolu ortak, kalici bir Optuna study'sini
      (data/optuna/optuna_studies.db) yeniden kullaniyordu -> diger pencerelerde
      (gelecek dahil) skorlanmis trial'lar `best_params`'i belirliyordu (leakage).
P0-8: `_compute_strategy_returns` fiyat modunda sinyali gerceklesen getiriden
      turetiyordu (|realized| = mukemmel ongorü).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")
pytest.importorskip("xgboost")

from src.models import _tuning  # noqa: E402


def _synthetic(seed: int, n: int = 60, f: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = 0.5 * X[:, 0] + rng.normal(scale=0.5, size=n)
    return X, y


@pytest.fixture
def create_study_spy(monkeypatch, tmp_path):
    """optuna.create_study cagrilarini kaydeder; kalici storage'i tmp'e yonlendirir
    (gercek data/optuna DB'si hicbir kosulda kirletilmesin)."""
    calls: list[dict] = []
    studies: list = []
    real_create = optuna.create_study

    def _spy(*args, **kwargs):
        calls.append(dict(kwargs))
        kw = dict(kwargs)
        if kw.get("storage") is not None:
            redirected = tmp_path / "redirected.db"
            kw["storage"] = f"sqlite:///{redirected.as_posix()}"
        study = real_create(*args, **kw)
        studies.append(study)
        return study

    monkeypatch.setattr(optuna, "create_study", _spy)
    return calls, studies


def _persistent_db_mtime():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "optuna", "optuna_studies.db")
    return os.path.getmtime(path) if os.path.exists(path) else None


def test_default_tuning_uses_in_memory_study_no_reuse(create_study_spy):
    from src.models.xgboost_model import XGBoostModel

    calls, studies = create_study_spy
    n_trials = 2
    for seed in (1, 2):
        X, y = _synthetic(seed)
        XGBoostModel().tune_and_train(X, y, n_trials=n_trials, n_splits=2)

    assert len(studies) == 2
    for kw in calls:
        assert kw.get("storage") is None, "varsayilan yol kalici storage kullanmamali"
    # Ikinci cagri onceki cagrinin trial'larini gormemeli.
    assert len(studies[1].trials) == n_trials


def test_random_forest_default_tuning_no_reuse(create_study_spy):
    from src.models.random_forest_model import RandomForestModel

    calls, studies = create_study_spy
    n_trials = 2
    for seed in (3, 4):
        X, y = _synthetic(seed)
        RandomForestModel().tune_and_train(X, y, n_trials=n_trials, n_splits=2)

    for kw in calls:
        assert kw.get("storage") is None
    assert len(studies[1].trials) == n_trials


def test_default_path_does_not_touch_persistent_db(create_study_spy):
    from src.models.xgboost_model import XGBoostModel

    before = _persistent_db_mtime()
    X, y = _synthetic(5)
    XGBoostModel(tune_on_fit=True, tune_n_trials=2, tune_n_splits=2).train(X, y)
    assert _persistent_db_mtime() == before


def test_storage_without_study_name_raises(tmp_path):
    from src.models.random_forest_model import RandomForestModel
    from src.models.xgboost_model import XGBoostModel

    X, y = _synthetic(6)
    with pytest.raises(ValueError):
        XGBoostModel().tune_and_train(
            X, y, n_trials=1, n_splits=2, study_storage=f"sqlite:///{(tmp_path / 'x.db').as_posix()}"
        )
    with pytest.raises(ValueError):
        RandomForestModel().tune_and_train(
            X, y, n_trials=1, n_splits=2, study_storage=f"sqlite:///{(tmp_path / 'x.db').as_posix()}"
        )


def test_explicit_storage_and_name_is_persisted(create_study_spy):
    calls, studies = create_study_spy
    _tuning.run_optuna_study(
        lambda trial: trial.suggest_float("a", 0.0, 1.0),
        n_trials=2,
        n_splits=2,
        random_state=0,
        study_name="explicit_name",
        log_prefix="t",
        study_storage="sqlite:///ignored.db",
    )
    assert calls[0]["storage"] == "sqlite:///ignored.db"
    assert calls[0]["study_name"] == "explicit_name"
    assert len(studies[0].trials) == 2


def test_tuned_tree_models_study_name_unique_per_run():
    from src.pipeline import training_workflows as tw

    seen: list[tuple[str, str]] = []

    class _FakeModel:
        def tune_and_train(self, X, y, **kw):
            seen.append((kw["study_name"], kw["study_storage"]))

    fake_trainer = SimpleNamespace(XGBoostModel=_FakeModel, RandomForestModel=_FakeModel)

    def _run(run_id):
        ctx = SimpleNamespace(
            stock_symbol="AKBNK",
            dataset_metadata={"run_id": run_id} if run_id else {},
            selected_models=[],
        )
        state = SimpleNamespace(trained_models={})
        helpers = SimpleNamespace(skip=lambda name: False)
        wf = tw.SingleSplitTrainingWorkflow(ctx, state, helpers)
        tensors = {"X_train_s": np.zeros((4, 2)), "y_train_s": np.zeros(4)}
        orig = tw.importlib.import_module
        tw.importlib.import_module = lambda name: fake_trainer
        try:
            wf._train_tuned_tree_models(tensors, "sqlite:///dummy.db")
        finally:
            tw.importlib.import_module = orig

    _run("runA")
    _run("runB")
    _run(None)
    _run(None)
    names = [n for n, _ in seen]
    assert len(names) == len(set(names)), f"study adlari tekrar ediyor: {names}"


# --------------------------------------------------------------------------
# P0-8
# --------------------------------------------------------------------------
def test_price_mode_strategy_returns_no_foresight():
    from src.validation.walk_forward import _compute_strategy_returns

    prev_close = np.array([100.0, 100.0, 100.0, 100.0])
    y_true_price = np.array([101.0, 99.0, 102.0, 98.0])  # +, -, +, -
    # Her zaman ters yon tahmin
    y_pred_price = np.array([99.0, 101.0, 98.0, 102.0])
    out = _compute_strategy_returns(
        np.array([]),
        np.array([]),
        y_true_price=y_true_price,
        prev_close=prev_close,
        target_mode="price",
        y_pred_price=y_pred_price,
    )
    assert out.size == 4
    assert float(np.mean(out)) < 0
    assert np.all(out < 0)


def test_price_mode_strategy_returns_correct_direction_positive():
    from src.validation.walk_forward import _compute_strategy_returns

    prev_close = np.array([100.0, 100.0])
    y_true_price = np.array([101.0, 99.0])
    out = _compute_strategy_returns(
        np.array([]),
        np.array([]),
        y_true_price=y_true_price,
        prev_close=prev_close,
        target_mode="price",
        y_pred_price=np.array([100.5, 99.5]),
    )
    assert np.all(out > 0)


def test_price_mode_requires_y_pred_price():
    from src.validation.walk_forward import _compute_strategy_returns

    with pytest.raises(ValueError):
        _compute_strategy_returns(
            np.array([]),
            np.array([]),
            y_true_price=np.array([101.0]),
            prev_close=np.array([100.0]),
            target_mode="price",
        )
