# -*- coding: utf-8 -*-
"""
Project-level pytest configuration.

Optional heavy dependencies are stubbed only when a real import fails. This
keeps dl_env validation honest while still allowing lightweight CI/sandbox runs.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import uuid
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PYTEST_TMP_ROOT = os.path.join(_PROJECT_ROOT, "outputs", "_pytest_tmp")
os.makedirs(_PYTEST_TMP_ROOT, exist_ok=True)
os.environ.setdefault("TMP", _PYTEST_TMP_ROOT)
os.environ.setdefault("TEMP", _PYTEST_TMP_ROOT)
os.environ.setdefault("TMPDIR", _PYTEST_TMP_ROOT)


class _WorkspaceTemporaryDirectory:
    """Permission-tolerant TemporaryDirectory replacement for Windows tests."""

    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        ignore_cleanup_errors: bool = False,
    ) -> None:
        base_dir = dir or _PYTEST_TMP_ROOT
        os.makedirs(base_dir, exist_ok=True)
        name = f"{prefix or 'tmp'}{uuid.uuid4().hex}{suffix or ''}"
        self.name = os.path.join(base_dir, name)
        self.ignore_cleanup_errors = ignore_cleanup_errors
        os.makedirs(self.name, exist_ok=False)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


def _patch_tempfile() -> None:
    import tempfile

    tempfile.tempdir = _PYTEST_TMP_ROOT
    tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory


def _stub(name: str) -> MagicMock:
    mock = MagicMock()
    mock.__name__ = name
    mock.__package__ = name.rpartition(".")[0] or name
    mock.__spec__ = ModuleSpec(name, loader=None)
    mock.__path__ = []
    sys.modules[name] = mock
    return mock


def _real_import_available(name: str) -> bool:
    if name in sys.modules and not isinstance(sys.modules[name], MagicMock):
        return True
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _stub_tensorflow() -> None:
    _tf = _stub("tensorflow")
    _keras = _stub("tensorflow.keras")
    _tf.keras = _keras
    for submodule in [
        "tensorflow.keras.models",
        "tensorflow.keras.layers",
        "tensorflow.keras.callbacks",
        "tensorflow.keras.optimizers",
        "tensorflow.keras.regularizers",
        "tensorflow.keras.backend",
        "tensorflow.keras.utils",
    ]:
        module = _stub(submodule)
        setattr(_keras, submodule.split("tensorflow.keras.")[-1], module)

    models = sys.modules["tensorflow.keras.models"]
    for symbol in ("Sequential", "Model", "load_model"):
        setattr(models, symbol, MagicMock)

    layers = sys.modules["tensorflow.keras.layers"]
    for symbol in (
        "LSTM",
        "Dense",
        "Dropout",
        "Bidirectional",
        "Input",
        "Permute",
        "Multiply",
        "Flatten",
        "RepeatVector",
        "Lambda",
        "Layer",
    ):
        setattr(layers, symbol, MagicMock)

    callbacks = sys.modules["tensorflow.keras.callbacks"]
    for symbol in ("EarlyStopping", "ReduceLROnPlateau"):
        setattr(callbacks, symbol, MagicMock)


def _stub_ta() -> None:
    ta_module = _stub("ta")
    for submodule in ("ta.momentum", "ta.trend", "ta.volatility", "ta.volume", "ta.others"):
        module = _stub(submodule)
        setattr(ta_module, submodule.rpartition(".")[-1], module)
    setattr(ta_module, "add_all_ta_features", MagicMock(return_value=MagicMock()))


def _stub_xgboost() -> None:
    module = _stub("xgboost")
    setattr(module, "XGBRegressor", MagicMock)


def _stub_lightgbm() -> None:
    module = _stub("lightgbm")
    setattr(module, "LGBMRegressor", MagicMock)


def _stub_prophet() -> None:
    module = _stub("prophet")
    setattr(module, "Prophet", MagicMock)


def _stub_optuna() -> None:
    module = _stub("optuna")
    setattr(module, "create_study", MagicMock())
    setattr(module, "Trial", MagicMock)
    _stub("optuna.integration")
    _stub("optuna.samplers")


def _install_optional_dependency_stubs() -> None:
    optional_modules = {
        "tensorflow": _stub_tensorflow,
        "yfinance": lambda: _stub("yfinance"),
        "xgboost": _stub_xgboost,
        "ta": _stub_ta,
        "lightgbm": _stub_lightgbm,
        "prophet": _stub_prophet,
        "optuna": _stub_optuna,
    }
    for name, installer in optional_modules.items():
        if not _real_import_available(name):
            installer()


_patch_tempfile()
_install_optional_dependency_stubs()
