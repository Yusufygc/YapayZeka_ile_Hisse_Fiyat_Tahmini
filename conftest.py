# -*- coding: utf-8 -*-
"""
conftest.py -- Proje koku pytest konfigurasyonu
================================================
Sandbox / CI ortamlarinda kurulmamis opsiyonel kutuphane bagimliliklar
(tensorflow, yfinance, xgboost, ta, lightgbm, prophet, optuna, torch) icin
sys.modules duzeyinde MagicMock taslaklari olusturur.

Pytest bu dosyayi HERHANGI bir test modulu import edilmeden once yukler;
dolayisiyla tum smoke testleri icin koruma saglar.
"""

import sys
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock


def _stub(name: str) -> MagicMock:
    """Verilen isimle sys.modules'a kaydedilmis bir MagicMock modulü doner."""
    mock = MagicMock()
    mock.__name__    = name
    mock.__package__ = name
    mock.__spec__    = ModuleSpec(name, loader=None)  # find_spec() icin gecerli spec
    mock.__path__    = []
    sys.modules[name] = mock
    return mock


# ---- TensorFlow -------------------------------------------------------------
if "tensorflow" not in sys.modules:
    _tf = _stub("tensorflow")
    _keras = _stub("tensorflow.keras")
    _tf.keras = _keras

    for _sub in [
        "tensorflow.keras.models",
        "tensorflow.keras.layers",
        "tensorflow.keras.callbacks",
        "tensorflow.keras.optimizers",
        "tensorflow.keras.regularizers",
        "tensorflow.keras.backend",
        "tensorflow.keras.utils",
    ]:
        _m = _stub(_sub)
        _attr = _sub.split("tensorflow.keras.")[-1]
        setattr(_keras, _attr, _m)

    _km = sys.modules["tensorflow.keras.models"]
    for _s in ("Sequential", "Model", "load_model"):
        setattr(_km, _s, MagicMock)

    _kl = sys.modules["tensorflow.keras.layers"]
    for _s in ("LSTM", "Dense", "Dropout", "Bidirectional", "Input",
               "Permute", "Multiply", "Flatten", "RepeatVector", "Lambda", "Layer"):
        setattr(_kl, _s, MagicMock)

    _kc = sys.modules["tensorflow.keras.callbacks"]
    for _s in ("EarlyStopping", "ReduceLROnPlateau"):
        setattr(_kc, _s, MagicMock)


# ---- yfinance ---------------------------------------------------------------
if "yfinance" not in sys.modules:
    _stub("yfinance")


# ---- xgboost ----------------------------------------------------------------
if "xgboost" not in sys.modules:
    _xgb = _stub("xgboost")
    setattr(_xgb, "XGBRegressor", MagicMock)


# ---- ta (Technical Analysis Library) ----------------------------------------
if "ta" not in sys.modules:
    _ta = _stub("ta")
    for _sub in ("ta.momentum", "ta.trend", "ta.volatility", "ta.volume", "ta.others"):
        _stub(_sub)
    setattr(_ta, "add_all_ta_features", MagicMock(return_value=MagicMock()))


# ---- LightGBM ---------------------------------------------------------------
if "lightgbm" not in sys.modules:
    _lgb = _stub("lightgbm")
    setattr(_lgb, "LGBMRegressor", MagicMock)


# ---- Prophet ----------------------------------------------------------------
if "prophet" not in sys.modules:
    _pr = _stub("prophet")
    setattr(_pr, "Prophet", MagicMock)


# ---- Optuna -----------------------------------------------------------------
if "optuna" not in sys.modules:
    _opt = _stub("optuna")
    setattr(_opt, "create_study", MagicMock())
    setattr(_opt, "Trial", MagicMock)
    _stub("optuna.integration")
    _stub("optuna.samplers")
