# -*- coding: utf-8 -*-
"""Serving katmaninin ortak yol yardimcilari (yalniz stdlib).

`default_serving_db_path` tek yerde tanimlidir; API, GUI ve nightly ayni yardimciyi kullanir.
"""
from __future__ import annotations

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def project_root() -> str:
    return _PROJECT_ROOT


def default_data_dir() -> str:
    return os.path.join(_PROJECT_ROOT, "data")


def default_serving_db_path() -> str:
    return os.path.join(default_data_dir(), "serving_pool.db")
