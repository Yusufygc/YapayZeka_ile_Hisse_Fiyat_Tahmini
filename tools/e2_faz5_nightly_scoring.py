# -*- coding: utf-8 -*-
"""Geriye uyumlu sarmalayici: yalniz skorlama (src/serving/nightly.py::score_main)."""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving.nightly import score_main  # noqa: E402
if __name__ == "__main__":
    raise SystemExit(score_main())
