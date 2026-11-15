# -*- coding: utf-8 -*-
"""Geriye uyumlu sarmalayici: asil kod src/serving/nightly.py (python -m src.serving.nightly)."""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving.nightly import main  # noqa: E402
if __name__ == "__main__":
    raise SystemExit(main())
