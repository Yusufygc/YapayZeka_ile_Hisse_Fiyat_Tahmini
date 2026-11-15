# -*- coding: utf-8 -*-
"""
reproducibility.py — Ensures deterministic behavior across the pipeline.
"""
import os
import random
import sys

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Sets the global random seed for Python, NumPy (and PyTorch when it is already
    loaded) to ensure reproducible experiment results.
    """
    # 1. Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Set Python built-in random seed
    random.seed(seed)

    # 3. Set NumPy random seed
    np.random.seed(seed)

    # 4. Seed PyTorch only if it is already imported (pooled path); the per-symbol
    #    path never imports it, so we do not pay the import cost here.
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)

    print(f"  [INFO] Global random seed set to: {seed}")
