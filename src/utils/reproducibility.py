# -*- coding: utf-8 -*-
"""
reproducibility.py — Ensures deterministic behavior across the pipeline.
"""
import os
import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed: int = 42) -> None:
    """
    Sets the global random seed for Python, NumPy, and TensorFlow to ensure
    reproducible experiment results.
    """
    # 1. Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 2. Set Python built-in random seed
    random.seed(seed)
    
    # 3. Set NumPy random seed
    np.random.seed(seed)
    
    # 4. Set TensorFlow random seed
    tf.random.set_seed(seed)
    
    # Optional: Force TensorFlow to use deterministic operations (may degrade performance slightly)
    # tf.config.experimental.enable_op_determinism()
    
    print(f"  [INFO] Global random seed set to: {seed}")
