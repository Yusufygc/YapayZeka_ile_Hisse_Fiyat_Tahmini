# -*- coding: utf-8 -*-
"""
tft_v2 - Tam Temporal Fusion Transformer (v2)
from src.models.tft_v2 import TFTModel
from src.models.tft_v2 import StaticCovariateEncoder
from src.models.tft_v2 import MultiHorizonHead
"""

from src.models.tft_v2.model import TFTModel
from src.models.tft_v2.encoders import StaticCovariateEncoder
from src.models.tft_v2.output_heads import MultiHorizonHead

__all__ = ["TFTModel", "StaticCovariateEncoder", "MultiHorizonHead"]


# --- Registry tescili (Faz 1) -------------------------------------------
from src.pipeline.model_registry import ModelSpec, register_model  # noqa: E402

register_model(ModelSpec(
    name="TFT",
    factory=lambda **kw: TFTModel(**kw),
    category="seq",
    role="candidate",
    ensemble_eligible=True,
    requires=("torch",),
    needs_config_keys=("tft",),
    default_candidate=True,
    description="Temporal Fusion Transformer v2; multi-horizon sequence forecaster.",
))
