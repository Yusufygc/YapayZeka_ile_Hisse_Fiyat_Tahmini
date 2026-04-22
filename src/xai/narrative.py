# -*- coding: utf-8 -*-
"""
narrative.py - Convert numeric XAI signals into plain Turkish statements.
"""

from __future__ import annotations

import numpy as np

from src.xai.feature_dictionary import describe_feature


def direction_label(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "nötr"
    if value > 0:
        return "yukarı"
    if value < 0:
        return "aşağı"
    return "nötr"


def contribution_sentence(feature_name: str, contribution: float | None, approximate: bool = False) -> str:
    base = describe_feature(feature_name)
    prefix = "Yaklaşık olarak " if approximate else ""
    direction = direction_label(contribution)
    if direction == "yukarı":
        return f"{prefix}{base}, model tahminini yukarı yönde etkileyen sinyaller arasında."
    if direction == "aşağı":
        return f"{prefix}{base}, model tahminini aşağı yönde etkileyen sinyaller arasında."
    return f"{prefix}{base}, model tahmininde dikkat edilen sinyaller arasında."


def model_summary_sentence(model_name: str, pred_target: float | None, pred_price: float | None, actual_price: float | None = None) -> str:
    direction = direction_label(pred_target)
    if direction == "yukarı":
        move = "yukarı yönlü"
    elif direction == "aşağı":
        move = "aşağı yönlü"
    else:
        move = "yatay veya zayıf"

    price_part = ""
    if pred_price is not None and np.isfinite(pred_price):
        price_part = f" Tahmini fiyat seviyesi yaklaşık {pred_price:.4f}."
    if actual_price is not None and np.isfinite(actual_price):
        price_part += f" Gerçekleşen fiyat {actual_price:.4f}."
    return f"{model_name} modeli bu adimda {move} bir hareket bekledi.{price_part}"


def uncertainty_sentence(low: float | None, mid: float | None, high: float | None) -> str:
    values = [low, mid, high]
    if any(value is None or not np.isfinite(value) for value in values):
        return ""
    width = abs(high - low)
    rel_width = width / max(abs(mid), 1e-9)
    if rel_width >= 0.08:
        band = "geniş"
    elif rel_width <= 0.03:
        band = "dar"
    else:
        band = "orta genişlikte"
    return f"TFT tahmin aralığı {band}; bu, modelin belirsizlik seviyesini gösterir."
