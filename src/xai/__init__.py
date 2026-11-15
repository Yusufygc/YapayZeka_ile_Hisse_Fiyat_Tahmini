# -*- coding: utf-8 -*-
"""Kol-B serving XAI yardimcilari (ozellik sozlugu, strateji, grup ozeti)."""

__all__ = ["XAIBackgroundProvider"]


def __getattr__(name):
    if name == "XAIBackgroundProvider":
        from src.xai.background import XAIBackgroundProvider

        return XAIBackgroundProvider
    raise AttributeError(name)
