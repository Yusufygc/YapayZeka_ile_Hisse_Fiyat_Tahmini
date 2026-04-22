# -*- coding: utf-8 -*-
"""Explainability helpers for model result reports."""

__all__ = ["XAIExplainer", "XAIReportWriter"]


def __getattr__(name):
    if name == "XAIExplainer":
        from src.xai.explainer import XAIExplainer

        return XAIExplainer
    if name == "XAIReportWriter":
        from src.xai.report_writer import XAIReportWriter

        return XAIReportWriter
    raise AttributeError(name)
