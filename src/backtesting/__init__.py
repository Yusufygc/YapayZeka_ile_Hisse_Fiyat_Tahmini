# -*- coding: utf-8 -*-
"""Backtesting paketi.

Motor (run_backtest), metrikler (summarize_backtest) ve raporlama
yardımcılarını lazy (__getattr__) olarak dışa açar; ağır importlar yalnızca
kullanıldığında yüklenir.
"""

__all__ = [
    "plot_equity_curves",
    "run_backtest",
    "save_backtest_report",
    "save_fold_backtest_report",
    "save_order_report",
    "save_trade_logs",
    "summarize_backtest",
]


def __getattr__(name: str):
    if name == "run_backtest":
        from src.backtesting.engine import run_backtest

        return run_backtest
    if name == "summarize_backtest":
        from src.backtesting.metrics import summarize_backtest

        return summarize_backtest
    if name in {"plot_equity_curves", "save_backtest_report", "save_fold_backtest_report", "save_order_report", "save_trade_logs"}:
        from src.backtesting.reporting import (
            plot_equity_curves,
            save_backtest_report,
            save_fold_backtest_report,
            save_order_report,
            save_trade_logs,
        )

        return {
            "plot_equity_curves": plot_equity_curves,
            "save_backtest_report": save_backtest_report,
            "save_fold_backtest_report": save_fold_backtest_report,
            "save_order_report": save_order_report,
            "save_trade_logs": save_trade_logs,
        }[name]
    raise AttributeError(name)
