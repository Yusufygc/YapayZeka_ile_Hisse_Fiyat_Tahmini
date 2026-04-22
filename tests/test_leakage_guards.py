# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd

from src.backtesting.signals import SignalConfig, generate_professional_signals
from src.backtesting.engine import run_backtest
from src.evaluation.financial_metrics import compute_financial_metrics
from src.features.macro_pipeline import MacroPipeline


class LeakageGuardTests(unittest.TestCase):
    def test_directional_accuracy_uses_target_returns(self):
        prev_close = np.array([100.0, 100.0, 100.0])
        true_target = np.array([0.01, -0.02, 0.03])
        pred_target = np.array([0.02, -0.01, 0.04])
        y_true_price = prev_close * np.exp(true_target)
        y_pred_price = prev_close * np.exp(pred_target)

        metrics = compute_financial_metrics(
            y_true_price,
            y_pred_price,
            y_true_target=true_target,
            y_pred_target=pred_target,
            prev_close=prev_close,
            target_mode="log_return",
        )

        self.assertEqual(metrics["Dir_Acc"], 100.0)
        self.assertGreater(metrics["BuyHold_Sharpe"], -100.0)

    def test_price_mode_directional_accuracy_uses_price_returns(self):
        prev_close = np.array([100.0, 100.0, 100.0])
        y_true_price = np.array([101.0, 99.0, 102.0])
        y_pred_price = np.array([102.0, 98.0, 103.0])

        metrics = compute_financial_metrics(
            y_true_price,
            y_pred_price,
            prev_close=prev_close,
            target_mode="price",
        )

        self.assertEqual(metrics["Dir_Acc"], 100.0)
        self.assertLess(metrics["Return_RMSE"], 0.02)

    def test_professional_signals_ignore_realized_price(self):
        pred_target = np.array([0.02, 0.02, 0.02, -0.05, 0.02])
        prev_close = np.full(5, 100.0)
        pred_price = prev_close * np.exp(pred_target)
        observed_returns = np.zeros(5)
        cfg = SignalConfig(min_directional_accuracy=0.0, min_composite_score=0.0)

        no_realized = generate_professional_signals(
            pred_target,
            pred_price,
            prev_close,
            "log_return",
            observed_returns=observed_returns,
            commission_bps=0.0,
            slippage_bps=0.0,
            config=cfg,
        )
        with_realized = generate_professional_signals(
            pred_target,
            pred_price,
            prev_close,
            "log_return",
            observed_returns=observed_returns,
            realized_price=np.array([1.0, 1000.0, 1.0, 1000.0, 1.0]),
            commission_bps=0.0,
            slippage_bps=0.0,
            config=cfg,
        )

        self.assertEqual(no_realized["Decision"].tolist(), with_realized["Decision"].tolist())
        self.assertEqual(no_realized["Position"].tolist(), with_realized["Position"].tolist())

    def test_backtest_same_bar_realized_price_does_not_change_same_bar_decision(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        prev_close = np.full(5, 100.0)
        pred_target = np.full(5, 0.02)
        pred_price = prev_close * np.exp(pred_target)
        cfg = SignalConfig(
            min_directional_accuracy=0.0,
            min_composite_score=0.0,
            min_holding_bars=1,
            volatility_window=2,
        )

        base = run_backtest(
            dates=dates,
            prediction_dates=dates - pd.Timedelta(days=1),
            y_true_price=np.array([101.0, 100.0, 100.0, 100.0, 100.0]),
            pred_price=pred_price,
            prev_close=prev_close,
            pred_target=pred_target,
            model_name="Model",
            validation_mode="test",
            target_mode="log_return",
            signal_mode="professional",
            signal_config=cfg,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        shocked = run_backtest(
            dates=dates,
            prediction_dates=dates - pd.Timedelta(days=1),
            y_true_price=np.array([50.0, 100.0, 100.0, 100.0, 100.0]),
            pred_price=pred_price,
            prev_close=prev_close,
            pred_target=pred_target,
            model_name="Model",
            validation_mode="test",
            target_mode="log_return",
            signal_mode="professional",
            signal_config=cfg,
            commission_bps=0.0,
            slippage_bps=0.0,
        )

        self.assertEqual(
            base["equity_curve"]["Decision"].iloc[0],
            shocked["equity_curve"]["Decision"].iloc[0],
        )
        self.assertEqual(
            base["equity_curve"]["Observed_Return_At_Decision"].iloc[0],
            shocked["equity_curve"]["Observed_Return_At_Decision"].iloc[0],
        )

    def test_backtest_applies_decision_to_next_bar_return(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        prev_close = np.full(3, 100.0)
        pred_price = np.array([101.0, 101.0, 101.0])

        result = run_backtest(
            dates=dates,
            y_true_price=np.array([200.0, 110.0, 100.0]),
            pred_price=pred_price,
            prev_close=prev_close,
            model_name="Model",
            validation_mode="test",
            target_mode="price",
            signal_mode="legacy",
            commission_bps=0.0,
            slippage_bps=0.0,
        )

        curve = result["equity_curve"]
        self.assertEqual(curve["Desired_Position"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(curve["Position"].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(curve["Net_Return"].iloc[0], 0.0)
        self.assertAlmostEqual(curve["Net_Return"].iloc[1], 0.10)

    def test_backtest_accounts_entry_exit_commission_and_slippage_separately(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        prev_close = np.full(3, 100.0)
        pred_price = np.array([101.0, 99.0, 99.0])

        result = run_backtest(
            dates=dates,
            y_true_price=np.array([100.0, 100.0, 100.0]),
            pred_price=pred_price,
            prev_close=prev_close,
            model_name="Model",
            validation_mode="test",
            target_mode="price",
            signal_mode="legacy",
            commission_bps=10.0,
            slippage_bps=5.0,
        )

        curve = result["equity_curve"]
        self.assertAlmostEqual(curve["Entry_Transaction_Cost"].sum(), 0.0015)
        self.assertAlmostEqual(curve["Exit_Transaction_Cost"].sum(), 0.0015)
        self.assertAlmostEqual(curve["Commission_Cost"].sum(), 0.0020)
        self.assertAlmostEqual(curve["Slippage_Cost"].sum(), 0.0010)

    def test_macro_release_lag_shifts_monthly_features(self):
        raw = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "INTEREST_RATE": [10.0, 12.0],
        })
        mp = MacroPipeline(rate_release_lag_days=3, cpi_release_lag_days=15)
        feats = mp._engineer_monthly_rate(raw)
        feats["Date"] = feats["Date"] + pd.to_timedelta(mp.rate_release_lag_days, unit="D")

        self.assertEqual(feats["Date"].iloc[0], pd.Timestamp("2024-01-04"))
        self.assertEqual(feats["Rate_Change"].iloc[1], 2.0)


if __name__ == "__main__":
    unittest.main()
