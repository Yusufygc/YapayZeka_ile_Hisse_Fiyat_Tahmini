# -*- coding: utf-8 -*-

import os
import unittest
from unittest.mock import patch

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
            risk_free_annual=0.05,
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
        self.assertEqual(curve["Position"].tolist(), [1.0, 1.0, 1.0])
        self.assertAlmostEqual(curve["Net_Return"].iloc[0], 1.0)
        self.assertAlmostEqual(curve["Net_Return"].iloc[1], 0.10)

    def test_perfect_one_step_signal_is_applied_to_same_aligned_return_row(self):
        dates = pd.date_range("2024-01-02", periods=3, freq="D")
        prediction_dates = dates - pd.Timedelta(days=1)
        prev_close = np.array([100.0, 110.0, 99.0])
        y_true_price = np.array([110.0, 99.0, 108.9])
        pred_price = y_true_price.copy()

        result = run_backtest(
            dates=dates,
            prediction_dates=prediction_dates,
            y_true_price=y_true_price,
            pred_price=pred_price,
            prev_close=prev_close,
            model_name="Perfect",
            validation_mode="test",
            target_mode="price",
            signal_mode="legacy",
            commission_bps=0.0,
            slippage_bps=0.0,
        )

        curve = result["equity_curve"]
        self.assertEqual(curve["Position"].tolist(), [1.0, 0.0, 1.0])
        np.testing.assert_allclose(curve["Net_Return"].to_numpy(), np.array([0.10, 0.0, 0.10]))

    def test_simple_signal_generates_buy_sell_buy_orders_without_costs(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        prev_close = np.full(3, 100.0)
        pred_target = np.array([0.02, -0.02, 0.03])

        result = run_backtest(
            dates=dates,
            prediction_dates=dates - pd.Timedelta(days=1),
            y_true_price=np.array([105.0, 95.0, 110.0]),
            pred_price=prev_close * np.exp(pred_target),
            prev_close=prev_close,
            pred_target=pred_target,
            model_name="Simple",
            validation_mode="test",
            target_mode="log_return",
            signal_mode="simple",
        )

        curve = result["equity_curve"]
        self.assertEqual(curve["Executable_Order_TR"].tolist(), ["AL", "SAT", "AL"])
        self.assertEqual(curve["Decision"].tolist(), ["BUY", "EXIT", "BUY"])
        self.assertEqual(curve["Position"].tolist(), [1.0, 0.0, 1.0])
        self.assertAlmostEqual(float(curve["Transaction_Cost"].sum()), 0.0)
        self.assertAlmostEqual(float(curve["Commission_Cost"].sum()), 0.0)
        self.assertAlmostEqual(float(curve["Slippage_Cost"].sum()), 0.0)
        np.testing.assert_allclose(curve["Net_Return"].to_numpy(), np.array([0.05, 0.0, 0.10]))

    def test_simple_signal_holds_after_initial_buy_and_stays_flat_on_negative_start(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        prev_close = np.full(3, 100.0)

        positive = run_backtest(
            dates=dates,
            y_true_price=np.full(3, 101.0),
            pred_price=np.full(3, 102.0),
            prev_close=prev_close,
            model_name="Simple",
            validation_mode="test",
            target_mode="price",
            signal_mode="simple",
        )["equity_curve"]
        self.assertEqual(positive["Executable_Order_TR"].tolist(), ["AL", "TUT", "TUT"])
        self.assertEqual(positive["Decision"].tolist(), ["BUY", "HOLD", "HOLD"])

        negative = run_backtest(
            dates=dates,
            y_true_price=np.full(3, 99.0),
            pred_price=np.full(3, 98.0),
            prev_close=prev_close,
            model_name="Simple",
            validation_mode="test",
            target_mode="price",
            signal_mode="simple",
        )["equity_curve"]
        self.assertEqual(negative["Executable_Order_TR"].tolist(), ["TUT", "TUT", "TUT"])
        self.assertEqual(negative["Position"].tolist(), [0.0, 0.0, 0.0])

    def test_professional_recommendation_marks_flat_negative_signal_as_sat_without_short(self):
        pred_target = np.array([-0.03, 0.03, -0.03])
        prev_close = np.full(3, 100.0)
        pred_price = prev_close * np.exp(pred_target)
        frame = generate_professional_signals(
            pred_target,
            pred_price,
            prev_close,
            "log_return",
            observed_returns=np.zeros(3),
            commission_bps=0.0,
            slippage_bps=0.0,
            config=SignalConfig(min_holding_bars=1, max_holding_bars=2, min_entry_threshold=0.001),
        )

        self.assertEqual(frame.loc[0, "Decision"], "NO_TRADE")
        self.assertEqual(frame.loc[0, "Recommendation_TR"], "SAT")
        self.assertEqual(frame.loc[0, "Position"], 0.0)
        self.assertEqual(frame.loc[1, "Recommendation_TR"], "AL")

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
        self.assertAlmostEqual(result["trades"]["Net_Return"].iloc[0], curve["Equity"].iloc[1] - 1.0)

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

    def test_evds_interest_rate_fetch_normalizes_json_response(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "items": [
                        {"Tarih": "01-01-2024", "TP_PPK_H01": "42,50", "UNIXTIME": 1},
                        {"Tarih": "01-02-2024", "TP_PPK_H01": "45.0", "UNIXTIME": 2},
                    ]
                }

        mp = MacroPipeline()
        with patch.dict(os.environ, {"TCMB_EVDS_API_KEY": "test-key"}):
            with patch("src.features.macro_pipeline.requests.get", return_value=FakeResponse()) as get_mock:
                df = mp._fetch_evds_series("TP.PPK.H01", "2024-01-01", "2024-02-01", "INTEREST_RATE")

        self.assertEqual(list(df.columns), ["Date", "INTEREST_RATE"])
        self.assertEqual(list(df["INTEREST_RATE"]), [42.5, 45.0])
        self.assertEqual(get_mock.call_args.kwargs["params"]["startDate"], "01-01-2024")
        self.assertEqual(get_mock.call_args.kwargs["headers"]["key"], "test-key")

    def test_evds_interest_rate_fetch_skips_without_api_key(self):
        mp = MacroPipeline()
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.features.macro_pipeline.requests.get") as get_mock:
                df = mp._fetch_evds_series("TP.PPK.H01", "2024-01-01", "2024-02-01", "INTEREST_RATE")

        self.assertIsNone(df)
        get_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
