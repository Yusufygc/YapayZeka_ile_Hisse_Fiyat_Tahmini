# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd

from src.backtesting.contracts import (
    compute_return_frame,
    empty_backtest_result,
    prepare_backtest_inputs,
)
from src.backtesting.engine import run_backtest


class BacktestEngineContractTests(unittest.TestCase):
    def test_prepare_backtest_inputs_tail_aligns_to_common_length(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        prediction_dates = dates - pd.Timedelta(days=1)

        frame = prepare_backtest_inputs(
            dates=dates,
            prediction_dates=prediction_dates,
            y_true_price=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
            pred_price=np.array([99.0, 100.0, 101.0, 102.0]),
            prev_close=np.full(5, 100.0),
            pred_target=np.array([0.01, 0.02, 0.03]),
            fold_ids=np.array([0, 0, 1, 1, 2]),
            market_regime=np.array([10, 20, 30, 40, 50], dtype=float),
        )

        self.assertEqual(frame.n, 3)
        self.assertEqual(frame.dates.tolist(), dates[-3:].tolist())
        self.assertEqual(frame.prediction_dates.tolist(), prediction_dates[-3:].tolist())
        np.testing.assert_allclose(frame.prices, np.array([102.0, 103.0, 104.0]))
        np.testing.assert_allclose(frame.pred_price, np.array([100.0, 101.0, 102.0]))
        np.testing.assert_allclose(frame.pred_target, np.array([0.01, 0.02, 0.03]))
        self.assertEqual(frame.fold_ids.tolist(), [1, 1, 2])
        np.testing.assert_allclose(frame.market_regime, np.array([30.0, 40.0, 50.0]))

    def test_prepare_backtest_inputs_defaults_prediction_dates_to_dates(self):
        dates = pd.date_range("2024-02-01", periods=3, freq="D")

        frame = prepare_backtest_inputs(
            dates=dates,
            y_true_price=np.array([10.0, 11.0, 12.0]),
            pred_price=np.array([10.5, 11.5, 12.5]),
            prev_close=np.array([9.0, 10.0, 11.0]),
        )

        self.assertTrue(frame.prediction_dates.equals(frame.dates))
        self.assertEqual(frame.fold_ids.tolist(), ["all", "all", "all"])
        np.testing.assert_allclose(frame.market_regime, np.zeros(3, dtype=float))

    def test_compute_return_frame_uses_prev_close_and_lagged_observed_return(self):
        returns = compute_return_frame(
            prices=np.array([110.0, 99.0, 108.9]),
            prev_close=np.array([100.0, 110.0, 99.0]),
        )

        np.testing.assert_allclose(returns.realized_returns, np.array([0.10, -0.10, 0.10]))
        np.testing.assert_allclose(returns.observed_returns, np.array([0.0, 0.10, -0.10]))
        np.testing.assert_allclose(returns.buy_hold_equity, np.array([1.10, 0.99, 1.089]))

    def test_empty_backtest_result_schema_is_stable(self):
        result = empty_backtest_result("Empty", "test")

        self.assertEqual(
            result["equity_curve"].columns.tolist(),
            ["Date", "Equity", "BuyHold_Equity", "Position", "Desired_Position", "Signal", "Net_Return"],
        )
        self.assertEqual(
            result["trades"].columns.tolist(),
            [
                "Model",
                "Fold",
                "Entry_Date",
                "Exit_Date",
                "Entry_Price",
                "Exit_Price",
                "Gross_Return",
                "Net_Return",
                "Holding_Period",
            ],
        )
        self.assertEqual(result["series"], {})

    def test_run_backtest_preserves_date_contract_columns(self):
        dates = pd.date_range("2024-01-02", periods=3, freq="D")
        prediction_dates = dates - pd.Timedelta(days=1)

        result = run_backtest(
            dates=dates,
            prediction_dates=prediction_dates,
            y_true_price=np.array([110.0, 99.0, 108.9]),
            pred_price=np.array([111.0, 98.0, 109.0]),
            prev_close=np.array([100.0, 110.0, 99.0]),
            model_name="Dates",
            validation_mode="test",
            target_mode="price",
            signal_mode="legacy",
            commission_bps=0.0,
            slippage_bps=0.0,
        )

        curve = result["equity_curve"]
        self.assertEqual(pd.to_datetime(curve["Prediction_Date"]).tolist(), prediction_dates.tolist())
        self.assertEqual(pd.to_datetime(curve["Date"]).tolist(), dates.tolist())
        self.assertEqual(pd.to_datetime(curve["Execution_Date"]).tolist(), dates.tolist())
        self.assertEqual(pd.to_datetime(curve["Realized_Return_Date"]).tolist(), dates.tolist())
        np.testing.assert_allclose(curve["Realized_Return"].to_numpy(), np.array([0.10, -0.10, 0.10]))

    def test_simple_signal_regression_is_unchanged(self):
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
        np.testing.assert_allclose(curve["Net_Return"].to_numpy(), np.array([0.05, 0.0, 0.10]))


if __name__ == "__main__":
    unittest.main()
