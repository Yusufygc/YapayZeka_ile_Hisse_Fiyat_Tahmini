# -*- coding: utf-8 -*-

import os
import shutil
import unittest

import numpy as np
import pandas as pd

from src.models.arima_model import ARIMAModel
from src.models.linear_sequence_model import DLinearSequenceModel, NLinearSequenceModel


class Phase4ModelTests(unittest.TestCase):
    def test_dlinear_and_nlinear_sequence_baselines_predict_expected_shape(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(24, 8, 3))
        y = rng.normal(size=24)
        X_test = rng.normal(size=(5, 8, 3))

        for cls in (DLinearSequenceModel, NLinearSequenceModel):
            model = cls(alpha=0.1)
            model.train(X, y)
            preds = model.predict(X_test)
            self.assertEqual(preds.shape, (5,))
            self.assertTrue(np.all(np.isfinite(preds)))

    def test_linear_sequence_baseline_roundtrip_save_load(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(18, 6, 2))
        y = rng.normal(size=18)
        X_test = rng.normal(size=(3, 6, 2))

        model = DLinearSequenceModel(alpha=0.5)
        model.train(X, y)
        expected = model.predict(X_test)

        tmp = os.path.abspath(os.path.join("outputs", "_test_phase4_models"))
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        try:
            path = os.path.join(tmp, "dlinear.pkl")
            model.save(path)
            loaded = DLinearSequenceModel()
            loaded.load(path)
            actual = loaded.predict(X_test)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        np.testing.assert_allclose(actual, expected)

    def test_arima_auto_order_selects_from_candidate_set(self):
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            self.skipTest("statsmodels minimal runtime'da kurulu degil")

        y = np.array([0.01, -0.02, 0.015, 0.0, 0.01, -0.005, 0.002, 0.004, -0.003, 0.001])
        candidates = [(1, 0, 0), (0, 0, 1)]
        model = ARIMAModel(auto_order=True, candidate_orders=candidates)
        model.train(np.arange(len(y)).reshape(-1, 1), y)
        self.assertIn(model.order, candidates)
        preds = model.predict(np.zeros((2, 1)))
        self.assertEqual(preds.shape, (2,))

    def test_attention_layer_bias_is_scalar_and_broadcasts(self):
        try:
            import tensorflow as tf
            from src.models.lstm_model import AttentionLayer
        except Exception as exc:
            self.skipTest(f"tensorflow minimal runtime'da kurulu degil: {exc}")

        layer = AttentionLayer()
        x = tf.ones((2, 4, 3), dtype=tf.float32)
        y = layer(x)

        self.assertEqual(tuple(layer.b.shape), (1,))
        self.assertEqual(tuple(y.shape), (2, 3))

    def test_prophet_model_uses_available_macro_regressors(self):
        import src.models.prophet_model as prophet_module
        from src.models.prophet_model import ProphetModel

        class FakeProphet:
            def __init__(self, **kwargs):
                self.regressors = []
                self.train_df = None
                self.future_df = None

            def add_regressor(self, name):
                self.regressors.append(name)

            def fit(self, df):
                self.train_df = df.copy()

            def predict(self, df):
                self.future_df = df.copy()
                return pd.DataFrame({"yhat": np.full(len(df), 0.01)})

        original = prophet_module.Prophet
        prophet_module.Prophet = FakeProphet
        try:
            model = ProphetModel(
                use_regressors=True,
                regressor_names=["USDTRY_Return", "Missing_Macro"],
                feature_names=["USDTRY_Return", "Other"],
            )
            X = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])
            y = np.array([0.01, 0.02, 0.03])
            dates = pd.date_range("2024-01-01", periods=3)

            model.train(X, y, dates_train=pd.Series(dates))
            preds = model.predict(X, dates_test=pd.Series(dates))
        finally:
            prophet_module.Prophet = original

        self.assertEqual(model.regressors_used, ["USDTRY_Return"])
        self.assertEqual(model.regressors_missing, ["Missing_Macro"])
        self.assertEqual(preds.shape, (3,))


if __name__ == "__main__":
    unittest.main()
