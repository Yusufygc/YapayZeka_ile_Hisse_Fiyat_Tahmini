# -*- coding: utf-8 -*-

import os
import shutil
import unittest

import numpy as np

from src.models.arima_model import ARIMAModel
from src.models.linear_sequence_model import DLinearSequenceModel, NLinearSequenceModel, PatchTSTExperimentalModel


class Phase4ModelTests(unittest.TestCase):
    def test_dlinear_and_nlinear_sequence_baselines_predict_expected_shape(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(24, 8, 3))
        y = rng.normal(size=24)
        X_test = rng.normal(size=(5, 8, 3))

        for cls in (DLinearSequenceModel, NLinearSequenceModel, PatchTSTExperimentalModel):
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


if __name__ == "__main__":
    unittest.main()
