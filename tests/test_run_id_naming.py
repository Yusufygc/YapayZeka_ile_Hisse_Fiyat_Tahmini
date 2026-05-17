# -*- coding: utf-8 -*-

import unittest

from src.pipeline.orchestrator import ForecastingPipeline


class RunIdNamingTests(unittest.TestCase):
    def test_single_model_slug_is_readable(self):
        slug = ForecastingPipeline._model_slug_for_run_id(["LSTM"])

        self.assertEqual(slug, "model-LSTM")

    def test_multiple_model_slug_keeps_model_names(self):
        slug = ForecastingPipeline._model_slug_for_run_id(
            ["XGBoost", "Random Forest", "LightGBM Return"]
        )

        self.assertEqual(slug, "models-XGBoost-RandomForest-LightGBMReturn")

    def test_long_model_slug_is_capped_but_identifiable(self):
        slug = ForecastingPipeline._model_slug_for_run_id(
            ["Prophet", "XGBoost", "Random Forest", "LightGBM Return", "LSTM", "ARIMA"]
        )

        self.assertTrue(slug.startswith("models-Prophet-XGBoost-RandomForest-plus3-"))
        self.assertLessEqual(len(slug), 60)


if __name__ == "__main__":
    unittest.main()
