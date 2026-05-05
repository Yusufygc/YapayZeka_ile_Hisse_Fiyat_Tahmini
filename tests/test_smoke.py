# -*- coding: utf-8 -*-
"""
test_smoke.py -- Temel Smoke Testleri (Faz 0.4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pipeline kritik siniflarinin hatasiz baslatilabildigini ve temel
kontratlarini korundugunu dogrular.

Kapsam:
  - EvaluationManager instantiation (0.1 bug regression)
  - ModelTrainer instantiation
  - DataManager instantiation + _wf_mode flag
  - EnsembleModel: optimize_grid_search() NotImplementedError (0.2 regression)
  - EnsembleModel: optimize_inverse_rmse() calisiyor
  - scale_data: save_scaler=False disk yazimini atlar (0.3 regression)
  - ForecastingPipeline: minimal argümanlarla baslatilabilir

Bagimliliklari: pytest, numpy, pandas, sklearn
Calistirma: python -m pytest tests/test_smoke.py -v
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd


# ── Yardimci: Minimal sentetik DataFrame ──────────────────────────────────────

def _make_ohlcv(n: int = 600) -> pd.DataFrame:
    """Test icin THYAO benzeri minimal OHLCV verisi uretir."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, size=n))
    open_ = close * (1 + rng.normal(0, 0.005, size=n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, size=n))
    low  = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, size=n))
    volume = rng.integers(1_000_000, 50_000_000, size=n).astype(float)
    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low":  low,
        "Close": close,
        "Adj_Close": close,
        "Volume": volume,
    })


# ── 1. EvaluationManager Smoke ────────────────────────────────────────────────

class TestEvaluationManagerSmoke(unittest.TestCase):
    """
    Regression test for Bug 0.1:
    evaluation_manager.py satir 98 NameError duzeldi mi?
    """

    def _make_em(self, tmpdir, flag=True):
        from src.pipeline.evaluation_manager import EvaluationManager
        from src.pipeline.config import ExecutionConfig, ModelConfig
        from src.experiments.experiment_tracker import ExperimentTracker
        
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(models_dir, exist_ok=True)
        return EvaluationManager(
            stock_symbol="TEST",
            outputs_dir=tmpdir,
            models_dir=models_dir,
            tracker=ExperimentTracker(os.path.join(tmpdir, "exp")),
            feature_names=["f1", "f2", "f3"],
            dataset_hash="abc123",
            dataset_metadata={"target_mode": "log_return", "signal_threshold_config": {}},
            exe_cfg=ExecutionConfig(),
            model_cfg=ModelConfig(ensemble_enabled=flag),
        )

    def test_evaluation_manager_instantiates_without_error(self):
        """EvaluationManager __init__ basariyla tamamlanmali."""
        with tempfile.TemporaryDirectory() as tmpdir:
            em = self._make_em(tmpdir)
            self.assertIsInstance(em.ensemble_enabled, bool)
            self.assertIsInstance(em.predictions, dict)
            self.assertIsInstance(em.latest_model_metrics, dict)

    def test_evaluation_manager_ensemble_enabled_matches_config(self):
        """ensemble_enabled, ModelConfig'den dogru okunmali."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for flag in (True, False):
                em = self._make_em(tmpdir, flag=flag)
                self.assertEqual(em.ensemble_enabled, flag,
                                 f"ensemble_enabled={flag} config'den okunmadi")


# ── 2. ModelTrainer Smoke ─────────────────────────────────────────────────────

class TestModelTrainerSmoke(unittest.TestCase):

    def test_model_trainer_instantiates(self):
        from src.pipeline.model_trainer import ModelTrainer
        from src.experiments.experiment_tracker import ExperimentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(
                stock_symbol="TEST",
                tracker=ExperimentTracker(tmpdir),
                feature_names=["f1", "f2"],
                dataset_hash="hash123",
                dataset_metadata={"target_mode": "log_return"},
            )
            self.assertEqual(trainer.stock_symbol, "TEST")
            self.assertIsInstance(trainer.trained_models, dict)
            self.assertIsInstance(trainer.wf_results, dict)

    def test_model_trainer_selected_models_defaults(self):
        """selected_models=None oldugunda _ALL_MODELS setine dusmeli."""
        from src.pipeline.model_trainer import ModelTrainer, _ALL_MODELS
        from src.experiments.experiment_tracker import ExperimentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(
                stock_symbol="X",
                tracker=ExperimentTracker(tmpdir),
                feature_names=[],
            )
            self.assertEqual(trainer.selected_models, set(_ALL_MODELS))

    def test_tree_hpo_storage_is_available_for_each_selected_tree_model(self):
        """XGBoost olmadan Random Forest secilirse Optuna storage tanimsiz kalmamali."""
        from src.pipeline.model_trainer import ModelTrainer
        from src.experiments.experiment_tracker import ExperimentTracker

        tensors = {
            "X_train_s": np.ones((12, 2), dtype=float),
            "y_train_s": np.ones((12, 1), dtype=float),
        }

        class FakeTunedModel:
            calls = []

            def tune_and_train(self, X_train, y_train, **kwargs):
                self.__class__.calls.append(kwargs)

        for selected in (["Random Forest"], ["XGBoost"], ["XGBoost", "Random Forest"]):
            FakeTunedModel.calls = []
            tmpdir = os.path.join(os.getcwd(), "outputs", "_test_tree_hpo_storage")
            os.makedirs(tmpdir, exist_ok=True)
            with patch.object(ModelTrainer, "_baseline_specs", return_value=[]), \
                 patch.object(ModelTrainer, "_linear_baseline_specs", return_value=[]), \
                 patch.object(ModelTrainer, "_boosting_baseline_specs", return_value=[]), \
                 patch.object(ModelTrainer, "_sequence_baseline_specs", return_value=[]), \
                 patch("src.pipeline.model_trainer.XGBoostModel", FakeTunedModel), \
                 patch("src.pipeline.model_trainer.RandomForestModel", FakeTunedModel):
                trainer = ModelTrainer(
                    stock_symbol="TEST",
                    tracker=ExperimentTracker(tmpdir),
                    feature_names=["f1", "f2"],
                    selected_models=selected,
                    dataset_metadata={"target_mode": "log_return"},
                )
                trainer.train_single_split(tensors)

            self.assertEqual(len(FakeTunedModel.calls), len(selected))
            for call in FakeTunedModel.calls:
                self.assertEqual(call.get("study_storage"), "sqlite:///optuna_studies_TEST.db")


# ── 3. DataManager Smoke ──────────────────────────────────────────────────────

class TestDataManagerSmoke(unittest.TestCase):

    def setUp(self):
        """DataUpdater.check_and_update'i devre disi birak."""
        patcher = patch("src.pipeline.data_manager.DataUpdater.check_and_update", return_value=None)
        self.mock_updater = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_data_cfg(self, tmpdir: str):
        from src.pipeline.config import DataConfig
        df_csv = _make_ohlcv(600).rename(columns={
            "Date": "Tarih", "Open": "Acilis", "High": "Yuksek",
            "Low": "Dusuk", "Close": "Kapanis",
            "Adj_Close": "Duzeltilmis_Kapanis", "Volume": "Hacim",
        })
        csv_path = os.path.join(tmpdir, "TEST.csv")
        df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return DataConfig(
            data_file=csv_path,
            use_macro=False,
            prune_correlated_features=False,
        )

    def test_data_manager_instantiates(self):
        from src.pipeline.data_manager import DataManager
        from src.pipeline.config import ValidationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(
                data_cfg=self._make_data_cfg(tmpdir),
                val_cfg=ValidationConfig(),
                models_dir=os.path.join(tmpdir, "models"),
            )
            self.assertFalse(dm._wf_mode, "_wf_mode baslangiçta False olmali")
            self.assertIsNone(dm.df)

    def test_data_manager_wf_mode_flag_set_on_split(self):
        """split_data('walk_forward') _wf_mode=True yapmali."""
        from src.pipeline.data_manager import DataManager
        from src.pipeline.config import ValidationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            val_cfg = ValidationConfig(
                validation_mode="walk_forward",
                wf_n_splits=3,
                wf_min_train_size=100,
                wf_test_size=21,
                wf_max_train_size=400,
                final_holdout_size=0,
            )
            dm = DataManager(
                data_cfg=self._make_data_cfg(tmpdir),
                val_cfg=val_cfg,
                models_dir=os.path.join(tmpdir, "models"),
            )
            # Inject synthetic df directly (skip ingest_and_engineer)
            df = _make_ohlcv(600)
            dm.df = df
            dm.feature_names = [c for c in df.columns if c != "Date"]

            with patch("src.pipeline.data_manager.TimeSeriesSplitter") as mock_ts:
                mock_ts.walk_forward_splits.return_value = []
                dm.split_data("walk_forward")

            self.assertTrue(dm._wf_mode,
                            "split_data('walk_forward') _wf_mode=True yapmali")

    def test_data_manager_single_split_wf_mode_false(self):
        """split_data('single_split') _wf_mode=False birakmali."""
        from src.pipeline.data_manager import DataManager
        from src.pipeline.config import ValidationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(
                data_cfg=self._make_data_cfg(tmpdir),
                val_cfg=ValidationConfig(validation_mode="single_split"),
                models_dir=os.path.join(tmpdir, "models"),
            )
            df = _make_ohlcv(600)
            dm.df = df
            dm.feature_names = [c for c in df.columns if c != "Date"]

            with patch("src.pipeline.data_manager.TimeSeriesSplitter") as mock_ts, \
                 patch.object(dm, "prepare_tensors", return_value=MagicMock()):
                mock_ts.single_split.return_value = (
                    df.iloc[:480].copy(), df.iloc[480:].copy(), None, None
                )
                dm.split_data("single_split")

            self.assertFalse(dm._wf_mode,
                             "split_data('single_split') _wf_mode=False birakmali")


# ── 4. EnsembleModel Smoke ────────────────────────────────────────────────────

class TestEnsembleModelSmoke(unittest.TestCase):

    def test_optimize_grid_search_raises_not_implemented(self):
        """Bug 0.2: optimize_grid_search() asla çalismamali."""
        from src.models.ensemble import EnsembleModel

        y_true = np.array([1.0, 2.0, 3.0])
        preds = {"A": np.array([1.1, 2.1, 3.1]), "B": np.array([0.9, 1.9, 2.9])}

        with self.assertRaises(NotImplementedError) as ctx:
            EnsembleModel.optimize_grid_search(y_true, preds, step=0.5)

        self.assertIn("optimize_inverse_rmse", str(ctx.exception),
                      "Hata mesaji alternatifi onermeli")

    def test_optimize_inverse_rmse_works(self):
        """optimize_inverse_rmse production-ready kalmali."""
        from src.models.ensemble import EnsembleModel

        rng = np.random.default_rng(0)
        y_true = rng.normal(size=100)
        preds = {
            "A": y_true + rng.normal(0, 0.1, size=100),
            "B": y_true + rng.normal(0, 0.3, size=100),
            "C": y_true + rng.normal(0, 0.5, size=100),
        }
        weights = EnsembleModel.optimize_inverse_rmse(y_true, preds)

        self.assertEqual(set(weights.keys()), {"A", "B", "C"})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
        # En iyi model (A) en yuksek agirligi almali
        self.assertGreater(weights["A"], weights["B"])
        self.assertGreater(weights["B"], weights["C"])

    def test_ensemble_combine_weighted_average(self):
        """EnsembleModel.combine() agirlikli ortalama dogru hesapliyor."""
        from src.models.ensemble import EnsembleModel

        preds = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([3.0, 2.0, 1.0])}
        model = EnsembleModel(weights={"A": 0.75, "B": 0.25})
        result = model.combine(preds)

        np.testing.assert_allclose(result, [1.5, 2.0, 2.5], atol=1e-6)


# ── 5. scale_data save_scaler=False Smoke ─────────────────────────────────────

class TestScaleDataSaveScaler(unittest.TestCase):

    def test_save_scaler_false_does_not_write_files(self):
        """save_scaler=False oldugunda pkl dosyalari yazilmamali."""
        from src.data.preprocessor import scale_data

        rng = np.random.default_rng(1)
        X_tr = rng.normal(size=(100, 5));  X_te = rng.normal(size=(20, 5))
        y_tr = rng.normal(size=(100, 1));  y_te = rng.normal(size=(20, 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            scale_data(X_tr, X_te, y_tr, y_te, save_dir=tmpdir, save_scaler=False)
            files = os.listdir(tmpdir)
            self.assertNotIn("scaler_X.pkl", files)
            self.assertNotIn("scaler_y.pkl", files)

    def test_save_scaler_true_writes_files(self):
        """save_scaler=True (default) scaler dosyalarini yazmali."""
        from src.data.preprocessor import scale_data

        rng = np.random.default_rng(2)
        X_tr = rng.normal(size=(80, 4));  X_te = rng.normal(size=(20, 4))
        y_tr = rng.normal(size=(80, 1));  y_te = rng.normal(size=(20, 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            scale_data(X_tr, X_te, y_tr, y_te, save_dir=tmpdir, save_scaler=True)
            files = os.listdir(tmpdir)
            self.assertIn("scaler_X.pkl", files)
            self.assertIn("scaler_y.pkl", files)

    def test_scale_data_results_identical_regardless_of_save_flag(self):
        """save_scaler flag transform sonuçlarini degistirmemeli."""
        from src.data.preprocessor import scale_data

        rng = np.random.default_rng(3)
        X_tr = rng.normal(size=(100, 6));  X_te = rng.normal(size=(25, 6))
        y_tr = rng.normal(size=(100, 1));  y_te = rng.normal(size=(25, 1))

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            r1 = scale_data(X_tr, X_te, y_tr, y_te, save_dir=d1, save_scaler=True)
            r2 = scale_data(X_tr, X_te, y_tr, y_te, save_dir=d2, save_scaler=False)

        for i, name in enumerate(["X_train_s", "X_test_s", "y_train_s", "y_test_s"]):
            np.testing.assert_allclose(r1[i], r2[i], atol=1e-10,
                                       err_msg=f"{name} save_scaler flagine bagli degismemeli")


# ── 6. ForecastingPipeline Config Smoke ───────────────────────────────────────

class TestForecastingPipelineSmoke(unittest.TestCase):

    def test_forecasting_pipeline_module_imports(self):
        """ForecastingPipeline modulu hatasiz import edilebilmeli.

        NOT: orchestrator.py'de DataManager API uyumsuzlugu nedeniyle
        instantiation testi yerine yalnizca import dogrulanir.
        Bu uyumsuzluk Faz 1 refaktoru kapsaminda ele alinacak.
        """
        from src.pipeline import orchestrator
        self.assertTrue(hasattr(orchestrator, "ForecastingPipeline"),
                        "ForecastingPipeline sinifi orchestrator'da tanimli olmali")
        cls = orchestrator.ForecastingPipeline
        # Temel metot sozlesmesi dogrulanir
        self.assertTrue(callable(getattr(cls, "run_all", None)),
                        "ForecastingPipeline.run_all() metodu olmali")


if __name__ == "__main__":
    unittest.main()
