# -*- coding: utf-8 -*-
"""
orchestrator.py - Ana ForecastingPipeline sinifi
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

from src.database.stock_model_db import StockModelDB
from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry
from src.pipeline.data_manager import DataManager
from src.pipeline.evaluation_manager import EvaluationManager
from src.pipeline.model_trainer import ModelTrainer
from src.utils.reproducibility import set_global_seed


class ForecastingPipeline:
    def __init__(
        self,
        data_file: str,
        test_ratio: float = 0.20,
        time_steps: int = 30,
        validation_mode: str = "single_split",
        selected_models: list = None,
        target_mode: str = "log_return",
        feature_mode: str = "stationary_features",
        scaling_mode: str = "robust_x_standard_y_clip",
        backtest_enabled: bool = True,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        initial_capital: float = 100000.0,
    ):
        self.data_file = data_file
        self.validation_mode = validation_mode
        self.selected_models = selected_models
        self.target_mode = target_mode
        self.feature_mode = feature_mode
        self.scaling_mode = scaling_mode
        self.backtest_enabled = backtest_enabled
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.initial_capital = initial_capital

        set_global_seed(42)

        self.stock_symbol = os.path.splitext(os.path.basename(self.data_file))[0]
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.outputs_dir = os.path.join(self.project_root, "outputs", self.stock_symbol)
        self.models_dir = os.path.join(self.outputs_dir, "models")
        self.experiment_dir = os.path.join(self.outputs_dir, "experiments")
        self.registry_dir = self.models_dir

        self.tracker = ExperimentTracker(self.experiment_dir)
        self.registry = ModelRegistry(self.registry_dir)

        db_path = os.path.join(self.project_root, "stock_models.db")
        self.stock_db = StockModelDB(db_path)

        self.data_manager = DataManager(
            data_file,
            test_ratio,
            time_steps,
            self.models_dir,
            use_macro=True,
            macro_cache_dir=os.path.join(self.project_root, "data", "macro"),
            target_mode=self.target_mode,
            feature_mode=self.feature_mode,
            scaling_mode=self.scaling_mode,
        )

        self.model_trainer = None
        self.evaluation_manager = None
        self.registry_version = "v5"
        self.run_dataset_metadata = {}
        self.run_dataset_hash = "N/A"

    def setup_environment(self) -> None:
        for directory in [self.models_dir, self.outputs_dir, self.experiment_dir, self.registry_dir]:
            os.makedirs(directory, exist_ok=True)

        print(f"\n  [INFO] Pipeline Modu: {self.validation_mode}")
        print(f"  [INFO] Hisse Sembolu: {self.stock_symbol}")
        print(f"  [INFO] Target Mode : {self.target_mode}")
        print(f"  [INFO] Feature Mode: {self.feature_mode}")
        print(f"  [INFO] Scaling Mode: {self.scaling_mode}")
        print(f"  [INFO] Backtest    : {'acik' if self.backtest_enabled else 'kapali'}")

    def run_all(self) -> None:
        self.setup_environment()

        self.data_manager.ingest_and_engineer()
        self.data_manager.split_data(self.validation_mode)
        self.run_dataset_metadata, self.run_dataset_hash = self.data_manager.build_run_metadata(
            self.validation_mode
        )

        self.model_trainer = ModelTrainer(
            self.stock_symbol,
            self.tracker,
            self.registry,
            self.data_manager.feature_names,
            self.selected_models,
            dataset_hash=self.run_dataset_hash,
            dataset_metadata=self.run_dataset_metadata,
            registry_version=self.registry_version,
        )
        self.evaluation_manager = EvaluationManager(
            self.stock_symbol,
            self.outputs_dir,
            self.models_dir,
            self.tracker,
            self.registry,
            self.data_manager.feature_names,
            self.run_dataset_hash,
            self.run_dataset_metadata,
            selected_models=self.selected_models,
            registry_version=self.registry_version,
            stock_db=self.stock_db,
            backtest_enabled=self.backtest_enabled,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            initial_capital=self.initial_capital,
        )

        print("\n" + "=" * 60)
        print("  ADIM 4 | Model Egitimi ve Tracking (Orchestrator)")
        print("=" * 60)

        if self.validation_mode == "single_split":
            self.model_trainer.train_single_split(self.data_manager.tensors)
            self.evaluation_manager.generate_predictions(self.model_trainer.trained_models, self.data_manager.tensors)
            self.evaluation_manager.evaluate_single_split(self.model_trainer.trained_models)
        elif self.validation_mode == "walk_forward":
            self.model_trainer.train_walk_forward(self.data_manager.wf_splits, self.data_manager)
            self.evaluation_manager.evaluate_walk_forward(
                self.model_trainer.wf_results,
                self.model_trainer.wf_predictions,
                self.model_trainer.wf_y_true,
                self.model_trainer.wf_backtest_inputs,
            )

        print("\n  [OK] Pipeline completed successfully!")
