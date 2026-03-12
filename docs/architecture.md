# Quantitative Research Pipeline Architecture

This document describes the flow and structure of the upgraded Production-Grade Financial Time Series Forecasting Pipeline.

## Core Principles
1. **Strict Data Leakage Prevention:** Ensuring that future data never leaks into model training. Achieved through chronological splitting and fitting scalers *only* on the training data.
2. **Reproducibility:** A global seed function controls the randomness of Python, Numpy, and TensorFlow to ensure deterministic backtests.
3. **Traceability:** Automatic logging of metrics, hyperparameters, and model paths via the `ExperimentTracker` and `ModelRegistry`.
4. **Validation Diversity:** Both classical Single Split (holdout test) and Rolling Window Walk-Forward cross-validation are natively supported.

## Directory Structure
- `src/utils/`
  - `data_splitter.py`: Chronological Time Series split algorithms.
  - `reproducibility.py`: Global seed enforcement.
- `src/features/`
  - `feature_pipeline.py`: Automatically generates returns, MAs, Volatility, momentum (RSI/MACD) dynamically.
- `src/models/`
  - `base_model.py`: Universal interface for all estimators.
  - `tft_model.py`: Temporal Fusion Transformer (Keras).
  - LSTM, XGBoost, Prophet, Random Forest.
- `src/validation/`
  - `walk_forward.py`: Orchestrates multi-window model sequential training.
- `src/evaluation/`
  - `financial_metrics.py`: Computes financial domain-specific scores (Directional Accuracy, approximated Sharpe, Hit Rate).
- `src/experiments/` & `src/model_registry/`
  - `experiment_tracker.py`: Logs parameter variations and scores to a CSV.
  - `model_registry.py`: Stores deployed model states to a versioned JSON list.

## Pipeline Lifecycle (pipeline_manager.py)
1. **Initialization:** Defines modes (`single_split` vs `walk_forward`). Configures paths. Triggers `set_global_seed()`.
2. **Data Ingestion:** Downloads missing external updates and pumps data through `FeaturePipeline` to produce lagging financial indicators.
3. **Preprocessing:** TimeSeriesSplitter executes strict time constraints over the data. Scale transformers are fit on the train domain. Time sequences generated.
4. **Training (Single vs Walk-Forward):** Modeller instances fit datasets dynamically. Walk-forward triggers isolated backtesting instances per window to measure robustness over varying timeframes.
5. **Prediction & Transformations:** In holdout modes, standard array predictions map back from their scaled spaces to raw prices via `inverse_transform`.
6. **Evaluation & Registration:** Merges financial metrics (MAE, RMSE, Hit_Rate, Sharpe). Injects into CSV trackers and exports final model weights locally while appending an entry to `registry.json`.
