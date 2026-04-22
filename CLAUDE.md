# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ts_forecasting_lab` is a production-grade time series forecasting pipeline for Turkish stock market (BIST) equities. It supports 11 model types across baseline, tree-based, deep learning, and experimental sequence categories. Validation modes are single-split and walk-forward. Model lifecycle is managed via JSON + SQLite registries.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (interactive CLI — prompts for stock, validation mode, and model selection)
python main_pipeline.py

# Run tests
python -m pytest tests/
```

The pipeline is fully interactive. It no longer uses hardcoded constants — stock, validation mode, and model selection are chosen at runtime via numbered menus.

### Key runtime defaults (set in main_pipeline.py):

| Parameter | Default |
|---|---|
| `target_mode` | `log_return` |
| `feature_mode` | `stationary_features` |
| `scaling_mode` | `robust_x_standard_y_clip` |
| `signal_mode` | `professional` |
| `wf_n_splits` | 12 |
| `wf_test_size` | 21 bars |
| `wf_max_train_size` | 756 bars (sliding window) |
| `final_holdout_size` | 60 bars |
| `time_steps` | 30 |
| `test_ratio` | 0.20 |

## Architecture

The pipeline follows a **Facade + Strategy** pattern. `ForecastingPipeline` in `src/pipeline/orchestrator.py` is the top-level facade; three sub-managers handle distinct concerns:

```
ForecastingPipeline (orchestrator.py)
├── DataManager          (data_manager.py)   — load, feature-engineer, split, scale, sequence
├── ModelTrainer         (model_trainer.py)  — train selected models; walk-forward or single-split
└── EvaluationManager    (evaluation_manager.py) — metrics, registry, logging, plots
```

Supporting subsystems:
```
src/backtesting/         — signal generation, backtest engine, financial metrics, reporting
src/xai/                 — explainability layer (XAIExplainer, feature dictionary, narrative, report writer)
src/features/
├── feature_pipeline.py  — 20+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
└── macro_pipeline.py    — macro context: USD/TRY, BIST100, TCMB rate, CPI (via yfinance + FRED)
src/experiments/         — ExperimentTracker (CSV logs per run)
src/model_registry/      — ModelRegistry (registry.json)
src/database/            — StockModelDB (SQLite)
src/validation/          — walk_forward.py (sliding/expanding window CV)
src/utils/
├── data_splitter.py     — strict chronological train/test split
└── reproducibility.py   — global seeds (Python, NumPy, TF, PyTorch)
```

### Data Flow

1. CSV loaded → Turkish column names mapped to English, zero-volume rows dropped
2. `FeaturePipeline` adds 20+ technical indicators; `MacroPipeline` appends macro features (USD/TRY, BIST100, rate, CPI)
3. `DataSplitter` performs strict chronological train/test split (no shuffling, no leakage)
4. `robust_x_standard_y_clip` scaling: RobustScaler on X, StandardScaler+clipping on y; scaler fit **only on train**
5. 3D sequences built for LSTM/TFT/sequence models (`[samples, TIME_STEPS, features]`)
6. Selected models trained; predictions are log-return based and inverse-transformed to price
7. Financial metrics computed: Directional Accuracy, Sharpe approximation, Hit Rate, benchmark comparison
8. Backtest engine generates signals (`professional` or `legacy` mode), simulates trades with commission + slippage
9. XAI layer generates feature-importance narratives and HTML/text reports
10. Models persisted to `outputs/{SYMBOL}/models/` (`.pkl` / `.keras` / `.pt`)
11. Metadata logged to `registry.json`, `stock_models.db` (SQLite), and CSV experiment logs

### Model Implementations

All models inherit from `BaseModel` (`src/models/base_model.py`) which enforces `train()`, `predict()`, `save()`, `load()`.

#### Baseline / Reference Models
| Class | File | Notes |
|---|---|---|
| `NaiveLastValueModel` | `src/models/naive_model.py` | Repeats last observed value |
| `NaiveZeroReturnModel` | `src/models/naive_model.py` | Always predicts zero return |
| `NaiveDriftModel` | `src/models/naive_model.py` | Linear trend extrapolation |
| `ARIMAModel` | `src/models/arima_model.py` | Configurable order; auto_order supported |
| `ProphetModel` | `src/models/prophet_model.py` | Univariate, Close price only |

#### Tree-Based Models
| Class | File | Notes |
|---|---|---|
| `XGBoostModel` | `src/models/xgboost_model.py` | Optuna HPO |
| `RandomForestModel` | `src/models/random_forest_model.py` | Optuna HPO |
| `LightGBMReturnModel` | `src/models/gradient_boosting_model.py` | Optional; skipped gracefully if not installed |

#### Linear / Regularized Models
| Class | File | Notes |
|---|---|---|
| `RidgeReturnModel` | `src/models/linear_model.py` | Ridge regression on log returns |
| `ElasticNetReturnModel` | `src/models/linear_model.py` | ElasticNet on log returns |

#### Deep Learning Models
| Class | File | Notes |
|---|---|---|
| `LSTMModel` / `AttentionLSTMModel` | `src/models/lstm_model.py` | Bidirectional LSTM + attention, Keras |
| `TFTModel` | `src/models/tft_model.py` | Temporal Fusion Transformer, PyTorch |

#### Experimental Sequence Baselines
| Class | File | Notes |
|---|---|---|
| `DLinearSequenceModel` | `src/models/linear_sequence_model.py` | Lightweight linear over 3D sequences |
| `NLinearSequenceModel` | `src/models/linear_sequence_model.py` | Normalised linear over 3D sequences |
| `PatchTSTExperimentalModel` | `src/models/linear_sequence_model.py` | Patch-based; evaluation path prepared, not production |

All model classes are exported lazily from `src/models/__init__.py` — optional dependencies (Prophet, TF, PyTorch, LightGBM) do not break unrelated imports.

### Backtesting Subsystem (`src/backtesting/`)

| Module | Purpose |
|---|---|
| `engine.py` | `run_backtest()` — simulates trades from signals, applies commission + slippage |
| `signals.py` | `generate_professional_signals()` / `generate_long_flat_signals()` with `SignalConfig` |
| `metrics.py` | Backtest-specific financial metrics (total return, Sharpe, max drawdown, win rate) |
| `reporting.py` | Formats and saves backtest reports |

`signal_mode="professional"` uses directional accuracy and volatility gates before entry. Gate thresholds (`min_directional_accuracy`, `max_rmse_vs_benchmark`, `min_composite_score`) are configurable at pipeline level.

### XAI Subsystem (`src/xai/`)

| Module | Purpose |
|---|---|
| `explainer.py` | `XAIExplainer` — model-specific SHAP/importance explanations |
| `feature_dictionary.py` | Human-readable feature descriptions and group labels |
| `narrative.py` | Sentence generators for contribution and uncertainty |
| `report_writer.py` | Writes HTML/text XAI reports |

### Key Design Rules

- **No data leakage**: scaler and feature stats are always fit on train only; `DataSplitter` always splits chronologically.
- **Reproducibility**: `src/utils/reproducibility.py` sets global seeds (Python, NumPy, TF, PyTorch) before any model is instantiated.
- **Walk-forward validation**: `src/validation/walk_forward.py` rolls a fixed-size training window forward in time — do not break the ordering invariant.
- **Log-return target**: all models predict log returns (`target_mode="log_return"`); predictions are inverse-transformed to price before evaluation.
- **Stationary features**: `feature_mode="stationary_features"` is the default; raw price levels are not used as features directly.
- **Optional dependencies**: LightGBM, Prophet, TF, PyTorch are optional — the pipeline degrades gracefully when they are absent.

### Output Structure

```
outputs/{SYMBOL}/
├── models/          # trained model files (.pkl / .keras / .pt)
├── experiments/     # CSV experiment logs per run
└── registry.json    # model version metadata
stock_models.db      # central SQLite (project root)
```

### Tests

Four test modules exist in `tests/`:

| File | Coverage |
|---|---|
| `test_leakage_guards.py` | Data leakage prevention |
| `test_phase4_models.py` | Baseline + new model classes |
| `test_reporting_metrics.py` | Metric computation correctness |
| `test_validation_protocol.py` | Walk-forward ordering invariants |

Run with `python -m pytest tests/`. No linter is configured.

### Documentation

Extended architecture notes and a technical audit live in `docs/architecture.md` and `docs/technical_audit_report.md`. Future roadmap (in Turkish) is in `docs/YapilmasiPlanlananlar.txt` and `docs/İleri Yönlü Plan.md`.
