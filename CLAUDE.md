# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ts_forecasting_lab` is a production-grade time series forecasting pipeline for Turkish stock market (BIST) equities. It trains 5 model types (Prophet, XGBoost, RandomForest, Bidirectional LSTM with Attention, TFT), supports single-split and walk-forward validation, and manages model lifecycle via JSON + SQLite registries.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (trains all models on a configured stock)
python main_pipeline.py

# To change the target stock or validation mode, edit main_pipeline.py lines ~26–35:
#   DATA_FILE = "data/TUPRS.csv"   # TUPRS, AKSA, ASELS, EREGL, SASA, THYAO
#   VALIDATION_MODE = "walk_forward"  # or "single_split"
#   TIME_STEPS = 30
#   TEST_RATIO = 0.20
```

No test suite or linter is configured yet.

## Architecture

The pipeline follows a **Facade + Strategy** pattern. `ForecastingPipeline` in `src/pipeline/orchestrator.py` is the top-level facade; the three sub-managers handle distinct concerns:

```
ForecastingPipeline (orchestrator.py)
├── DataManager          (data_manager.py)   — load, feature-engineer, split, scale, sequence
├── ModelTrainer         (model_trainer.py)  — train all 5 models; walk-forward or single-split
└── EvaluationManager    (evaluation_manager.py) — metrics, registry, logging, plots
```

### Data Flow

1. CSV loaded → Turkish column names mapped to English, zero-volume rows dropped
2. `FeaturePipeline` adds 20+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
3. `DataSplitter` performs strict chronological train/test split (no shuffling, no leakage)
4. `MinMaxScaler` fit **only on train**, applied to both splits
5. 3D sequences built for LSTM/TFT (`[samples, TIME_STEPS, features]`)
6. All 5 models trained; XGBoost and RandomForest use Optuna for HPO
7. Predictions inverse-transformed → financial metrics computed (Directional Accuracy, Sharpe approximation, Hit Rate)
8. Models persisted to `outputs/{SYMBOL}/models/` (`.pkl` / `.keras` / `.pt`)
9. Metadata logged to `registry.json`, `stock_models.db` (SQLite), and a CSV experiment log

### Model Implementations

All models inherit from `BaseModel` (`src/models/base_model.py`) which enforces `train()`, `predict()`, `save()`, `load()`. Implementations:

| Class | File | Notes |
|---|---|---|
| `ProphetModel` | `src/models/prophet_model.py` | Univariate, uses Close price only |
| `XGBoostModel` | `src/models/xgboost_model.py` | Optuna HPO |
| `RandomForestModel` | `src/models/random_forest_model.py` | Optuna HPO |
| `AttentionLSTMModel` | `src/models/attention_lstm_model.py` | Bidirectional LSTM + attention, Keras |
| `TFTModel` | `src/models/tft_model.py` | Temporal Fusion Transformer, Keras/PyTorch |

### Key Design Rules

- **No data leakage**: scaler and feature stats are always fit on train only; `DataSplitter` always splits chronologically.
- **Reproducibility**: `src/utils/reproducibility.py` sets global seeds (Python, NumPy, TF, PyTorch) before any model is instantiated.
- **Walk-forward validation**: `src/validation/walk_forward.py` rolls a fixed-size training window forward in time — do not break the ordering invariant.

### Output Structure

```
outputs/{SYMBOL}/
├── models/          # trained model files
├── experiments/     # CSV experiment logs per run
└── registry.json    # model version metadata
stock_models.db      # central SQLite (project root)
```

### Documentation

Extended architecture notes and a technical audit live in `docs/architecture.md` and `docs/technical_audit_report.md`. Future roadmap (in Turkish) is in `docs/YapilmasiPlanlananlar.txt` and `docs/İleri Yönlü Plan.md`.
