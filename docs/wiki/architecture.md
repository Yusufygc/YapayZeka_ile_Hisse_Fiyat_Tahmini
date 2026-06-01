---
title: Architecture
type: concept
status: active
last_updated: 2026-06-01
owner: llm
source_count: 14
---

# Architecture

`ts_forecasting_lab` is a production-oriented time series forecasting lab for BIST
equities. It combines data ingestion, feature engineering, model training,
walk-forward validation, final-holdout selection, financial backtesting,
explainability, model registries, and forward forecast persistence.

## Design Pattern

The core application follows a Facade + Strategy style:

```text
ForecastingPipeline                         src/pipeline/orchestrator.py
|-- DataManager                             src/pipeline/data_manager.py
|   |-- DataLoader / DataUpdater            src/data/
|   |-- Preprocessor                        src/data/preprocessor.py
|   |-- FeaturePipeline                     src/features/feature_pipeline.py
|   |-- MacroPipeline                       src/features/macro_pipeline.py
|   |-- FeatureCache                        src/features/feature_cache.py
|   `-- TimeSeriesSplitter                  src/utils/data_splitter.py
|-- ModelTrainer                            src/pipeline/model_trainer.py
|   |-- Model implementations               src/models/
|   `-- WalkForwardValidator                src/validation/walk_forward.py
`-- EvaluationManager                       src/pipeline/evaluation_manager.py
    |-- PredictionService                   src/pipeline/evaluation_services.py
    |-- BacktestService                     src/pipeline/evaluation_services.py
    |-- SignalCalibrationService            src/pipeline/evaluation_services.py
    `-- MetricsReportingService             src/pipeline/evaluation_services.py
```

## Pipeline Service Decomposition

The 2026-05-17 Phase 3 refactor keeps the main pipeline classes as public
facades while moving stage orchestration into owner-backed workflow/service
classes:

```text
DataManager
|-- DataIngestionService
|-- TensorPreparationService
|-- ValidationSplitService
`-- DataQualityReportingService

ModelTrainer
|-- SingleSplitTrainingWorkflow
|-- WalkForwardTrainingWorkflow
`-- FinalHoldoutTrainingWorkflow

EvaluationManager
|-- SingleSplitEvaluationWorkflow
|-- WalkForwardEvaluationWorkflow
`-- FinalHoldoutEvaluationWorkflow
```

These classes are internal implementation details, not new public APIs. The
facades still own state and preserve existing method names such as
`ingest_and_engineer`, `split_data`, `prepare_tensors`, `train_*`, and
`evaluate_*`. This keeps compatibility while reducing the complexity of the
three largest pipeline files.

### Evaluation Services: Owner-Forward → Explicit DI (E1 epic, in progress)

The original evaluation services (`PredictionService`, `BacktestService`,
`SignalCalibrationService`, `MetricsReportingService`) were *owner-backed*:
they inherited `_OwnerBackedService`, whose `__getattr__`/`__setattr__` forwarded
every attribute access to the owning `EvaluationManager`. The
[E1 Owner-Forward Removal Epic](e1-owner-forward-epic.md) replaces that "magic"
with explicit dependency injection: each service takes `(ctx, state)` in its
constructor, where `EvaluationContext` is the read-only config/identity bag and
`EvaluationState` is the mutable runtime-output bag (both in
`evaluation_services.py`). The mixin bodies now read/write `self.ctx.X` /
`self.state.X` instead of forwarding to the owner.

Migration status (2026-06-01): `PredictionService` (Faz 3.1) and
`BacktestService` (Faz 3.2) are converted to DI and no longer inherit
`_OwnerBackedService`. `SignalCalibrationService` and `MetricsReportingService`
are still owner-backed pending Faz 3.3/3.4. `EvaluationManager` exposes
`manager.X ⇄ manager.context.X` / `manager.state.X` property forwards so the
remaining owner-backed services and all workflows keep reading the same
context/state. Behavior is unchanged and locked by characterization golden tests
(`tests/test_owner_forward_contract.py`).

## Operational Hardening Phase

The 2026-05-20 operational hardening work keeps the public facades but changes
the production-serving path in three important ways:

- `ForecastingPipeline.run_all()` can automatically create a fresh five-day
  forecast after training, controlled by
  `ExecutionConfig.auto_generate_forecast_after_training`.
- Forward forecasts load saved model/scaler/metadata sidecars instead of
  silently retraining at serving time. The forecast artifact package is written
  next to the model file during single-split and final-holdout evaluation.
- `GET /analysis/{symbol}` can queue a SQLite-backed refresh job when the best
  model has no matching forecast or the forecast is stale. The job updates data,
  refreshes macro features, resolves old forecast points, and generates a new
  forecast in a background worker when serving from the project default DB.

Operational support modules added in this phase:

| Module | Responsibility |
|---|---|
| `src/api/runtime_config.py` | Local-first CORS settings from `AI_CORE_CORS_ORIGINS` |
| `src/api/observability.py` | JSON-line rotating log events under `logs/ai_core.log` |
| `src/api/services/data_refresh_service.py` | Analysis refresh job orchestration |
| `src/cli/db_maintenance.py` | SQLite summary and backup-reset commands |
| `src/forecasting/artifacts.py` | Forecast model/scaler/metadata sidecar persistence |
| `src/forecasting/bist_calendar.py` | Deterministic BIST calendar generation and merge with manual overrides |

## Modular Extraction Phase

The 2026-05-16 modular refactor keeps the public facades stable while moving
low-risk responsibilities into smaller modules:

| Facade / Caller | Extracted module | Responsibility |
|---|---|---|
| `src/backtesting/signals.py` | `src/backtesting/signal_validation.py` | Signal config validation helpers |
| `src/backtesting/engine.py` | `src/backtesting/execution.py` | Execution arrays, costs, and executable order columns |
| `src/backtesting/engine.py` | `src/backtesting/equity.py` | Equity-curve dataframe construction and signal column attachment |
| `src/backtesting/engine.py` | `src/backtesting/trades.py` | Long/flat trade-log extraction |
| `src/backtesting/signals.py` | `src/backtesting/signal_math.py` | Expected-return, recommendation, volatility, regime, and volatility-threshold math |
| `src/features/feature_pipeline.py` | `src/features/correlation_pruning.py` | Correlation graph pruning strategy |
| `src/features/macro_pipeline.py` | `src/features/macro_transforms.py` | Pure macro date filtering, lag, and merge transforms |
| `src/features/macro_pipeline.py` | `src/features/macro_feature_engineering.py` | Monthly rate/CPI and daily macro feature engineering |
| `src/pipeline/signal_calibrator.py` | `src/pipeline/signal_calibration/grid.py` | Calibration grid generation, deterministic sampling, and coverage expansion |
| `src/pipeline/signal_calibrator.py` | `src/pipeline/signal_calibration/selection.py` | Calibration trial summaries, ranking, and confirmed-row selection |
| `src/pipeline/model_trainer.py` | `src/pipeline/model_factory.py` | Model constants, specs, and stage-aware factory helpers |

The old import paths and private compatibility methods remain during the
transition period. Large orchestration methods such as `DataManager`,
`EvaluationManager`, `StockModelDB`, `ForecastRunner`, and XAI reporting are
still planned decomposition targets; they require broader characterization
tests because they own persistence, leakage, or user-facing output contracts.

## Main Runtime Flow

1. `python -m src.cli.interactive` asks for stock, validation mode, and model preset.
2. `PipelineConfig` groups all runtime settings into `DataConfig`, `ValidationConfig`, `ModelConfig`, and `ExecutionConfig`.
3. `ForecastingPipeline.run_all()` sets output directories, seeds randomness, loads data, engineers features, splits data, trains models, evaluates models, writes reports, and syncs `latest/`.
4. In `single_split`, selected models are trained once and evaluated on the chronological test split.
5. In `walk_forward`, each fold is trained independently, the best model is selected from validation behavior, and a final holdout can be evaluated.
6. Results are written to run-scoped output directories, CSV experiment logs, JSON model registries, and SQLite.

## Technology Stack

| Area | Technologies |
|---|---|
| Data processing | pandas, numpy |
| Technical indicators | ta |
| Classical ML | scikit-learn, xgboost, optional lightgbm |
| Statistical models | statsmodels, optional Prophet |
| Deep learning | TensorFlow/Keras for LSTM |
| HPO | Optuna with SQLite warm-start behavior |
| Explainability | SHAP where supported, project-level XAI reports |
| Backtesting | Custom engine in `src/backtesting/` |
| Persistence | JSON registry, CSV logs, SQLite |
| API | FastAPI, Pydantic |
| Market calendars | Deterministic local BIST calendar with optional `pandas-market-calendars` dependency |
| Testing | pytest/unittest style tests under `tests/` |
| Formatting/static config | black, isort, mypy configured in `pyproject.toml` |

## Configuration Objects

`src/pipeline/config.py` is the central configuration layer.

| Class | Responsibility |
|---|---|
| `DataConfig` | Data file, target mode, feature mode, scaling, macro options, training window, universe, data update options |
| `ValidationConfig` | `single_split` or `walk_forward`, fold sizes, sliding/expanding windows, embargo, final holdout |
| `ModelConfig` | Selected models, registry version, ensemble flag, model-specific settings |
| `ExecutionConfig` | Backtest settings, costs, signal mode, calibration, diagnostics, report writing |
| `PipelineConfig` | Root container passed into `ForecastingPipeline` |

## Database Schema

The main SQLite database path used by the pipeline is `data/stock_models.db`.

| Table | Purpose |
|---|---|
| `experiments` | Every model run with metrics, feature metadata, dataset hash, model path, candidate flag, and run id |
| `best_models` | Current best production candidate per symbol |
| `forecast_runs` | Forward forecast run metadata and source experiment link |
| `forecast_points` | Horizon-level bounded forecast points and eventual actuals |
| `forecast_accuracy_summary` | Resolved forecast accuracy summary per forecast run |
| `analysis_refresh_jobs` | Queued/running/completed/failed API refresh jobs |

`StockModelDB` maintains schema migrations by adding missing columns when needed.

## Output Structure

Run outputs are scoped by stock symbol and run id:

```text
outputs/{SYMBOL}/
|-- runs/{RUN_ID}/
|   |-- models/
|   |-- experiments/
|   |-- xai/
|   `-- reports and plots
|-- latest/                 copied from the most recent successful run
`-- forecast_models/         scaler/model workspace for forward forecasts

outputs/batch_summaries/     batch CSV/JSON summaries
data/optuna/                 Optuna warm-start SQLite files
tools/reports/               local report generation helpers
```

The run id includes timestamp, symbol, validation mode, and a compact selected-model slug.

## Important Invariants

- Time order must never be shuffled for validation.
- Scalers must be fit only on the training slice currently being evaluated.
- Walk-forward fold data must not leak into final holdout selection.
- Final holdout must not be used for signal threshold optimization.
- Benchmarks are used for comparison and selection pressure; they are not production candidates.
- `outputs/{SYMBOL}/latest/` sync must stay inside the symbol output root.

## Related Pages

- [Data Pipeline](data-pipeline.md)
- [Model Catalog](model-catalog.md)
- [Validation and Backtesting](validation-and-backtesting.md)
- [Persistence and API](persistence-and-api.md)
- [Testing and Quality](testing-and-quality.md)
- [E1 Owner-Forward Removal Epic](e1-owner-forward-epic.md)

