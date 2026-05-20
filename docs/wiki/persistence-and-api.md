---
title: Persistence and API
type: concept
status: active
last_updated: 2026-05-20
owner: llm
source_count: 11
---

# Persistence and API

The project uses layered persistence: run-scoped output files, CSV experiment
logs, JSON model registries, and a central SQLite database.

## Run Output Layout

`ForecastingPipeline` writes per-symbol output under:

```text
outputs/{SYMBOL}/
|-- runs/{RUN_ID}/
|   |-- models/
|   |-- experiments/
|   |-- xai/
|   `-- reports, plots, diagnostics
`-- latest/
```

After a successful run, `_sync_latest_output()` copies the run directory to
`outputs/{SYMBOL}/latest/`. It checks that both source and target stay inside the
symbol output root before deleting/replacing `latest/`.

## Experiment Tracker

`src/experiments/experiment_tracker.py` writes run-level CSV experiment logs.
These are useful for local inspection and reproducibility.

## JSON Registry

`src/model_registry/model_registry.py` manages per-run or per-symbol model
metadata in JSON form. The active registry version defaults to `v5` in pipeline
metadata.

## SQLite Registry

The main database path is:

```text
data/stock_models.db
```

`StockModelDB` owns:

- Schema creation and migration.
- Experiment logging.
- Best-model updates.
- Leaderboard queries.
- Forward forecast persistence.
- Forecast resolution against actual closes.

The 2026-05-17 Phase 4 refactor keeps `StockModelDB` as the public facade and
moves implementation into internal repositories:

- `SchemaRepository`: table creation, additive migrations, and legacy best-model refresh.
- `ExperimentRepository`: experiment insert and comparison queries.
- `BestModelRepository`: production-best upsert and leaderboard queries.
- `ForecastRepository`: idempotent forecast run and point persistence.
- `ForecastResolutionRepository`: actual-close resolution and accuracy summary refresh.

The SQLite contract remains unchanged. Forecast runs are still idempotent by
`run_key`, and forecast points are unique by `(run_id, target_date)`.

Main tables:

| Table | Description |
|---|---|
| `experiments` | Training/evaluation records and model metrics |
| `best_models` | Current best eligible model per stock |
| `forecast_runs` | Forward forecast run headers |
| `forecast_points` | Per-horizon predicted and actual prices |
| `forecast_accuracy_summary` | Resolved forecast accuracy summary |
| `analysis_refresh_jobs` | Non-blocking analysis refresh job state |

Best-model maintenance uses score-aware replacement. A final-holdout production
candidate or curated production ensemble updates `best_models` only when it
beats the current row under the selection guard:

- an `eligible` candidate can replace an ineligible current best;
- an ineligible candidate cannot replace an eligible current best;
- within the same eligibility class, the higher `composite_score` wins.

Schema refresh reconstructs `best_models` from the highest-scored production
experiment per symbol, not from the latest inserted experiment.

## Analysis Refresh Jobs

`analysis_refresh_jobs` records API-triggered refresh work. Jobs are de-duplicated
by `(symbol, reason)` while a matching job is `queued` or `running`, so repeated
analysis requests do not spawn duplicate refresh work.

Important facade methods on `StockModelDB`:

- `create_or_get_refresh_job()`
- `update_refresh_job()`
- `get_refresh_job()`
- `get_latest_refresh_job()`
- `get_schema_status()` and `get_table_counts()`

`DataRefreshService` owns the work behind the job: ensure/generate the BIST
calendar, call `DataUpdater.check_and_update()`, refresh macro features, resolve
old forecast points from CSV actuals, and generate a new five-day forecast when
a best model exists.

## Forward Forecasting

`python -m src.cli.forecast` is the CLI entrypoint. It uses `ForecastRunner` in
`src/forecasting/runner.py`.

Example commands:

```bash
python -m src.cli.forecast --stocks TUPRS,ASELS --horizon-days 5
python -m src.cli.forecast --stocks TUPRS --model "Ridge Return"
python -m src.cli.forecast --stocks TUPRS --use-macro
python -m src.cli.forecast --stocks TUPRS --resolve
```

Forward forecast behavior:

- Reads best model from SQLite unless `--model` forces one.
- Avoids baseline production models by finding a trainable replacement.
- Loads saved production model/scaler/metadata sidecars from the selected
  experiment's model path.
- Generates BIST trading-day horizon points recursively. Each step freezes
  exogenous features but updates close/return lag state before the next horizon.
- Applies BIST price tick and band rules.
- Saves run and horizon points to SQLite.
- Can resolve old forecasts with actual closes from CSV.

Forecast artifact sidecars are written next to the model file:

- `{model}.forecast_metadata.json`
- `{model}.scaler_X.pkl`
- `{model}.scaler_y.pkl`

Missing or incompatible sidecars fail the forecast path explicitly through
`ForecastArtifactError`; serving code should not silently retrain in that case.

Production ensemble leaders are now allowed only for the curated methods
`Ensemble Inverse RMSE` and `Ensemble Cash-Gated`. Their forecast runs persist
`ensemble_metadata_json`, `ensemble_direction_agreement`, `forecast_strategy`,
`artifact_mode`, and forecast warnings such as frozen exogenous features.

`ForecastRunner` is a facade over internal workflows for best-model resolution,
data preparation, production training, latest-target prediction, and roll-forward
point generation. The CLI lives under `src.cli`; root-level wrapper scripts are
not retained.

## BIST Rules

`src/forecasting/bist_rules.py` provides:

- Trading calendar handling.
- Price tick rounding.
- Price band limits.
- Trend threshold and trend label logic.

Calendar metadata comes from `data/meta/bist_calendar.csv`.

## FastAPI Service

`src/api/main.py` exposes the registry over HTTP.

Run:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Important endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | DB health and service status |
| GET | `/symbols` | Registered stock symbols |
| GET | `/best-model/{symbol}` | Best model for one stock |
| GET | `/experiments/{symbol}` | Experiment history |
| GET | `/metrics/{symbol}` | Model comparison |
| GET | `/leaderboard` | Cross-symbol leaderboard |
| POST | `/run/{symbol}` | Trigger background pipeline run |
| GET | `/run/status/{job_id}` | Poll background run status |
| GET | `/refresh/status/{job_id}` | Poll analysis refresh job status |

`/health` now includes schema status, table counts, latest refresh job metadata,
CORS mode, project root, and Python executable. CORS is local-first by default:
`http://localhost` and `http://127.0.0.1` are allowed, and extra origins must be
provided with `AI_CORE_CORS_ORIGINS`.

`GET /analysis/{symbol}` still returns HTTP 200 for `no_model` and
`no_forecast`, but its response now includes refresh status fields and a
`forecast_source` block. If the best model lacks a forecast matching its
`source_experiment_id` and latest observed date, the service queues a refresh
with reason `missing_forecast_for_best_model`. Stale forecasts queue
`stale_market_data`.

The current job tracker is in-memory. For production multi-process deployment,
the API comments note that Redis would be more appropriate.

## Related Pages

- [Architecture](architecture.md)
- [Model Catalog](model-catalog.md)
- [Validation and Backtesting](validation-and-backtesting.md)

