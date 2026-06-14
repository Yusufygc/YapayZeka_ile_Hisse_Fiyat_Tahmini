---
title: Persistence and API
type: concept
status: active
last_updated: 2026-06-14
owner: llm
source_count: 14
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
|   |-- model_results/{model_slug}/
|   |-- experiments/
|   |-- xai/
|   `-- reports, plots, diagnostics
`-- latest/
```

`model_results/{model_slug}/` is an inspection layer only. It keeps
model-scoped metrics, prediction rows, fold metrics for walk-forward runs, and
artifact manifests so a "train all models" run can be reviewed model by model
without launching separate manual runs. The forecast-serving contract still uses
the canonical model paths under `models/` and their sidecars.

After a successful run, `_sync_latest_output()` copies the run directory to
`outputs/{SYMBOL}/latest/`. This is a convenience copy only; run-level analysis
must read `outputs/{SYMBOL}/runs/{RUN_ID}` as the source of truth. The sync step
copies into a temporary `latest.__tmp__{RUN_ID}` directory first, then replaces
`latest/` under a symbol-local `.latest_sync.lock` file so nearby runs do not
delete the target while another run is copying into it.

As of the 2026-06-14 XAI audit, `xai/` also carries machine-readable run
metadata:

- `xai_manifest_{suffix}.json` and alias `xai_manifest.json`: method counts,
  fallback/approximate ratio, background scope, row/feature counts,
  top-feature stability, dictionary coverage, `run_id`, and `created_at`.
- `xai_sequence_heatmap_{suffix}.csv`: optional feature-lag attribution output
  for sequence models. Existing aggregate `xai_top_reasons_*.csv` and
  `xai_daily_reasons_*.csv` stay backward-compatible.

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

As of 2026-05-20, signal/trade quality fields are first-class SQLite metrics.
`experiments` and `best_models` both carry `rmse_vs_benchmark`, `net_return`,
`buyhold_return`, `max_drawdown`, `trade_count`, and `signal_diagnosis`.
Final-holdout, walk-forward, and single-split workflows merge the selected
backtest fields into the metrics dict before tracker/DB logging, while keeping
prediction metrics such as RMSE and directional accuracy as the primary model
selection fields.

Existing run outputs can be backfilled without retraining:

```powershell
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m src.cli.db_maintenance backfill-run-metrics --symbol ASELS
```

The command reads run-scoped `csv/backtest_report_{suffix}.csv` files, tolerates
semicolon-delimited UTF-8/BOM CSVs, updates matching `experiments` rows by
`symbol + run_id + model_name + validation_mode`, then rebuilds `best_models`
with the selection guard.

Run-level holdout diagnostics can be produced without retraining:

```powershell
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m src.cli.run_leaderboard --symbol ARDYZ --format json
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m src.cli.run_leaderboard --symbols ARDYZ,ASELS,LOGO --out outputs/research/run_leaderboard.csv
```

The command reads only `outputs/{SYMBOL}/runs/*`, never `latest/`, and reports
final-holdout completeness, walk-forward/final return gap, benchmark-clone
flags, trade sufficiency, and `leader_reliability_class` values such as
`stable`, `defensive`, `unstable`, `invalid`, and `incomplete`. Multi-symbol
mode adds history bucket, sector, prediction rank, trading rank, sector summary,
and history-effect summary CSVs. History is diagnostic only, not an exclusion
filter: `long_history` means the symbol meets the 10-year reference threshold,
`mid_history` means 5-10 years, `short_history` means less than 5 years, and
`missing_data`/`unknown` make unavailable history explicit.

`run_manifest.json` now includes `final_holdout_status`. A successful final
holdout records `status=success` and the selected model. If final-holdout
evaluation fails, the manifest records `status=failed`, `model_name`,
`error_type`, and `error` so the failure is visible after the warning scrolls
past in the console.

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
- `--model NAME` (forced): `BestModelResolver.resolve` looks up the latest
  experiment for that model whose artifact file exists on disk
  (`latest_member_experiment`) and uses its `model_path` + training config
  (target/feature/scaling/dataset_hash). If no saved artifact exists for the
  forced model, it falls back to the best model's metadata with an empty path.
  (2026-06-02 fix: previously the forced branch returned no `model_path`, so
  `ProductionTrainingWorkflow` tried to load an empty artifact path and raised
  `artifact model file not found`.)
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
- `{model}.interval_calib.json` — **opsiyonel** olasılıksal interval kalibrasyonu
  (B2 residual σ + C conformal q̂). Eksikse forecast bozulmaz, interval atlanır
  (`_OPTIONAL_SIDECARS`). Walk-forward residual'larından `_build_interval_calibration`
  ile üretilir (final holdout kullanılmaz).

Missing or incompatible sidecars (zorunlu olanlar) fail the forecast path explicitly
through `ForecastArtifactError`; serving code should not silently retrain in that case.

### Olasılıksal forward interval (B2/C)

- `forecast_points` ek kolonlar: `p10_close`, `p50_close`, `p90_close`,
  `predicted_return_p10/p50/p90`, `interval_method`
  (`quantile_model | residual_b2 | conformal | null`). p10=alt, p50=nokta, p90=üst.
  Quantile model yolu da aynı alanları doldurur; `interval_method` üreteni ayırır.
  Geriye uyumlu: eski satırlar/interval'siz tahminler `null`.
- `forecast_accuracy_summary` ek kolonlar: `interval_coverage` (%),
  `interval_avg_width`, `nominal_coverage`. Resolve sırasında
  `actual_close ∈ [p10, p90]` oranından hesaplanır (`forecast_resolution.py`).
- Kolonlar additive migration ile mevcut DB'lere eklenir
  (`SchemaRepository._ensure_forecast_point_columns` /
  `_ensure_forecast_accuracy_columns`). DDL `stock_model_db.py`.
- API `ForecastPoint` şeması p10/p50/p90 + `predicted_return_p*` + `interval_method`
  alanlarını taşır (`_build_forecast_block` map eder); desktop/UI tüketebilir.
- Karşılaştırma: `tools/interval_coverage_report.py` (B2 vs conformal coverage tablosu).
  Detay: [Validation and Backtesting](validation-and-backtesting.md).

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

XAI summaries are resolved against the current best model's run first. The API
passes `best_models.run_id` and `model_path` into `build_xai_product_summary()`,
which searches `outputs/{SYMBOL}/runs/{RUN_ID}/xai` before falling back to
`latest/xai`. This avoids stale `latest/` XAI tables from a later non-best run
masking the best model's explanations. The XAI product summary now reads
`xai_manifest.json` when available and exposes `xai.status`, `method_detail`,
`approximate_ratio`, `feature_stability_top`, `generated_at`, `run_id`,
`background_scope`, `dictionary_coverage`, and `group_summaries` without
changing legacy factor arrays.

The current job tracker is in-memory. For production multi-process deployment,
the API comments note that Redis would be more appropriate.

## Advisory Audit Log (Sprint 9 — 2026-05-26)

Every `GET /analysis/{symbol}` response is appended to
`data/advisory_history.csv` via `src/api/services/advisory_audit.py`.
Failures here never break the response (audit best-effort).

Schema (one row per response):

| Column | Type | Source |
|---|---|---|
| `timestamp_utc` | ISO-8601 string | `datetime.now(tz=UTC)` |
| `symbol` | string | `response.symbol` |
| `horizon_days` | int \| null | `forecast.horizon_days` |
| `model_name` | string \| null | `model.model_name` |
| `trend_label` | string \| null | `forecast.trend_label` |
| `p50_return` | float \| null | first point `predicted_return_p50` (fallback `predicted_return`) |
| `p10_return` | float \| null | first point `predicted_return_p10` |
| `p90_return` | float \| null | first point `predicted_return_p90` |
| `confidence_label` | enum | `confidence.label` |
| `analysis_status` | enum | `response.analysis_status` |

Backfill job (future): at `T + horizon_days` fetch realized return and
emit `realized_dir_correct` boolean for calibration drift dashboards.

## Response Cache (Sprint 9 A9.2)

`src/api/services/response_cache.py:ResponseCache` provides an in-memory
TTL cache keyed by uppercase `symbol`. Default TTL: 24h
(`AI_CORE_RESPONSE_CACHE_TTL_SECONDS` env override; `0` disables).
`AI_CORE_RESPONSE_CACHE_DISABLED=1` also disables. The cache is lazy-eviction
(no background sweeper) and thread-safe.

Cache invalidation: callers must invoke `cache.invalidate(symbol)` after a
new training run or forecast persistence (TODO: wire into refresh job
completion in Sprint 10).

## Rate Limit (Sprint 9 A9.3)

`src/api/services/rate_limit.py:RateLimiter` is a fixed-window IP rate
limiter. Default: 60 req/min/IP (`AI_CORE_RATE_LIMIT_PER_MINUTE`; `0`
disables). Trusted IPs in `AI_CORE_RATE_LIMIT_TRUSTED_IPS` (comma-separated;
default `127.0.0.1`) bypass the limit. Over-limit requests get HTTP 429
with `Retry-After: 60`. The middleware is added at app construction time
only when the limit is enabled.

## Timezone-Aware Datetimes (Sprint 9 A9.4)

All API-layer timestamps (health, audit log) use `datetime.now(tz=timezone.utc)`
with `isoformat(timespec="seconds")`. Database writes that pre-date Sprint 9
continue to use local wall-clock strings; gradual migration is in scope for
Sprint 10.

## Peer Serving Store (E2 Faz 5–8)

The E2 pooled global model serves through an **isolated** SQLite DB
`data/serving_pool.db` (`src/serving/peer_store.py`, class `PeerStore`) that
**never touches `best_models`**. A nightly batch writes it; the API reads it.

Two tables:

- `global_model_runs` — one row per training/scoring run: `run_id` (PK), `created_at`,
  `as_of_date`, `model_name`, `data_snapshot_hash`, `n_symbols`, `n_rows`,
  `horizon`, `ic_mean`, `icir`, `pct_ic_positive`, `config_json`.
- `peer_scores` — one row per (run, symbol), `UNIQUE(run_id, symbol)` (upsert):
  `peer_score` (−1..1), `peer_percentile` (0..100), `peer_label`
  (outperform/inline/underperform/unknown), `raw_pred`, `universe_size`,
  `segment_liq/vol/sector`, `segment_icir` (composite, tradability-aware),
  `confidence_label` (low/medium/high), `confidence_reasons/warnings` (JSON), and
  **trend tendency (Faz 7b)**: `trend_label` (yukarı/yatay/aşağı/belirsiz),
  `trend_prob_up`, `trend_expected_return`; **Kol-B XAI (Faz 10)**:
  `xai_top_features` (JSON: `{method, approximate, caveat, top_positive[], top_negative[]}`),
  `xai_method`, `xai_approximate`, `xai_error`, `xai_generated_at`;
  and **Kol-B Price Band (Phase 5)**: `kolb_price_p50` (expected absolute price),
  `kolb_price_low` (lower absolute price band), `kolb_price_high` (upper absolute price band),
  `kolb_horizon_days` (horizon days, default 5), and `kolb_band_level` (nominal coverage, default 0.8).

`PeerStore._migrate` runs idempotent `ALTER TABLE ADD COLUMN` on open, so
pre-existing DBs (run_id ≤ 2, no trend/xai/kolb price band columns) upgrade in place; old rows return
NULL trend/xai/kolb values -> API surfaces `None`/`xai_available=False`
(graceful, never breaks).

**Kol-B XAI (Faz 10):** `tools/e2_faz5_nightly_scoring.py` →
`score_latest_universe` (cfg `enable_xai`) computes per-symbol peer-rank XAI via
`src/serving/peer_xai.py::compute_peer_xai`. Ensemble models use ensemble-level
permutation sensitivity; LightGBM-leg SHAP can remain as diagnostic metadata.
Scoring persists `xai_method`, `xai_approximate`, `xai_error`, and
`xai_generated_at` so XAI failures are observable without breaking the peer
score row. `PeerEnrichmentService` parses these fields plus `xai_top_features`
into `PeerBlock.xai_available/xai_method/xai_approximate/xai_error/
xai_generated_at/xai_caveat/xai_top_positive/xai_top_negative` (reusing
`XaiFactorItem`).

**API surface:** `src/api/services/peer_service.py` (`PeerEnrichmentService`)
reads the latest run for a symbol and attaches an additive `peer` block to
`AnalysisResponse` (`PeerBlock` in `src/api/schemas/analysis.py`). Missing
DB/symbol → silent no-op; the existing absolute forecast/confidence blocks are
never modified. See [Analysis API Contract](analysis-api-contract.md) and
[E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md).

**Nightly automation (Faz 8):** `tools/e2_nightly_pipeline.py` orchestrates
trading-day gate → universe data refresh (`DataUpdater.check_and_update` loop) →
scoring batch (`tools/e2_faz5_nightly_scoring.py`). `scripts/nightly_serving.ps1`
is the Windows Task Scheduler target (daily 21:00, logs to `logs/nightly_*.log`);
`scripts/register_nightly_task.ps1` registers it.

## Related Pages

- [Architecture](architecture.md)
- [Model Catalog](model-catalog.md)
- [Validation and Backtesting](validation-and-backtesting.md)
- [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md)

