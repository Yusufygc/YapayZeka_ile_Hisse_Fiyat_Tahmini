---
title: Persistence and API
type: concept
status: active
last_updated: 2026-11-16
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

After a successful run, `_sync_latest_output()` (-> `src/utils/output_paths.py::sync_latest_output`)
refreshes `outputs/{SYMBOL}/latest/` as a LIGHTWEIGHT view (2026-11-15): only small
report files (`.json/.csv/.png/.md/.txt/.html`, <= 5 MB) are copied, and model
artifacts (`.pkl`, `.keras`, ...) are NOT duplicated. `latest/LATEST_RUN.json`
(`{run_id, path: "runs/{RUN_ID}", created_at}`) points to the real run. Readers that
need a model/artifact from `latest/` use
`resolve_latest_artifact(symbol_root, "models/x.pkl")` (checks `latest/` first, then
the pointer's run dir; rejects `..`/absolute names). Run-level analysis must still
read `outputs/{SYMBOL}/runs/{RUN_ID}` as the source of truth. The sync step builds a
temporary `latest.__tmp__{RUN_ID}` directory first, then replaces `latest/` under a
symbol-local `.latest_sync.lock` so nearby runs do not delete the target while
another run is writing into it. Only known `latest/` reader in `src/` is
`src/xai/product_summary.py` (small xai CSVs; still copied, run-dir candidates
are tried first).

### Model compression, run manifest status, retention (2026-11-15)

- Every `joblib.dump` of a model/scaler (`src/models/*`, `src/forecasting/artifacts.py`)
  uses `compress=3` (RandomForest ~50 MB -> ~7 MB). `joblib.load` reads both
  compressed and legacy uncompressed files; Keras saves are unchanged.
- `run_manifest.json` is written at run START with `status="running"` and
  `started_at`, and rewritten at the end as `completed` (plus `finished_at`), or
  `failed`/`interrupted` with `error{type,message}` when a Python exception /
  Ctrl-C escapes `run_all`. A hard kill/crash leaves `status="running"`. A failed
  run never touches `latest/`.
- `tools/prune_outputs.py` (logic in `src/utils/output_retention.py`): per symbol keeps
  the N newest runs (`--keep`, default 5), runs referenced by SQLite
  `best_models`/`experiments.model_path` (matched by the `runs/{RUN_ID}` segment,
  Windows or POSIX separators; DB opened read-only), the `LATEST_RUN.json` run, and
  recent `running` runs (24 h). Default is dry-run; `--apply` deletes.
  `--slim-latest` converts legacy full-copy `latest/` dirs to the pointer layout,
  `--compress-existing` rewrites uncompressed `.pkl` >= 1 MB with `compress=3`
  (reload + `joblib.hash` equality check, atomic `os.replace`), `--clean-tmp`
  removes `.codex_tmp`.

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
experiment per symbol, not from the latest inserted experiment. Kaldırılmış modeller
(`REMOVED_MODELS`: LSTM, Random Forest, ElasticNet, Cash-Gated, ...) bu sıralamada
dışlanır (2026-11-16); destekli üretim satırı yoksa mevcut davranış korunur.

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

## Persistence Hardening (2026-11)

- **Önbellekli DB**: `src/database/stock_model_db.py::get_db(path)` yol başına tek, thread-safe `StockModelDB` döner; API (`main._get_db`, `AnalysisService._get_db`) bunu kullanır. Şema/migrasyon/`best_models` yeniden inşası süreçte yol başına bir kez çalışır (`StockModelDB.ensure_schema(force=False)`; DB dosya kimliği değişirse yeniden kurulur). DDL sabitleri `src/database/schema_ddl.py`.
- **Pragma**: `src/database/connection.py::open_db` her bağlantıda `busy_timeout=5000`, `foreign_keys=ON` ve WAL değilse bir kez `journal_mode=WAL` ayarlar. `StockModelDB`, `PeerStore` ve GUI `DBHelper` bunu kullanır.
- **Atomiklik**: `log_experiment` deney satırı + `best_models` güncellemesini tek transaction'da yazar (hata -> ikisi de geri alınır). `PeerStore.write_run(meta, scored)` run başlığı + peer skorlarını tek transaction'da yazar (`insert_run`/`insert_peer_scores` opsiyonel `conn=` alır).
- **forecast_points upsert**: yeniden kayıt çözümlenmiş kolonları (`actual_close`, `actual_return`, `abs_error`, `direction_correct`, `resolved_at`) artık silmez.
- **resolve_forecasts**: sembolün tüm noktaları tek sorguda çekilir (N+1 yok), zaten çözülmüş noktalar atlanır, accuracy özeti yalnızca değişen run'lar için yenilenir; dönüş değeri YENİ çözülen nokta sayısıdır.
- **forecast_runs.status**: `pending` = "tahmin üretildi, gerçekleşme bekleniyor"; hiçbir yazıcı `completed` yazmaz. GUI (`DBHelper.get_forecasts_by_symbol`) `status IN ('pending','completed')` okur.
- **Taşınabilir model yolu**: `src/database/paths.py` (`to_storable_path`, `resolve_model_path`, `migrate_stored_path`). `experiments/best_models.model_path` proje köküne göreceli (`outputs/...`, `/` ayraçlı) yazılır; okurken (`get_best_model`, `get_experiments`, workflows, analysis_service, xai product_summary) proje köküne çözülür. Başka makineden kalan mutlak yol yerelde yoksa son `outputs` segmenti üzerinden yerel outputs köküne yeniden eşlenir. `SchemaRepository.migrate_model_paths` idempotent migrasyonu mutlak `.../outputs/...` yollarını göreceliye çevirir (bu makinede var olan, proje dışı mutlak yollara dokunmaz).
- `compute_psi_30d` (analysis_service çağırır) artık ortak `src/data/ohlcv_clean.clean_ohlcv_csv` ile okur; `Tarih/Kapanış` başlıkları eşlenir. Önceden hep `missing_required_columns` dönüyor, PSI kapısı fiilen kapalıydı. Açıldıktan sonra ilk 200 sembolde dağılım: major_drift 86, moderate 66, stable 48 (eski eşikler 0.10/0.25 gerçek veride placebo null medyanının bile altındaydı -> gürültü). **Kalibrasyon (2026-11-15, karar verildi):** tahmin edici referans-kantil bin (252g referansın %10/35/65/90 kesimleri, 5 açık uçlu bin) + Laplace yumuşatma (+1), 3 feature'ın **ortalaması** (max değil); eşikler 581 sembol x ~87 placebo pencere (50.600, her 21 iş günü) null dağılımından: moderate >= 0.40 (q90), major >= 0.80 (q99). Güncel pencerede 585 CSV: stable 527 / moderate 51 / major 7 / unavailable 1. `major_drift` güven etiketini `low` yapar (yanıt reddedilmez). Ayrıntı: `src/api/services/data_quality_monitor.py` sabit yorumları.
- Test: `tests/test_persistence_fixes_2026_11.py`.

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

**2026-11-16:** Özellik listesi eski bir artefaktla uyuşmuyorsa
`_validate_artifact_features` (`src/forecasting/workflows.py`) açık bir "yeniden eğit"
hatası verir. Silinmiş modeller için (`LSTM`, `Prophet`, `ARIMA`, ... bkz.
[Model Catalog](model-catalog.md#removed-models-2026-11-16)) tahmin
`ModelNoLongerSupportedError` fırlatır (reason `model_no_longer_supported`, action
`retrain`). Hata `src/forecasting/artifacts.py` içinde tanımlıdır; refresh servisi
(`src/api/services/data_refresh_service.py`) bu istisnayı yakalar. Eski özellik listesi
artık `ArtifactFeaturesOutdatedError` (reason `artifact_features_outdated`, action `retrain`)
verir ve refresh servisi bunu genel `forecast_generation_failed` yerine aynı yapısal
"yeniden eğit" sonucuna çevirir. Kaldırılmış best model için refresh önce piyasa verisini
günceller, sonra yapısal retrain hatasını döner (veri bayatlamaz).

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
  2026-11-16: quantile-model yolu kaldırıldı; `p10/p50/p90` alanları nullable olarak kalır ve
  yalnızca model-agnostik kalibrasyondan dolar; `interval_method` üreteni ayırır.
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

Production ensemble leaders are now allowed only for `Ensemble Inverse RMSE`
(2026-11-16: `Ensemble Cash-Gated` kaldırıldı). Their forecast runs persist
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
provided with `AI_CORE_CORS_ORIGINS`. 2026-11-16: `/health` ayrıca `nightly_run`
anahtarını döner (`serving_health.nightly_run_status`; `data/nightly_status.json`
özeti, `interrupted` bayrağı dahil).

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
`scripts/register_nightly_task.ps1` registers it. 2026-11-16 sağlamlaştırması için
aşağıdaki bölüme bakın.

## Nightly Automation (2026-11-16)

**Teşhis:** `ts_forecasting_nightly` zamanlanmış görevi 2026-09-16'dan sonra kayıtlı
değildi. Ağustos-Eylül koşuları ~150-160 / 585 sembolden sonra ölmüştü (alfabetik
A1CAP..DOAS 2026-09-16'ya kadar güncellendi, DOCO..ZRGYO 2026-06-02'de kaldı). Fark
edilmedi.

**Değişiklikler:**

- **Durum dosyası:** `src/serving/run_status.py` → `data/nightly_status.json`. Kosu
  başında ve her fazda atomik (geçici dosya + `os.replace`) yazılır; kosu normal
  bitmezse `completed: false` kalır. `/health.nightly_run` bunu `interrupted` olarak gösterir.
- **Bayat-pay kapısı:** başarısız + bayat sembol oranı `--max-stale-fraction` (varsayılan
  0.20) üstünde ise skorlama atlanır, çıkış kodu 3, son iyi PeerStore koşusu korunur.
- **Zaman sınırları:** `--refresh-deadline-s` (varsayılan `DataUpdater.update_many`
  deadline 2700 s); yfinance `timeout=30`; `call_with_deadline` tekil 120 s, toplu 300 s
  (`src/data/market_providers.py`). `--status-file` durum dosyasını değiştirir.
  Zaman aşımıyla terk edilen bir sağlayıcı çağrısı (asılı yfinance thread'i) varsa
  `python -m src.serving.nightly` durum/log yazıldıktan sonra `os._exit(rc)` ile sonlanır
  (`exit_after_run`; yalnız `__main__` yolu).
- **Görev kaydı:** `scripts/register_nightly_task.ps1` `Register-ScheduledTask` kullanır;
  pilde başlar, pil modunda durmaz, uygun olunca başlar (StartWhenAvailable), 2 saat
  zaman sınırı, `MultipleInstances IgnoreNew`.
- **Log:** `scripts/nightly_serving.ps1` UTF-8 log yazar ve bitiş satırını her durumda yazar.
- **Tarih:** `src/data/updater_frames.py` değer-bazlı ayrıştırıcı kullanır; tüm yazıcılar ISO
  yazar. Diskte `dd/mm/yyyy` kalan `data/ASELS.csv`, `data/EREGL.csv` doğru okunur (bkz.
  [Data Pipeline](data-pipeline.md)).

**Bilinen açık:** Makro veri (`USDTRY`, `BIST100`, `INTEREST_RATE`) nightly işte
yenilenmez; bu bir açıktır.

**Sonraki adım:** Veri yenilenmeden DB skorları ve Kol-B serving havuzu yeniden üretilmelidir.

## Serving Havuzu Tazelik Alanı (K6, 2026-11-10)

`GET /health` additive olarak `serving_pool` döner: `available`, `as_of_date`, `age_days`, `stale`
(`age_days > 7`), `n_symbols`, `n_rows`, `created_at`. `src/api/services/serving_health.py`
`data/serving_pool.db`yi **salt-okunur** açar (dosyayı oluşturmaz); yoksa/bozuksa
`{"available": false, "reason": ...}` döner, istisna metnini sızdırmaz. Mevcut `/health`
alanları değişmedi.

## Related Pages

- [Architecture](architecture.md)
- [Model Catalog](model-catalog.md)
- [Validation and Backtesting](validation-and-backtesting.md)
- [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md)

