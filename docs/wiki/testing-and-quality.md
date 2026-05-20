---
title: Testing and Quality
type: concept
status: active
last_updated: 2026-05-20
owner: llm
source_count: 6
---

# Testing and Quality

The repository has a meaningful test suite under `tests/`. The earlier skeleton
claim that no test suite exists is stale and should not be repeated.

## Test Command

Run the full suite:

```bash
python -m pytest tests
```

Run targeted suites:

```bash
python -m pytest tests/test_smoke.py -v
python -m pytest tests/test_validation_protocol.py -v
python -m pytest tests/test_forecasting.py -v
```

## Test Areas

| Test file | Coverage intent |
|---|---|
| `test_smoke.py` | Basic construction and smoke coverage for pipeline components |
| `test_validation_protocol.py` | Validation and leakage protocol behavior |
| `test_leakage_guards.py` | Explicit leakage guard checks |
| `test_model_scope_production.py` | Candidate vs benchmark production selection rules |
| `test_evaluation_services.py` | Service composition in evaluation manager |
| `test_data_services.py` | DataManager service composition, train-only scaler reporting, final holdout split exclusion |
| `test_training_workflows.py` | ModelTrainer workflow composition, delegation, and selected-model skip policy |
| `test_forecasting.py` | BIST forecast rules, calendar behavior, forecast persistence |
| `test_reporting_metrics.py` | Reported metrics behavior |
| `test_phase*_*.py` | Acceptance/regression tests for staged refactors |
| `test_macro_cache_schema.py` | Macro cache schema behavior |
| `test_run_id_naming.py` | Run id naming contracts |
| `test_xai_routing.py` | XAI output routing for validation modes |
| `test_stock_model_db_repositories.py` | StockModelDB repository composition, schema idempotency, forecast idempotency/resolution |
| `test_forecast_workflows.py` | ForecastRunner workflow composition and forecast helper characterization |
| `test_xai_strategies.py` | SHAP/LIME strategy behavior and fallback coverage |
| `test_analysis_endpoint.py` | Analysis API status, refresh-job, forecast-source, and CORS contract coverage |
| `test_operational_hardening.py` | Data updater failures, analysis refresh failures, and DB backup-reset |

## Static Tooling

`pyproject.toml` configures:

- `black` line length 100
- `isort` profile `black`
- `mypy` Python 3.10 settings and missing-import ignores for optional ML libraries

These tools may need to be installed separately depending on the environment.

Common commands:

```bash
python -m black .
python -m isort .
python -m mypy src
```

## Code Complexity Review

The 2026-05-16 review uses the `dl_env` Python runtime for static checks:

```powershell
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m vulture src tests --min-confidence 80
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m radon cc src -s -a
```

Conservative cleanup decisions from that review:

- Backtest CSV/Markdown/order reports must not depend on successful plot rendering.
- Matplotlib users should prefer a headless-safe backend because `dl_env` can fail on Tk.
- The old standalone Monte Carlo, Kelly sizing, and permutation-test research helpers were removed from the active codebase after import/reference checks.
- `src/pipeline/report_writer.py` was a tombstoned deprecated file with no active imports.

## Modular Refactor Gate

The first modular extraction phase is intentionally compatibility-first:

- Public facades and import paths remain stable while extracted modules take
  single responsibilities.
- New or changed helper modules should stay at radon `C` or better.
- Refactors that touch leakage, persistence, forecast output, or XAI routing
  must keep characterization tests ahead of behavior changes.

Current targeted gate for this phase:

```powershell
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_smoke.py tests\test_evaluation_services.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_leakage_guards.py tests\test_phase7_acceptance.py tests\test_phase8_acceptance.py tests\test_reporting_metrics.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_forecasting.py tests\test_model_scope_production.py tests\test_macro_cache_schema.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m radon cc src -s -a
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m vulture src tests --min-confidence 80
```

## Pipeline Decomposition Gate

Phase 3 adds service-boundary tests around the public pipeline facades:

```powershell
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_smoke.py tests\test_evaluation_services.py tests\test_data_services.py tests\test_training_workflows.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_leakage_guards.py tests\test_phase7_acceptance.py tests\test_phase8_acceptance.py tests\test_reporting_metrics.py tests\test_phase6_backtest_standard.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_forecasting.py tests\test_model_scope_production.py tests\test_macro_cache_schema.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m radon cc src/pipeline/evaluation_manager.py src/pipeline/data_manager.py src/pipeline/model_trainer.py src/pipeline/evaluation_workflows.py src/pipeline/data_services.py src/pipeline/training_workflows.py -s -a
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m vulture src tests --min-confidence 80
```

The intended standard for newly extracted workflow/service code is radon `C` or
better, with facade methods staying at simple delegation level.

## Persistence, Forecasting, and XAI Gate

Phase 4 adds repository/workflow/strategy boundaries for SQLite persistence,
forward forecasts, and model-family XAI:

```powershell
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_forecasting.py tests\test_model_scope_production.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_xai_routing.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_smoke.py tests\test_evaluation_services.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_stock_model_db_repositories.py tests\test_forecast_workflows.py tests\test_xai_strategies.py -q
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m radon cc src/database src/forecasting src/xai -s -a
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m vulture src tests --min-confidence 80
```

## Operational Hardening Gate

The 2026-05-20 serving changes add targeted regression coverage for refresh job
deduplication, matching forecasts to the current best experiment, stale/missing
forecast refresh reloads, failed refresh reason propagation, local-only CORS
defaults, recursive forecast state updates, artifact-sidecar requirements,
production ensemble eligibility, and `AttentionLSTM v2` attention weights.

Focused command:

```powershell
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe' -m pytest tests\test_analysis_endpoint.py tests\test_forecasting.py tests\test_forecast_workflows.py tests\test_model_scope_production.py tests\test_operational_hardening.py -q
```

## Graphify Knowledge Graph

Graphify is installed in the `dl_env` conda environment via the PyPI package
`graphifyy`; the CLI binary is:

```powershell
C:\Users\ysfygc\anaconda3\envs\dl_env\Scripts\graphify.exe
```

Use the AST-only refresh command after code changes when no LLM API key is
available:

```powershell
& 'C:\Users\ysfygc\anaconda3\envs\dl_env\Scripts\graphify.exe' update .
```

This writes `graphify-out/graph.json`, `graphify-out/graph.html`, and
`graphify-out/GRAPH_REPORT.md`. Full semantic extraction requires one of
`GEMINI_API_KEY`, `GOOGLE_API_KEY`, `MOONSHOT_API_KEY`, `ANTHROPIC_API_KEY`, or
`OPENAI_API_KEY`.

## Runtime Dependencies

`requirements.txt` includes:

- pandas, numpy, scikit-learn
- xgboost, optional lightgbm
- prophet
- tensorflow
- torch
- ta
- matplotlib
- joblib
- optuna
- yfinance
- requests
- pandas-datareader
- pandas-market-calendars
- statsmodels
- shap

FastAPI and uvicorn are required for the API service but are not listed in the
visible `requirements.txt` snapshot; install them separately if running
`src/api/main.py`.

## Quality Invariants To Protect

- Chronological order in all train/test and fold splits.
- Train-only scaler fitting.
- No final-holdout tuning.
- Benchmarks excluded from production leader selection.
- Run-scoped outputs stay under the symbol output root.
- Forward forecast points respect BIST calendar and price rules.
- Optional dependencies should fail gracefully where code expects optional behavior.

## Pre-Change Checklist For Agents

1. Read [Wiki Index](index.md).
2. Identify which subsystem page applies.
3. Inspect source code before changing behavior.
4. Add or update tests when the change affects leakage boundaries, model selection, persistence, or public API behavior.
5. Update the wiki if the change adds a decision, bug solution, or new feature plan.
6. Add a top entry to [Log](log.md).
7. When a commit is requested, use a clear Turkish commit message with correct Turkish characters.


