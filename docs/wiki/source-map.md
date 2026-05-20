---
title: Source Map
type: source-map
status: active
last_updated: 2026-05-09
owner: llm
---

# Source Map

This page maps raw source layers to the wiki pages they inform. Use it before
deep inspection so source reads stay targeted.

## Primary Source Layers

| Source | Role | Wiki Pages |
|---|---|---|
| `src/pipeline/` | Orchestration, config, training/evaluation services | [Architecture](architecture.md), [Validation and Backtesting](validation-and-backtesting.md) |
| `src/data/` | CSV loading, schema normalization, corporate-action checks, updating | [Data Pipeline](data-pipeline.md) |
| `src/features/` | Technical indicators, macro features, feature caching | [Data Pipeline](data-pipeline.md) |
| `src/models/` | Model implementations and `BaseModel` contract | [Model Catalog](model-catalog.md) |
| `src/validation/` | Walk-forward split execution | [Validation and Backtesting](validation-and-backtesting.md) |
| `src/backtesting/` | Signals, execution simulation, metrics, trade extraction, reporting | [Validation and Backtesting](validation-and-backtesting.md) |
| `src/evaluation/` | Forecast metrics, benchmark enrichment, and plots | [Validation and Backtesting](validation-and-backtesting.md) |
| `src/database/` | SQLite experiment and forecast registry | [Persistence and API](persistence-and-api.md) |
| `src/model_registry/` | Per-run JSON registry | [Persistence and API](persistence-and-api.md) |
| `src/experiments/` | CSV experiment logging | [Persistence and API](persistence-and-api.md) |
| `src/forecasting/` | Forward forecast runner, BIST rules, persistence | [Persistence and API](persistence-and-api.md) |
| `src/api/` | FastAPI service around registry data and async pipeline jobs | [Persistence and API](persistence-and-api.md) |
| `src/xai/` | Explainability, feature dictionary, report writer, narratives | [Architecture](architecture.md), [Testing and Quality](testing-and-quality.md) |
| `tests/` | Behavioral contracts and regression coverage | [Testing and Quality](testing-and-quality.md) |

## Entrypoints

| Entrypoint | Purpose |
|---|---|
| `python -m src.cli.interactive` | Interactive training/evaluation entrypoint |
| `python -m src.cli.batch` | Batch execution helper |
| `python -m src.cli.forecast` | BIST-compliant forward forecast command |
| `src/api/main.py` | HTTP service entrypoint |

## Reference Documents

| File/Directory | Use |
|---|---|
| `README.md` | Broad Turkish project overview and phase history |
| `docs/glossary.md` | Domain glossary |
| `docs/MyDocs/` | Planning, research, and report artifacts; currently ignored by git |
| `AGENTS.md` | Agent schema and wiki maintenance rules |
| `RULES.md` | Repository change, wiki-update, and Turkish commit-message rules |

## Data and Metadata

| Path | Meaning |
|---|---|
| `data/*.csv` | OHLCV stock files, plus universe metadata |
| `data/bist_universe.csv` | Stock universe metadata used for survivorship and listing checks |
| `data/meta/bist_calendar.csv` | BIST trading calendar for forward forecasts |
| `data/macro/` | Macro cache generated from external sources |
| `data/feature_cache/` | Feature cache generated from raw data and config |
| `data/stock_models.db` | SQLite registry, generated runtime artifact |

## Source-of-Truth Rule

If the wiki and source code disagree, trust source code first, then update the
wiki and add a log entry. If source code and tests disagree, inspect both and
surface the contradiction before editing.

