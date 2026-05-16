# Project and Wiki Guidelines

You are the lead developer of this project and the sole maintainer of the knowledge base (Wiki) located in the `docs/wiki/` directory.

## Core Rules
1. **Consult the Wiki:** Whenever a question is asked or a new feature is requested, you must read the `docs/wiki/index.md` file before touching any code. Understand the architectural decisions and the current state from there.
2. **Self-Update (Ingest):** If we make a new technology decision, find a solution to a bug, or plan a new feature during our conversation, do not just keep it in your memory. Update the relevant markdown file within `docs/wiki/` or create a new one.
3. **Keep a Log (Log):** Append every significant architectural change or wiki update to the very top of the `docs/wiki/log.md` file with a timestamp (using the `## [YYYY-MM-DD] Action | Topic` format).

## Wiki Directory (`docs/wiki/`)
- `index.md`: The map and cross-references of all wiki pages.
- `log.md`: The chronological record of all updates made.
- `architecture.md`: Technologies used, database schema, and overall structure.

# AGENTS.md

This file is the schema layer for the repository's LLM-maintained wiki. The
wiki follows the "LLM Wiki" pattern: raw project sources are compiled into a
persistent, cross-linked markdown knowledge base that improves over time.
Repository-level change and commit rules live in `RULES.md`.

## LLM Wiki Operating Model

There are three layers:

1. **Raw sources:** Source code, tests, README, existing docs, data metadata, and user-provided decisions. Source code and tests are the primary truth for behavior.
2. **Wiki:** `docs/wiki/*.md`. The LLM owns this layer: summaries, cross-links, decisions, contradiction notes, and maintenance logs.
3. **Schema:** This `AGENTS.md` file. It defines how agents should operate on the wiki and codebase.
4. **Rules:** `RULES.md`. It defines how system changes, wiki updates, and Turkish commit messages are handled.

When the wiki and source code disagree, trust the source code, update the wiki,
and log the correction.

## Mandatory Wiki Workflow

Before code or architecture work:

1. Read `docs/wiki/index.md`.
2. Read the relevant linked wiki pages.
3. Inspect source code or tests when exact behavior matters.

When a decision, bug solution, feature plan, or durable explanation appears:

1. Update the relevant wiki page or create a new one.
2. Update `docs/wiki/index.md` if page navigation changes.
3. Add a new top entry to `docs/wiki/log.md`.
4. Follow `RULES.md` for wiki updates and Turkish commit messages.

For detailed wiki conventions, use `docs/wiki/wiki-guidelines.md`.

## Wiki Pages

- `docs/wiki/index.md`: Wiki map and current project state.
- `docs/wiki/wiki-guidelines.md`: LLM Wiki rules, frontmatter, ingest/query/lint workflows.
- `docs/wiki/change-management.md`: Change, wiki-update, and Turkish commit-message workflow.
- `docs/wiki/source-map.md`: Raw source map and where facts should be filed.
- `docs/wiki/architecture.md`: System architecture, technology stack, config, outputs, database schema.
- `docs/wiki/data-pipeline.md`: Data loading, features, scaling, targets, leakage boundaries.
- `docs/wiki/model-catalog.md`: Benchmarks, candidates, model families, production scope.
- `docs/wiki/validation-and-backtesting.md`: Single split, walk-forward, final holdout, signals, metrics.
- `docs/wiki/persistence-and-api.md`: Outputs, SQLite, forecast persistence, FastAPI.
- `docs/wiki/testing-and-quality.md`: Test areas, commands, quality invariants.
- `docs/wiki/log.md`: Append-first chronological update log.

## Project Overview

`ts_forecasting_lab` is a production-oriented time series forecasting platform
for BIST equities. It supports data updating, technical and macro feature
engineering, single-split and walk-forward validation, model benchmarking,
financial backtesting, XAI reporting, model registries, SQLite persistence, and
BIST-compliant forward forecasts.

The top-level facade is `ForecastingPipeline` in `src/pipeline/orchestrator.py`.

```text
ForecastingPipeline
|-- DataManager
|-- ModelTrainer
`-- EvaluationManager
    |-- PredictionService
    |-- BacktestService
    |-- SignalCalibrationService
    `-- MetricsReportingService
```

## Main Commands

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Run interactive pipeline
python main_pipeline.py

# Run tests
python -m pytest tests

# Run selected test file
python -m pytest tests/test_smoke.py -v

# Run BIST-compliant forward forecasts
python run_forecast.py --stocks TUPRS,ASELS --horizon-days 5

# Serve registry/API data
uvicorn src.api.main:app --reload --port 8000
```

`fastapi` and `uvicorn` may need to be installed separately if they are missing
from the active environment.

## Architecture Rules

- Preserve chronological ordering in all validation logic.
- Fit scalers only on the training slice currently being evaluated.
- Do not tune on final holdout data.
- Treat naive models as benchmarks, not production candidates.
- Keep production model scope aligned with `src/pipeline/model_scope.py`.
- Keep run outputs under `outputs/{SYMBOL}/runs/{RUN_ID}/` and sync `latest/`
  only inside that symbol output root.
- Keep SQLite registry behavior aligned with `src/database/stock_model_db.py`.
- Prefer existing config dataclasses in `src/pipeline/config.py` over new flat
  argument lists.

## Editing Guidance

- Follow existing module boundaries before introducing new abstractions.
- For data behavior, update or add tests around leakage, chronological splits,
  target conversion, or scaler fitting.
- For model selection, update tests around candidate/benchmark eligibility.
- For persistence/API behavior, update tests around SQLite schema, forecast
  records, endpoint contracts, or idempotency.
- Keep generated runtime artifacts (`outputs/`, model files, caches, SQLite DBs)
  out of source control unless explicitly requested.
- When a commit is requested, write the commit message in clear Turkish and use
  Turkish characters correctly.

## Documentation Discipline

Use the wiki as the durable memory of the project. Chat history is temporary;
the wiki is the maintained knowledge base.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
