---
title: Wiki Index
type: index
status: active
last_updated: 2026-05-20
owner: llm
---

# Wiki Index

This wiki is the canonical, LLM-maintained knowledge base for `ts_forecasting_lab`.
Every implementation question and feature request starts here, then drills into the
linked pages below.

## Operating Model

- [Wiki Guidelines](wiki-guidelines.md): LLM Wiki structure, ownership rules, frontmatter, cross-linking, and maintenance workflows.
- [Change Management](change-management.md): Change, wiki-update, and Turkish commit-message rules.
- [Source Map](source-map.md): Raw source layers the wiki is compiled from.
- [Log](log.md): Chronological record of ingests, architecture updates, query-derived pages, and lint passes.

## System Knowledge

- [Architecture](architecture.md): Main components, technology stack, database schema, and runtime structure.
- [Data Pipeline](data-pipeline.md): OHLCV loading, corporate action handling, macro features, scaling, leakage controls, and tensors.
- [Model Catalog](model-catalog.md): Benchmarks, candidate models, legacy/comparison models, and production selection rules.
- [Validation and Backtesting](validation-and-backtesting.md): Single split, walk-forward, final holdout, signal gates, backtest metrics, and leakage boundaries.
- [Backtest Signal Improvement Plan](backtest-signal-improvement-plan.md): Draft plan for improving signal logic and model tiers before API integration or cost modelling.
- [Persistence and API](persistence-and-api.md): Outputs, SQLite registry, forecast persistence, FastAPI endpoints, and downstream integration.
- [Testing and Quality](testing-and-quality.md): Test suites, quality gates, smoke tests, configured tools, and known verification commands.
- [Code Quality and Refactoring](code-quality-and-refactoring.md): Code thresholds, file/class size limits, input validation guidelines, error handling, and datetime policy.
- [Product Decision Support Design](product-decision-support-design.md): Desktop AI decision-support product boundary, target architecture, MVP scope, and phase roadmap.
- [Analysis API Contract](analysis-api-contract.md): `GET /analysis/{symbol}` response schema, status codes, and confidence label definition.
- [Confidence and Risk Policy](confidence-and-risk-policy.md): Confidence label derivation rules, signal-diagnosis mapping, eligibility status, and data-quality gates.
- [LLM Explanation Policy](llm-explanation-policy.md): AI explanation layer role, forbidden actions, response structure, system prompt skeleton, and disclaimer.

## Current Project State

- The project is a production-oriented research platform for BIST equity forecasting.
- The active orchestration facade is `ForecastingPipeline` in `src/pipeline/orchestrator.py`.
- Main orchestration responsibilities are split across `DataManager`, `ModelTrainer`, and `EvaluationManager`.
- Evaluation logic is now service-composed via `PredictionService`, `BacktestService`, `SignalCalibrationService`, and `MetricsReportingService`.
- The default production candidate set is defined in `src/pipeline/model_scope.py`
  and the model registry; in the current source tree `TFT` is not registered and
  `src/models/tft_v2/` is absent.
- Cheap naive models are treated as benchmarks, not production candidates.
- The main database is `data/stock_models.db`; trained run artifacts live under `outputs/{SYMBOL}/runs/{RUN_ID}/`.
- `outputs/{SYMBOL}/latest/` is synchronized from the latest completed run.
- The analysis API now tracks non-blocking refresh jobs in SQLite, restricts
  CORS to local origins plus explicit environment configuration, and can queue
  data/forecast refresh work when the best model has no current forecast or the
  market data is stale.
- Forward forecasts are served from saved production artifacts and recursive
  horizon generation. Production ensemble leaders are limited to `Ensemble
  Inverse RMSE` and `Ensemble Cash-Gated`, with member metadata persisted.
- `AttentionLSTM v2` is an opt-in sequence candidate with temporal attention
  XAI export; it is not part of the default production candidate set.
- Repository-wide change discipline is captured in `RULES.md` and [Change Management](change-management.md).

## Navigation Rules for Agents

1. Read this file before code changes or architecture answers.
2. Use [Source Map](source-map.md) to decide which raw source files to inspect.
3. If the answer requires system behavior, inspect source code as the primary source of truth.
4. If a decision, bug solution, or feature plan is produced, update the relevant wiki page.
5. Append a new entry to the top of [Log](log.md) for every significant wiki or architecture update.
6. Commit completed changes with clear Turkish explanations when the user asks for a commit.
