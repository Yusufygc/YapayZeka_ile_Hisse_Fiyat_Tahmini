---
title: Validation and Backtesting
type: concept
status: active
last_updated: 2026-05-25
owner: llm
source_count: 6
---

# Validation and Backtesting

Validation is designed around time order, benchmark-relative scoring, and an
inspectable trading simulation. The default backtest answers "what would have
happened if these AL/SAT/TUT signals were traded?" with a long/flat, cost-free
simulation. More complex cost-aware signal modes remain available as opt-in
research paths.

## Validation Modes

### Walk-Forward (default, 2026-05-25)

`walk_forward` is now the **only production validation mode**. Defaults:

- `wf_n_splits = 12`, `wf_test_size = 21`, `wf_max_train_size = 756`,
  `wf_window_type = "sliding"` → ~252 OOS trading days (~1 year coverage).
- `wf_embargo_size = None` is auto-resolved to `max(200, time_steps)` at
  runtime (`_resolve_wf_embargo_size` in `src/pipeline/data_manager.py`). This
  prevents `Market_Regime_SMA200` and similar 200-bar rolling features from
  leaking train data into the test slice.

Flow:

1. Generate chronological folds.
2. Train each model per fold.
3. Evaluate fold predictions.
4. Aggregate model behavior across folds.
5. Select a best production candidate.
6. Optionally train/evaluate that model on final holdout.

`ValidationConfig` supports sliding and expanding windows. Sliding windows can
be capped by `wf_max_train_size`; expanding windows set that cap to `None`.

### (Removed from production) Single Split — research-only as of 2026-05-25

`single_split` is **no longer a production validation mode**. It is retained
only as a research/debug path accessible via `python -m src.cli.batch
--debug-quick`. When that flag is set the orchestrator calls
`_run_research_single_split()`, marks the run as `production_eligible=false`,
and stores `research_policy="debug_quick_single_split"` plus
`research_metadata.research_only=true` in the run manifest and dataset
metadata. Production leaderboards, advisory APIs, and registry promotion
ignore these runs.

Reason for removal:

- Ensemble weight optimization (`_add_single_split_ensembles`) historically
  ran `optimize_inverse_rmse`, `optimize_by_sharpe`, etc. against the
  full test-set `y_true_aligned` → in-sample / look-ahead leakage.
- A single chronological split provides no fold variance → no statistical
  confidence on Sharpe / Dir_Acc.
- The CLI prompt that asked users to pick `single_split` vs `walk_forward`
  silently routed casual runs into the leakage path.

The minimum-invasive Sprint 0 fix flags the leakage scope as
`ensemble_weight_scope[name] = "in_sample_test_set_research_only"`. The
proper train-tail validation-slice fix is scheduled for Sprint 4 (probabilistic
forecasting + multi-horizon target work introduces the slice naturally).

## Final Holdout

Final holdout is reserved confirmation data. It should not be used to tune:

- Model hyperparameters
- Signal thresholds
- Candidate selection rules
- Feature engineering decisions for the current run

`ExecutionConfig.calibration_scope` defaults to `wf_train`; any other value is
treated as a leakage risk by the calibration code.

## Evaluation Services

`EvaluationManager` is intentionally thin. Its business logic is delegated to
services:

| Service | Responsibility |
|---|---|
| `PredictionService` | Prediction alignment, target-to-price conversion, ensemble prediction |
| `BacktestService` | Convert predictions to trading signals and run backtests |
| `SignalCalibrationService` | Calibrate signal thresholds within allowed scope |
| `MetricsReportingService` | Reports, plots, CSVs, registry/db logging |

This service composition is covered by `tests/test_evaluation_services.py`.

## Metrics

Core prediction metrics include:

- MAE
- RMSE
- MAPE
- Directional accuracy
- Hit rate
- Sharpe-like financial metrics
- Benchmark-relative fields such as RMSE vs benchmark

Quantile-capable models can also produce interval/quantile metrics.

## Composite Score

`src/database/stock_model_db.py` computes a 0-100 composite score.

The score rewards:

- Benchmark-relative error performance
- Directional accuracy
- Buy-and-hold excess Sharpe behavior
- Useful trading activity

It penalizes:

- Models worse than benchmark on RMSE
- Excessive neutral/no-trade behavior
- Ineligible leader records

## Signal Generation

Signals live in `src/backtesting/signals.py`.

Modes:

- `simple`: default long/flat AL/SAT/TUT mode. A positive expected return above
  `buy_threshold` opens a long position, a negative expected return below
  `-sell_threshold` closes an existing long position, and all other cases keep
  the current state. Defaults are `buy_threshold=0.0` and
  `sell_threshold=0.0`.
- `legacy`: historical direction-only long/flat behavior.
- `professional`: opt-in research mode with quality gates, expected return
  thresholds, volatility gates, holding-period controls, take-profit/stop-loss
  rules, and market-regime inputs.

`SignalConfig` is embedded in `ExecutionConfig`.

## Signal Calibration

Execution-parameter calibration is behavior-preserving and restricted to
walk-forward training inputs. The calibration flow is decomposed into trial
generation, trial evaluation, adaptive expansion, OOS confirmation, report-frame
construction, and summary metadata updates. This keeps the final holdout outside
the tuning loop while making the sampler and rejection policy independently
testable.

The production sampler remains deterministic for a fixed grid and seed. The
research profile still runs the full grid. OOS confirmation can reject all
trials and mark execution inactive without changing the selected candidate's
reported calibration diagnostics.

## Backtest Engine

`src/backtesting/engine.py` converts signals into simulated P&L.

The default `ExecutionConfig` uses `signal_mode="simple"` with
`commission_bps=0.0` and `slippage_bps=0.0`. In this mode the engine is
long/flat only: `AL` opens a long position, `SAT` closes an existing long
position, and `TUT` preserves the current state. `SAT` never opens a short
position, and leverage, options, warrants, commission, and slippage are outside
the default scope.

The engine records:

- Initial capital
- Position state
- Previous and new position state
- Executable order (`AL`, `SAT`, `TUT`)
- Expected and realized return
- Commission, slippage, and transaction-cost columns
- Drawdowns
- Trade logs
- Blocked/no-trade signal states

Backtest summaries now feed the persistent registry before experiment logging.
The workflow merges only the durable signal/trade fields into model metrics:
`Net_Return`, `BuyHold_Return`, `Max_Drawdown`, `Trade_Count`, and
`Signal_Diagnosis`. It deliberately does not overwrite prediction-side RMSE,
directional accuracy, or model-selection Sharpe with the trading report values.
This keeps final holdout untouched for tuning while still making eligibility
and confidence decisions reflect the actual simulated trade count.

With default simple-mode costs, `Transaction_Cost`, `Commission_Cost`, and
`Slippage_Cost` remain zero. Non-zero cost accounting is still available by
explicitly configuring commission/slippage values.

## Order Reports

Every pipeline backtest writes a daily order file under the run CSV directory:

- `csv/backtest_orders_{suffix}.csv`

The report includes model, prediction date, execution date, `AL/SAT/TUT`,
previous position, new position, expected return, realized return, thresholds,
risk state, and the signal/order reason. This file is the primary audit trail
for checking whether the system actually generated actionable buy/sell/hold
orders instead of silently becoming buy-and-hold.

## Advanced Backtest Metrics

`src/backtesting/metrics.py` includes:

- Sharpe
- Sortino
- Max drawdown
- VaR/CVaR
- Deflated Sharpe
- Omega ratio
- Recovery factor
- Consecutive loss metrics
- Information ratio
- Trade efficiency

Standalone Monte Carlo bootstrap, Kelly sizing, and independent permutation-test
helpers were removed from the active codebase. The default production scope is
cost-free long/flat AL/SAT/TUT signal simulation with no leverage, shorting, or
position scaling.

## Selection Boundary

Production leader selection should consider only eligible production candidates,
not benchmarks. This is explicitly covered by tests in
`tests/test_model_scope_production.py`.

## Related Pages

- [Data Pipeline](data-pipeline.md)
- [Model Catalog](model-catalog.md)
- [Testing and Quality](testing-and-quality.md)

