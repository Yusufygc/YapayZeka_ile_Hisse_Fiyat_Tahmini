---
title: Backtest Signal Improvement Plan
type: feature-plan
status: draft
last_updated: 2026-05-24
owner: llm
source_count: 4
---

# Backtest Signal Improvement Plan

This page captures the current design focus for improving BIST strategy
backtests. API integration and transaction-cost modelling are intentionally out
of scope for the first iteration.

## Research Scope Decision

As of 2026-05-23, transaction-cost modelling remains explicitly out of scope
for the current research phase. Commission and slippage can stay at zero while
the project investigates whether the forecast-to-position layer produces
repeatable, leakage-safe signal behavior. The near-term decision criterion is
not cost realism; it is whether a signal policy can create enough confirmed
trades, preserve directional edge, and avoid fragile one-period holdout wins.

## Current Goal

The near-term goal is not to build a full trading or advisory API. The goal is
to improve the signal mechanism so model-driven long/flat backtests can beat
buy-and-hold and then confirm on final holdout without tuning on final holdout.

## Diagnosis

The weak backtest result should not be interpreted as "too few models" by
default. The first suspected bottleneck is the layer that turns return forecasts
into positions:

- The default execution mode is `simple`, with zero buy/sell thresholds.
- The richer `professional` signal mode, shadow comparisons, gate diagnostics,
  and execution calibration exist, but calibration is disabled by default.
- Reports often need to be read by diagnosis label first:
  `underperform_buyhold`, `insufficient_trades`, `model_signal_weak`, and
  `gate_too_strict`.
- A no-trade result may be correct when buy-and-hold is negative, but it is not
  evidence of predictive edge.

## Bank Sector Signal Diagnosis

The 2026-05-23 AKBNK, GARAN, and ISCTR review showed a recurring pattern:
strong final-holdout candidates often carried `gate_too_strict` or
`model_signal_weak` diagnoses. In this context `gate_too_strict` means the
model can pass quality checks, but the conversion from forecast to position is
too conservative or too binary. This should be treated as a signal policy issue
before changing the model catalog.

For this sector profile, the next signal work should compare three families of
rules using only walk-forward calibration data:

- Softer entry gates that target a minimum trade band rather than simply
  maximizing net return.
- Rank-based or percentile-based entries, where the model trades only its
  strongest forecast days instead of using a fixed zero threshold.
- Sector-relative confirmation, where a long signal is easier to accept when
  the stock has positive relative strength versus the sector or BIST benchmark.

The detailed cross-sector research plan is now tracked in this wiki page rather
than in ignored local plan files. It extends the bank-sector findings to
holding, industrial, and technology stocks, and includes the ARDYZ run-level
final-holdout durability rules.

## Industrial Sector Signal Diagnosis

The 2026-05-23 EREGL, ERBOS, and FROTO review showed that industrial stocks do
not share one simple signal profile:

- EREGL behaved like a trend-capture case. Buy-and-hold was strongly positive
  on final holdout, and `Random Forest` produced a usable `ok` diagnosis with
  enough trades and excess return.
- ERBOS behaved like a defensive/selective-entry case. Buy-and-hold was mildly
  negative on final holdout; `AttentionLSTM v2` led the scoreboard but only
  with two trades, while `XGBoost` was the more credible research candidate
  despite `model_signal_weak`.
- FROTO behaved like a drawdown-reduction case. Buy-and-hold was deeply
  negative on final holdout, and the best practical result was not profit but
  loss reduction from `ElasticNet Return`/`Ridge Return`; all final candidates
  still carried `model_signal_weak`.

For this sector profile, signal improvement should be regime-aware before it is
model-family-aware. The same industrial label contains trend-following,
defensive, and drawdown-reduction problems. Selection should therefore score
models separately for positive-trend capture, negative-regime loss reduction,
and enough confirmed trades, instead of ranking only by final composite score.

## Technology Mid-History Diagnosis

The 2026-05-24 ARDYZ review is now treated as a `mid_history` technology
diagnostic cohort, not as an exclusion from the research set. The 10-year line
is a reference threshold for comparing data-history effects. ARDYZ showed a
new failure pattern: `Random Forest` and `LightGBM Return` led walk-forward
backtests, but both weakened sharply on final holdout. `DLinear` was the most
defensible final-holdout research candidate, but it still did not beat
buy-and-hold. `ARIMA` matched buy-and-hold with only one trade, so it should be
classified as benchmark-like rather than as a real trading leader.

This adds four rules to the signal-improvement work:

- Read `outputs/{SYMBOL}/runs/{RUN_ID}` as the analysis source of truth;
  `latest/` is a convenience copy and can be overwritten by nearby runs.
- Exclude runs without final-holdout reports from leaderboards and mark them
  `incomplete_final_holdout`.
- Report the walk-forward to final-holdout performance gap for each model so
  unstable walk-forward winners are visible.
- Treat one-trade or buy-and-hold-clone results as invalid leadership even when
  final net return is high.
- Report `history_bucket` (`long_history`, `mid_history`, `short_history`,
  `missing_data`, `unknown`) so weak results can be compared against data
  history before blaming model quality alone.

## Implemented Diagnostic Automation

As of 2026-05-24, the run-level diagnostic automation is implemented and the
Plan 1 closure command path is available:

- `python -m src.cli.run_leaderboard` reads run-scoped outputs for one symbol,
  selected symbols, or all symbols under `outputs/`.
- The leaderboard includes final-holdout completeness, WF/final gap,
  buy-and-hold excess, trade sufficiency, benchmark-clone detection,
  reliability class, history bucket, sector, prediction rank, and trading rank.
- `latest/` is not used as an analysis source; it remains a convenience copy.
- `Prophet-ML/DL Hybrid` now follows the same date-aware prediction path as
  `Prophet` in final holdout and forward forecast flows.
- `run_manifest.json` records final-holdout success, skip, or failure status.
- `python -m src.cli.signal_research` provides universe checks, `ensure-data`
  restoration from `data/old`, symbol/model/policy matrices, sequential
  `run --resume` execution, and summaries from completed runs.
- Research runs write `research_policy`, `research_phase`,
  `uses_final_holdout_for_selection=false`, and policy/history/sector metadata
  into `run_manifest.json`.
- ISO history dates are parsed year-first, so `2020-02-06` is not shifted by
  day-first Turkish CSV parsing.

The implemented V1-V4 policy profiles are executable configuration-level
research variants: V1 soft professional gate, V2 percentile-entry-biased
thresholding on walk-forward calibration, V3 trade/exposure-band calibration,
and V4 sector-relative confirmation metadata with the dynamic
`Sector_Relative_Strength` feature. Long-running backtest campaigns still need
to be launched deliberately because the full matrix can be expensive.

## Model Catalog Direction

Model simplification should be done by promotion tiers, not by deleting all
underperformers at once.

- Active core: `Ridge Return`, `ElasticNet Return`, `LightGBM Return`,
  `XGBoost`, `DLinear`, and naive benchmarks.
- Conditional active: `Random Forest` only when its per-stock diagnostics show
  stable excess return or useful diversification.
- Research shelf: `LSTM`, `NLinear`, `Prophet`, and `ARIMA` until they show
  repeated walk-forward and final-holdout value.

The reason is BIST stock structure diversity: liquid large caps, volatile
speculative names, shallow names, and trend-heavy names can favor different
model families. The catalog should preserve experimentation, while default runs
stay smaller and easier to interpret.

## Signal Improvement Direction

Signal work should come before adding more model families.

1. Use walk-forward reports to classify failure mode per stock/model.
2. Compare `simple`, `professional_current`, `professional_soft_gate`, and
   `legacy_directional` via shadow backtests.
3. If `gate_too_strict` appears, relax signal gates or use soft gates.
4. If `insufficient_trades` appears with positive edge, lower entry thresholds
   or use horizon/holding rules that allow more trades.
5. If `model_signal_weak` appears, do not force trading; demote that model for
   that stock profile.
6. Use market regime and volatility multipliers as context filters, not as
   standalone prediction models.

## Evaluation Rules

- Final holdout remains confirmation only.
- Signal thresholds, gate settings, model defaults, and stock-profile rules must
  be selected from walk-forward data only.
- Beating buy-and-hold is meaningful only when supported by enough trades and
  repeated folds, not by one terminal hold or zero-trade avoidance.
- Cost modelling can stay disabled for the research iteration. Turnover and
  trade count must still be tracked, but they are used to diagnose signal
  behavior rather than to reject otherwise promising strategies on cost grounds.

## ClaudeGelistirme Integration

The local `ClaudeGelistirme/` analysis was reviewed and only the parts aligned
with the current product plan were integrated into `yeniTasarim/`.

Accepted into the design:

- Classify weak backtests by diagnosis labels before changing the model catalog.
- Reject naive benchmark leadership when trade count is insufficient.
- Use cross-run leaderboard, shadow backtests, and signal diagnostics as
  selection inputs.
- Add rolling holdout, fold stability, and distribution-shift gates as core
  follow-up work.
- Keep adaptive regime-aware gating as a signal-confidence mechanism, not as an
  automatic trading rule.
- Treat ensemble directional agreement as support unless independently validated
  as a production strategy.

Deferred from the current design:

- Default transaction-cost modelling.
- Portfolio construction, long-short, market-neutral, and tax modelling.
- API deployment, monitoring, notification, and Docker work.
- CPCV, SPA/reality-check, and panel modelling until simpler robustness gates
  are in place.

## Related Pages

- [Validation and Backtesting](validation-and-backtesting.md)
- [Model Catalog](model-catalog.md)
- [Product Decision Support Design](product-decision-support-design.md)
