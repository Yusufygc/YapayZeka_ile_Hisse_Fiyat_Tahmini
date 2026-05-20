---
title: Backtest Signal Improvement Plan
type: feature-plan
status: draft
last_updated: 2026-05-18
owner: llm
source_count: 4
---

# Backtest Signal Improvement Plan

This page captures the current design focus for improving BIST strategy
backtests. API integration and transaction-cost modelling are intentionally out
of scope for the first iteration.

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
- Cost modelling can stay disabled for the first iteration, but turnover and
  trade count must still be tracked because future costs will punish noisy
  strategies.

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
