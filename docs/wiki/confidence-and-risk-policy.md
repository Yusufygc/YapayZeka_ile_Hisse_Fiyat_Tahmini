---
title: Confidence and Risk Policy
type: concept
status: active
last_updated: 2026-05-25
owner: llm
source_count: 4
---

# Confidence and Risk Policy

Defines how the `confidence.label` field in `GET /analysis/{symbol}` is
derived, what inputs feed it, and how data quality and risk signals affect
model eligibility.

## Confidence Label Levels

Three levels only — never a percentage:

| Label | When |
|---|---|
| `low` | Any hard block applies (see below), or multiple soft degradations |
| `medium` | No hard block; modest performance with acceptable caveats |
| `high` | Strong historical signal across all quality gates |

## Input Sources (Priority Order)

### Hard Blocks → Always `low`

1. `eligibility_status != 'eligible'` — model is naive leader with few trades,
   insufficient total trades, failed benchmark guard, or no production
   candidate found.
2. `data_freshness = 'stale_data'` — data beyond staleness threshold.
3. `directional_accuracy < 50` or `rmse_vs_benchmark >= 1.0` — model does not
   beat naive directional chance or benchmark RMSE.
4. `psi_high = True` or `corporate_action_anomaly = True` (data quality flags
   from `src/data/quality.py`).
5. `model_status = 'degraded'` — rolling live resolution dir_acc fell below 50
   in the last 60 trading days (Faz 2).

### Soft Degradations → Cap or Lower One Level

| Signal | Effect |
|---|---|
| `signal_diagnosis` contains `insufficient_trades` | cap at `medium` |
| `signal_diagnosis` contains `gate_too_strict` | cap at `medium` |
| `signal_diagnosis` contains `model_signal_weak` | cap at `medium` |
| `signal_diagnosis` contains `underperform_buyhold` | cap at `medium` |
| `stability_score < threshold` (fold Sharpe instability) | lower one level |
| `rolling_positive_window_ratio < 0.5` (Faz 2 rolling holdout) | lower one level |
| `ensemble_direction_agreement < 0.5` (Faz 2) | cap at `medium` |
| Regime misalignment with forecast direction (Faz 2) | lower one level |
| `risk_free_unavailable = True` (Sprint 1 A1.1) | lower one level — Sharpe/Sortino NaN; advisory için risk-adjusted skor güvensiz |

### `high` Conditions

All of the following must hold simultaneously:
- No hard block.
- No soft degradation that has applied.
- `directional_accuracy >= 55`.
- `stability_score >= upper_threshold` (fold Sharpe stable).
- `rmse_vs_benchmark < 1.0` meaningfully (composite_score >= threshold).
- `ensemble_direction_agreement >= 5/7` (Faz 2, if available).

## Signal Diagnosis Labels

Produced by `SignalCalibrationService` and stored on experiment rows. The
confidence engine reads these labels from the best-model's source experiment.

| Label | Meaning |
|---|---|
| `underperform_buyhold` | Backtest return below buy-and-hold |
| `insufficient_trades` | Too few trades for statistical significance |
| `model_signal_weak` | Model directional accuracy near chance |
| `gate_too_strict` | Professional gate produced near-zero trades |
| `rejected_no_trade` | Signal rejected all positions |

These labels are not shown to the user as raw strings. They feed the confidence
level and appear in `confidence.reasons` and `confidence.warnings` as
human-readable text.

## Model Eligibility Status

Stored in `best_models.eligibility_status`:

| Value | Meaning |
|---|---|
| `eligible` | Production candidate; passes all guards |
| `naive_low_trades` | Naive benchmark is leader and trade count is too low |
| `insufficient_trades` | Trade count below minimum threshold |
| `no_candidate` | No production-eligible model found for this symbol |
| `benchmark_failed` | Candidate did not beat the benchmark RMSE ratio guard |

When `no_candidate`, `analysis_status` is `no_model`.

Production ensembles are eligible only when their stored ensemble method is
`Inverse RMSE` or `Cash-Gated`. Other ensemble rows are forced to
`no_candidate` by the selection guard even if they are reportable.

## Naive Leader Rejection

If the top-scoring model for a symbol is a naive benchmark (`is_baseline=True`)
**and** total walk-forward trade count is below `min_trades`, the symbol has
`eligibility_status = 'naive_low_trades'` and no production model is surfaced.

User-facing message in this case:
```
Bu hisse için geçmiş doğrulamada güvenilir production model bulunamadı.
```

## Data Quality and Distribution Shift

Computed by `src/data/quality.py`:

| Flag | Threshold | Effect |
|---|---|---|
| `psi_high` | PSI (train vs holdout features) > 0.25 | Hard block → `low` confidence |
| `corporate_action_anomaly` | Detected via `tools/audit_corporate_actions.py` (any `|log_return| >= 0.30` day in last 252 BD) OR Adj_Close anomaly | Hard block → `low` confidence |
| `clip_rate` | Fraction of prices clipped by BIST band rules | Warning; shown in `confidence.warnings` |
| `survivorship_warning` / `survivorship_bias_report.warning` | `short_history` (< 2y) or `delisted_or_late_listing` (gap > 10d) | Warning; shown in `confidence.warnings` |

### PSI Threshold Tiers (Sprint 2 A2.4 — exposed via API in Sprint 7)

`compute_psi()` returns per-feature PSI; `psi_max` aggregates the worst.

| `psi_max` | Status | Effect |
|---|---|---|
| `< 0.10` | `stable` | no effect |
| `0.10 - 0.25` | `moderate_drift` | soft degradation (lower one level) |
| `≥ 0.25` | `major_drift` | hard block → `low` |

### Live PSI 30d Monitor (Sprint 7 — 2026-05-25)

`src/api/services/data_quality_monitor.py:compute_psi_30d()` performs an
on-the-fly PSI check at API request time on stationary OHLCV-derived
features (`log_return`, `range_pct`, `volume_log_change`). The holdout is the
last 30 trading days; the train window is the prior 252 trading days. Bins=3
keeps the small-sample noise floor below `0.10` for iid distributions.

The result is published in `data_quality.psi_30d` / `psi_status` on the
analysis response. `major_drift` forces confidence to `low` and appends a
`data_drift_major:` warning string; `moderate_drift` downgrades `high` to
`medium`.

## Stability Score

Computed from walk-forward fold metrics at experiment insert time:

```
stability_score = positive_fold_ratio - 0.5 * std(fold_sharpe)
```

Where `positive_fold_ratio` is the fraction of folds with positive net return
above buy-and-hold.

Rolling holdout inputs (Faz 2):

```
rolling_positive_window_ratio  = fraction of 60-bar holdout windows with net return > 0
rolling_median_net_return       = median net return across all windows
rolling_iqr_net_return          = IQR (spread of dispersion)
```

## Freshness Threshold

Computed by `src/api/services/analysis_freshness.py` using the BIST trading
calendar from `data/meta/bist_calendar.csv`.

Default: `staleness_days > 1 BIST trading day` → `stale_data`.

When stale, the response still contains the model and forecast data but
`analysis_status = 'stale_data'` and a warning appears in
`confidence.warnings`.

The analysis service also queues a refresh job when stale data is detected.
Failed refresh jobs surface their failure reason through `refresh_reason`, but
confidence still uses the freshness value from the forecast that is actually
returned.

## Peer / Segment Confidence (E2 Faz 5–7)

The pooled-model `peer` block carries its **own** confidence
(`src/serving/confidence.py`, `peer_confidence`), independent of the absolute
forecast confidence above:

- Signal strength = **composite segment ICIR** (`composite_icir`, weighted across
  liquidity/volatility/sector buckets). ICIR ≥ 1.0 → `high`, ≥ 0.5 → `medium`,
  else `low`.
- **Hard gates always force `low`** regardless of ICIR: not tradeable
  (median turnover below the liquidity floor), stale data, thin universe, or
  `unknown` peer label. This encodes the Faz 6 tension: the strongest
  cross-sectional signal lives in the least-liquid (hardest-to-trade) names.
- Faz 7 validated that within the tradeable universe `high` is genuinely more
  directionally accurate than `medium`; `low` also contains the untradeable
  strong-signal tail (correctly flagged, not actionable).
- The peer **trend tendency** (yukarı/yatay/aşağı) is a calibrated probabilistic
  tilt; this confidence label governs how much to trust it.

## Related Pages

- [Analysis API Contract](analysis-api-contract.md)
- [Product Decision Support Design](product-decision-support-design.md)
- [Backtest Signal Improvement Plan](backtest-signal-improvement-plan.md)
- [Validation and Backtesting](validation-and-backtesting.md)
- [Model Catalog](model-catalog.md)
- [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md)
