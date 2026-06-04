---
title: Data Pipeline
type: concept
status: active
last_updated: 2026-05-25
owner: llm
source_count: 11
---

# Data Pipeline

The data layer turns BIST OHLCV CSV files into model-ready tabular and sequence
tensors while enforcing chronological order and leakage controls.

## Raw Inputs

- Stock CSV files live under `data/`.
- `python -m src.cli.interactive` filters CSVs so metadata files such as `bist_universe.csv`
  and `bist_calendar.csv` are not shown as stocks.
- `DataUpdater` can update stock files when `DataConfig.auto_update_data` is enabled.
- Macro data is optional and controlled by `DataConfig.use_macro`.

`DataUpdater.check_and_update()` now returns a structured `DataUpdateResult`
instead of only printing side effects. The result records status, before/after
latest dates, rows added, and error text. It also accepts a `MarketDataProvider`
interface, with `YFinanceProvider` as the default, so refresh behavior is
testable without direct network calls.

## Loading and Cleaning

`src/data/data_loader.py` handles raw OHLCV normalization.

Key behaviors:

- Turkish column names are mapped to English (`Date`, `Open`, `High`, `Low`, `Close`, `Adj_Close`, `Volume`).
- Dates are parsed with mixed-format tolerance and sorted chronologically.
- Zero-volume rows can be dropped to remove non-trading days.
- `Adj_Close` is used only when it passes anomaly checks.
- If adjusted close is missing or anomalous, raw `Close` remains the price source and the decision is recorded in dataframe attributes.

## Corporate Action Handling

`load_and_clean()` stores a `corporate_action_report` in `df.attrs`.

Important fields include:

- `adj_close_available`
- `price_source`
- `adjusted_price_trusted`
- `corporate_action_anomaly`
- `max_abs_adj_close_diff_pct`
- `warning`

This metadata is later folded into run metadata by `DataManager`.

### Corporate Action Audit (Sprint 2 — 2026-05-25)

`tools/audit_corporate_actions.py` scans every CSV under `data/` and flags
days where `|log_return| >= 0.30`. Such jumps usually mean an unadjusted
split, a large dividend ex-day, or a data error. Output:

- `outputs/_audits/corporate_action_audit_{timestamp}.csv`
- `outputs/_audits/corporate_action_audit_latest.csv`

Columns: `Symbol, Date, prev_close, close, log_return, abs_log_return,
auto_adjust_active, severity (high/extreme), notes`.

Usage:

```bash
python tools/audit_corporate_actions.py
python tools/audit_corporate_actions.py --universe data/bist_universe.csv
python tools/audit_corporate_actions.py --threshold 0.20 --out outputs/_audits
```

`src/data/quality.py` reads the latest audit and sets
`corporate_action_anomaly=True` whenever the current symbol has an
anomaly within the last `252` business days. This flag is a hard block
in the analysis API confidence chain
([Confidence and Risk Policy](confidence-and-risk-policy.md)).

### yfinance Auto-Adjust Hard Control (Sprint 2 A2.1)

`YFinanceProvider.AUTO_ADJUST = True` is required. yfinance applies the
split/dividend correction at fetch time; without it, a `1:2` split day
produces `log_return ≈ -0.69` and corrupts every trained model.
Production code must not lower this flag.

### Survivorship Bias Report (Sprint 2 A2.3)

`load_and_clean()` also writes a `survivorship_bias_report` to
`df.attrs`. Shape:

```json
{
  "symbol": "TUPRS",
  "actual_start": "2015-05-01",
  "actual_end": "2026-05-23",
  "span_days": 4036,
  "row_count": 2715,
  "max_gap_days": 4,
  "short_history_warning": false,
  "delisted_or_late_listing_warning": false,
  "warning": ""
}
```

Triggers:

- `short_history_warning`: total span < 2 years.
- `delisted_or_late_listing_warning`: a single gap > 10 calendar days
  between consecutive observations (suggests delisting, late listing,
  or data hole).

`compute_quality_flags()` exposes the report under
`survivorship_bias_report` so downstream confidence/warning chains can
relay the rationale to the user.

### Population Stability Index (PSI)

`src/data/quality.compute_psi(train_df, holdout_df)` returns a per-column
PSI dictionary. `_psi_one_feature` builds a shared 10-bin histogram from
the combined sample, then computes:

```
PSI = Σ (f_holdout - f_train) * log(f_holdout / f_train)
```

`compute_quality_flags()` sets `psi_high=True` whenever
`psi_max > 0.25`. This is a hard block. Thresholds (Sprint 7 will add
`psi_status` enum to the analysis API):

| `psi_max` | Status | Action |
|---|---|---|
| < 0.10 | `stable` | no effect |
| 0.10 - 0.25 | `moderate_drift` | soft degradation (lower confidence) |
| ≥ 0.25 | `major_drift` | hard block → `low` confidence |

## Recursive Forecast Feature Recompute (Sprint 5 — 2026-05-25)

`FeaturePipeline.recompute_close_dependent(frame)` rewrites every
close-derived technical indicator after a recursive future-forecast row is
appended. Without this rewrite, SMA/EMA/RSI/MACD/Bollinger/ATR/ADX/MFI/OBV
values from the original last row would be carried forward unchanged and
the model would predict against stale technical context — multi-day
horizons (5+ days) degrade fast.

Order of operations during `ForecastPointGenerator.roll_forward_recursive`:

1. Predict next-day target.
2. Bound predicted close with BIST band rules.
3. Append recursive row (`_append_recursive_row`) — Close, Return,
   Log_Return, LogRet_Lag_* shifted in.
4. **`recompute_close_dependent(frame)`** — every close-derived feature
   regenerated through the same `_add_*` chain used in training.
5. **`MacroForwardProjector.project_last_row`** — macro features ARIMA
   projected (see below).

Macro, sector, and lag features are not touched by step 4. Lag handled in
step 3, macro in step 5, sector inherits projected macro.

## Macro Forward Projection (Sprint 5 — 2026-05-25)

`src/features/macro_forward_projection.py:MacroForwardProjector` advances
known macro columns one step using ARIMA(1,1,1) over the last
`history_window=252` days. ARIMA failure falls back to a 20-day mean-delta
trend extrapolation.

Macro whitelist (`_KNOWN_MACRO_COLUMNS`):

- `USDTRY`, `USDTRY_Return`, `EURTRY`
- `BIST100`, `BIST100_Return`
- `BIST_*_Return` sector indices
- `VIX`, `INTEREST_RATE`, `CPI`, `BRENT`, `GOLD`

Projection is applied only to the last (recursive) row. Historical macro
rows are not modified — backtest and training data stay deterministic.

The forecast warnings enum updated from `frozen_exogenous_features` to
`projected_exogenous_features`, signaling macro inputs are advanced
deterministically rather than held constant. See
[Analysis API Contract](analysis-api-contract.md) `forecast_source.warnings`.

If `statsmodels` is unavailable the projector silently falls back to trend
extrapolation; the forecast loop never raises.

## Calendar Features (Sprint 7 — 2026-05-25)

`FeaturePipeline._add_calendar_features` derives stationary calendar signals
from the `Date` column. Always recomputed (Date-only) and safe to reapply on
recursive forecast rows.

| Column | Definition |
|---|---|
| `day_of_week` | 0=Monday … 4=Friday |
| `day_of_month` | 1..31 |
| `days_to_month_end` | days remaining in current month |
| `days_to_quarter_end` | days remaining in current quarter |
| `is_quarter_end_week` | 1 if `days_to_quarter_end <= 5` else 0 |
| `days_to_next_fomc` | days until next FOMC date, `365` if not found |

FOMC dates are read from `data/meta/fomc_calendar.csv` (manually maintained).
Missing CSV degrades to a constant `365.0` placeholder. Toggle via
`FeaturePipeline(enable_calendar_features=False)`.

## Cross-Sectional Momentum (Sprint 7 — 2026-05-25)

`FeaturePipeline._add_cross_sectional_momentum` runs inside `_merge_macro`
when sector/market returns are available. Compares the symbol's 60-day
momentum against sector index and BIST100.

| Column | Definition |
|---|---|
| `momentum_60d` | `Close.pct_change(60)` |
| `market_momentum_60d` | cumprod-equivalent of `BIST100_Return` over 60d |
| `sector_momentum_60d` | cumprod-equivalent of sector index return (falls back to market when sector missing) |
| `relative_momentum_60d` | `momentum_60d - sector_momentum_60d` |
| `relative_to_market_60d` | `momentum_60d - market_momentum_60d` |

Missing macro return columns make the related rows silently absent — no
exception is raised. Toggle via
`FeaturePipeline(enable_cross_sectional_momentum=False)`.

## Feature Engineering

Feature generation is split across:

- `src/data/data_loader.py`: base stationary technical features and lagged log returns.
- `src/features/feature_pipeline.py`: expanded technical feature set.
- `src/features/macro_pipeline.py`: macro context features.
- `src/features/feature_cache.py`: cache by data/config key.
- `src/features/sector_mapping.py`: dynamic stock-to-sector-index mapping from `data/bist_universe.csv`.

The project favors stationary or normalized features:

- Relative moving-average features instead of raw moving-average price levels.
- Normalized ATR (`NATR_14`) and MACD-style features (`MACD_norm`, `MACD_Signal_norm`, `MACD_Diff_norm`).
- Log-return lags instead of raw close lags.
- Stationary momentum and volume indicators like Money Flow Index (`MFI_14`), Average Directional Index (`ADX_14`), and Chaikin Money Flow (`CMF_20`).
- Macro lag controls for rate and CPI data to avoid lookahead.
- Sektörel Göreli Güç (`Sector_Relative_Strength`), calculating the difference between target stock return and its corresponding sector index return (falling back to BIST100 if the index is missing).
- Reduced correlation pruning threshold from `0.98` to `0.88` for stricter feature filtering.

`Sector_Relative_Strength` is now driven by `data/bist_universe.csv`
`Sector_Index` metadata rather than a hard-coded stock-to-sector dictionary.
Missing symbols, missing or unsupported sector indexes, or unavailable sector
return columns fall back to `BIST100_Return`; if `BIST100_Return` is also
missing, the feature is skipped without failing the pipeline. Feature cache keys
include the universe file path, timestamp, and content hash so mapping changes
invalidate stale engineered-feature caches. Dataset metadata and cache metadata
include a `sector_mapping` report with matched/fallback status and fallback
reason.

`MacroPipeline.get_macro_features()` now acts as an orchestration layer around
separate cache refresh, cache loading, date filtering, monthly release-lag,
daily/global merge, monthly ffill, and final feature-engineering helpers. Public
inputs and output schema are unchanged; the split exists to keep EVDS/FRED/manual
CSV fallback behavior and release-lag leakage controls independently testable.

### Macro feature-cache poisoning guard (2026-06-02 fix)

The feature-cache key already includes `use_macro`, but a transient macro-fetch
failure used to poison the cache: when macro requested but unavailable, the
engineered frame was still written under the `use_macro=True` key, so later
(healthy) runs served the macro-less frame on a cache hit. A real trigger was a
Windows `cp1252` `UnicodeEncodeError` on the `→` character in `MacroPipeline`
log prints — the exception bubbled into `DataIngestionService._fetch_macro`,
which swallowed it and returned an empty frame, silently dropping macro. Two
guards in `DataIngestionService._engineer_features_cached`
(`src/pipeline/data_services.py`):

- **Write guard:** when `use_macro` is True but the resolved `macro_df` is empty/
  None, the degraded frame is **not** cached (a transient failure cannot poison).
- **Self-heal on read:** a cache hit whose `feature_names` contain no macro
  feature (`_has_macro_features`, checked against
  `MacroPipeline.macro_feature_names`) while macro is expected is evicted and
  recomputed, so already-poisoned entries recover automatically.

The root print crash was also removed (`→` replaced with `->` in
`macro_pipeline.py` runtime prints) so macro no longer drops on cp1252 stdout.

## DataManager Responsibilities

`DataManager` in `src/pipeline/data_manager.py` owns:

- Optional data update.
- Raw data loading.
- Training-window filtering.
- Macro fetch and cache.
- Feature cache lookup/write.
- Feature pruning metadata.
- Dynamic sector mapping from `DataConfig.universe_file` and `sector_mapping`
  metadata propagation.
- Survivorship/listing checks.
- Chronological splitting.
- Tensor preparation.
- Dataset hash and run metadata construction.

## Splitting

Splitting is chronological. No shuffle-based split is valid for this project.

Modes:

- `single_split`: one chronological train/test split using `DataConfig.test_ratio`.
- `walk_forward`: fold generation using `ValidationConfig` and `WalkForwardValidator`.
- `final_holdout`: last reserved block used only after walk-forward selection.

Important walk-forward settings:

- `wf_n_splits`
- `wf_min_train_size`
- `wf_test_size`
- `wf_max_train_size`
- `wf_window_type`: `sliding` or `expanding`
- `wf_embargo_size`
- `final_holdout_size`

## Calendar Maintenance

`src/forecasting/bist_calendar.py` maintains `data/meta/bist_calendar.csv`.
`ensure_bist_calendar()` generates a rolling BIST trading calendar with Turkish
fixed-date closures, weekends, and deterministic session labels. Existing rows
are treated as manual overrides and merged back into the generated range, so
religious holidays or half-day sessions can be added to the CSV without changing
consumer code.

The analysis refresh service calls calendar generation before freshness checks
and forecast generation. `analysis_freshness.compute_freshness()` warns when the
calendar range does not cover the requested freshness window and falls back to a
weekday count.

## Scaling

`src/data/preprocessor.py` enforces train-only fitting.

Supported scaling modes:

| Mode | X scaler | y scaler | Clip |
|---|---|---|---|
| `robust_x_standard_y_clip` | RobustScaler | StandardScaler | yes |
| `robust` | RobustScaler | RobustScaler | yes |
| `standard` | StandardScaler | StandardScaler | no |
| `minmax` | MinMaxScaler | MinMaxScaler | no |

The default modern mode is `robust_x_standard_y_clip`.

Leakage rule:

- `fit_transform` is allowed only on the training slice.
- Test, fold-test, and final-holdout slices must use `transform`.
- Walk-forward fold scalers are not written repeatedly to the same inference path.

## Targets

The modern target mode is `log_return`.

Prediction conversion:

- `reconstruct_prices_from_logret()` converts log-return predictions to prices using previous actual close.
- `reconstruct_prices_from_return()` converts simple-return predictions to prices.
- Clipping protects against pathological out-of-distribution predictions.

## Sequence Tensors

`create_sequences()` creates sequence tensors for sequence-style models such as
LSTM and the lightweight linear sequence baselines:

```text
X: (samples, features)
to
X_seq: (samples - time_steps + 1, time_steps, features)
```

The target aligns with the final row of each sequence window. The intended
semantic is one-step-ahead forecasting with features known at the end of day `t`
and target for the next move.

## Leakage Boundaries

- No random train/test split.
- No scaler fit on test, fold-test, or final holdout.
- Macro features must respect configured lags.
- Signal calibration must not optimize on final holdout.
- Final holdout is for confirmation, not model selection training feedback.

## Related Pages

- [Architecture](architecture.md)
- [Validation and Backtesting](validation-and-backtesting.md)
- [Testing and Quality](testing-and-quality.md)
