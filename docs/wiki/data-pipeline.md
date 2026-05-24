---
title: Data Pipeline
type: concept
status: active
last_updated: 2026-05-24
owner: llm
source_count: 7
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
