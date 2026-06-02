---
title: E2 Faz 2 — Pooled Loader + Group-Purged CV Design
type: feature-plan
status: active
last_updated: 2026-06-02
owner: llm
branch: feat/e2-pooled-global-model
---

# E2 Faz 2 — Pooled Loader + Group-Purged Multi-Window CV (Design)

Detailed technical design for the heart of the E2 epic: load all ~585 stock
CSVs into one panel, and validate a global model with a **leakage-free,
date-based, purged, multi-window** cross-validation that produces a distribution
of out-of-sample (OOS) metrics per symbol. Parent: [E2 Pooled Global Model
Epic](e2-pooled-global-model-epic.md).

## Goals / Non-Goals

**Goals**
- One pooled long panel `(symbol, date, features…, target, conditioning…)`.
- A CV that is provably free of the three panel leaks (below).
- Multi-window rolling OOS → a *distribution*, not a single holdout number.
- Per-symbol OOS metrics from the *global* model (feeds the existing confidence
  policy / `GET /analysis/{symbol}` unchanged).
- Reuse: `FeaturePipeline` for features, `DataConfig.target_horizon` (h=5),
  existing scaling utilities.

**Non-Goals (this phase)**
- The global model architecture/training itself (Faz 3).
- Per-symbol fine-tune (Faz 4).
- Horizon-aware backtest/forecast (Faz 1b).
- Cross-sectional ranking / portfolio (out of scope for the per-symbol product).

## Why the existing splitter is not enough

`TimeSeriesSplitter.walk_forward_splits` (`src/utils/data_splitter.py`) is
**row-indexed and single-symbol**: `test_start = n - i*test_size` slices by row
position. On an interleaved panel that mixes symbols and dates, row slicing
would put the same calendar date in train for one symbol and test for another →
massive cross-symbol leakage. Faz 2 needs a **calendar-date-based** panel
splitter. The embargo concept is reused; the slicing is rewritten.

## Leakage taxonomy (what we must prevent)

1. **Cross-symbol same-date leak.** Date *d* appearing in train for symbol A and
   test for symbol B. → Prevented by splitting on the **global date axis**: every
   symbol shares the same train/test date boundary.
2. **Symbol-future (horizon) leak.** A train row at date *t* whose target uses
   `close[t+h]` reaches into the test window. → Prevented by **purge + embargo**:
   train keeps only dates `t` with `t + h < test_start − buffer`.
3. **Feature-lookahead leak.** A feature at *t* using data from `> t`. → Already
   prevented by `FeaturePipeline` (causal features); re-asserted by tests.

Survivorship is handled naturally: delisted symbols contribute rows only up to
their last trading date; a date-based split never invents post-delisting rows.
The loader **must include** delisted/stale symbols' history (incl. the 7 no-data
symbols' retained old CSVs) so the model is not survivor-biased.

## Pooled data model (long panel)

```text
columns:
  symbol            str      (e.g. "EREGL")
  Date              datetime (ISO, uniform after Faz 0.5)
  <feature cols>    float    (FeaturePipeline stationary features, per symbol)
  target            float    (log(close[t+h]/close[t]); h from DataConfig)
  sector            category (GICS, from bist_universe.csv; "Unknown" allowed)
  liq_bucket        int      (turnover decile, dynamic-but-causal; see below)
  vol_bucket        int      (trailing realized-vol decile)
  symbol_id         int      (stable code for optional NN embedding)
row identity: (symbol, Date) unique
```

- Features are engineered **per symbol** (same as today) then concatenated; no
  cross-symbol contamination at feature time.
- `target` reuses `build_target_series` semantics (h-day forward), computed
  per symbol; last `h` rows per symbol have NaN target → dropped.
- Conditioning columns are **causal**: `sector`/`symbol_id` are static;
  `liq_bucket`/`vol_bucket` use trailing windows only. Market-cap is not in the
  data → use **median TL turnover (close×volume)** over a trailing window as the
  liquidity/size proxy (documented limitation).

## Module plan

```text
src/data/pooled_loader.py
    @dataclass PooledLoaderConfig:
        data_dir: str
        universe_file: str
        target_horizon: int = 5
        feature_mode: str = "stationary_features"
        use_macro: bool = False          # Faz 2: keep simple; macro = Faz 3 opt-in
        min_rows: int = 60               # below -> cold-start only (still loaded)
        include_delisted: bool = True
        liq_lookback: int = 63
        vol_lookback: int = 63

    class PooledPanelLoader:
        def load(self) -> pd.DataFrame            # the long panel above
        def _engineer_symbol(self, sym) -> DataFrame | None
        def _add_conditioning(self, panel) -> DataFrame
        # robust: skips unreadable CSVs, logs a per-symbol report

src/validation/pooled_cv.py
    @dataclass PooledCVConfig:
        target_horizon: int = 5
        embargo_buffer: int = 5          # extra days on top of h
        window_len: int = 63             # ~3 trading months per test window
        n_windows: int = 6               # rolling OOS windows
        step: int | None = None          # default = window_len (non-overlapping)
        max_train_days: int | None = None  # None=expanding, int=sliding
        min_train_days: int = 504
        final_holdout: bool = True       # reserve newest window, never selected on

    @dataclass PooledFold:
        index: int
        train_mask: np.ndarray
        test_mask: np.ndarray
        test_date_start: Timestamp
        test_date_end: Timestamp
        embargo_days: int
        is_final_holdout: bool

    class PooledPurgedWalkForward:
        def split(self, panel: DataFrame) -> list[PooledFold]
```

## CV algorithm (date-based, purged, multi-window)

```
D = sorted(unique(panel.Date))
embargo E = target_horizon + embargo_buffer        # purge horizon overlap + buffer
reserve the newest `window_len` dates as final_holdout (if enabled)
choose n_windows test windows walking backward from the end of the selectable
range, each of `window_len` consecutive trading dates, stepped by `step`:

for window k with test dates [a_k, b_k]:
    test_mask  = panel.Date in [a_k, b_k]
    cutoff     = a_k shifted back by E trading days
    train_mask = panel.Date <= cutoff
                 AND (max_train_days is None
                      OR panel.Date >= cutoff - max_train_days)   # sliding
    drop folds whose train date span < min_train_days
```

Properties:
- **No cross-symbol leak** — boundary is a single global date `a_k`.
- **No horizon leak** — `E ≥ h` guarantees every train row's `t+h < a_k`.
- **Multi-window** — K folds → per-fold OOS metrics → distribution
  (`rolling_positive_window_ratio`, median, IQR — the inputs the existing
  [Confidence and Risk Policy](confidence-and-risk-policy.md) already consumes).
- **Final holdout** — newest window reserved; never used to pick the model
  (champion/challenger promotion uses fold distribution, not the holdout).

## Scaling policy

- Fit scalers on the **pooled train rows of the current fold only** (train-only
  scope invariant preserved).
- Default: global robust-X / standard-Y (matches existing `scaling_mode`).
- Option (flag): **cross-sectional per-date standardization** of features
  (z-score within each date across symbols) — strong for panels (removes
  market-wide regime level), evaluated in Faz 3. Causal: uses only that date's
  cross-section, no future.

## Metric aggregation (per-symbol from a global model)

For each fold, predict the test rows, then group test rows **by symbol** and
compute Dir_Acc / RMSE / MAE / edge-over-base-rate per symbol per fold. Aggregate
across folds → per-symbol OOS distribution. This is what populates
`performance.*` and the confidence inputs for `GET /analysis/{symbol}`; the API
contract does not change. Backtest/Sharpe stays out (Faz 1b horizon-aware
prerequisite).

## Test plan (`tests/test_pooled_cv.py`, `tests/test_pooled_loader.py`)

- **Leak — cross-symbol:** for every fold, `max(train.Date) + E_days < min(test.Date)`.
- **Leak — horizon:** no train row `t` with `t + h >= a_k`.
- **No overlap:** train_mask & test_mask disjoint; folds' test windows disjoint
  (when step = window_len).
- **Final holdout:** newest window flagged, excluded from the selectable folds.
- **Determinism:** same panel → same folds.
- **Loader:** panel `(symbol, Date)` unique; delisted symbol present with only
  pre-delist dates; thin symbol kept; NaN target rows dropped; conditioning
  columns causal (no future).
- **Survivorship:** a synthetic delisted symbol contributes to early folds'
  train, absent from post-delist test.

## Acceptance criteria

- Pooled panel builds over the real `data/` universe (≈585 symbols, >1M rows)
  without cross-symbol/horizon/feature leaks (tests green).
- `PooledPurgedWalkForward.split` yields K deterministic folds + 1 final holdout,
  each leak-asserted.
- Per-symbol OOS metrics computable from a dummy global model end-to-end.
- No change to `GET /analysis/{symbol}` schema or existing suite.

## Open questions (need a decision)

1. **Window length / count:** `window_len=63` (≈3m) × `n_windows=6` ≈ 1.5y OOS.
   Bigger windows = more stable, fewer folds. Acceptable default?
2. **Expanding vs sliding train:** expanding (all history) vs `max_train_days`
   (regime-local). Default expanding for Faz 2?
3. **Liquidity/size proxy:** TL turnover trailing median as cap proxy — OK, or
   source a real market-cap/free-float feed later?
4. **Cross-sectional standardization:** default off in Faz 2 (measure in Faz 3),
   or on from the start?
5. **Delisted inclusion:** include all retained-history symbols (survivorship-
   correct) — confirm we want delisted names in the training pool.

## Related Pages

- [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md)
- [Validation and Backtesting](validation-and-backtesting.md)
- [Data Pipeline](data-pipeline.md)
- [Confidence and Risk Policy](confidence-and-risk-policy.md)
