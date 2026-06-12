---
title: Refactor Analysis 2026-06-12
type: audit
status: active
last_updated: 2026-06-12
owner: llm
source_count: 18
---

# Refactor Analysis 2026-06-12

This page records the source-grounded refactor and quantitative-finance audit
performed on 2026-06-12. It complements [Staged Refactor Plan](refactor-plan.md)
and updates the risk picture after E1/E2 work.

## Executive Findings

- **Leakage controls are mostly strong on the default path.** Chronological
  splitting, train-only scaler fitting, walk-forward embargo, final-holdout
  exclusion, and pooled date-based purge are explicit in source and tests.
- **The main leakage-adjacent gap is horizon awareness.** `DataConfig.target_horizon`
  documents that `h > 1` backtest/forecast semantics are not production-ready.
  Source confirms target arrays use `h`, but `dates_prediction` / `dates_test`
  in `TensorPreparationService.prepare_tensors` still use one-step slices. Keep
  `target_horizon=1` for Kol-A production until horizon-aware evaluation is
  implemented.
- **Modeling direction is correct: Kol-A and Kol-B solve different problems.**
  Kol-A remains per-symbol absolute forecasting; Kol-B is pooled cross-sectional
  ranking with IC/ICIR as the correct validation lens. The two outputs should
  stay separate in API language and confidence handling.
- **The largest remaining code-quality risk is stateful orchestration, not a
  simple missing helper.** Evaluation services and `ForecastRunner` moved to DI,
  but training/evaluation/DataManager workflows still inherit
  `_OwnerBackedService` and write owner state. Fail-loud guards reduce typo risk,
  but SRP/test isolation is still weaker than explicit state-passing.
- **Pooled serving needs stricter contract guards.** `EnsemblePooledModel.predict`
  assumes one cross-section/date. That is correct for nightly serving, but the
  model API should fail loudly if multi-date batches are accidentally passed.
- **Memory pressure exists in pooled deep training.** `TorchMLPModel.fit` converts
  the full feature matrix to one Torch tensor, and ensemble training repeats this
  for multiple seeds. This is acceptable for offline nightly runs today, but it
  should be moved toward dataset/dataloader batching if the universe or features
  grow.

## Critical Refactor Priorities

### P0 - Horizon-Aware Alignment

Fix Kol-A `target_horizon > 1` before interpreting any multi-day Sharpe,
backtest, or forecast output:

- Change `dates_train`, `dates_prediction`, and `dates_test` alignment from
  one-step slices to `h`-aware slices.
- Add regression tests for `h=5` covering tensor lengths, dates, previous close,
  prediction date, realized target date, and backtest order dates.
- Add a runtime guard that blocks `target_horizon > 1` production backtests until
  signal execution semantics are explicitly defined.

### P1 - Validation and Calibration Consistency

- For pooled serving, compute OOS IC/ICIR for the same model family used in
  final nightly scoring. Current documentation notes segment ICIR/confidence is
  LGB-only while serving may score with the LGB+MLP ensemble; this is fast and
  deterministic, but it creates a calibration mismatch.
- Add a cheap ensemble-OOS calibration mode or record the mismatch explicitly in
  `global_model_runs.config` and API caveats.
- Use non-overlapping IC sampling (`sample_gap_days ~= target_horizon`) wherever
  ICIR is reported as a model-quality number.

### P2 - Stateful Workflow Decomposition

- Convert the remaining `_OwnerBackedService` consumers in
  `evaluation_workflows.py`, `training_workflows.py`, and `data_services.py` to
  explicit context/state objects in small slices.
- Prioritize `TensorPreparationService` and `ValidationSplitService` because
  they own leakage and tensor contracts.
- Keep characterization tests ahead of each migration; do not rewrite all
  workflows in one PR.

### P3 - Backtest and Signal Engine Slimming

- Split `run_backtest` into a thin orchestrator plus data-contract helpers for
  alignment, signal frame, execution arrays, cost arrays, trade rows, and equity
  output.
- Keep default `simple` long/flat semantics unchanged.
- Add tests that prove `prediction_date`, `execution date`, and realized return
  use the intended horizon.

### P4 - Memory and Performance

- Move `TorchMLPModel.fit` to `TensorDataset` / `DataLoader` batching rather than
  materializing the entire matrix as one tensor.
- Downcast dense pooled numeric features to `float32` after train-only
  preprocessing where precision loss is acceptable.
- Avoid repeated dataframe copies in pooled feature assembly and nightly scoring
  unless mutation isolation is required.

## Recommended Execution Order

1. **Horizon safety patch:** fix or guard `target_horizon > 1` on Kol-A.
2. **Serving model contract patch:** fail loudly when pooled ensemble predict is
   called on multiple dates or an undersized cross-section.
3. **Ensemble OOS calibration:** align peer confidence calibration with the
   production scorer or persist a clear LGB-only calibration caveat.
4. **Workflow DI slice:** convert tensor preparation to explicit context/state.
5. **Backtest slimming:** make date/return alignment separately testable.
6. **Pooled memory work:** introduce minibatched Torch training and float32
   feature matrices for nightly scale.

## Source Anchors

- `src/pipeline/data_services.py`: target construction, tensor/date alignment,
  split/final-holdout handling.
- `src/data/preprocessor.py`: train-only scaler fitting and clipping.
- `src/utils/data_splitter.py`: chronological walk-forward splits and embargo.
- `src/validation/pooled_cv.py`: date-based pooled purge.
- `src/data/cross_sectional.py`: within-date rank target and cross-sectional
  features.
- `src/models/global_pooled_model.py`: target-column exclusion and LightGBM
  pooled model.
- `src/models/ensemble_pooled_model.py`: one-cross-section rank-blend assumption.
- `src/models/torch_mlp_model.py`: full-matrix Torch training.
- `src/backtesting/engine.py`: long/flat execution and realized-return contract.
- `src/pipeline/signal_calibrator.py`: final-holdout calibration guard.
