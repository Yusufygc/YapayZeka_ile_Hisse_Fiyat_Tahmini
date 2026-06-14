---
title: Refactor Analysis 2026-06-12
type: audit
status: active
last_updated: 2026-06-14
owner: llm
source_count: 21
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
- **The largest remaining code-quality risk is now memory/performance scale, not
  owner-forward magic.** Evaluation services, ForecastRunner, all four
  DataManager services, and all six training/evaluation workflows now use
  explicit context/state DI. P3 also slimmed the backtest engine behind explicit
  data contracts without changing validation semantics.
- **Pooled serving needs stricter contract guards.** `EnsemblePooledModel.predict`
  assumes one cross-section/date. That is correct for nightly serving, but the
  model API should fail loudly if multi-date batches are accidentally passed.
- **Memory pressure is reduced on the pooled model path.** P4-A moved
  `TorchMLPModel` away from full normalized Torch tensor allocation toward
  float32 mini-batch fit/predict. P4-B then moved pooled OOS, serving scoring,
  and pooled model wrappers onto shared contiguous `float32` matrix helpers.

## Critical Refactor Priorities

## Implementation Update 2026-06-14

- **P0 guard dalgasi tamamlandi:** Kol-A ana `ForecastingPipeline` akisi artik
  `target_horizon > 1` degerini production backtest/forecast semantigi
  tamamlanana kadar fail-loud engeller. `src.cli.forecast --horizon-days` bu
  guard'dan etkilenmez; o parametre forward forecast ufkudur, `DataConfig.target_horizon`
  degildir.
- **Tensor tarih metadatasi h-aware oldu:** `TensorPreparationService` icinde
  `dates_train`, `dates_prediction` ve `dates_test` artik `h` ufkuna gore
  hizalanir. `h=1` davranisi korunur; `h>1` icin helper/research seviyesinde
  tarih metadatasi dogrudur fakat production backtest hala guard altindadir.
- **P1 serving contract guard eklendi:** `score_latest_universe()` default
  strict modda multi-date paneli reddeder; eski latest-date secimi yalniz
  `PeerScoringConfig(strict_single_date=False)` ile acilir. `EnsemblePooledModel.predict()`
  2D input ve en az iki satirli cross-section bekler.
- **P2-A DataManager DI dilimi tamamlandi:** `TensorPreparationService` ve
  `ValidationSplitService` artik `_OwnerBackedService` mirasi kullanmaz; explicit
  `DataManagerContext` + `DataManagerState` ile calisir. `DataManager` public
  attribute kontrati property alias'larla korunur. Ingestion, data-quality,
  training workflows ve evaluation workflows owner-backed olarak bilincli sekilde
  P2-B/P2-C'ye ertelendi.
- **P2-B DataManager ingestion/data-quality DI dilimi tamamlandi:** `DataIngestionService`
  ve `DataQualityReportingService` de owner-forward mirasindan cikarildi. `DataManagerContext`
  stock/project/macro/universe alanlarini, `DataManagerState` dataset/report
  alanlarini tasir; `DataManager` public attribute kontrati property alias'larla
  korunur. `src/pipeline/data_services.py` artik `_OwnerBackedService` import
  etmez.
- **P2-C workflow DI dilimi tamamlandi:** 3 training workflow ve 3 evaluation
  workflow explicit context/state/services DI sozlesmesine tasindi. `_OwnerBackedService`
  tamamen silindi; kalan owner-forward kapsam 0 siniftir.
- **P3 backtest slimming tamamlandi:** `run_backtest()` public API'si korunarak
  hizalama, realized/observed return, execution/cost ve bos sonuc sozlesmeleri
  `src/backtesting/contracts.py` icindeki test edilebilir data-contract
  helper'larina tasindi. Default `simple` long/flat semantigi, rapor kolonlari,
  maliyet hesaplari ve leakage guard davranisi degismedi.
- **P4-A Torch MLP batching tamamlandi:** `TorchMLPModel.fit()` ve `predict()`
  artik full standardized matrix/full Torch tensor olusturmak yerine contiguous
  `float32` NumPy + `TensorDataset/DataLoader` batch akisini kullanir. Train-only
  `mu/sd` `float32` saklanir; embedding mimarisi, seed determinism ve
  `EnsemblePooledModel` public kontrati degismedi.
- **P4-B pooled matrix downcast tamamlandi:** pooled OOS ve serving scoring
  akislari `src/data/pooled_matrix.py` uzerinden fold/latest-slice bazli
  contiguous `float32` feature/target matrix uretir. `GlobalPooledModel` ve
  `EnsemblePooledModel` artik bu input'u tekrar `float64`'a upcast etmez.
- **Kalan is:** P0'un tam h-aware backtest/forecast semantigi, P1 ensemble-OOS
  calibration ve daha genis panel assembly/profiling temizligi henuz yapilmadi.

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

- **Completed 2026-06-14:** all four `DataManager` services and all six
  training/evaluation workflows now use explicit context/state DI.
- `_OwnerBackedService` was deleted from `evaluation_services.py`; future cleanup
  should focus on shrinking orchestration/backtest functions rather than
  forwarding removal.

### P3 - Backtest and Signal Engine Slimming

- **Completed 2026-06-14:** `run_backtest` is now a thin orchestrator. Input
  tail-alignment, default prediction-date handling, realized/observed return
  calculation, execution/cost arrays, and empty-result schema are isolated in
  `src/backtesting/contracts.py`.
- Default `simple` long/flat semantics, `legacy`/`professional` signal modes,
  trade/equity output shape, fold propagation, transaction-cost columns, and
  order-report date columns are unchanged.
- Regression coverage lives in `tests/test_backtest_engine_contract.py` plus the
  existing phase/leakage/report suites.

### P4 - Memory and Performance

- **P4-A completed 2026-06-14:** `TorchMLPModel.fit` and `predict` now use
  contiguous `float32` input arrays plus `TensorDataset` / `DataLoader`
  mini-batches. Numeric features are standardized per batch from train-only
  `mu/sd`; categorical id columns remain raw for embeddings.
- Public sklearn-like model API, seed determinism, multi-seed ensemble blend,
  and serving cross-section contract are unchanged.
- **P4-B completed 2026-06-14:** pooled OOS and nightly scoring now build
  contiguous `float32` matrices through `src/data/pooled_matrix.py`. Fold-level
  train/test matrices are created only for the active fold, and serving latest
  cross-section scoring uses the same helper.
- `GlobalPooledModel` and `EnsemblePooledModel` consume `float32` matrices
  without re-upcasting to `float64`; public prediction outputs, peer scoring
  schema, XAI behavior, and rank-blend semantics are unchanged.

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
