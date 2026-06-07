---
title: Validation and Backtesting
type: concept
status: active
last_updated: 2026-05-25
owner: llm
source_count: 8
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

## PurgedKFold and CPCV (Sprint 3 — 2026-05-25)

Opt-in validators that augment walk-forward by tightening leakage control
and producing statistical confidence intervals on Sharpe.

### PurgedKFold (`src/validation/purged_kfold.py`)

Per López de Prado, *Advances in Financial Machine Learning* Ch. 7.
Classical KFold is unsafe for time series (test fold may sit in the middle
of train; rolling features see test data). PurgedKFold:

- Folds are produced in chronological order — no shuffle.
- `purge_window` train samples around each test fold are dropped (default
  recommendation `max(rolling_feature_window, time_steps)` ≈ `max(200, time_steps)`).
- `embargo` additional train samples after the test fold are dropped
  (overlapping label leakage for `h>1` forward-return targets).

Constructor:

```python
PurgedKFold(n_splits=5, purge_window=200, embargo=10)
```

Returns an iterator of `(train_idx, test_idx)` numpy arrays.

### Combinatorial Purged CV (`src/validation/cpcv.py`)

Per López de Prado AFML Ch. 12. Splits data into `n_groups` chronological
groups, then for every `C(n_groups, k_test)` combination uses `k_test` groups
as test and the rest as train (with purge + embargo around each test group).

Defaults exposed via `ValidationConfig`:

```python
use_purged_kfold = False
use_cpcv         = False
cpcv_n_groups    = 6     # C(6, 2) = 15 paths
cpcv_k_test      = 2
```

`C(6, 2) = 15` paths means each backtest produces 15 different OOS return
series → empirical 95% CI on Sharpe via bootstrap (see Concat-Sharpe below).

### Concat-Sharpe Aggregation (Sprint 3 A3.3)

`WalkForwardValidator.run()` now collects per-fold strategy returns and
appends three new keys to `aggregated_metrics`:

| Field | Meaning |
|---|---|
| `Sharpe_Concat` | Sharpe computed on concatenated daily strategy returns across all folds (not fold-mean Sharpe — statistically correct) |
| `Sharpe_CI_95_Low` | Bootstrap (1000 resamples) lower bound at 95% CI |
| `Sharpe_CI_95_High` | Bootstrap upper bound at 95% CI |
| `Concat_Returns_N` | Number of concatenated daily returns |

If `risk_free_unavailable` (Sprint 1 A1.1), all four collapse to `NaN`
and confidence label is degraded one level via the existing
`risk_free_unavailable` rule in
[Confidence and Risk Policy](confidence-and-risk-policy.md).

Bootstrap parameters (`_BOOTSTRAP_RESAMPLES = 1000`,
`_BOOTSTRAP_SEED = 20260525`) are module-level constants in
`src/validation/walk_forward.py` and deterministic across runs.

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
- Sharpe-like financial metrics (Sharpe / Sortino / Deflated Sharpe / Calmar)
- Benchmark-relative fields such as RMSE vs benchmark

Quantile-capable models can also produce interval/quantile metrics.

### Metric Priority for Advisory (Sprint 1 — 2026-05-25)

Because the product is an advisory/recommendation service (no automated
trading) and the default backtest uses `commission_bps=0` /
`slippage_bps=0`, monetary fields (`Net_Return`, `BuyHold_Return`,
`Profit_TL`, `End_Capital`) are **footnotes only**. They test "would the
direction have been right?" — not "what is the realized P&L?"

`save_metrics_report` (`src/evaluation/evaluator.py`) orders columns as:

1. **Advisory primary**: `Dir_Acc`, `Hit_Rate`, `DirAcc_vs_benchmark`,
   `Composite_Score`.
2. **Error metrics**: `RMSE_vs_benchmark`, `RMSE`, `MAE`, `Return_RMSE`,
   `Return_MAE`, `RMSE_vs_zero_return`.
3. **Risk-adjusted**: `Calmar`, `Deflated_Sharpe`,
   `Sharpe_Probabilistic_Score`, `Sharpe`, `BuyHold_Sharpe`,
   `Sharpe_excess_vs_buy_hold`, `Risk_Free_Unavailable`,
   `Sharpe_Warning`.
4. **Probabilistic**: `Pinball_Loss`, `P10_P90_Coverage`,
   `Avg_Interval_Width`, `Winkler_Score`.
5. **Benchmark flags**: `Benchmark_Model`, `Beats_*`,
   `Eligible_For_Leader`, `Neutral_Rate`, `MAPE`.
6. **Raw returns (footnote)**: `Net_Return`, `BuyHold_Return`,
   `RMSE_Fark_Delta`, `RMSE_Fark_Yuzde`.

### Risk-Free Rate Fail-Loud (Sprint 1 A1.1)

`src/utils/risk_free_rate.get_current_risk_free_rate` no longer falls
back to the legacy `0.40` constant. Priority:

1. `data/macro/INTEREST_RATE.csv` (cache from `MacroPipeline`).
2. `RISK_FREE_RATE_ANNUAL` environment variable.
3. Explicit `fallback=` argument (default `None`).

When none resolves the function returns `None`. Downstream callers
(`compute_financial_metrics`, `summarize_backtest`) then:

- Set `Sharpe`, `Sortino`, `BuyHold_Sharpe`, `Deflated_Sharpe` to `NaN`.
- Add `Risk_Free_Unavailable=True` and
  `Sharpe_Warning="risk_free_unavailable"` to the metric dict.

The flag flows into the analysis API confidence chain in Sprint 8 (see
[Confidence and Risk Policy](confidence-and-risk-policy.md) — Soft
Degradations).

### Backtest Cost Disclaimer (Sprint 1 A1.3)

`save_metrics_report` and the CLI summary block automatically prepend:

> ⚠ Backtest sonuçları işlem maliyeti (commission/slippage) İÇERMEZ.
> ⚠ Bu çıktı kişisel yatırım tavsiyesi değildir; nihai karar
>   kullanıcıya aittir.

`src/api/constants.INVESTMENT_DISCLAIMER` carries the same message into
the analysis API response.

## Composite Score

`compute_composite_score` (in `src/database/stock_model_db.py`) computes a
0-100 score. Sprint 1 A1.4 reweighted the formula for advisory use:

| Bileşen | Ağırlık | Açıklama |
|---|---|---|
| `RMSE_vs_benchmark` | 0.30 | Hata-bazlı temel skor (was 0.45) |
| `DirAcc_vs_benchmark` | 0.25 | Göreli yön üstünlüğü |
| `Dir_Acc` (raw) | 0.20 | Mutlak yön doğruluğu (was 0.10) |
| `Hit_Rate` (raw) | 0.15 | Kazançlı işlem oranı (yeni) |
| `Sharpe_excess_vs_buy_hold` | 0.10 | rf-bağımlı; NaN ise nötr 50 puan (was 0.20) |

- `Net_Return` formüle dahil **değildir** (advisory için yatırımsal
  yorumlanmamalı — cost=0).
- `Neutral_Rate > 0` ise küçük ceza uygulanır.
- `RMSE_vs_benchmark > 1.0` veya `Eligible_For_Leader=False` ise skor
  49.0 ile sertçe kapatılır (ineligible model lider olamaz).
- Sharpe NaN olduğunda formül crash etmez; `sharpe_relative_score=50`
  (nötr) alınır.

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

## Pooled Cross-Sectional Validation (E2)

The E2 pooled global model uses a different validation lens from the per-symbol
splits above — cross-sectional, not absolute per-symbol:

- **Group-purged date-based walk-forward** (`src/validation/pooled_cv.py`,
  `PooledPurgedWalkForward`): splits are on **dates** across all symbols at once;
  no cross-symbol same-date leakage, and the horizon is purged via
  `target_date < window_start`.
- **Per-symbol OOS aggregation** (`src/validation/pooled_oos.py`): pools fold
  predictions, then reports the **daily cross-sectional IC** — within each date,
  `spearman(pred, true)` across symbols — and **ICIR = mean/std**. Overlapping
  h-day target windows inflate ICIR (autocorrelation), so
  `daily_cross_sectional_ic(..., sample_gap_days=h)` sub-samples non-overlapping
  dates for an honest ICIR.
- **Segment stratification** (`src/validation/segment_ic.py`): IC/ICIR per
  liquidity/volatility/sector bucket, feeding the serving confidence.
- **Leak guard**: any `target*` column is excluded from features (regression test
  `test_target_variants_never_become_features`).

Current full-universe result: IC ≈ 0.099, ICIR ≈ 1.55. See
[E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md) and
[E2 Faz 2 Pooled CV Design](e2-faz2-pooled-cv-design.md).

## Olasılıksal Forward Interval (Kademe B2 → C)

Forward forecast (`ForecastRunner`) tek-nokta (p50) yerine **olasılıksal aralık**
üretebilir: "p10'a iner, p90'a çıkar, p50 merkez". Aralık **model-agnostik**'tir
(sadece quantile model değil, her üretim modeli için). İki kademe `src/forecasting/
interval_calibration.py` içinde:

### Kademe B2 — residual band (parametrik baseline)

- **Kaynak**: walk-forward fold residual'ları (target uzayı),
  `residual = y_true_target − y_pred_target`. Final holdout KULLANILMAZ
  (out-of-sample sızıntısız). `compute_residual_calibration(fold_records)`.
- **σ**: örneklem standart sapması (ddof=1); opsiyonel **rejim-koşullu σ**
  (`sigma_by_regime`, walk-forward `market_regime` etiketinden) → volatil rejimde
  geniş, sakin rejimde dar band (heteroskedastisiteyi kısmen yakalar).
- **Band**: `p50 ± z·σ·√h` (`residual_band`). z iki-taraflı normal (0.8→1.28,
  0.9→1.64, 0.95→1.96); `√h` random-walk horizon ölçeği.
- **Varsayım**: yaklaşık normal + homoskedastik → kapsama **garantisi yok**, ampirik
  ölçülür. Fat-tail'i kısmen kaçırır. Hızlı, açıklaması kolay → tez baseline'ı.

### Kademe C — conformal (dağılımdan bağımsız, kapsama-garantili)

- **Split-conformal**: nonconformity skoru `s_i = |y_true − y_pred|`; ampirik
  quantile `q̂ = quantile(scores, ceil((n+1)·level)/n)` (sonlu-örneklem kapsama
  garantisi). `compute_conformal_calibration(fold_records, level=0.9)`.
- **Band**: `p50 ± q̂` (`conformal_band`; opsiyonel `√h` horizon ölçeği).
- **ACI-lite** (`adaptive_conformal_update`): zaman-serisi exchangeability kırılımı
  için `q̂`'i kayan kapsama sapmasına göre online ayarlar
  (`q_new = q̂·(1 + γ·(target − recent))`). Kapsama düşükse band genişler.
- **Üstünlük**: dağılım varsayımı yok → fat-tail'i doğal kapsar, **teorik + sonlu
  örneklem coverage garantisi** (tezde en güçlü iddia).

### Kalibrasyon depolama & serving

- Eğitimde `_build_interval_calibration` (`src/pipeline/evaluation_workflows.py`)
  WF residual'larından B2 + C'yi birlikte hesaplar; **model yanı sidecar artifact**
  `<model>.interval_calib.json`'a yazılır (`save_forecast_artifact_package`,
  opsiyonel). Serve'de yeniden eğitim yok — `load_forecast_artifact_package`
  metadata'ya `interval_calibration` olarak gömer (yoksa None → interval atlanır).
- `roll_forward_recursive` quantile dalı yoksa ve kalibrasyon varsa model-agnostik
  dalı çağırır (`_apply_model_agnostic_interval`): band hedef uzayında kurulur →
  `target_to_price` → BIST clip → `p10_close/p50_close/p90_close` +
  `predicted_return_p*` + `interval_method` (quantile_model | residual_b2 |
  conformal). Aktif üreteç `model_settings["interval_method"]` ile seçilir
  (`resolve_active_calibration`); varsayılan `residual_b2`.

### Coverage backtest (kalibrasyon kanıtı)

- `forecast_resolution.py` resolve sırasında `actual_close ∈ [p10, p90]` oranını
  hesaplar → `forecast_accuracy_summary.interval_coverage` (%) +
  `interval_avg_width`. Ampirik kapsama = tez metriği.
- `tools/interval_coverage_report.py` üreteç (B2 vs conformal) bazında ortalama
  kapsama + band genişliği tablolar → **tez "naive band %X vs conformal %Y, hedef
  %90" karşılaştırma tablosu**. B2 = baseline/ablation, C = aday.
- Yorum sınırı: aralık göreli/koşullu fiyat aralığıdır, kesin değer değil;
  dar band + hedefe yakın kapsama = iyi kalibre + bilgili.

### Ensemble forecast interval + kalibrasyon akışı

- **Kalibrasyon kaynağı (kritik akış):** interval_calib sidecar'ı final-holdout
  artifact yazımında (`_build_interval_calibration`, `evaluation_workflows.py`)
  walk-forward residual'larından üretilir. Residual'lar `wf_backtest_inputs`'ta
  (ModelTrainer) tutulur; `evaluate_walk_forward` bunu owner'a (EvaluationManager)
  stash eder ki FinalHoldout workflow okuyabilsin. Bu stash olmadan kalibrasyon
  None döner ve sidecar yazılmaz (tek-model forecast'ta p10/p90 boş kalır).
- **Ensemble bacağı:** ensemble forecast (`combine_member_points`) üye p10/p90'larını
  ağırlıklı birleştirir (mevcut üyeler üzerinde renormalize); p50 = ensemble bounded
  close, band p50 etrafında clamp. Karışık yöntemde `interval_method="ensemble"`;
  hiç üyede interval yoksa no-op. Üye interval'i ancak o üyenin artifact'ında
  interval_calib sidecar varsa dolar.
- **Doğrulanmış uçtan uca (GARAN, Ridge Return):** σ=0.0256 (log-getiri),
  rejim-koşullu (bear 0.0278 / bull 0.0247), conformal q̂=0.0392, 240 residual.
  Forecast h1 `[125.10, 133.60]` → h5 `[119.30, 138.10]` (√h ile genişler).

Testler: `tests/test_interval_calibration.py` (σ/q̂/band/ACI/validasyon),
`tests/test_forecast_interval_persist.py` (roll_forward dalı + ensemble kombinasyon +
persistence round-trip + coverage + rapor).

## Related Pages

- [Data Pipeline](data-pipeline.md)
- [Model Catalog](model-catalog.md)
- [Testing and Quality](testing-and-quality.md)
- [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md)

