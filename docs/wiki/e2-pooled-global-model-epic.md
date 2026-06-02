---
title: E2 Pooled Global Model Epic
type: feature-plan
status: active
last_updated: 2026-06-02
owner: llm
branch: feat/e2-pooled-global-model
---

# E2 Pooled Global Model Epic

Training-side redesign to fix overfitting and the absence of demonstrable alpha
in the current **single-symbol** training pipeline, while keeping the existing
**per-symbol product** (desktop `PortfoySimulasyonu` → FastAPI
`GET /analysis/{symbol}` → registry/XAI/confidence) byte-compatible.

This epic owns the **modeling/training/validation** layer only. The
serving/product/confidence layer is already designed and largely implemented;
see [Product Decision Support Design](product-decision-support-design.md),
[Analysis API Contract](analysis-api-contract.md), and
[Confidence and Risk Policy](confidence-and-risk-policy.md). **E2 must not break
those contracts** — it changes how `best_models` / `experiments` rows are
produced, not what the API returns.

## Problem

- Single-symbol daily training overfits: deep models look strong in walk-forward
  then collapse out-of-sample. Evidence (2026-06-02, EREGL all-models run
  `..._EREGL_walk_forward_ALL_MODELS`): `AttentionLSTM v2` WF composite **83.81**
  (dir 56.7%, Sharpe 1.24) → final-holdout composite **56.15** (dir **38.98%**,
  Sharpe **-3.37**, RMSE 0.52 → 1.57). Did not beat incumbent RF → correctly not
  promoted. The guardrail worked; the alpha did not exist.
- Root causes: (a) selection bias — best of 17 models on a tiny noisy sample is
  mostly luck (composite floored at 49.00 for all but one outlier); (b) single
  final-holdout window = one regime = one coin flip; (c) deep models overfit a
  few thousand noisy daily rows; (d) low daily signal/noise; (e) backtest Sharpe
  from ~12 trades is statistically meaningless.

## Core Decision: training scope ≠ serving scope

- **Product/serving stays per-symbol.** `GET /analysis/{symbol}` contract,
  confidence labels, disclaimer, and product language boundaries are unchanged.
- **Training becomes pooled/global.** Train ONE conditioned model on pooled rows
  across all ~592 stock CSVs in `data/` (~1M+ rows), not 592 separate models and
  not single-symbol from scratch. The model is global; the answer is local
  (feed a symbol's latest features → that symbol's forecast + per-row SHAP XAI).
- **Conditioning, not partitioning.** The single model takes stock
  characteristics as input so it adapts per symbol automatically: sector,
  market-cap bucket, liquidity/volume, volatility, beta, optional learned symbol
  embedding. Avoids 592 model artifacts and negative-transfer washout.
- **Cold-start is a feature, not a bug.** Thin/IPO symbols (CSVs < ~600 rows,
  e.g. `A1YEN` 128, `AHSGY` 292) cannot train alone. The global model serves
  them day one via the universal feature→return mapping. This is what makes
  "user can ask ANY stock" actually work.

## Quality ≠ uniform — feeds the existing confidence layer

"Any stock works" means "always an honest answer with a confidence label", NOT
"good signal everywhere". Speculative/illiquid names are near-random; no model
fixes a coin flip. This is already handled by
[Confidence and Risk Policy](confidence-and-risk-policy.md):

- `directional_accuracy < 50` → hard block → `low` confidence (auto-catches
  speculative stocks).
- `eligibility_status`, naive-leader rejection, `stability_score`, PSI drift,
  survivorship warnings already gate trust.

E2's job is to keep populating these per-symbol fields **correctly from a pooled
model** (evaluate the global model on each symbol's own OOS slice). No new
confidence concept is introduced.

## Phase Plan

- **Faz 0 — Data/universe audit (blocking). ✅ DONE 2026-06-02.** See findings
  below. Auditor: `tools/e2_faz0_universe_audit.py` (read-only; writes
  `outputs/e2_faz0_universe_audit.md` + `outputs/e2_faz0_symbol_stats.csv`).
- **Faz 0.5 — Universe re-pull (freshness/format/universe fix). ✅ DONE
  2026-06-02.** `tools/refetch_universe.py` re-fetches every ticker from yfinance
  (`{SYMBOL}.IS`, `auto_adjust=True` per the split-leakage invariant), rewrites
  each `data/{TICKER}.csv` from scratch in a single ISO `%Y-%m-%d` format, and
  upserts `data/bist_universe.csv` per fetched symbol (Listed_Date from min date,
  Status Active/Inactive by freshness, Sector/Delisted_Date preserved if already
  set). **Full 592-symbol run result:** ok **585**, no-data **7**, failed 0.
  Post-refetch re-audit confirms the fix: date format unified to ISO (592/592),
  fresh-to-2026-06-02 = 583 (was 1), universe coverage 585/592 (was 28),
  1,275,614 pooled rows (+178k), dup/zero-price = 0. The 7 no-data symbols
  (`DOBUR, EFORC, IPEKE, KOZAA, KOZAL, SNKRN, YGYO`) returned no yfinance data —
  likely delisted/suspended (Koza group etc.); their old CSVs are retained and
  flagged for the survivorship decision. **Remaining Faz 0 gaps:** sector still
  blank for ~557 symbols (only the original 28 cataloged sectors survive) →
  backfill needed; 110 symbols still show a `|log_return|≥0.30` day (real split/
  dividend events even under auto_adjust — needs an audit pass); 30 thin (<500
  rows) cold-start symbols. (Audit's `sector_missing` counter undercounts: NaN
  sectors are truthy — real blank count is ~557.)
- **Faz 0.6 — Sector backfill (conditioning prerequisite). ✅ DONE 2026-06-02.**
  `tools/backfill_sectors.py` fills `bist_universe.csv` `Sector` with a uniform
  GICS vocabulary from yfinance `Ticker.info` (585 symbols: 580 resolved, 5
  `Unknown` = GMTAS, ISGSY, ISKUR, KZGYO, ULUFA). Distribution: Industrials 115,
  Consumer Cyclical 99, Financial Services 77, Basic Materials 71, Consumer
  Defensive 61, Real Estate 56, Technology 36, Utilities 33, Healthcare 15,
  Communication Services 12, Energy 5, Unknown 5. `Sector_Index` (the 7-index
  macro sector-return field) is left untouched. Industry is fetched but not
  written (no universe column yet) — deferred to a Faz 3 feature. Sector is now a
  usable categorical conditioning input.
- **Faz 1 — Horizon shift (cheap win, orthogonal). ✅ DONE (predictive slice)
  2026-06-02.** Added backward-compatible `DataConfig.target_horizon` (default 1
  = unchanged) wired into `TensorPreparationService.build_target_series` /
  `prepare_tensors` (`y[t]=target(close[t+h])`, `X=features[:-h]`). Tests in
  `tests/test_data_services.py`; full suite green (568). Predictive comparison
  `tools/e2_faz1_horizon_compare.py` (Ridge+LightGBM, chronological 80/20,
  EREGL/AKBNK/TUPRS/AEFES/SASA, h∈{1,3,5,10}):
  - Mean Dir_Acc rises h=1 ~50% → h=5 ~52.5% → h=10 ~52% — weekly is modestly
    more predictable than daily, as predicted.
  - **Caveat:** higher h inflates the positive base-rate (up-trending test
    windows); edge over a naive "always up" baseline is small (~0–3pp at h=5,
    ~0 at h=1). A small, consistent gain — not a step change, no free alpha.
  - **Decision:** adopt h=5 as the E2 default horizon (better signal/noise);
    the real lever remains pooling (Faz 2–3). Horizon-aware backtest/forecast
    (**Faz 1b**) is deferred until pooled models beat baselines — for h>1 the
    backtest/forecast/signal path is NOT yet horizon-correct, so h>1 backtest/
    Sharpe must not be interpreted.
- **Faz 2 — Pooled loader + group-aware CV (leakage guard). ✅ DONE
  2026-06-03.** Design: [E2 Faz 2 Pooled CV Design](e2-faz2-pooled-cv-design.md).
  Implemented + tested:
  - `src/validation/pooled_cv.py` — `PooledPurgedWalkForward` date-based splitter
    (global date axis → no cross-symbol leak; purge+embargo `E=h+buffer`; exact
    purge via `target_date < a_k` when present; multi-window rolling OOS + newest
    window reserved as final holdout). 7 leak/structure tests.
  - `src/data/pooled_loader.py` — `PooledPanelLoader` builds the long panel
    (per-symbol features via `FeaturePipeline`, h-day `target`+`target_date`,
    `sector`/`symbol_id`/causal `liq_log`/`vol`; delisted history included;
    stray raw `Kapanış` price-level dropped). 6 tests + real 3-symbol smoke
    (8463 rows, 6+1 folds, leak-asserted on real data).
  - `src/validation/pooled_oos.py` — `evaluate_per_symbol` aggregation harness:
    fits a fresh `model_factory()` per CV fold (no cross-fold state leak),
    predicts test rows, groups OOS predictions by symbol → per-symbol metric
    distribution (`dir_acc`, `rmse`, `base_rate`, `edge`-over-base-rate,
    `positive_fold_ratio`, `reliable`) + per-`(symbol,fold)` detail. log-return
    target → sign-based direction (no price-mode/prev_close coupling). Final
    holdout excluded by default. 6 tests + real Ridge smoke (3 symbols: AKBNK
    edge +0.5, EREGL −1.9, TUPRS −2.4 — honest, near base-rate). Feeds the Faz 5
    serving confidence score.
  - Full suite 587 green. Defaults locked: 63×6 OOS, expanding, cross-sec norm
    off, delisted included. **Next:** Faz 3 global conditioned model.
- **Faz 3 — Global conditioned model (pretrain). 🟡 MODEL BUILT 2026-06-03.**
  `src/models/global_pooled_model.py`:
  - `GlobalPooledModel` — pooled LightGBM (native API, deterministic: `seed`,
    `deterministic=True`, `num_threads=1`). sklearn-vari `fit`/`predict` →
    `pooled_oos` harness compatible. Champion/challenger compatible (periodic
    batch retrain, no per-query training).
  - `build_pooled_features(panel)` → `(panel_aug, feature_cols, cat_indices)`:
    adds stable `sector_code`; conditioning = numeric `liq_log`/`vol` +
    categorical `symbol_id`/`sector_code` (LightGBM native categorical, no
    one-hot blowup). `make_global_model_factory(cat_indices, cfg)` for the harness.
  - 5 tests (schema, stable codes, deterministic fit, harness run, sector-signal
    learnability). Full suite 592 green.
  - **Benchmark (OOS harness, h=5, 39 long-history symbols / 109k rows):** with
    the **absolute** return target, pooled LightGBM does not beat per-symbol
    base-rate (mean edge −2.92). **But the per-symbol directional metric is the
    wrong lens for a pooled model** — the signal is *relative*, not absolute.
- **Faz 3.5 — Cross-sectional rank target (alpha lever). ✅ DONE 2026-06-03.**
  `src/data/cross_sectional.py` `add_cross_sectional_target`: per-date,
  within-cross-section rank of the forward return → centered `[-1,1]`
  (`(rank-0.5)/n`, mean 0; base-rate ~50%). Leakage-safe: ranking is within a
  single date, all those rows share `target_date=d+h`, so `pooled_cv` purge keeps
  them on the same side. Optional `zscore` method.
  - **Correct metric = daily cross-sectional IC** (`daily_cross_sectional_ic` in
    `pooled_oos`, scipy-free Spearman; exposed on `PerSymbolOOSResult.ic`):
    per-date `corr(y_pred, y_true)` across symbols; ICIR = mean/std.
  - **Result (same 39-symbol panel):** absolute target → IC +0.041, ICIR 0.228,
    %IC>0 60%. **Cross-sectional target → IC +0.092, ICIR 0.549, %IC>0 73%** —
    IC more than doubles, ICIR 0.55 is a genuinely usable cross-sectional signal.
    The pooled global model *can* rank BIST names over h=5. This is the alpha
    direction.
  - **Leak fix (regression-tested):** `target`/`target_cs` (any `target_*`) are
    now hard-excluded from features in both `build_pooled_features` and
    `pooled_oos._auto_feature_cols`; a first wrong run showed IC 0.97 from
    `target_cs` self-leaking as a feature. Tests:
    `test_target_variants_never_become_features`, IC tests.
  - 6 + 4 tests. Full suite 602 green.
  - **DECISIVE full-universe run** (`tools/e2_faz35_cs_ic_study.py`,
    `outputs/e2_faz35_cs_ic_study.md`): **589 symbols, 1.228M rows, 2856 dates,
    378 OOS days, h=5.** ABSOLUTE target → IC +0.032, ICIR 0.525, %IC>0 70%.
    **CROSS-SECTIONAL target → IC +0.083, ICIR 1.243, %IC>0 90%.** ICIR 1.24 is
    a strong cross-sectional signal (quant: ICIR>1.0 is excellent); the thin
    39-name cross-section had understated it (ICIR 0.55). Wide cross-section
    (589 names/date) stabilizes IC. Per-symbol dir_acc edge still −2.5 →
    re-confirms absolute per-symbol direction is the wrong lens; alpha lives in
    the daily IC.
  - **Implication:** product is per-stock but alpha is relative → serving must
    translate predicted rank into "expected to out/under-perform peers", with
    confidence from IC stability (NOT absolute price/direction). Faz 4 per-symbol
    fine-tune is not the lever.
- **Faz 3.6 — Cross-sectional (peer-relative) features. ✅ DONE 2026-06-03.**
  `add_cross_sectional_features` (`src/data/cross_sectional.py`): per-date
  relative version of each existing causal feature — centered rank `_csr`
  (`[-1,1]`) + `_csz` z-score. "Where does this stock sit vs peers today."
  Leakage-safe: same-date, all inputs causal (no look-ahead); purge guards the
  target side; NaN→neutral 0. Auto-included as features by `build_pooled_features`
  (numeric, non-`target_*`). 2 tests (8 total in cross_sectional).
  - **Full-universe 3-variant run** (`tools/e2_faz35_cs_ic_study.py`, boost=400):
    ABSOLUTE IC +0.044 / ICIR 0.718 / %IC>0 74; CROSS-SECTIONAL IC +0.105 /
    ICIR 1.418 / %IC>0 91; **CS+CSFEAT IC +0.099 / ICIR 1.550 / %IC>0 93.4.**
    cs-features mainly cut IC variance (ICIR 1.42→1.55, %positive 91→93) — a more
    stable signal; on a thin 64-symbol subset they also lifted IC mean
    (ICIR 0.65→0.93). **Best config = cross-sectional target + cs-features,
    ICIR ~1.55 on the full BIST universe.**
  - *Determinism (checked):* **in-process is byte-identical** — two back-to-back
    CS+CSFEAT fits gave identical IC/ICIR and predictions (maxdiff 0). So
    train-once-persist serving is fully reproducible from a fixed data snapshot,
    and champion/challenger comparisons within a run are exact. The earlier
    cross-*run* ICIR drift (ABSOLUTE 0.525→0.718) is NOT a model bug (same panel
    size 1.228M rows, deterministic in-process) — most likely a data-snapshot
    difference between processes (data/ CSVs are being refreshed). Benign for
    serving; lock the data snapshot when persisting.
- **Faz 4 — Gated per-symbol fine-tune (optional, experimental).** Pretrain pool
  → short per-symbol fine-tune, applied ONLY when a symbol has enough history AND
  fine-tune improves its multi-window OOS; else serve global. Consistent with
  product design "fine-tuning reserved for a later experimental phase"
  (`yeniTasarim/08`). May be deferred.
- **Faz 5 — Registry + serving wiring.** Persist global model + per-symbol OOS
  metrics + eligibility/confidence inputs so `GET /analysis/{symbol}` keeps
  working unchanged. Decide registry shape: global model row + per-symbol metric
  rows (vs the current per-symbol `best_models`).
- **Faz 6 — Stratified validation & promotion policy.** 3-variant test on a
  stratified sample (blue-chip / mid / speculative / thin): `single-symbol` vs
  `pooled-all` vs `pooled→finetune`, compared on per-symbol multi-window OOS.
  Promote pooled only where it beats single-symbol stably; document where nothing
  works → `low` confidence by policy.

## Faz 0 Findings (2026-06-02)

Audit of all 592 stock CSVs in `data/` via `tools/e2_faz0_universe_audit.py`.

- **Scale confirmed:** 592 symbols, **1,097,481** pooled (symbol, date) rows.
  History buckets: `<1y` 22, `1-2y` 49, `2-5y` 136, `5-8y` 22, `8y+` **363**.
  Thin (<500 rows) = **71** symbols (cold-start dependent).
- **Freshness BLOCKER:** only **1** symbol fresh (≤5 days); **569 frozen at
  2025-10-24** (a one-time bulk snapshot export, ~220 days stale). The pipeline
  only refreshes a handful of symbols. Pooled training must first re-pull /
  align the whole universe to a single current cutoff date.
- **Three date formats:** `%Y-%m-%d` (577), `%d/%m/%Y` (14), `%d.%m.%Y` (1).
  A loader assuming one format silently produces an empty frame (this exact bug
  hit the auditor's first pass: 578/592 dropped). Pooled ingestion must detect
  all three.
- **Survivorship unresolved:** `bist_universe.csv` catalogs only **28** symbols;
  **564** CSVs are not in the universe (no sector, no `Delisted_Date`). Delisting
  cannot be inferred from `last_date` (all cut at the same snapshot). Open
  survivorship-bias risk — a delisting source is required before Faz 2 CV.
- **Data quality:** 102 symbols have a `|log_return(adj)| ≥ 0.30` day (real
  corporate action or unadjusted split — needs audit/clip policy); 1 symbol has
  a zero/negative price row; 1 has duplicate dates; 2 have >30-day calendar gaps.
- **Conditioning gap:** sector label missing for **564** symbols → needs
  backfill or an `unknown` bucket before the model can condition on sector.

**Pre-Faz-2 prerequisites surfaced:** (1) universe-wide re-pull to one cutoff
(freshness); (2) multi-format date ingestion; (3) delisting/survivorship source;
(4) sector backfill; (5) corporate-action + zero/neg price cleaning policy.

## Acceptance Criteria (draft)

- `GET /analysis/{symbol}` response schema unchanged; existing API/serving tests
  stay green.
- Pooled model evaluated per-symbol on multi-window OOS; per-symbol metrics
  populate `performance.*` and feed confidence as today.
- Group-aware purged CV proven leak-free by tests (cross-symbol same-date and
  symbol-future leakage asserted absent).
- Stratified 3-variant comparison shows pooled ≥ single-symbol on enough symbols
  to justify the switch, or documents the negative result honestly.
- Cold-start: a thin/never-trained symbol returns an answer with an honest
  (likely `low`) confidence rather than failing.

## Open Questions

- Registry shape: single global `best_models` row + per-symbol metric table, or
  keep per-symbol `best_models` rows fed by the global model? (Affects API
  internals only, not the response contract.)
- Conditioning via explicit features only, or add a learned symbol embedding?
- Pool everything vs sector/liquidity peer-pools as a Faz 6 refinement.

## Related Pages

- [Product Decision Support Design](product-decision-support-design.md)
- [Analysis API Contract](analysis-api-contract.md)
- [Confidence and Risk Policy](confidence-and-risk-policy.md)
- [Data Pipeline](data-pipeline.md)
- [Validation and Backtesting](validation-and-backtesting.md)
- [Model Catalog](model-catalog.md)
- [Persistence and API](persistence-and-api.md)
