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
- **Faz 1 — Horizon shift (cheap win, orthogonal).** Daily → weekly (5-day)
  forward return target. Higher signal/noise; reuses existing pipeline. Measure
  EREGL + a stratified sample before/after.
- **Faz 2 — Pooled loader + group-aware CV (leakage guard).** Pooled
  (symbol, date) data loader. Purged + embargo, **group-aware** splits: no
  same-calendar-date leakage across symbols between train/test, no symbol-future
  leakage. Replace single final-holdout with **multi-window rolling OOS** →
  report a distribution, not one number. Feeds `stability_score` /
  `rolling_positive_window_ratio` (already in confidence policy).
- **Faz 3 — Global conditioned model (pretrain).** One model + conditioning
  features/embedding, trained on pooled rows. Champion/challenger compatible
  (matches existing training policy: periodic batch retrain, no per-query
  training).
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
