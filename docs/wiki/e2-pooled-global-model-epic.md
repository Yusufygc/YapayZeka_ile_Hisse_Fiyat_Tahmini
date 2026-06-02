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

- **Faz 0 — Data/universe audit (blocking).** Survivorship & look-ahead across
  the 592-symbol universe (delisted symbols, universe drift over time), IPO
  cold-start handling, split/adjustment consistency, per-symbol history length
  distribution. Decide minimum-history policy for pooling vs cold-start.
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
