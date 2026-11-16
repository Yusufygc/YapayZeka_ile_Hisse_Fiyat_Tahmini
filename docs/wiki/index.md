---
title: Wiki Index
type: index
status: active
last_updated: 2026-11-16
owner: llm
---

# Wiki Index

This wiki is the canonical, LLM-maintained knowledge base for `ts_forecasting_lab`.
Every implementation question and feature request starts here, then drills into the
linked pages below.

## Operating Model

- [Wiki Guidelines](wiki-guidelines.md): LLM Wiki structure, ownership rules, frontmatter, cross-linking, and maintenance workflows.
- [Change Management](change-management.md): Change, wiki-update, and Turkish commit-message rules.
- [Source Map](source-map.md): Raw source layers the wiki is compiled from.
- [Log](log.md): Chronological record of ingests, architecture updates, query-derived pages, and lint passes.

## System Knowledge

- [Architecture](architecture.md): Gecelik havuz servisi mimarisi, dizinler, teknoloji, `serving_pool.db` şeması, salt-okunur okuyucu kuralı.
- [Risk Modeli](risk-model.md): Havuzlu oynaklık modeli (hedef, özellikler, Ridge, bias faktörü, konformal z, kabul kapısı, gerçek veri doğrulaması, durumlar).
- [Data Pipeline](data-pipeline.md): Güncel havuz paneli, kürate makrosuz özellik seti, K8 karantina (ASELS kök nedeni), tazelik; altında Kol-A dönemi tarihsel bölümler.
- [Model Catalog](model-catalog.md): Kalan modeller (Kol-B LightGBM/MLP/ensemble + risk Ridge) ve emekli Kol-A.
- [Validation and Backtesting](validation-and-backtesting.md) **[tarihsel]**: Single split, walk-forward, final holdout, signal gates, backtest metrics, and leakage boundaries.
- [Backtest Signal Improvement Plan](backtest-signal-improvement-plan.md) **[tarihsel]**: Draft plan for improving signal logic and model tiers before API integration or cost modelling.
- [Persistence and API](persistence-and-api.md): PeerStore, `nightly_status.json`, API v2 şekli/durumları (`unavailable` 503, `no_data`), kaldırılan endpoint'ler, önbellek, güvenlik.
- [Analytics Layer](analytics-layer.md) **[tarihsel]**: Salt-okunur DuckDB analiz katmanı (çapraz-koşu/çapraz-kaynak analitik; servis için değil), view'lar, adlandırılmış sorgular, CLI, ağdan eklenti yok.
- [Testing and Quality](testing-and-quality.md): Güncel test alanları, komut ve kalite değişmezleri.
- [Staged Code Review Guide](code-review-stages.md) **[tarihsel]**: Project split into 8 dependency-ordered stages for staged code review; per-stage files, checklist, tests, and dependency notes.
- [Code Quality and Refactoring](code-quality-and-refactoring.md): Code thresholds, file/class size limits, input validation guidelines, error handling, and datetime policy.
- [Code Quality Audit (2026-05-31)](code-quality-audit.md) **[tarihsel]**: God-object/SOLID/DRY findings, bloated file/function metrics, and the phased docstring/comment plan.
- [Staged Refactor Plan (2026-05-31)](refactor-plan.md) **[tarihsel]**: Per-stage god-object/complexity/SOLID-KISS-DRY findings mapped to the 8 review stages, behavior-preserving refactor actions, cross-cutting epics (owner-forward removal, DRY, god constructors), and risk-tiered execution order.
- [Refactor Analysis 2026-06-12](refactor-analysis-2026-06-12.md) **[tarihsel]**: Source-grounded quant/architecture audit after E1/E2; horizon-aware alignment, pooled serving calibration, remaining owner-state workflow, backtest, and memory priorities.
- [XAI Audit 2026-06-12](xai-audit-2026-06-12.md) **[tarihsel]**: Source-grounded explainability audit covering Kol-A XAI, walk-forward attribution, sequence/attention diagnostics, Kol-B peer XAI, API contract gaps, and phased improvement plan.
- [E1 Owner-Forward Removal Epic](e1-owner-forward-epic.md) **[tarihsel]**: Tier 3'un kalan kismi — owner-forward magic'i tamamen kaldirip servisleri `EvaluationContext`/`EvaluationState` DI'ya cevirme; karakterizasyon testi stratejisi + 7 fazli plan. Dal: `refactor/e1-owner-forward-di`.
- [E2 Faz 2 Pooled CV Design](e2-faz2-pooled-cv-design.md): Pooled panel loader + tarih-bazli purged coklu-pencere CV detayli tasarimi; leakage taksonomisi (capraz-sembol / horizon / feature), modul plani (`src/data/pooled_loader.py`, `src/validation/pooled_cv.py`), test plani, acceptance, acik sorular.
- [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md): Egitim tarafi redesign — tek-hisse overfit'i gidermek icin ~592 hisseyi havuzlayan tek kosullu global model; per-symbol urun/`GET /analysis/{symbol}` kontrati korunur; grup-purged coklu-pencere OOS, cold-start, opsiyonel per-symbol fine-tune. Dal: `feat/e2-pooled-global-model`.
- [Product Decision Support Design](product-decision-support-design.md) **[tarihsel]**: Desktop AI decision-support product boundary, target architecture, MVP scope, and phase roadmap.
- [Desktop GUI Application](desktop-gui-application.md): PySide6 tek tablo (salt-okunur PeerReader).
- [Analysis API Contract](analysis-api-contract.md) **[tarihsel]**: `GET /analysis/{symbol}` response schema, status codes, and confidence label definition.
- [Confidence and Risk Policy](confidence-and-risk-policy.md) **[tarihsel]**: Confidence label derivation rules, signal-diagnosis mapping, eligibility status, and data-quality gates.
- [LLM Explanation Policy](llm-explanation-policy.md): AI explanation layer role, forbidden actions, response structure, system prompt skeleton, and disclaimer.

## Current Project State

- **A+C (2026-11-16):** Kol-A (hisse başına yön modelleri), `ForecastingPipeline`, backtest, makro katman
  ve `stock_models.db` kaldırıldı. Ürün yalnız gecelik havuz çıktısını sunar: Kol-B yön/göreli sinyal +
  havuzlu risk modeli (`src/risk/`, %80 fiyat bandı). **API v2** (`/health`, `/symbols`,
  `/analysis/{symbol}`; durumlar `ok|stale_data|low_confidence|no_data|unavailable`), GUI tek tablo.
  Okuyucular salt-okunur; risk adımı durumu `nightly_status.json`/`/health`'e yansır; `as_of` geriliği,
  kapı değişimi ve risk tahmini olmayan sembol nedenleri izlenir. Sayfalar: [Architecture](architecture.md),
  [Risk Modeli](risk-model.md), [Persistence and API](persistence-and-api.md). **Aşağıdaki maddelerden
  Kol-A/`ForecastingPipeline`/model-scope/`stock_models.db`/owner-forward anlatanlar tarihseldir.**
- The project is a production-oriented research platform for BIST equity forecasting.
- **Özellik ve model temizliği (2026-11-16, `fix/deep-audit-2026-11`):** model özellik seti
  63'ten 19'a indi (IC denetimi; sektör endeksi ve makro'nun USDTRY/BIST100 dışı kolonları
  kaldırıldı); model kataloğu 3 aday + 3 naive benchmark + tek ensemble (`Ensemble Inverse RMSE`)
  ile sınırlandı, LSTM/Prophet/ARIMA/DLinear vb. silindi (`ModelNoLongerSupportedError`);
  gece işi için durum dosyası, bayat-pay kapısı ve zaman sınırları eklendi (2026-09-16'dan
  beri zamanlanmış görev kayıtlı değildi). **Mevcut DB skorları ve Kol-B serving havuzu eski
  semantikle üretildi; önce veri yenilenmeli, sonra yeniden koşulmalıdır.** Bkz.
  [Data Pipeline](data-pipeline.md), [Model Catalog](model-catalog.md),
  [Persistence and API](persistence-and-api.md#nightly-automation-2026-11-16).
- **Veri yönetimi düzeltmeleri (2026-11-15, `fix/deep-audit-2026-11`):** ortak tarih ayrıştırıcı (`src/data/dates.py`), ortak OHLCV temizleyici (`src/data/ohlcv_clean.py`), tazelik kapısı, seans-bilinçli + toplu + kilitli `DataUpdater` (nightly `update_many` kullanır), SQLite `get_db` önbelleği + WAL/busy_timeout + atomik yazımlar + göreli `model_path`, sıkıştırılmış model kayıtları + hafif `latest/` + `tools/prune_outputs.py`. Bkz. [Data Pipeline](data-pipeline.md), [Persistence and API](persistence-and-api.md).
- The active orchestration facade is `ForecastingPipeline` in `src/pipeline/orchestrator.py`.
- Main orchestration responsibilities are split across `DataManager`, `ModelTrainer`, and `EvaluationManager`.
- Evaluation logic is now service-composed via `PredictionService`, `BacktestService`, `SignalCalibrationService`, and `MetricsReportingService`.
- **Owner-forward removal (E1 epic, ✅ CLOSED 2026-06-01 on `refactor/e1-owner-forward-di`):**
  all four evaluation services (`PredictionService`, `BacktestService`,
  `SignalCalibrationService`, `MetricsReportingService`) were converted from
  `_OwnerBackedService` (`__getattr__`/`__setattr__` forwarding) to explicit
  `(ctx, state)` dependency injection (`EvaluationContext` read-only config +
  `EvaluationState` mutable runtime). Faz 4 trimmed `EvaluationManager` to a
  thinner orchestrator (deleted 10 dead delegations, 1035 → 979 lines). Faz 5
  moved `ForecastRunner` to `ForecastContext` DI and **deleted**
  `_OwnerBackedForecastService`. Faz 6 made the 4 `DataManager` services
  fail-loud. P2-A converted `TensorPreparationService` and
  `ValidationSplitService` to explicit `DataManagerContext`/`DataManagerState`
  DI; P2-B then converted `DataIngestionService` and
  `DataQualityReportingService` to the same explicit DI contract. P2-C converted
  the 3 training workflows and 3 evaluation workflows to explicit context/state
  DI and deleted `_OwnerBackedService`. Faz 7 (cleanup) deleted the temporary
  `tools/owner_forward_inventory.py` and updated docs. The remaining
  owner-forward scope is now **0 classes**; behavior is unchanged and locked by
  golden tests in `tests/test_owner_forward_contract.py`.
  See [E1 Owner-Forward Removal Epic](e1-owner-forward-epic.md).
- **E2 pooled global model (branch `feat/e2-pooled-global-model`, Faz 2–8 ✅,
  2026-06-04):** one conditioned LightGBM trained across ~589 BIST stocks with a
  **cross-sectional rank target** (within-date rank of forward return). Group-
  purged date walk-forward gives daily cross-sectional **IC ≈ 0.099 / ICIR ≈
  1.55**. Serving stays per-symbol via an additive `peer` block on
  `GET /analysis/{symbol}` (NOT per-query training): a nightly batch scores the
  universe → `PeerStore` (`data/serving_pool.db`, isolated from `best_models`).
  Each symbol gets peer_score/percentile/label, segment (liq/vol/sector) +
  composite-ICIR confidence (tradability-gated), and a **trend tendency**
  (`yukarı/yatay/aşağı/belirsiz` + calibrated P(up) + expected return; Faz 7
  finding: peer rank carries a modest-but-real monotone absolute-direction tilt,
  Q5 54% up vs Q1 43%). **Faz 8** adds a Windows nightly job (Task Scheduler
  21:00, BIST trading-day gated) that refreshes universe data then re-scores. The
  honest framing (relative, probabilistic, decision-support — not a buy/sell bot)
  is preserved. See [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md).
- The default production candidate set is defined in `src/pipeline/model_scope.py`
  and the model registry; in the current source tree `TFT` is not registered and
  `src/models/tft_v2/` is absent.
- Cheap naive models are treated as benchmarks, not production candidates.
- The main database is `data/stock_models.db`; trained run artifacts live under `outputs/{SYMBOL}/runs/{RUN_ID}/`.
- `outputs/{SYMBOL}/latest/` is synchronized from the latest completed run.
- The analysis API now tracks non-blocking refresh jobs in SQLite, restricts
  CORS to local origins plus explicit environment configuration, and can queue
  data/forecast refresh work when the best model has no current forecast or the
  market data is stale.
- Forward forecasts are served from saved production artifacts and recursive
  horizon generation. The only production ensemble leader is `Ensemble Inverse
  RMSE` (2026-11-16: `Ensemble Cash-Gated` kaldırıldı), with member metadata persisted.
- Repository-wide change discipline is captured in `RULES.md` and [Change Management](change-management.md).
- **Validation mode** (as of 2026-05-25): default and production-only mode is
  `walk_forward`. `single_split` is research-only via `--debug-quick`. WF
  embargo auto-resolves to horizon + 5 (h+5) when left unset (2026-11-16).
- **Metric priority (Sprint 1, 2026-05-25)**: reports lead with `Dir_Acc`,
  `Hit_Rate`, `Composite_Score`; `Net_Return` / `BuyHold_Return` move to a
  footnote because the default backtest is cost-free. Risk-free rate
  fallback (`0.40`) was removed — Sharpe/Sortino return `NaN` and a
  `Risk_Free_Unavailable` flag is raised whenever macro
  `INTEREST_RATE.csv` and `RISK_FREE_RATE_ANNUAL` env are both missing.
  Backtest CSV/MD reports automatically prepend the cost + advisory
  disclaimer.

- **Kol-B denetimi K1–K9 (2026-11-10):** doğrudan sızıntı bulunmadı; başlık IC tradable alfa
  sayılmaz (likidite yoğunlaşması, hayatta kalma yanlılığı, çoklu OOS seçimi). Skorlama artık en
  güncel piyasa gününe yapılır, holdout/örtüşmesiz/tradable IC raporlanır, segment kovaları
  nedensel, `/health.serving_pool` tazelik gösterir. Ayrıntı:
  [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md) (Kol-B Denetim bölümü).

- **Derin denetim (2026-11-15, dal `fix/deep-audit-2026-11`):** rapor
  `../analizler/derin_denetim_2026-11-15.md`. P0 sızıntılar (Optuna paylaşımı, WF ensemble
  in-sample ağırlık, makro yayın gecikmesi, ABD aynı gün verisi, holdout kapısı) ve P1 ölçüm
  hataları (Sortino, DSR, rf, Dir_Acc, Composite, embargo, maliyet, interval kalibrasyonu,
  Kol-B NW/örtüşmesiz IC) düzeltildi; seçim Diebold–Mariano kapılı; varsayılan adaylar
  XGBoost + Ridge Return + LightGBM Return; Python 3.13; ~2.800 satır ölü kod silindi.
  **Mevcut DB skorları eski semantikle üretildi, yeniden koşu gerekir.**

## Navigation Rules for Agents

1. Read this file before code changes or architecture answers.
2. Use [Source Map](source-map.md) to decide which raw source files to inspect.
3. If the answer requires system behavior, inspect source code as the primary source of truth.
4. If a decision, bug solution, or feature plan is produced, update the relevant wiki page.
5. Append a new entry to the top of [Log](log.md) for every significant wiki or architecture update.
6. Commit completed changes with clear Turkish explanations when the user asks for a commit.
