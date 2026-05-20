## [2026-05-20] Wiki Update | Operasyonel sertleştirme, forecast artifact ve analiz refresh

- Analiz API'si için yerel-first CORS, JSON loglama, SQLite tabanlı
  `analysis_refresh_jobs`, `/refresh/status/{job_id}` ve stale/missing forecast
  durumlarında otomatik refresh kuyruğu dokümante edildi.
- Forward forecast hattının production artifact sidecar'larından model/scaler
  yüklediği, recursive horizon ürettiği, forecast source metadata yazdığı ve
  üretim ensemble kapsamını `Ensemble Inverse RMSE` ile `Ensemble Cash-Gated`
  ile sınırladığı kaydedildi.
- `AttentionLSTM v2` opt-in sequence modeli, temporal attention XAI export'u,
  minimum sequence eşiği ve yeni operasyonel test kapısı wikiye işlendi.
- Veri güncelleme sonucunun `DataUpdateResult` olarak dönmesi, BIST calendar
  üretimi, DB backup-reset bakımı ve güncel `ASELS.csv` veri ekleri not edildi.

## [2026-05-19] test | test_phase5_data_quality.py patch yollari guncellendi

- 3 kırık test onarıldı: patch yolu `data_manager` → `data_services`, `universe_auto_sync=False` eklendi.
- Tam suite: 332/332 yeşil.

## [2026-05-19] Adim 2.5 | Forecast Resolution Rolling Takibi

- `forecast_runs.live_status TEXT DEFAULT 'healthy'` kolonu eklendi (additive migration).
- `ForecastResolutionRepository.get_rolling_resolution_accuracy()`: son 60 günlük
  gerçekleşmiş forecast_points üzerinde rolling dir_acc + rolling MAE hesaplar.
  dir_acc < 50 ise `model_status='degraded'` ve en son forecast_run güncellenir.
- `StockModelDB.get_rolling_resolution_accuracy()` facade metodu eklendi.
- `AnalysisService.build()` bu sonucu okuyarak `model_status` parametresini
  `compute_confidence()`'a iletir → degraded ise confidence 'low' olur.

## [2026-05-19] Adim 2.4 | XAI Fold-Stability Skoru

- `strategies.py`: `compute_feature_stability_scores()` fold bazlı top-K özellik
  sayısı → fold_ratio döner.
- `XaiProductSummary.feature_stability_top` alanı eklendi.

## [2026-05-19] Adim 2.3 | Piyasa Rejimi ve Trend Bağlamı

- `src/pipeline/regime_context.py`: `compute_market_regime()` (SMA50/SMA200+slope),
  `compute_relative_strength()` (60-gün), `compute_regime_context()` tam payload.
- `regime_misalignment=True` → confidence_calculator seviye düşürür (soft gate).

## [2026-05-19] Adim 2.2 | Ensemble Yön Uzlaşısı

- `prediction_engine.py`: `compute_ensemble_direction_agreement()` statik metot.
- `forecast_runs.ensemble_direction_agreement REAL` kolonu eklendi.
- `log_forecast_run()`, `ForecastPersistence.save_run()`, `StockModelDB` facade
  güncellendi. `AnalysisResponse.forecast.ensemble_agreement` DB'den doldu.

## [2026-05-19] Adim 2.1 | Rolling Holdout

- `src/validation/rolling_holdout.py`: 60-bar pencereler (adim=20) üzerinde
  median_net_return, positive_window_ratio, iqr_net_return üretir.
- Confidence calculator bu metrikleri kullanıyor (1.3'te eklendi).

## [2026-05-19] Adim 1.9 | Veri Kalite ve Distribution Shift

- `src/data/quality.py` yeni modül: `compute_quality_flags()` ve `compute_psi()`.
- `corporate_action_anomaly`: df.attrs['corporate_action_report']'dan okunur.
- `survivorship_warning`: tarih serisi < 2 yıl veya > 10 gün boşluk tespiti.
- `psi_high`: train vs holdout PSI > 0.25 eşiği (per-feature, max alınır).
- `clip_rate`: df.attrs'dan okunur (preprocessor tarafından doldurulur).
- `tests/test_data_quality.py`: 10 test (PSI hesabı, flag tespiti) — tümü yeşil.
- `confidence_calculator` bu bayrakları girdi olarak kabul ediyor (1.3'te eklenmişti).

## [2026-05-19] Adim 1.8 | HPO Stability-Aware Objective

- XGBoost ve RandomForest Optuna objective: RMSE → -(mean_sharpe - 0.5*std_sharpe).
- Minimum trial sayısı 30 → 40 (xgboost_model, random_forest_model, training_workflows).
- Final holdout HPO kullanmıyor; mevcut davranış korundu (FinalHoldoutTrainingWorkflow.run() .train() çağırıyor).

## [2026-05-19] Adim 1.7 | Run Manifest + Seed Logging

- `src/pipeline/orchestrator.py`: `_write_run_manifest()` metodu eklendi; her run sonunda
  `outputs/{SYMBOL}/runs/{RUN_ID}/run_manifest.json` üretir.
- Manifest içeriği: run_id, generated_at, stock_symbol, data_hash (MD5), feature_pipeline_version,
  model_config_hash (SHA-256), signal_config_hash, random_seed (42), model_list,
  validation_protocol, git_commit (subprocess), python_version, lib_versions.
- `_sync_latest_output` manifest'i `latest/` klasörüne otomatik kopyalar.
- `tests/test_run_manifest.py`: 5 test (dosya varlığı, zorunlu alanlar, değer doğruluğu,
  lib_versions dict, JSON geçerliliği) — tümü yeşil.

## [2026-05-19] Wiki Ingest | yeniTasarim Design Notes

- Enriched `product-decision-support-design.md`: product position, correct/forbidden
  language, regulatory boundary, target architecture, MVP scope (Faz 1 and 2
  items), out-of-scope items, and phase roadmap through Faz 3.
- Created `analysis-api-contract.md`: full `GET /analysis/{symbol}` JSON schema,
  analysis_status codes, status priority hierarchy, confidence label table,
  freshness definition, and XAI caveat requirement.
- Created `confidence-and-risk-policy.md`: hard blocks always producing `low`,
  soft degradation rules, signal_diagnosis label table, eligibility_status
  values, naive-leader rejection rule, data-quality flags (PSI, corporate action,
  clip_rate, survivorship), stability_score formula, rolling holdout metrics
  (Faz 2), and freshness threshold.
- Created `llm-explanation-policy.md`: LLM role definition, permitted/forbidden
  actions, response structure (8 sections), system prompt skeleton with
  `{payload_json}` placeholder, verbatim disclaimer, XAI language rules, and
  note that actual LLM call is out of scope for Faz 1.
- Updated `index.md` with links to all four new/updated pages.

## [2026-05-18] Fix | Interactive LSTM Lite Selection

- Added `LSTM Lite` to the interactive CLI model list and the deep-learning
  preset so manual runs can actually select the new candidate.
- Routed `LSTM Lite` through sequence XAI handling alongside `LSTM`.

## [2026-05-18] Feature | LSTM Lite Candidate

- Added the implementation plan result for `LSTM Lite`: a selected-only,
  smaller unidirectional LSTM sequence candidate for single-symbol BIST runs.
- Documented its separate `lstm_lite_min_sequence_samples=252` gate,
  train-only optional HPO scope, and multi-metric evaluation expectations.
- Existing `LSTM` remains unchanged and stays in the default candidate set.

## [2026-05-18] Audit | LSTM Model Process Review

- Reviewed the LSTM data preparation, sequence alignment, model factory,
  walk-forward training, final-holdout prediction, and recent output reports.
- Recorded that no direct sequence off-by-one or train/test scaler leakage was
  found, but current LSTM capacity is too high for the observed single-symbol
  fold sizes and feature count.
- Added realistic improvement guidance to `docs/wiki/model-catalog.md`: raise
  LSTM sample gates, shrink architecture, add train-only HPO, prune/stationarize
  sequence features, and require rolling holdout plus trade-quality stability
  before promotion.

## [2026-05-18] Feature Plan | ClaudeGelistirme Design Integration

- Reviewed `ClaudeGelistirme/` and integrated only the plan items aligned with
  the current `yeniTasarim/` scope: signal diagnosis, naive-leader rejection,
  cross-run leaderboard, shadow backtest selection inputs, rolling holdout,
  stability score, and distribution-shift gating.
- Kept transaction costs, portfolio construction, API deployment/monitoring,
  CPCV/SPA, and panel modelling out of the near-term design.
- Updated `docs/wiki/backtest-signal-improvement-plan.md` with the accepted vs
  deferred integration boundary.

## [2026-05-18] Wiki Correction | TFT Removed From Active Model Catalog

- Corrected stale wiki references after source inspection showed `src/models/tft_v2/`
  is absent and the active model registry has no `TFT` model spec.
- Updated model catalog, architecture, source map, testing notes, and the
  backtest signal improvement plan so TFT is no longer described as an active or
  research-shelf model.
- Old TFT mentions can still appear in historical `outputs/` artifacts, but they
  are not active source-code models.
## [2026-05-18] Wiki Update | Ensemble MekanizmasÄ±

- `docs/wiki/model-catalog.md` iÃ§indeki ensemble bÃ¶lÃ¼mÃ¼ kaynak kodla uyumlu ÅŸekilde geniÅŸletildi.
- Ensemble tahminlerinin eÄŸitim sonrasÄ± sentezlendiÄŸi, aday model filtresi kullandÄ±ÄŸÄ±, tek split ve walk-forward akÄ±ÅŸlarÄ±nda aynÄ± yedi ensemble ailesini eklediÄŸi kaydedildi.
- Forward forecast tarafÄ±nda ensemble/baseline en iyi model seÃ§ilirse trainable replacement aranmasÄ± kuralÄ± netleÅŸtirildi.

## [2026-05-17] Refactor | CLI Restructuring & Connection Fixes

- KÃ¶k dizindeki scriptler (`main_pipeline.py`, `run_batch.py`, `run_forecast.py`), `src/cli/` altÄ±na modÃ¼ler biÃ§imde taÅŸÄ±ndÄ± (`interactive.py`, `batch.py`, `forecast.py`).
- CLI dizinindeki `interactive.py` betiÄŸinde `sys.path` yolu eklentisi yapÄ±larak, dÄ±ÅŸ dizinlerden (Ã¶rneÄŸin kÃ¶k dizin) Ã§alÄ±ÅŸtÄ±rÄ±ldÄ±ÄŸÄ±nda alÄ±nan `ModuleNotFoundError` Ã§Ã¶zÃ¼ldÃ¼.
- `AGENTS.md` iÃ§erisindeki Ã§alÄ±ÅŸtÄ±rma komutlarÄ± yeni `python -m src.cli...` yapÄ±sÄ±na uygun ÅŸekilde gÃ¼ncellendi.
- `StockModelDB` sÄ±nÄ±fÄ± ve orkestrasyon yÃ¶neticileri, daha temiz bir mimari iÃ§in repository ve servis desenlerine ayrÄ±ldÄ±. Eski araÅŸtÄ±rma modÃ¼lleri temizlendi.

## [2026-05-17] Refactor | Root organization and dead file cleanup
- Root CLI scripts were relocated into `src.cli` modules with no compatibility wrappers: `interactive`, `batch`, and `forecast`.
- Runtime artifacts were moved out of the root: batch summaries now live under `outputs/batch_summaries/`, Optuna warm-start databases under `data/optuna/`, and local report tooling under `tools/reports/`.
- Unused standalone research helpers for Monte Carlo bootstrap, Kelly sizing, and independent permutation tests were removed after import/reference checks.
- README and wiki references were updated to the new CLI commands and simplified active AL/SAT/TUT product scope.

## [2026-05-17] Refactor | Persistence forecasting and XAI phase 4 plan

- `StockModelDB` was thinned into a facade backed by schema, experiment, best-model, forecast, and forecast-resolution repositories.
- `ForecastRunner` now delegates best-model resolution, data preparation, production training, latest-target prediction, and roll-forward point generation to internal workflows.
- `XAIExplainer` now dispatches model-family explanations through SHAP/LIME strategy helpers while preserving TFT attention/variable-selection and permutation fallbacks.
- Phase 4 gates passed under `dl_env`: forecasting/model-scope, XAI routing/TFT, smoke/evaluation services, new repository/workflow/strategy tests, radon, and vulture.

## [2026-05-17] Refactor | Pipeline service decomposition phase 3

- `EvaluationManager`, `DataManager`, and `ModelTrainer` were thinned into public facades that delegate stage logic to owner-backed workflow/service classes.
- New modules `pipeline/evaluation_workflows.py`, `pipeline/data_services.py`, and `pipeline/training_workflows.py` now hold single-split, walk-forward, final-holdout, ingestion, tensor preparation, validation split, data quality, and training workflows.
- Service boundary tests were added for evaluation, data, and training composition while preserving train-only scaler fitting and final-holdout exclusion from model selection.
- Phase 3 gates passed under `dl_env`: smoke/service boundary tests, leakage/phase/reporting/backtest acceptance, forecasting/model-scope/macro-cache tests, radon, and vulture.

## [2026-05-16] Refactor | Lowest-risk extraction phase 2 continued

- `backtesting.engine` was further thinned by moving equity-curve construction and trade-log extraction into `backtesting.equity` and `backtesting.trades`.
- `backtesting.signals` now delegates numerical signal helpers to `backtesting.signal_math`; `macro_pipeline` delegates daily/monthly feature engineering to `macro_feature_engineering`.
- Signal calibration trial summary, ranking, and confirmed-selection helpers moved into `pipeline.signal_calibration.selection` while keeping existing compatibility wrappers.
- Phase 2 regression gates passed under `dl_env`: smoke/evaluation/forecast/model-scope/macro-cache, leakage/phase/reporting/backtest acceptance, radon, and vulture.

## [2026-05-16] Refactor | Modular extraction phase 1

- Backtesting, signal validation, macro transforms, feature correlation pruning, signal calibration grid sampling, and model factory responsibilities were extracted into smaller modules while preserving existing public facades.
- Compatibility wrappers remain in `signals.py`, `engine.py`, `feature_pipeline.py`, `macro_pipeline.py`, `signal_calibrator.py`, and `model_trainer.py` so existing tests and import paths continue to work.
- The phase gate passed under `dl_env`: smoke/evaluation services, leakage/phase/reporting, forecasting/model scope/macro cache, radon, and vulture.

## [2026-05-16] Refactor | Signal calibration and macro pipeline decomposition

- `signal_calibrator` execution calibration was decomposed into trial append, adaptive expansion, OOS confirmation, report-frame ranking, and summary snapshot helpers while preserving final-holdout leakage boundaries.
- `MacroPipeline.get_macro_features()` was reduced to an orchestration method around cache refresh/load, date filtering, release-lagged monthly features, daily/global merges, and final feature engineering.
- Characterization coverage was expanded for deterministic calibration sampling, macro cache-only execution, finalized macro schema, and manual monthly CSV fallback.

## [2026-05-16] Fix | Backtest reporting and conservative cleanup

- Backtest plot writing was separated from CSV/Markdown/order report survival so headless matplotlib failures do not suppress core reports.
- `dl_env` static review decisions were recorded: Monte Carlo, Kelly position sizing, and permutation importance remain research helpers outside the active default pipeline.
- The deprecated `src/pipeline/report_writer.py` tombstone was removed after confirming it had no active imports.
- README defaults were aligned with the current simple AL/SAT/TUT signal mode and zero default transaction costs.

## [2026-05-16] Decision | Simple buy-sell-hold signal mode

- Varsayilan sinyal modu `simple` olarak belirlendi; sistem artik uzun/nakit calisir ve `AL`, `SAT`, `TUT` emirlerini uretir.
- Varsayilan komisyon ve slippage degerleri `0.0` yapildi; maliyet kolonlari basit modda sifir kalir.
- Her backtest icin `csv/backtest_orders_{suffix}.csv` gunluk emir raporu uretilecegi kaydedildi.
- `professional` sinyal modu arastirma ve ileri seviye kullanim icin korunur, ancak varsayilan akis disina alindi.

## [2026-05-16] Wiki Update | Graphify Kurulumu

- `dl_env` ortamÄ±nda Graphify CLI'nin `graphifyy` paketiyle kullanÄ±lacaÄŸÄ± kaydedildi.
- Codex Graphify skill kurulumu `C:\Users\ysfygc\.agents\skills\graphify\SKILL.md` konumuna yapÄ±ldÄ±.
- API anahtarÄ± yokken `graphify update .` komutunun AST-only graph yenileme yolu olduÄŸu ve `graphify-out/` Ã§Ä±ktÄ±larÄ± Ã¼rettiÄŸi belgelendi.

## [2026-05-09] Decision | DeÄŸiÅŸiklik ve Commit KurallarÄ±

- `RULES.md` dosyasÄ± oluÅŸturuldu.
- Her anlamlÄ± sistem deÄŸiÅŸikliÄŸinde ilgili wiki sayfalarÄ±nÄ±n gÃ¼ncellenmesi kuralÄ± eklendi.
- Gerekirse `docs/wiki/` altÄ±nda yeni Markdown dosyasÄ± oluÅŸturulacaÄŸÄ± ve `index.md` Ã¼zerinden baÄŸlanacaÄŸÄ± netleÅŸtirildi.
- Commit mesajlarÄ±nÄ±n TÃ¼rkÃ§e karakterlere dikkat edilerek aÃ§Ä±k ve anlaÅŸÄ±lÄ±r TÃ¼rkÃ§e aÃ§Ä±klamalarla yazÄ±lmasÄ± kararÄ± kaydedildi.

## [2026-05-09] Wiki Update | Expanded LLM Wiki Knowledge Base

- Expanded the wiki from a skeleton into a project knowledge base with architecture, data, model, validation, persistence, testing, source-map, and operating-guideline pages.
- Added explicit LLM Wiki operating rules: raw sources, generated wiki layer, schema layer, ingest/query/lint workflows, frontmatter, cross-linking, and log discipline.
- Updated the wiki to reflect the current repository state: tests exist, the model catalog is broader than the initial 5-model summary, SQLite lives under `data/stock_models.db`, and outputs use run-scoped directories plus `latest/`.

## [2026-05-09] Wiki Update | Initial Wiki Skeleton

- Created the initial wiki structure under `docs/wiki/`.
- Added `index.md`, `architecture.md`, and `log.md` as the baseline knowledge base files.
- Added mandatory wiki maintenance rules to the top of `AGENTS.md`.
## [2026-05-18] Feature Plan | Desktop AI Decision Support Design

- Yeni desktop AI sayfasi icin karar destek urunu sinirlari belgelendi: sistem kisisel yatirim tavsiyesi degil, model/forecast/XAI ciktisini aciklayan analitik destek katmani olarak konumlandirildi.
- Kok dizinde yerel ve git disi `yeniTasarim/` tasarim calisma alani olusturuldu; urun konumu, hedef mimari, API sozlesmesi, AI yanit politikasi, egitim stratejisi, ensemble/trend/XAI kullanimi, overfit riskleri ve MVP kapsami ayri Markdown dosyalarina ayrildi.
- Wiki'ye `product-decision-support-design.md` sayfasi eklendi ve indeks uzerinden baglandi.
## [2026-05-18] Feature Plan | Backtest Signal Improvement Focus

- Added `backtest-signal-improvement-plan.md` to capture the revised near-term
  goal: API work and transaction-cost modelling are deferred, while the first
  focus is improving long/flat signal generation so walk-forward backtests can
  beat buy-and-hold and confirm on final holdout.
- Recorded the model-catalog direction as tiered promotion: keep a compact
  active core for default runs, demote unstable or expensive models to the
  research shelf, and use diagnostics before deleting model families.
- Linked the new page from the wiki index.

