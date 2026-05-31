---
title: Staged Code Review Guide
type: guide
status: active
last_updated: 2026-05-31
owner: llm
---

# Staged Code Review Guide

`ts_forecasting_lab` ~120 kaynak dosya, 16 modül, 68 test dosyası içerir. Tek
seferde review pratik değil. Bu rehber projeyi **veri akışına göre
bağımlılık-sıralı 8 aşamaya** böler. Her aşama önceki aşamaların çıktısına
dayanır; bu sırayla incelendiğinde her aşama izole ve tam anlaşılır.

## Nasıl kullanılır

1. Aşamaları **sırayla** (1 → 8) review et. Sonraki aşama önceki aşamanın
   sözleşmesine güvenir.
2. Her aşamada:
   - **Dosyalar + Amaç** tablosundaki dosyaları oku.
   - **Review Checklist** maddelerini tek tek doğrula (leakage / invariant / risk).
   - **İlgili Testler**i çalıştır; review bulgusu varsa testle teyit et.
3. Otomatik review için: `/code-review` (mevcut diff) ya da bir aşamanın
   dosyalarını okutup manuel inceleme.
4. Aşama testleri yeşilse bir sonraki aşamaya geç.

Kaynak doğruluk sırası: **kod > test > wiki**. Çelişkide kodu esas al, wiki'yi
güncelle, `log.md`'ye işle (`RULES.md`).

## Aşama bağımlılık grafiği

```text
Stage 1  Veri & Konfig & Feature
   │  (temiz, scaled, leakage-free tensörler + feature isimleri)
   ▼
Stage 2  Model Tanımları & Factory
   │  (BaseModel sözleşmeli model örnekleri, candidate/benchmark scope)
   ▼
Stage 3  Eğitim Workflow & Validation
   │  (walk-forward / final-holdout fold sonuçları, eğitilmiş modeller)
   ▼
Stage 4  Backtest & Sinyal Kalibrasyonu
   │  (kalibre sinyal config, backtest equity/trade frame'leri)
   ▼
Stage 5  Evaluation & Raporlama & XAI
   │  (metrikler, composite score, confidence label, XAI çıktıları)
   ▼
Stage 6  Persistence & Üretim Seçimi
   │  (SQLite registry, best_model, outputs/{SYMBOL}/latest sync)
   ▼
Stage 7  Forward Forecast & API
   │  (kayıtlı artifact'tan forward forecast, /analysis HTTP servisi)
   ▼
Stage 8  CLI & Orkestrasyon
      (interactive/batch/forecast giriş noktaları, run_all facade)
```

---

## Stage 1 — Veri & Konfigürasyon & Feature

Ham OHLCV CSV'den leakage-free, ölçeklenmiş tensörlere kadar tüm veri yolu.
Tüm aşamaların temeli — burada kaçan leakage tüm metrikleri bozar.

### Dosyalar + Amaç

| Dosya | Amaç |
|---|---|
| `src/pipeline/config.py` | `PipelineConfig` / `DataConfig` / `ValidationConfig` / `ModelConfig` / `ExecutionConfig` dataclass'ları |
| `src/data/data_loader.py` | OHLCV yükleme, Türkçe kolon çevirisi, sıfır-hacim filtresi, temel teknik indikatör + lag |
| `src/data/data_updater.py` | yfinance ile sembol güncelleme, kurumsal aksiyon, geçmiş derinliği validasyonu |
| `src/data/preprocessor.py` | Scaling (robust/standard/clip), LSTM sequence üretimi, fiyat geri-dönüşümü |
| `src/data/quality.py` | Kalite bayrakları (NaN, dup tarih, hacim anomali), PSI, kurumsal aksiyon denetimi |
| `src/data/universe_sync.py` | BIST evren metadata senkronu, survivorship/listing kontrol |
| `src/features/feature_pipeline.py` | Modüler feature üretimi (stationary-relative MA, momentum, takvim, cross-sectional momentum) |
| `src/features/macro_pipeline.py` | Makro feature yükleme + dönüştürme (USDTRY, BIST100, faiz, CPI) |
| `src/features/macro_transforms.py` | Saf makro tarih filtre / lag / merge transformları |
| `src/features/macro_feature_engineering.py` | Aylık rate/CPI + günlük makro feature mühendisliği |
| `src/features/macro_forward_projection.py` | Forecast ufku için makro feature ARIMA(1,1,1) ileri projeksiyon |
| `src/features/correlation_pruning.py` | Korelasyon graf pruning (eşik 0.88) |
| `src/features/sector_mapping.py` | Hisse→sektör endeksi eşleme (`data/bist_universe.csv`) |
| `src/features/feature_cache.py` | Hesaplanan feature'ları run'lar arası cache'leme |
| `src/utils/data_splitter.py` | `TimeSeriesSplitter` — embargo + sliding/expanding zaman-bilinçli split |

### Review Checklist

- [ ] **Zaman karıştırma yok** — hiçbir yerde rastgele train/test split; sadece kronolojik.
- [ ] **Scaler train-only fit** — `scale_data()` fit yalnız eğitim diliminde; test/fold/holdout sadece `transform()`.
- [ ] **Macro release-lag** — aylık CPI/faiz açıklama gecikmesi uygulanıyor (ileriye bakma yok).
- [ ] **Survivorship/listing** — kısa geçmiş, delisting, büyük boşluk uyarıları üretiliyor.
- [ ] **PSI eşikleri** — `<0.10` stable, `0.10–0.25` moderate, `≥0.25` hard block.
- [ ] **Korelasyon pruning 0.88** — eşik doğru, deterministik sıra.
- [ ] **Forward projection** — sadece son (recursive) satıra, geçmiş veri deterministik.
- [ ] **Config dataclass** — flat arg listesi yerine `config.py` dataclass'ları kullanılıyor.

### İlgili Testler + Komut

```bash
python -m pytest tests/test_data_services.py tests/test_data_quality.py \
  tests/test_audit_corporate_actions.py tests/test_psi.py \
  tests/test_feature_improvements.py tests/test_cross_sectional_momentum.py \
  tests/test_calendar_features.py tests/test_macro_cache_schema.py \
  tests/test_macro_forward_projection.py tests/test_phase5_data_quality.py -v
```

### Bağımlılık Notu

Aşamaların kökü — dışa bağımlılığı yok. **Çıktı:** `DataManager` üzerinden split
başına ölçeklenmiş tensörler (dict), `feature_names`, `wf_splits`,
`final_holdout_df`. Stage 2–3 bu sözleşmeye dayanır.

---

## Stage 2 — Model Tanımları & Factory

Tüm model implementasyonları ve hangi modelin üretim adayı / benchmark olduğunu
belirleyen scope kuralları.

### Dosyalar + Amaç

| Dosya | Amaç |
|---|---|
| `src/models/base_model.py` | `BaseModel` soyut sözleşme (train/predict/save/load) |
| `src/models/xgboost_model.py` ... `linear_model.py` | Üretim aday seti (XGBoost, RandomForest, GradientBoosting, Linear) |
| `src/models/lstm_model.py`, `lstm_lite_model.py`, `attention_lstm_v2_model.py` | Sequence modeller (LSTM, lite, temporal attention) |
| `src/models/naive_model.py`, `arima_model.py`, `prophet_model.py`, `prophet_hybrid_model.py`, `linear_sequence_model.py`, `quantile_lightgbm_model.py` | Benchmark / karşılaştırma / araştırma modelleri |
| `src/models/ensemble.py` | Ensemble (inverse-RMSE, cash-gated ağırlık) |
| `src/pipeline/model_factory.py` | Model sabitleri, spec, stage-aware factory; candidate/optional/legacy/benchmark setleri |
| `src/pipeline/model_scope.py` | Candidate vs benchmark seçim kuralları (registry-backed) |
| `src/pipeline/model_registry.py` | Run başına JSON registry (spec + metadata) |

### Review Checklist

- [ ] **BaseModel sözleşmesi** — her concrete model `train/predict/save/load` uyguluyor.
- [ ] **Candidate vs benchmark** — naive/ARIMA üretim adayı DEĞİL; benchmark olarak işaretli.
- [ ] **Scaler/metadata model yanında** — kaydedilen modelle birlikte sidecar.
- [ ] **Reproducibility** — seed numpy/TF/random'a uygulanıyor, deterministik.
- [ ] **Prophet kısıtı** — walk-forward desteklemeyen modeller doğru kısıtlanıyor.
- [ ] **Quantile desteği** — XGBoost/LightGBM quantile loss p10/p50/p90 üretiyor.

### İlgili Testler + Komut

```bash
python -m pytest tests/test_model_scope_production.py tests/test_model_registry.py \
  tests/test_model_filters.py tests/test_factory_registry_integration.py \
  tests/test_factory_spec_helpers.py tests/test_phase4_models.py \
  tests/test_dl_improvements.py tests/test_lstm_lite_model.py \
  tests/test_mc_dropout_lstm.py tests/test_quantile_lightgbm.py \
  tests/test_selection_guard.py tests/test_ensemble_weights.py -v
```

### Bağımlılık Notu

**Girdi:** Stage 1 feature isimleri (target/feature seti factory'ye gider).
**Çıktı:** Eğitilebilir model örnekleri + scope sözleşmesi. Stage 3 bu örnekleri
eğitir, Stage 6 scope'a göre üretim seçer.

---

## Stage 3 — Eğitim Workflow & Validation

Modelleri zaman-bilinçli protokollerle eğitir ve değerlendirir. Leakage'in en
kritik olduğu aşama.

### Dosyalar + Amaç

| Dosya | Amaç |
|---|---|
| `src/pipeline/training_workflows.py` | `SingleSplit` / `WalkForward` / `FinalHoldout` eğitim stratejileri |
| `src/pipeline/model_trainer.py` | `ModelTrainer` facade — workflow'lara delege |
| `src/validation/walk_forward.py` | `WalkForwardValidator` — concat-Sharpe + bootstrap CI |
| `src/validation/rolling_holdout.py` | Rolling pencere validasyon varyantı |
| `src/validation/purged_kfold.py` | Purged K-Fold (López de Prado, AFML Ch.7) |
| `src/validation/cpcv.py` | Combinatorial Purged CV (AFML Ch.12, multi-path OOS) |

### Review Checklist

- [ ] **Kronolojik fold bağımsızlığı** — her fold train'i kendi test'inden önce.
- [ ] **Embargo auto** — set edilmezse `max(200, time_steps)`; rolling-feature leakage engellenir.
- [ ] **Purge window** — test fold çevresindeki train örnekleri düşürülüyor.
- [ ] **Final holdout izolasyonu** — eğitimde/seçimde KULLANILMAZ, sadece teyit.
- [ ] **Concat-Sharpe determinizmi** — bootstrap seed sabit (`20260525`), 1000 resample.
- [ ] **Walk-forward default** — 12 split, 21 gün test, 756 gün max train (sliding).

### İlgili Testler + Komut

```bash
python -m pytest tests/test_training_workflows.py tests/test_validation_protocol.py \
  tests/test_walk_forward_default.py tests/test_embargo_auto.py \
  tests/test_purged_kfold.py tests/test_cpcv.py tests/test_concat_sharpe.py \
  tests/test_rolling_holdout.py tests/test_leakage_guards.py \
  tests/test_stability_score.py -v
```

### Bağımlılık Notu

**Girdi:** Stage 1 fold tensörleri + Stage 2 model örnekleri. **Çıktı:** Fold
başına tahminler + eğitilmiş modeller (`SingleSplitResult` / `WalkForwardResult`
/ `FinalHoldoutResult`). Stage 4 sinyal üretir, Stage 5 metriğe çevirir.

---

## Stage 4 — Backtest & Sinyal Kalibrasyonu

Tahminleri al/sat/tut sinyaline çevirir, maliyet/risk simülasyonu yapar, sinyal
parametrelerini kalibre eder.

### Dosyalar + Amaç

| Dosya | Amaç |
|---|---|
| `src/backtesting/engine.py` | `run_backtest()` — sinyal frame, trade execution, equity curve |
| `src/backtesting/signals.py` | `SignalConfig` + long/flat, simple, professional sinyal üretimi |
| `src/backtesting/signal_math.py` | Beklenen getiri / öneri / volatilite / regime matematiği |
| `src/backtesting/signal_validation.py` | Sinyal config validasyon kuralları |
| `src/backtesting/execution.py` | Execution dizileri, maliyet, emir kolonları |
| `src/backtesting/equity.py` | Equity-curve dataframe + sinyal kolon ekleme |
| `src/backtesting/trades.py` | Long/flat trade-log çıkarımı |
| `src/backtesting/metrics.py` | Backtest metrikleri (Sharpe, Sortino, max DD, win rate, VaR/CVaR) |
| `src/backtesting/reporting.py` | Backtest raporu (CSV/MD) + maliyet/risk disclaimer |
| `src/pipeline/signal_calibrator.py` | Sinyal eşik optimizasyonu (walk-forward leakage guard) |
| `src/pipeline/regime_context.py` | Sinyal bağlamı için regime tespiti |
| `src/pipeline/selection_guard.py` | Model seçimi leakage guard |

### Review Checklist

- [ ] **Final holdout kalibrasyonda KULLANILMAZ** — sinyal tuning sadece walk-forward train girdisi.
- [ ] **Walk-forward scope izolasyonu** — kalibrasyon fold dışına sızmıyor.
- [ ] **Maliyet/risk disclaimer** — varsayılan cost-free (commission=0, slippage=0); rapor disclaimer prepend ediyor.
- [ ] **Sinyal modları** — simple (AL/SAT/TUT), professional (gate'li), legacy doğru ayrışıyor.
- [ ] **Volatilite/regime gate** — professional sinyalde eşik + holding period + TP/SL uygulanıyor.

### İlgili Testler + Komut

```bash
python -m pytest tests/test_phase6_backtest_standard.py tests/test_signal_research.py \
  tests/test_regime_context.py tests/test_selection_guard.py -v
```

### Bağımlılık Notu

**Girdi:** Stage 3 fold tahminleri + Stage 1 fiyat serisi. **Çıktı:** Kalibre
`SignalConfig`, equity/trade frame'leri, backtest metrikleri. Stage 5 bunları
nihai metrik/confidence ile birleştirir.

---

## Stage 5 — Evaluation & Raporlama & XAI

Metrik hesabı, composite skor, confidence label türetimi ve açıklanabilirlik
çıktıları.

### Dosyalar + Amaç

| Dosya | Amaç |
|---|---|
| `src/pipeline/evaluation_workflows.py` | `SingleSplit` / `WalkForward` / `FinalHoldout` değerlendirme stratejileri |
| `src/pipeline/evaluation_services.py` | `PredictionService` / `BacktestService` / `SignalCalibrationService` / `MetricsReportingService` |
| `src/pipeline/evaluation_manager.py` | `EvaluationManager` facade — workflow/servis kompozisyonu |
| `src/pipeline/artifacts.py` | Rapor yazımı (validation, quality, manifest) + output sync |
| `src/pipeline/confidence_calculator.py` | Confidence label türetimi (degradation matrisi) |
| `src/evaluation/evaluator.py` | Metrik (MAE/RMSE/MAPE/Dir_Acc/composite), benchmark zenginleştirme, plot |
| `src/evaluation/financial_metrics.py` | Finansal metrik (Sharpe/Sortino/max DD), risk-free fallback davranışı |
| `src/xai/explainer.py`, `report_writer.py`, `feature_dictionary.py`, `narrative.py`, `product_summary.py`, `strategies.py` | SHAP/permütasyon açıklama, narrative, ürün özeti |

### Review Checklist

- [ ] **Metrik öncelik sırası** — advisory primary (Dir_Acc, Hit_Rate) → error → risk-adjusted → probabilistic → benchmark; Net_Return footnote.
- [ ] **Composite score ağırlıkları** — RMSE_vs_bench 0.30, DirAcc_vs_bench 0.25, Dir_Acc 0.20, Hit_Rate 0.15, Sharpe_excess 0.10.
- [ ] **Confidence degradation** — PSI high & corporate_action_anomaly hard block; moderate drift / risk_free_unavailable / clip_rate soft downgrade.
- [ ] **Risk-free fail-loud** — INTEREST_RATE.csv + env yoksa Sharpe/Sortino NaN + flag (fallback yok).
- [ ] **XAI fold stability** — feature importance fold'lar arası tutarlı; desteklenmeyen modelde fallback.

### İlgili Testler + Komut

```bash
python -m pytest tests/test_evaluation_services.py tests/test_reporting_metrics.py \
  tests/test_metrics_priority.py \
  tests/test_confidence_calculator.py tests/test_risk_free_fail_loud.py \
  tests/test_xai_routing.py tests/test_xai_strategies.py \
  tests/test_xai_product_summary.py tests/test_xai_fold_stability.py -v
```

### Bağımlılık Notu

**Girdi:** Stage 3 tahminleri + Stage 4 backtest sonuçları. **Çıktı:** Nihai
metrik tablosu, composite skor, confidence label, XAI raporları. Stage 6 bunları
persiste eder ve üretim lideri seçer.

---

## Stage 6 — Persistence & Üretim Seçimi

Sonuçları SQLite + JSON registry'ye yazar, üretim best-model'i seçer,
`outputs/{SYMBOL}/latest/` senkronlar.

### Dosyalar + Amaç

| Dosya | Amaç |
|---|---|
| `src/database/stock_model_db.py` | `StockModelDB` — merkezi SQLite registry facade |
| `src/database/repositories/schema.py` | Şema oluşturma + additive migrasyon |
| `src/database/repositories/experiment.py` | Experiment kaydı erişimi |
| `src/database/repositories/best_model.py` | Best-model upsert + selection guard + leaderboard |
| `src/database/repositories/forecast.py` | Idempotent forecast run/point persistence (`run_key`) |
| `src/database/repositories/forecast_resolution.py` | Actual-close çözümleme + accuracy summary |
| `src/database/repositories/helpers.py` | Ortak DB yardımcıları |
| `src/model_registry/model_registry.py` | Run başına JSON metadata + seçim işaretleri |
| `src/experiments/experiment_tracker.py` | CSV experiment loglama |
| `src/pipeline/orchestrator.py` | `ForecastingPipeline.run_all()` + outputs/ sync |
| `src/pipeline/model_result_exporter.py` | Model sonuç export (CSV/JSON) |

### Review Checklist

- [ ] **Atomik write-once artifact** — run çıktıları immutable, üzerine yazılmıyor.
- [ ] **latest/ sync izolasyonu** — sadece `outputs/{SYMBOL}/` kökü içinde senkron.
- [ ] **Forecast idempotency** — `run_key` ile tekrar yazım engelleniyor.
- [ ] **Best-model selection guard** — eligible, ineligible'ı yener; eşit sınıfta yüksek skor kazanır.
- [ ] **Şema migrasyon idempotency** — `_ensure_*_columns()` tekrar çalıştırmada güvenli.
- [ ] **Run ID/manifest sözleşmesi** — symbol/model/policy/timestamp adlandırması tutarlı.

### İlgili Testler + Komut

```bash
python -m pytest tests/test_stock_model_db_repositories.py tests/test_run_leaderboard.py \
  tests/test_run_manifest.py tests/test_run_id_naming.py \
  tests/test_model_result_exporter.py tests/test_smoke.py -v
```

### Bağımlılık Notu

**Girdi:** Stage 5 metrik/skor/confidence. **Çıktı:** SQLite `best_models` +
`experiments`, `outputs/{SYMBOL}/latest/`, JSON registry. Stage 7 forward
forecast üretirken bu kayıtlı best-model + artifact'ları okur.

---

## Stage 7 — Forward Forecast & API

Kayıtlı üretim artifact'larından ileri tahmin üretir ve HTTP analiz servisi
sunar. Yeni Sprint 7–9 özelliklerinin (PSI 30d, advisory audit, cache, rate
limit) çoğu burada.

### Dosyalar + Amaç

| Dosya | Amaç |
|---|---|
| `src/forecasting/runner.py` | `ForecastRunner` — kayıtlı artifact ile forward forecast orkestrasyonu |
| `src/forecasting/workflows.py` | Forecast iç servisleri (BestModelResolver, ForecastPointGenerator, vb.) |
| `src/forecasting/artifacts.py` | Forecast model/scaler/metadata sidecar persistence |
| `src/forecasting/persistence.py` | `ForecastPersistence` — DB + CSV forecast point kaydı |
| `src/forecasting/bist_rules.py` | `BistMarketRules` — seans saatleri, tatil |
| `src/forecasting/bist_calendar.py` | Deterministik BIST takvimi + manuel override merge |
| `src/api/main.py` | FastAPI app, CORS, rate-limit middleware, run/status job tracker |
| `src/api/routers/analysis.py` | `GET /analysis/{symbol}` + `/v1/analysis` (cache hit + audit append) |
| `src/api/schemas/analysis.py` | `AnalysisResponse` Pydantic şeması (forecast/confidence/data_quality/xai blokları) |
| `src/api/services/analysis_service.py` | `AnalysisService.build()` — best_model + forecast + confidence + PSI + XAI |
| `src/api/services/data_refresh_service.py` | Refresh job kuyruğu (data/macro/forecast) SQLite job takibi |
| `src/api/services/analysis_freshness.py` | Forecast/data bayatlık + refresh uygunluk kontrolü |
| `src/api/services/advisory_audit.py` | Advisory response geçmişi CSV audit (best-effort) |
| `src/api/services/response_cache.py` | In-memory 24h TTL response cache |
| `src/api/services/rate_limit.py` | Fixed-window IP rate limit (60/dk/IP) |
| `src/api/services/data_quality_monitor.py` | `compute_psi_30d()` — 30g holdout vs 252g train PSI |
| `src/api/runtime_config.py`, `observability.py` | CORS env config; JSON-line log |

### Review Checklist

- [ ] **Serve'de retrain YOK** — forecast kayıtlı model/scaler sidecar'dan; çalışma anında eğitim yok.
- [ ] **Refresh job idempotency** — symbol+reason queued/running iken de-dup.
- [ ] **Recursive feature recompute** — her ileri adımda SMA/EMA/RSI/MACD/Bollinger/ATR yeniden hesap.
- [ ] **BIST band clipping** — tahmin fiyat kuralları (tick, üst/alt band) uygulanıyor.
- [ ] **Response cache** — TTL 24h, key uppercase+strip, lazy eviction; env ile disable.
- [ ] **Rate limit** — 60/dk/IP, trusted IP muaf, over-limit 429 + Retry-After.
- [ ] **UTC datetime** — tüm `datetime.now()` → `datetime.now(tz=timezone.utc)`.
- [ ] **CORS** — local-first + `AI_CORE_CORS_ORIGINS` explicit; açık değil.
- [ ] **Advisory audit best-effort** — yazma hatası response'u bozmuyor.

### İlgili Testler + Komut

```bash
python -m pytest tests/test_forecasting.py tests/test_forecast_workflows.py \
  tests/test_prediction_date_aware.py tests/test_prophet_hybrid.py \
  tests/test_recompute_close_dependent.py tests/test_recursive_quantile_path.py \
  tests/test_multi_horizon_targets.py tests/test_psi_monitor.py \
  tests/test_analysis_endpoint.py tests/test_analysis_api_faz2.py \
  tests/test_response_cache.py tests/test_rate_limit.py \
  tests/test_advisory_audit.py tests/test_operational_hardening.py -v
```

Servisi elle test:

```bash
uvicorn src.api.main:app --reload --port 8000   # /docs Swagger
```

### Bağımlılık Notu

**Girdi:** Stage 6 best-model + kayıtlı artifact + SQLite registry. **Çıktı:**
Forecast point'leri (DB + CSV), `/analysis` HTTP response. En dış üretim katmanı;
sonraki aşama yalnız orkestrasyon.

---

## Stage 8 — CLI & Orkestrasyon

Kullanıcı giriş noktaları ve uçtan uca facade akışı.

### Dosyalar + Amaç

| Dosya | Amaç |
|---|---|
| `src/cli/interactive.py` | Menü tabanlı hisse/validation/model seçimi → `run_all()` |
| `src/cli/batch.py` | Çoklu hisse/konfig batch çalıştırma |
| `src/cli/forecast.py` | BIST-uyumlu forward forecast komutu |
| `src/cli/db_maintenance.py` | SQLite özet + backup-reset |
| `src/cli/run_leaderboard.py` | Leaderboard analiz CLI |
| `src/cli/signal_research.py` | Sinyal parametre araştırma CLI |
| `src/cli/_model_filters.py` | Interactive model filtre/seçim yardımcısı |
| `src/analysis/run_leaderboard.py`, `signal_research.py` | Run-seviyesi analiz araçları |
| `src/utils/reproducibility.py` | numpy/TF/random seed yönetimi |
| `src/utils/reporting_utils.py` | Output yol yönlendirme + rapor yazımı |
| `src/utils/risk_free_rate.py` | Risk-free oran fetch + fallback |

### Review Checklist

- [ ] **CLI parametre validasyonu** — geçersiz hisse/horizon/mode reddediliyor.
- [ ] **Reproducible run** — seed her çalıştırmada deterministik sonuç veriyor.
- [ ] **Config gruplama** — CLI flat arg yerine `config.py` dataclass'larına maplıyor.
- [ ] **Facade akışı** — `run_all()` data → train → eval → persist → sync sırasını koruyor.
- [ ] **Backward compat** — eski scope/sözleşme kırılmıyor.

### İlgili Testler + Komut

```bash
python -m pytest tests/test_interactive_cli_models.py tests/test_model_filters.py \
  tests/test_run_leaderboard.py tests/test_scope_backward_compat.py \
  tests/test_phase7_acceptance.py tests/test_phase8_acceptance.py -v
```

### Bağımlılık Notu

**Girdi:** Tüm Stage 1–7 katmanları (facade hepsini çağırır). **Çıktı:** Çalışan
CLI komutları + tam run pipeline. En üst orkestrasyon — bağımlılık ağacının tepesi.

---

## Çapraz-kesen endişeler

Belirli bir aşamaya ait olmayan, tüm aşamalara dokunan konular. Her aşama
review'ünde akılda tutulmalı.

- **Leakage guard'lar** — kronolojik sıra, train-only scaler fit, embargo/purge,
  final holdout izolasyonu, sinyal kalibrasyon scope'u. (Stage 1, 3, 4 yoğun;
  `tests/test_leakage_guards.py`.)
- **Reproducibility** — `src/utils/reproducibility.py` seed; bootstrap CI sabit
  seed; deterministik feature/macro projeksiyon.
- **UTC datetime** — Sprint 9'dan beri tüm zaman damgaları timezone-aware
  (`datetime.now(tz=timezone.utc)`); `RULES.md` datetime politikası.
- **CORS / rate-limit** — API güvenlik sınırı; local-first CORS, IP rate limit,
  trusted IP muafiyeti (Stage 7).
- **Disclaimer disiplini** — cost-free backtest + advisory uyarısı tüm rapor ve
  API response'larında (Stage 4, 5, 7).
- **Tam suite** — bir aşama biterken regresyon için:
  ```bash
  python -m pytest tests -q
  ```

## İlgili sayfalar

- [Architecture](architecture.md) — sistem mimarisi, teknoloji yığını, şema.
- [Data Pipeline](data-pipeline.md) — Stage 1 detay.
- [Model Catalog](model-catalog.md) — Stage 2 detay.
- [Validation and Backtesting](validation-and-backtesting.md) — Stage 3–4 detay.
- [Persistence and API](persistence-and-api.md) — Stage 6–7 detay.
- [Testing and Quality](testing-and-quality.md) — test süitleri + kalite gate'leri.
