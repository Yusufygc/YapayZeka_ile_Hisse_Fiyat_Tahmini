---
title: Staged Refactor Plan (2026-05-31)
type: plan
status: active
last_updated: 2026-05-31
owner: llm
source_count: 8
---

# Staged Refactor Plan — 2026-05-31

Bu sayfa, [Staged Code Review Guide](code-review-stages.md)'in **8 aşamasını**
temel alarak her aşamadaki god-object / şişkin dosya-fonksiyon / yüksek
karmaşıklık / SOLID-KISS-DRY ihlallerini listeler ve **davranış-koruyan** bir
refactor planı verir. Global bulgular [Code Quality Audit](code-quality-audit.md)
(B1–B4) sayfasındadır; bu plan onları aşamalara dağıtır ve uygulama sırası önerir.

**İlke:** Refactor kod davranışını DEĞİŞTİRMEZ. Her adım, aşamanın mevcut
testleriyle ([code-review-stages.md](code-review-stages.md) komutları) yeşil
kalarak doğrulanır. Test boşluğu olan adımda önce karakterizasyon testi yazılır.

## Eşikler ve etiketler

Eşikler ([Code Quality and Refactoring](code-quality-and-refactoring.md)):
dosya ≤500 satır, class ≤300 satır / ≤20 metod, fonksiyon ≤60 satır, siklomatik
karmaşıklık (CXTY proxy: dal+bool sayımı) hedef <12.

- Önem: 🔴 yüksek · 🟠 orta · 🟡 düşük
- Efor: **S** (<½ gün) · **M** (½–2 gün) · **L** (>2 gün, ayrı PR)
- Risk: davranış değişme riski (leakage/sözleşme kırma).

Ölçüm anı (AST, `src/`): en yüksek CXTY `_add_walk_forward_ensembles` **58**,
`_add_single_split_ensembles` **51**, `compute_confidence` **41**, `batch.main`
**37**, `compute_market_regime` **33**. En büyük dosya `signal_calibrator.py`
**997**. En büyük class `_SignalCalibratorMixin` **34 metod/898 satır**.

---

## Çapraz-kesen epikler (aşama-üstü)

Bu üçü tek bir aşamaya sığmaz; aşama-içi adımlardan **bağımsız ele alınır** ve
en yüksek mimari borçtur.

### E1 — Owner-forward kozmetik servis ayrımı (🔴 L) [B1]

`pipeline/evaluation_services.py` (`_OwnerBackedService`) ve
`forecasting/workflows.py` (`_OwnerBackedForecastService`): `__getattr__`/
`__setattr__` tüm attribute erişimini tek `owner`'a yönlendirir. "Servisler"
gerçek sorumluluk almaz, paylaşılan mutable owner state üzerinde çalışır
→ SRP/encapsulation yok, `__setattr__` typo sessizce owner'a yazar.

**Hedef:** owner-forward yerine açık bağımlılık enjeksiyonu. `EvaluationContext`
(salt-okunur girdi sözleşmesi) + dar protokoller; mixin'leri davranış-sahibi
servislere çevir. Adım adım: (1) owner'ın okunan/yazılan attribute'larını
envanterle, (2) `__getattr__`'ı kaldırıp açık ctor parametrelerine çevir, (3) her
servis kendi state'ini taşısın. **Stage 5 (eval) + Stage 7 (forecast) birlikte.**

### E2 — DRY birleştirmeleri [B4]

- 🔴 **İkiz ensemble builder** — `prediction_engine._add_single_split_ensembles`
  (176L/CXTY51) vs `_add_walk_forward_ensembles` (197L/CXTY58). İkisi de
  inverse-RMSE + sequence-model alt blokları, yakın kopya. Ortak
  `_assemble_ensemble(members, weights_strategy)` helper'a çıkar. **Stage 5.**
- 🟠 **Tree `tune_and_train` ikizi** — `xgboost_model` (119L) +
  `random_forest_model` (103L) + gradient boosting benzer Optuna döngüsü. Ortak
  `_tune_tree_model` template-method veya mixin. **Stage 2.**
- 🟠 **3× `run()` workflow** — `evaluation_workflows.py` SingleSplit/WalkForward/
  FinalHoldout `run()` (147/112/77L) aynı iskelet. Template-method taban sınıf.
  **Stage 5.**
- 🟡 **Model-instance factory ikizi** — `forecasting/runner._make_model_instance`
  (88L/CXTY18) vs `pipeline/model_factory`. Tek factory'de birleştir. **Stage 2/7.**

### E3 — God constructor'lar (🟠 M)

`orchestrator.ForecastingPipeline.__init__` (123L/~40 attribute) ve
`evaluation_manager.EvaluationManager.__init__` (104L). Builder/DI ile
parçalanır; E1 ile koordine. **Stage 5/6.**

---

## Aşama bazlı plan

### Stage 1 — Veri & Konfig & Feature  (17 dosya, 3707 LOC)

| Hedef | Bulgu | Aksiyon | Önem/Efor |
|---|---|---|---|
| `features/macro_pipeline.py` 693L, `MacroPipeline` 29m/560L | fetch (EVDS/yfinance) + transform + cache + projeksiyon tek sınıfta — SRP | `MacroFetcher` (IO) / feature-engineering / cache sorumluluklarını ayır; `macro_transforms` zaten saf transform — oraya kaydır | 🟠 M |
| `utils/data_splitter.py` `walk_forward_splits` 85L/**CXTY27** | sliding/expanding + embargo + window dallanması iç içe | embargo hesabı + pencere hesabını saf helper'lara çıkar (leakage-kritik, testli) | 🔴 S |
| `data/data_updater.py` `check_and_update` 129L/CXTY17 | indirme + kurumsal aksiyon + derinlik validasyonu tek fn | adım fonksiyonlarına böl (`_fetch`, `_audit_actions`, `_validate_depth`) | 🟠 S |
| `data/quality.py` `compute_quality_flags` 73L/CXTY17 | ardışık if-zinciri bayrak kuralları | tablo-güdümlü kural listesi (rule → flag) | 🟡 S |
| `data/data_loader.py` `load_and_clean` 114L/CXTY13 | kolon-çeviri + sıfır-hacim + indikatör tek fn | adım fonksiyonlarına böl | 🟡 S |
| `features/feature_pipeline.py` `FeaturePipeline` 17m/545L, `_merge_macro` 95L | sınır LOC aşımı; macro-merge ayrı endişe | `_merge_macro`'yu collaborator'a taşı | 🟡 M |

**Test guard:** `test_data_services test_data_quality test_psi test_feature_improvements test_audit_corporate_actions`. Leakage kritik — splitter/scaler değişiminde `test_leakage_guards`.

### Stage 2 — Model & Factory  (19 dosya, 3709 LOC)

| Hedef | Bulgu | Aksiyon | Önem/Efor |
|---|---|---|---|
| `xgboost_model.tune_and_train` 119L, `random_forest_model.tune_and_train` 103L | DRY: tree modelleri Optuna döngüsü ikiz | E2: ortak `_tune_tree_model` template-method | 🟠 M |
| `models/ensemble.py` 403L, `EnsembleModel` 14m | sınırda; `optimize_by_ridge_stacker` CXTY12 | düşük öncelik; ağırlık-stratejilerini ayır | 🟡 S |
| `attention_lstm_v2_model._tune_hyperparameters` 81L | sequence tuning uzun | yardımcıya böl | 🟡 S |

En temiz aşama. Öncelik düşük; sadece E2 tree-DRY anlamlı kazanım.

**Test guard:** `test_phase4_models test_quantile_lightgbm test_ensemble_weights test_factory_registry_integration`.

### Stage 3 — Eğitim & Validation  (7 dosya, 1131 LOC)

| Hedef | Bulgu | Aksiyon | Önem/Efor |
|---|---|---|---|
| `validation/walk_forward.py` `run` 171L/**CXTY28** | fold-döngüsü + concat-Sharpe + bootstrap-CI tek fn (en kritik leakage noktası) | `_run_folds`, `_concat_sharpe`, `_bootstrap_ci` helper'larına böl; determinizm (seed 20260525) korunur | 🔴 M |
| `training_workflows.py` `run` CXTY12, 308L | strateji dağıtımı kabul edilebilir | E2 ile uyumlu, müdahale yok | 🟡 — |
| `model_trainer.py` `ModelTrainer` 25m/181L | metod çok ama küçük; facade | bölme gerekmez (KISS) | 🟡 — |

**Test guard:** `test_walk_forward_default test_concat_sharpe test_embargo_auto test_purged_kfold test_cpcv test_leakage_guards`. Determinizm: bootstrap CI sabit-seed assertion'ları.

### Stage 4 — Backtest & Sinyal  (13 dosya, 3738 LOC) — en şişkin aşama

| Hedef | Bulgu | Aksiyon | Önem/Efor |
|---|---|---|---|
| `pipeline/signal_calibrator.py` **997L**, `_SignalCalibratorMixin` **34m/898L** | repo'nun en büyük dosyası + en büyük class; owner-forward mixin | sorumlulukları ayrı collaborator'lara böl: threshold-kalibrasyon / WF-parametre-arama (`_calibrate_walk_forward_signal_parameters` 118L/CXTY19) / OOS-confirmation / decision-MD raporlama. `signal_calibration/grid.py`+`selection.py` zaten var — oraya taşı | 🔴 L |
| `backtesting/engine.py` `run_backtest` 204L/CXTY16 | sinyal-frame + execution + equity + gate tek orkestratör fn | adımları çağıran ince orkestratör; `_build_signal_frame` zaten var, execution/equity/gate'i de ayır | 🔴 M |
| `backtesting/signals.py` `generate_professional_signals` 202L/**CXTY21** | gate + holding + TP/SL iç içe | gate/holding/TP-SL alt-kurallarını saf fonksiyonlara çıkar | 🟠 M |
| `backtesting/metrics.py` `summarize_backtest` 193L/**CXTY27** | tüm metrik aileleri tek fn | aile bazlı böl: return / risk-adj / drawdown / trade-quality | 🟠 M |
| `pipeline/regime_context.py` `compute_market_regime` **CXTY33** | repo'daki en yüksek tek-fn karmaşıklık (S4) | eşik-tablosu güdümlü; regime kuralları veri yapısına | 🟠 M |
| `backtesting/reporting.py` `save_backtest_report` 118L | rapor + disclaimer | şablon parçalarına böl | 🟡 S |

**Test guard:** `test_phase6_backtest_standard test_signal_research test_regime_context test_selection_guard`. **Leakage:** kalibrasyon final-holdout'a sızmamalı — WF-scope assertion'ları korunur.

### Stage 5 — Evaluation & XAI  (16 dosya, 4593 LOC) — god-object merkezi

| Hedef | Bulgu | Aksiyon | Önem/Efor |
|---|---|---|---|
| `pipeline/prediction_engine.py` 758L, `_PredictionEngineMixin` 10m/714L | `_add_walk_forward_ensembles` **CXTY58** + `_add_single_split_ensembles` **CXTY51** ikiz (B4) — repo'nun en karmaşık fn'leri | E2: ortak `_assemble_ensemble` helper; per-model-type blokları ayır | 🔴 L |
| `pipeline/evaluation_manager.py` `EvaluationManager` **48m/533L**, `__init__` 104L | de-facto god object (B1) | E1 + E3: DI ile servisleri davranış-sahibi yap | 🔴 L |
| `pipeline/evaluation_services.py` | owner-forward `_OwnerBackedService` (B1) | E1 | 🔴 L |
| `pipeline/confidence_calculator.py` `compute_confidence` 143L/**CXTY41** | degradation matrisi dev if-zinciri | tablo-güdümlü kural motoru (koşul → label/sebep); hard-block/soft-downgrade ayrı | 🟠 M |
| `xai/explainer.py` `XAIExplainer` 27m/554L, `_decision_explanation` CXTY25 | sınır aşımı + karmaşık | strateji bazlı böl (`strategies.py` zaten var) | 🟠 M |
| `evaluation/evaluator.py` `save_metrics_report` 154L | metrik tablosu + plot + yazım | hesap / formatla / yaz ayır | 🟡 M |
| `pipeline/evaluation_workflows.py` 3× `run()` (147/112/77L) | DRY (B4) | E2: template-method taban sınıf | 🟠 M |

**Test guard:** `test_metrics_priority test_composite_score test_confidence_calculator test_evaluation_services test_xai_*`.

### Stage 6 — Persistence & Üretim Seçimi  (17 dosya, 4184 LOC)

| Hedef | Bulgu | Aksiyon | Önem/Efor |
|---|---|---|---|
| `pipeline/data_services.py` **791L** | `prepare_tensors` 106L/CXTY23, `run` 123L/CXTY20, `check_survivorship_bias` 88L/CXTY19, `split_data` 72L | data-prep / scaling-report / survivorship / split ayrı collaborator'lara; leakage-kritik | 🔴 M |
| `database/stock_model_db.py` **807L**, `StockModelDB` 33m/513L | facade; `compute_composite_score` 84L heavy logic | skorlama mantığını ayrı modüle (evaluator ile DRY kontrolü); facade ince kalsın | 🟠 M |
| `pipeline/orchestrator.py` `run_all` 143L/CXTY15, `__init__` 123L | E3 god constructor; `ForecastingPipeline` 13m/511L | E3: builder/DI; `run_all` adımları (data→train→eval→persist→sync) ayrı metodlara | 🟠 M |
| `database/repositories/best_model.py` `upsert_best_from_values` 85L, `update_production_best_model` 82L | selection-guard mantığı uzun | guard kurallarını ayrı saf fonksiyona | 🟡 S |

**Test guard:** `test_stock_model_db_repositories test_run_manifest test_run_leaderboard test_smoke`. **İdempotency:** `run_key` forecast + şema migrasyon idempotency korunur.

### Stage 7 — Forward Forecast & API  (24 dosya, 3962 LOC)

| Hedef | Bulgu | Aksiyon | Önem/Efor |
|---|---|---|---|
| `forecasting/workflows.py` **881L** | `_OwnerBackedForecastService` owner-forward (B1); 7 workflow ikiz iskelet; `roll_forward_recursive` 83L | E1: DI; recursive forecast nedensel zinciri korunur | 🔴 L |
| `api/services/analysis_service.py` 692L, `build` 127L/CXTY15 | `_build_forecast_response` 111L + `_build_*` parça-üretici god-method | block-builder'lar zaten var; `build`'i ince orkestratöre indir | 🟠 M |
| `api/main.py` 425L | app + CORS + rate-limit middleware + job-tracker karışık | job-tracker'ı ayrı modüle, middleware kaydını ayır | 🟠 M |
| `forecasting/runner.py` `_make_model_instance` 88L/CXTY18 | E2 model-factory ikizi | tek factory'de birleştir | 🟡 M |

**Test guard:** `test_forecasting test_forecast_workflows test_analysis_endpoint test_analysis_api_faz2 test_recursive_quantile_path test_response_cache test_rate_limit`. **Sözleşme:** serve'de retrain YOK, UTC datetime, cache/rate-limit davranışı korunur.

### Stage 8 — CLI & Orkestrasyon  (14 dosya, 3476 LOC)

| Hedef | Bulgu | Aksiyon | Önem/Efor |
|---|---|---|---|
| `cli/batch.py` `main` 150L/**CXTY37** | en yüksek CLI karmaşıklığı; god-procedure | arg-parse / per-stock döngü / özet ayrı handler'lara | 🟠 M |
| `cli/signal_research.py` `main` 195L/CXTY23 | god-procedure | komut handler'larına böl | 🟠 M |
| `analysis/run_leaderboard.py` 739L | `_reliability_class` CXTY28, `_sector_for_symbol` CXTY19 | eşik/lookup tablo-güdümlü | 🟡 M |
| `analysis/signal_research.py` 669L | `run_research_matrix` 146L/CXTY23 | matris-üretimi alt fonksiyonlara | 🟡 M |
| `cli/{forecast,run_leaderboard}.py` `main` 85/98L | tüm CLI main'leri god-procedure | ortak komut-handler deseni | 🟡 S |

**Test guard:** `test_interactive_cli_models test_model_filters test_run_leaderboard test_scope_backward_compat test_phase8_acceptance`.

---

## Önerilen uygulama sırası (risk-ayarlı)

Aşama-sırası bağımlılık içindir; **uygulama sırası** risk/kazanım dengesidir.
Önce ucuz-güvenli fonksiyon parçalama, sonra DRY, en son mimari.

**Tier 0 — Fonksiyon parçalama (🟢 düşük risk, testle korunur):** mevcut testler
güçlü, davranış aynı. Sıra: `data_splitter.walk_forward_splits` (S1, leakage
ama testli) → `walk_forward.run` (S3) → `metrics.summarize_backtest` (S4) →
`compute_market_regime` (S4) → `confidence_calculator.compute_confidence` (S5) →
CLI main'leri (S8). Quick win, karmaşıklık düşer.

**Tier 1 — DRY birleştirme (🟠 orta):** E2. İkiz ensemble builder (S5, en yüksek
kazanım) → 3× `run()` workflow (S5) → tree `tune_and_train` (S2) → model-factory
ikizi (S2/S7). Kopya kod silinir, regresyon riski testle yönetilir.

**Tier 2 — Dosya/sorumluluk bölme (🟠 orta):** `signal_calibrator.py` (S4) →
`data_services.py` (S6) → `analysis_service.build` (S7) → `api/main.py` (S7) →
`macro_pipeline.py` (S1). Her biri ayrı PR.

**Tier 3 — Mimari (🔴 yüksek, ayrı epik):** E1 owner-forward kaldırma
(`EvaluationManager`+`evaluation_services` S5, `forecasting/workflows` S7) + E3
god constructor (orchestrator/evaluation_manager). Karakterizasyon testi şart;
en son.

## Durum

- 2026-05-31: Plan oluşturuldu (taze AST taraması: per-stage LOC/CXTY/god-class).
  Docstring Faz 3-4 ([code-quality-audit.md](code-quality-audit.md)) bu refactor
  sonrasına ertelendi (yapı değişeceği için yorum eklemek erken).
- 2026-05-31: **Tier 0 (fonksiyon parçalama) tamamlandı** — davranış-koruyan,
  helper çıkarma. 7 hedef fonksiyonun karmaşıklığı hedefe (<12) indi:

  | Fonksiyon | Dosya | CXTY önce→sonra | LOC önce→sonra |
  |---|---|---|---|
  | `walk_forward_splits` | utils/data_splitter | 27→6 | 85→70 |
  | `WalkForwardValidator.run` | validation/walk_forward | 28→7 | 171→43 |
  | `summarize_backtest` | backtesting/metrics | 27→10 | 193→130 |
  | `compute_market_regime` | pipeline/regime_context | 33→4 | — |
  | `compute_regime_context` | pipeline/regime_context | 24→7 | — |
  | `compute_confidence` | pipeline/confidence_calculator | 41→11 | 143→84 |
  | `batch.main` | cli/batch | 37→6 | 150→38 |

  Çıkarılan helper'lar: `_resolve_split_count`/`_window_bounds`/`_first_last_date`
  (splitter); `_run_single_fold`/`_aggregate_metrics` (WF); `_resolve_risk_free`/
  `_empty_metrics`/`_exposure_and_turnover`/`_cost_drags`/`_basic_trade_stats`
  (metrics); `_regime_short_only`/`_regime_full`/`_forecast_alignment` (regime);
  `_check_hard_blocks`/`_soft_degradation_reasons` (confidence); `_resolve_symbols`/
  `_resolve_mode`/`_resolve_models`/`_print_banner`/`_execute_batch`/
  `_print_and_save_summary` (batch). Doğrulama: py_compile + 121 ilgili test yeşil.
- 2026-05-31: **Tier 1 (E2 DRY) tamamlandı** — kopya kod tek kaynağa indirildi:
  - **İkiz ensemble builder:** `prediction_engine._add_single_split_ensembles`
    (CXTY 51→17) ve `_add_walk_forward_ensembles` (CXTY 58→21); ~140 satır kopya
    çekirdek `_compute_ensemble_blends` helper'ına çıkarıldı + ortak `payload`
    dilimleme `_slice_template_payload`'a alındı. Cash-gate length-guard ortaklaştı
    (normal durumda SS ile birebir, dejenere durumda strictly daha güvenli).
  - **Tree `tune_and_train`:** yeni `src/models/_tuning.py` — `run_optuna_study`
    (SQLite warm-start + fallback) + `stability_adjusted_cv_objective` (Sharpe CV).
    XGBoost + Random Forest artık bu ortak çekirdeği kullanır (modele özgü tek
    kısım param-space + fit yolu). ~50 satır × 2 kopya silindi.
  - **3× `run()` workflow:** İncelemede güçlü ikiz OLMADIĞI görüldü (workflow
    gövdeleri esasen farklı). Zorla template-method KASITLI uygulanmadı (B1
    owner-forward kozmetik-SRP tuzağına düşmemek için). Yalnızca gerçek tekrar olan
    metadata-attach çiftleri (`composite+scope`, `leakage+family`)
    `_attach_score_metadata`/`_attach_guard_metadata` modül helper'larına alındı.
  - Doğrulama: py_compile + 114 ilgili test + tune_and_train fonksiyonel smoke yeşil.
- 2026-05-31: **Tier 2 (dosya/sorumluluk bölme) tamamlandı** — her hedef ayrı
  commit. Davranış-koruyan; her birinde ilgili test süiti yeşil.
  - **`analysis_service.build`:** iki neredeyse-aynı refresh dalı (missing_forecast
    + stale_data) `_try_refresh_and_rebuild`'e DRY'landi. CXTY/LOC düştü.
  - **`api/main.py` (425→~340L):** POST /run job tracker (`RunRequest`/`RunStatus`,
    `_jobs`, inline `_bg_run`) yeni `src/api/services/pipeline_jobs.py`'ye taşındı.
    main artık yalnız route katmanı — gerçek SRP ayrımı.
  - **`macro_pipeline.py`:** `get_macro_features` global-gösterge döngüsü
    `_refresh_global_daily_frames`'e alındı (mevcut `_refresh_*` desenine uygun).
    **Bulgu:** MacroPipeline 29 metod ama çoğu <25L + yüksek kohezyon → zorla
    file-split YOK (cache state coupling riski, marjinal kazanım).
  - **`data_services.py`:** `DataIngestionService.run` (123→~30L) iki helper'a
    bölündü (`_engineer_features_cached` + `_print_ingestion_summary`).
    **Bulgu:** dosya zaten 4 SRP servise ayrık; **paket-split GÜVENSİZ** çünkü
    testler `data_services.load_data`/`DataUpdater`/`FeatureCache`'i monkeypatch
    ediyor (modül namespace değişimi patch hedeflerini kırar). In-place
    decomposition seçildi; `prepare_tensors` (scaling, yüksek sayısal risk) ertelendi.
  - **`signal_calibrator.py` (997L):** **bulgu** — zaten yoğun decompose: 34 küçük
    metod, grid logic önceden `signal_calibration/grid.py`'ye çıkarılmış,
    `_calibrate_walk_forward_signal_parameters` temiz orkestrasyon. Zararlı god
    object DEĞİL; büyüklük dağıtılmış kalibrasyon karmaşıklığından. Esas borç
    owner-forward mixin coupling (**B1/E1**). Zorla file-split kozmetik + owner-state
    riski (tam B1 tuzağı) olacağı için YAPILMADI; derin bölme **E1/Tier 3'e ertelendi**.
- **Genel ilke (Tier 2):** "büyük dosya" her zaman "kötü dosya" değildir.
  macro_pipeline + signal_calibrator zaten iyi-faktörlü; gerçek borç owner-forward
  mimari (E1). Kozmetik LOC-azaltma için zorla bölme yapılmadı.
- **Sıradaki:** Tier 3 (mimari — E1 owner-forward kaldırma + E3 god constructor).
  Karakterizasyon testi gerektiren ayrı epik; en yüksek risk.
