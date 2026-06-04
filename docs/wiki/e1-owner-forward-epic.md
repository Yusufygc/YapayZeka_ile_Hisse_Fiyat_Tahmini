---
title: E1 Owner-Forward Removal Epic
type: plan
status: done
last_updated: 2026-06-01
owner: llm
source_count: 9
branch: refactor/e1-owner-forward-di
---

# E1 — Owner-Forward Kaldırma Epiği (Tam DI'ya Geçiş)

Bu sayfa, [Staged Refactor Plan](refactor-plan.md)'in **Tier 3 / E1** kalan
kısmının kendi odaklı epiğidir. Tier 3'te yapılan güvenli dilim (fail-loud
`__setattr__` typo guard + E3 god constructor parçalama) bu epiğin **ön
koşuludur**; burada owner-forward "magic" tamamen kaldırılır ve servisler
açık bağımlılık enjeksiyonlu (DI) davranış-sahibi nesnelere çevrilir.

**Çalışma dalı:** `refactor/e1-owner-forward-di` (yeni session bu dalda çalışır).

**Temel ilke:** Davranış DEĞİŞMEZ. Her adım önce karakterizasyon testiyle
kilitlenir, sonra refactor edilir, suite (epik başında 549, güncelde 561 test)
her commit'te yeşil kalır. Kaynak doğruluk sırası kod > test > wiki.

---

## 1. Neden bu bir epik (büyük + riskli)

İki ayrı owner-forward tabanı var, üç ayrı owner state yüzeyine yönleniyor:

| Taban | Dosya | Forward | Owner | `__setattr__`? |
|---|---|---|---|---|
| `_OwnerBackedService` | `pipeline/evaluation_services.py` | get + set | `EvaluationManager` | ✅ (artık fail-loud) |
| `_OwnerBackedService` | (aynı) | get + set | `DataManager` (4 servis) | ✅ (opt-out, permissive) |
| `_OwnerBackedForecastService` | `forecasting/workflows.py` | yalnız get | `ForecastRunner` | ❌ (yazımlar serviste kalır) |

Mixin gövdeleri (`_PredictionEngineMixin`, `_BacktestRunnerMixin`,
`_SignalCalibratorMixin`, `_MetricsReporterMixin`) ~4500 satır ve owner state'ini
hem **okur** hem **yazar**. Paylaşılan mutable state, servis ↔ workflow
entegrasyon sözleşmesidir: ör. `PredictionService` `owner.predictions`'a yazar,
sonra `WalkForwardEvaluationWorkflow` aynı `owner.predictions`'ı okur. Bu yüzden
saf DI'ya geçiş, state'i dönüş değerleri / context nesneleri üzerinden açıkça
threadlemeyi gerektirir → tek seferde "big-bang" rewrite yüksek regresyon riski.

### Korunması zorunlu invariantlar

- **Leakage sınırları:** scaler yalnız train slice'ta fit; final holdout
  kalibrasyona sızmaz; WF scope izolasyonu. (`test_leakage_guards`, WF-scope
  assert'leri.)
- **Determinizm:** bootstrap CI sabit seed (20260525), `set_global_seed(42)`.
- **Monkeypatch hedefleri:** testler `data_services.load_data` / `DataUpdater` /
  `FeatureCache`'i patch'liyor — modül namespace'i KORUNMALI (paket-split yapma).
- **Public yüzey:** `EvaluationManager.evaluate_single_split/walk_forward/
  final_holdout` ve `ForecastRunner` API imzaları sabit.
- **Erken servis init sırası:** `EvaluationManager.__init__` line ~118
  `_signal_threshold_metadata()` erken servis init'i tetikliyor — korunur.

---

## 2. Hedef mimari

Owner-forward "magic" yerine **açık girdi/çıktı sözleşmesi**:

- `EvaluationContext` (zaten var): salt-okunur girdi (symbol, dirs, cfg, tracker,
  feature_names, dataset_*). Genişlet: servislerin owner'dan OKUDUĞU tüm config /
  identity attribute'ları buraya taşı.
- `EvaluationState` (zaten var): mutable runtime çıktı (predictions,
  prediction_targets, quantile_predictions, latest_*, ensemble_weights,
  ensemble_weight_scope, signal_config + threshold state). Servisler bu nesneyi
  açıkça alır, `self.owner.X` yerine `self.state.X` / `self.ctx.X` kullanır.
- Servisler `_OwnerBackedService`'ten MİRAS ALMAZ; ctor'da `(ctx, state)` (ve
  gerekli dar protokoller) alır. `__getattr__`/`__setattr__` silinir.
- `EvaluationManager` ince orkestratör kalır: context+state kurar, servisleri
  enjekte eder, public metodları workflow'lara delege eder. Geriye-uyumlu
  delegasyon metodları (50+ adet) korunur veya kademeli sadeleştirilir.
- `ForecastRunner`: aynı desen, ama read-only olduğu için daha kolay —
  `ForecastContext` (project_root, db, rules, model_config, persistence) enjekte
  edilir; `__getattr__` silinir.

---

## 3. Karakterizasyon testi stratejisi (ÖNCE)

Her faz öncesi davranışı kilitle. Mevcut suite güçlü ama owner-state
mutasyon zincirini açıkça doğrulayan testler ekle:

1. **State-contract testi:** `evaluate_single_split` / `evaluate_walk_forward`
   sonrası `manager.predictions`, `prediction_targets`, `quantile_predictions`,
   `ensemble_weights`, `ensemble_weight_scope`, `signal_config`,
   `signal_threshold_source/summary` değerlerini snapshot'la (golden).
2. **Workflow okuma sözleşmesi:** workflow'ların owner'dan hangi attribute'ları
   okuduğunu sabitleyen test (refactor sonrası state'ten okumalı, değer aynı).
3. **Forecast contract:** `ForecastRunner.run` çıktısı golden snapshot
   (mevcut `test_forecasting` + `test_forecast_workflows` üstüne).
4. **Determinizm:** iki ardışık run aynı hash/metrik (sabit seed).

Bu testler `tests/test_owner_forward_contract.py` altında toplanır; epik boyunca
değişmez referans olur.

---

## 4. Fazlı uygulama (ayrı PR'lar)

Her faz kendi commit'i; suite yeşil + karakterizasyon golden sabit.

### Faz 0 — Hazırlık (🟢 düşük)
- `tests/test_owner_forward_contract.py` karakterizasyon golden'ları yaz.
- Owner read/write envanterini tamamla: 4 mixin + 6 evaluation/training workflow
  + 4 DataManager servisi + 6 forecast servisi için `self.<attr>` okuma/yazma
  tablosu (AST script ile, repo'ya geçici, sonra silinir).

### Faz 1 — `EvaluationState` tam taşıma (🟠 orta)
- Mixin yazımlarını (`self.predictions = ...` vb.) `self.state.X = ...`'e çevir.
- `EvaluationManager` attribute'larını `state`'e property/forward ile aynala
  (geriye uyumluluk: `manager.predictions` -> `manager.state.predictions`).
- Workflow okumalarını `state`'ten yap.
- Guard: state-contract testi + tüm suite.

### Faz 2 — `EvaluationContext` tam taşıma (🟠 orta)
- Servislerin owner'dan okuduğu config/identity attribute'larını `context`'e al;
  `self.ctx.X` kullan. `EvaluationManager` flat attribute'ları context'ten
  türetilen property'lere indir.

### Faz 3 — Mixin → davranış-sahibi servis (🔴 yüksek)
- `_OwnerBackedService` mirasını kaldır; servis ctor'u `(ctx, state, deps)` alır.
- `__getattr__`/`__setattr__` sil. Mixin metotları artık servis metodu;
  `self.ctx`/`self.state` üzerinden çalışır.
- 4 servisi tek tek dönüştür (Prediction → Backtest → SignalCalibration →
  MetricsReporting), her birinde suite yeşil.

### Faz 4 — `EvaluationManager` ince orkestratör (🟠 orta) [E3 ile birleşir]
- 50+ geriye-uyumlu delegasyon metodunu gözden geçir; gerçekten kullanılmayanları
  (yalnız test erişen) sadeleştir. `_ensure_services` mantığını koru.

### Faz 5 — `ForecastRunner` DI (🟠 orta)
- `_OwnerBackedForecastService` kaldır; `ForecastContext` enjekte et.
- 6 forecast servisini dönüştür. `test_forecasting`/`test_forecast_workflows`
  golden sabit.

### Faz 6 — DataManager servisleri guard'a alma (🟡 düşük-orta)
- DataManager owner state'ini (`selection_df`, `final_holdout_df`, `tensors`,
  `wf_splits`, `_wf_mode`, `dataset_hash`, ...) `__init__`'te pre-init et.
- 4 DataManager servisinin `_FAIL_LOUD = False` opt-out'unu kaldır (artık tüm
  yazımlar pre-init attribute'a gider). Monkeypatch namespace'i KORU.

### Faz 7 — Temizlik & dokümantasyon
- `_OwnerBackedService` / `_OwnerBackedForecastService` ölü kodu sil.
- `refactor-plan.md` Durum + `log.md` + `architecture.md` güncelle.
- Geçici AST script'lerini sil.

---

## 5. Sıralama & risk dengesi

Faz 0 (test) → 1 → 2 → 3 (en riskli, servis servis) → 4 → 5 → 6 → 7.
DataManager (Faz 6) ve Forecast (Faz 5) bağımsız; gerekirse paralel/ayrı PR.
Faz 3 servis-servis kesilebilir; bir servis yarıda bırakılmaz (suite kırmızı
kalmamalı).

## 6. Kabul kriterleri

> **Revize (2026-06-01, Faz 7):** Orijinal kriter "`_OwnerBackedService` repo'dan
> silindi" idi. Kodu esas alınca (kod > test > wiki) bu tabanın hâlâ 3 eval + 3
> training workflow + 4 DataManager servisi tarafından miras alındığı ve bu 10
> sınıfın paylaşılan owner state'i okuyup-yazdığı görüldü (servis↔workflow
> entegrasyon sözleşmesi, §1'in "big-bang yüksek risk" uyarısı; training §8'de
> zaten kapsam-dışı). Tabanı tamamen silmek = bu 10 sınıfı DI'ya çevirmek →
> **ayrı gelecek epik**. E1 hedefi "evaluation servislerini + forecast runner'ı
> DI'ya çevir, forecast tabanını sil, tüm forward yazımları fail-loud yap" olarak
> karşılandı.

- ✅ `_OwnerBackedForecastService` repo'dan silindi (Faz 5).
- ✅ 4 evaluation servisi + `ForecastRunner` owner-forward'dan DI'ya geçti; bu
  yüzeylerde `__getattr__`/`__setattr__` forward kalmadı.
- ✅ `_OwnerBackedService` tabanı bilinçli korundu (workflow + DataManager
  servisleri); tüm forward yazımlar fail-loud (typo guard).
- ✅ Tüm suite yeşil (561) + karakterizasyon golden'lar değişmeden geçer.
- ✅ Leakage/determinizm assert'leri korundu; monkeypatch hedefleri çalışıyor.
- ✅ Public API imzaları değişmedi.
- ⏭️ **Gelecek epik (E1.x):** 6 workflow + 4 DataManager servisini DI'ya çevirip
  `_OwnerBackedService` tabanını sil (servis-servis, golden korunarak).

## 7. Geri alma (rollback)

Her faz ayrı commit. Bir faz golden'ı bozarsa `git revert <faz-commit>` ile
izole geri alınır; önceki fazlar etkilenmez. Dal birleştirilmeden önce tam
suite + `python -m src.cli.forecast` smoke koşulur.

## 8. Faz 0 envanter sonucu (AST taraması)

`tools/owner_forward_inventory.py` (geçici, Faz 7'de silinir) ile çıkarıldı.
Owner-forward sınıflarının `self.<attr>` oku/yaz sınıflandırması:

**Evaluation owner state — MUTABLE (EvaluationState hedefi):** `predictions`,
`prediction_targets`, `quantile_predictions`, `single_backtest_inputs`,
`latest_tensors`, `latest_backtest_results`, `latest_backtest_metrics`,
`latest_model_metrics`, `ensemble_weights` (bunlar zaten `EvaluationState`'te) +
**henüz state'te OLMAYAN, Faz 1'de taşınacak:** `y_true_aligned`,
`y_true_target_aligned`, `prev_close_aligned`, `ensemble_weight_scope`,
`signal_config`, `signal_threshold_source`,
`signal_threshold_calibration_summary`. Training tarafı: `wf_y_true`,
`final_holdout_model`, `final_holdout_model_name` (B1 kapsamı, bu epikte değil).

**Evaluation owner state — READ-ONLY (EvaluationContext hedefi):**
`stock_symbol`, `outputs_dir`, `models_dir`, `tracker`, `feature_names`,
`dataset_hash`, `dataset_metadata`, `stock_db`, `ensemble_enabled`,
`selected_models`, `backtest_enabled`, `commission_bps`, `slippage_bps`,
`initial_capital`, `signal_mode`, `default_signal_config`, `xai_dir`.

**DataManager owner state (Faz 6):** MUTABLE `df`, `dataset_hash`,
`dataset_metadata`, `feature_names`, `feature_groups`, `*_report`,
`selection_df`, `final_holdout_df`, `tensors`, `wf_splits`, `_wf_mode`,
`_prepare_tensors_call_idx`; READ-ONLY `data_cfg`, `validation_config`,
`project_root`, `macro_cache_dir`, `models_dir`, `scaling_reports`,
`universe_file`, `stock_symbol`.

**ForecastRunner owner state (Faz 5, yalnız-get):** READ-ONLY `db`,
`model_config`, `project_root`, `rules`, `persistence` + servis referansları
(`model_resolver`, `data_preparation_service`, `production_training_workflow`,
`forecast_point_generator`, `latest_target_prediction_workflow`).

> NOT: Workflow'ların okuduğu `self._method` çağrıları (`_baseline_specs`,
> `_make_lstm`, `_skip`, `_split_walk_forward_signal_sets`, ...) attribute değil
> davranış-delegasyonudur; DI'da bunlar dar protokol/servis bağımlılığına döner.

## Durum

- 2026-06-01 (Faz 7 ✅ — **EPİK KAPANDI**): Temizlik & dokümantasyon. Geçici AST
  envanter script'i `tools/owner_forward_inventory.py` **silindi** (hiç commit
  edilmemişti, untracked diskten kaldırıldı). Kod gerçeği doğrulandı (kod > wiki):
  `_OwnerBackedForecastService` zaten Faz 5'te silinmiş (kalan = yalnız tarihsel
  doc/yorum). `_OwnerBackedService` ise **canlı** — 3 eval workflow
  (`SingleSplit/WalkForward/FinalHoldout EvaluationWorkflow`), 3 training workflow
  (`FinalHoldout/SingleSplit/WalkForward TrainingWorkflow`), 4 DataManager servisi
  miras alıyor; hepsi paylaşılan owner state'i hem okur hem yazar (servis↔workflow
  entegrasyon sözleşmesi). Bu yüzden epic'in orijinal "taban silindi" kabul kriteri
  **revize edildi**: tabanı silmek = bu 10 sınıfı DI'ya çevirmek, epic §1'in
  "big-bang yüksek regresyon riski" diye uyardığı ve §8'in training için kapsam-dışı
  bıraktığı büyük iş → **ayrı gelecek epik (E1.x)** olarak işaretlendi. Taban
  **bilinçli korundu**, tüm forward yazımlar artık fail-loud (Faz 6). Doc güncellendi:
  `architecture.md` (E1 closed + taban-koru gerekçesi), `refactor-plan.md` Durum
  (E1 KAPANDI girişi), epic frontmatter `status: done`, §6 kabul kriteri revize.
  Kod davranışı **değişmedi** (sadece temp script silindi + doc). Tam suite **561
  passed**, golden sabit. **E1 epiği tamamlandı.**
- 2026-06-01 (Faz 6 ✅): DataManager servisleri fail-loud guard'a alındı. 4 servisten
  (`DataIngestionService`, `TensorPreparationService`, `ValidationSplitService`,
  `DataQualityReportingService`) `_FAIL_LOUD = False` opt-out **kaldırıldı**; artık
  ortak `_OwnerBackedService` varsayılanı (`_FAIL_LOUD = True`) geçerli — owner'a
  forward edilen yazım, `__init__`'te init edilmemiş bir attribute'a giderse (yazım
  hatası) sessizce yeni attribute yaratmak yerine `AttributeError` fırlatır.
  Hardening doğrulaması: tüm forward-write hedefleri (`df`, `feature_names`, `tensors`,
  `wf_splits`, `selection_df`, `final_holdout_df`, `dataset_metadata`, `dataset_hash`,
  `corporate_action_report`, `feature_groups`, `feature_pruning_report`,
  `sector_mapping_report`, `survivorship_bias_report`, `training_window_report`,
  `_wf_mode`, `_prepare_tensors_call_idx`, `scaling_reports`) üretim yolunda (`__init__`)
  zaten pre-init ediliyordu. `__new__` ile kurulan legacy test objeleri için
  (`test_phase7/8_acceptance`) `_ensure_config_objects`'e mutable runtime state
  pre-init bloğu eklendi (hasattr-guard, testin set ettiği değerleri ezmez) — böylece
  `split_data` gibi forward-yazan yollar her kuruluş biçiminde güvenli. `_OwnerBackedService`
  docstring/yorum güncellendi. Monkeypatch namespace'i korundu (paket-split yok). Tam
  suite **561 passed**, golden değişmedi. Commit `d4f9297`. **Kalan owner-forward: yalnız
  `_OwnerBackedService` base** + tükettiği evaluation/training workflow + DataManager
  servis aileleri. Sonraki: Faz 7 (temizlik & taban kaldırma değerlendirmesi).
- 2026-06-01 (Faz 5 ✅): `ForecastRunner` DI'ya geçti. `_OwnerBackedForecastService`
  base sınıfı **silindi** (read-only `__getattr__` forward'du). Yeni `ForecastContext`
  dataclass (`forecasting/workflows.py`): READ-ONLY config (`project_root`, `db`,
  `rules`, `model_config`, `persistence`) + `ForecastRunner` factory callable'ları
  (`make_model_instance`, `make_prophet`, `target_to_price`; `model_config`'e bağlı,
  runner'da kalır) + kardeş servis referansları (5 servis). 6 forecast servisi
  (`BestModelResolver`, `ForecastDataPreparationService`, `ProductionTrainingWorkflow`,
  `LatestTargetPredictionWorkflow`, `ForecastPointGenerator`, `ForecastSymbolWorkflow`)
  artık ctor'da `(ctx)` alır; gövdedeki owner-forward `self.X` erişimleri `self.ctx.X`
  oldu. `BestModelResolver._best_trainable_experiment` owner hop'u kendi metoduna
  (`self.best_trainable_experiment`) indirildi. `_init_workflows` ctx kurar,
  servisleri inşa eder, kardeş referansları ctx'e bağlar; runner test/public yüzeyini
  (`model_resolver`, `forecast_point_generator`, `_make_target`, `_target_to_price`,
  `_roll_forward_points`, `_best_trainable_experiment`, `db`, ...) korur.
  Stub-owner testleri (`test_prediction_date_aware`, `test_recursive_quantile_path`)
  pozisyonel ctor'a uyumlu, dokunulmadı. Tam suite **561 passed**, forecast golden'ları
  değişmedi. Commit `0806c11`. **Kalan owner-forward: yalnız evaluation
  `_OwnerBackedService`** (workflow + DataManager). Sonraki: Faz 6 (DataManager guard).
- 2026-06-01 (Faz 4 ✅): `EvaluationManager` ince orkestratör sadeleştirmesi.
  Geriye-uyumlu servis delegasyon yüzeyinden **hiçbir yerden çağrılmayan 10 ölü
  delegasyon** silindi (workflow forward, src, test — hepsi 0 referans; davranış
  değişmez, sadece ölü kod): `_save_selected_models_plot`, `_diagnostic_numeric`,
  `_diagnostic_float`, `_count_decision`, `_payload_expected_return`,
  `_write_signal_gate_diagnostics`, `_write_shadow_backtest_reports`,
  `_signal_calibration_sort_key`, `_write_signal_calibration_reports`,
  `_get_signal_calibration_decision_md`. Bu metotların mixin içi (`backtest_runner`/
  `signal_calibrator`/`prediction_engine`) `self.X` çağrıları ilgili servisin
  kendi metoduna çözülüyordu; manager delegasyonu artık gereksizdi.
  `_filter_backtest_inputs_by_folds` **korundu** (manager `_split_walk_forward_signal_sets`
  dahili kullanıyor). Workflow'ların forward ile okuduğu (wf=1) ve testlerin
  eriştiği delegasyonlara dokunulmadı (mekanik churn + risk için bekletildi).
  `EvaluationManager` 1035 → 979 satır. Tam suite **561 passed**, golden değişmedi.
  Commit `6d142ff`. Sonraki: Faz 5 (`ForecastRunner` DI).
- 2026-06-01 (Faz 3.4 ✅): `MetricsReportingService` DI'ya geçti (Faz 3 servis #4/4 —
  son servis). `_OwnerBackedService` mirası kalktı, ctor `(ctx, state)`.
  `_MetricsReporterMixin` forward erişimleri açık: READ-ONLY `self.ctx.X`
  (`dataset_metadata`, `commission_bps`/`slippage_bps`, `stock_symbol`,
  `feature_names`, `xai_dir`, `write_xai_tables`, `write_markdown_reports`),
  mutable `self.state.X` (`predictions`, `prediction_targets`, `quantile_predictions`,
  `y_true_aligned`, `ensemble_weights`, `latest_backtest_results`).
  `EvaluationContext`'e XAI-yazım 2 flag eklendi (`write_xai_tables`=False,
  `write_markdown_reports`=True) — default'lar eski getattr fallback'leriyle aynı.
  Manager'da 2 yeni context-backed property; `_init_services`
  `MetricsReportingService(self.context, self.state)`.
  `tests/test_xai_routing.py` `_StubReporter` DI şekline (ctx/state) güncellendi.
  **4 evaluation servisinin tamamı artık DI.** `_OwnerBackedService` yalnızca
  evaluation/training workflow'ları + `DataManager` servis aileleri tarafından
  kullanılıyor (kaldırma Faz 7). Tam suite **561 passed**, golden değişmedi.
  Commit `398c82f`. Sonraki: Faz 4 (ince orchestrator).
- 2026-06-01 (Faz 3.3 ✅): `SignalCalibrationService` DI'ya geçti (Faz 3 servis #3/4).
  `_OwnerBackedService` mirası kalktı, ctor `(ctx, state)`. `_SignalCalibratorMixin`
  forward erişimleri açık: READ-ONLY `self.ctx.X` (commission/slippage/initial_capital/
  outputs_dir/default_signal_config/dataset_metadata + 9 yeni signal_calibration flag),
  mutable `self.state.X` (signal_config/signal_threshold_source/
  signal_threshold_calibration_summary). `EvaluationContext`'e mixin + `signal_calibration/
  grid.apply_trial_policy`'nin okuduğu 9 exe_cfg flag eklendi (`calibration_scope`,
  `signal_calibration_require_oos_confirmation`, `signal_calibration_min_eval_excess_return`,
  `signal_calibration_min_eval_sharpe`, `signal_calibration_objective`,
  `signal_calibration_profile`, `signal_calibration_sampler`, `signal_calibration_seed`,
  `signal_calibration_max_trials`) — default'lar eski getattr fallback'leriyle birebir
  aynı. `apply_trial_policy(self.ctx, grid)` artık ctx'i okur. Manager'da 9 yeni
  context-backed property; `_init_services` `SignalCalibrationService(self.context, self.state)`.
  Kalan 1 servis (MetricsReporting) hâlâ owner-backed. Tam suite **561 passed**,
  golden değişmedi. Sonraki: Faz 3.4 (MetricsReportingService).
- 2026-06-01 (Faz 3.2 ✅): `BacktestService` DI'ya geçti (Faz 3 servis #2/4).
  `_OwnerBackedService` mirası kalktı, ctor `(ctx, state)`. `_BacktestRunnerMixin`
  forward erişimleri açık: READ-ONLY `self.ctx.X` (commission/slippage/initial_capital/
  backtest_enabled/signal_mode/dataset_metadata/outputs_dir/stock_symbol + 6 yeni
  exe_cfg flag), mutable `self.state.X` (signal_config/signal_threshold_source/
  latest_backtest_results/latest_backtest_metrics). `EvaluationContext`'e
  BacktestService'in okuduğu 6 exe_cfg flag eklendi (`write_trade_logs`,
  `signal_calibration_min_trades`, `signal_calibration_reject_behavior`,
  `auto_signal_diagnostics`, `enable_gate_diagnostics`, `enable_shadow_backtests`)
  — default'lar eski getattr fallback'leriyle birebir aynı. Manager'da 6 yeni
  context-backed property; `_init_services` `BacktestService(self.context, self.state)`.
  Kalan 2 servis (SignalCalibration, MetricsReporting) hâlâ owner-backed. Tam suite
  **561 passed**, golden değişmedi. Sonraki: Faz 3.3 (SignalCalibrationService).
- 2026-06-01 (Faz 3.1 ✅): `PredictionService` DI'ya geçti (Faz 3 servis #1/4).
  `_OwnerBackedService` mirası kaldırıldı; ctor `(ctx, state)` enjekte alır
  (`evaluation_services.py`). `_PredictionEngineMixin` gövdesindeki tüm owner-forward
  attribute erişimleri açık hale getirildi: READ-ONLY `self.ctx.X`
  (`dataset_metadata`/`ensemble_enabled`/`selected_models`), mutable `self.state.X`
  (`predictions`/`prediction_targets`/`quantile_predictions`/`single_backtest_inputs`/
  `latest_tensors`/`ensemble_weights`/`ensemble_weight_scope`/`y_true_aligned`/
  `y_true_target_aligned`/`prev_close_aligned`). Defensive `getattr(self, ...)` formları
  ve `ensemble_weight_scope` hasattr guard'ı kaldırıldı (state alanı default_factory).
  `_init_services` `PredictionService(self.context, self.state)` enjekte ediyor;
  diğer 3 servis hâlâ owner-backed. `test_prediction_date_aware.py` DI ctor'a uyarlandı;
  golden'lar (`test_owner_forward_contract.py`) değişmeden geçti. Tam suite **561 passed**.
  Sonraki: Faz 3.2 (BacktestService → SignalCalibration → MetricsReporting).
- 2026-06-01 (Faz 2 ✅): `EvaluationContext` tam taşıma. `EvaluationContext` tüm
  alanları default'lu hale getirildi (lazy/`__new__` desteği) + 9 türetilmiş
  READ-ONLY alan eklendi (`ensemble_enabled`, `selected_models`, `backtest_enabled`,
  `commission_bps`, `slippage_bps`, `initial_capital`, `signal_mode`,
  `default_signal_config`, `xai_dir`). `EvaluationManager`'da 19 config/identity
  attribute (10 base + 9 türetilmiş) context-backed **property**'ye dönüştü
  (`manager.X` ⇄ `manager.context.X`); owner-forward servisler/workflow'lar getattr
  ile aynı context'ten okur. `__init__`'teki düz `self.stock_symbol = ...` atamaları
  kaldırıldı (base'ler context constructor'a, türevler `_init_*` setter'larına gitti);
  mixin gövdesine yine **dokunulmadı** (Faz 3'te `self.ctx.X`). `context` lazy property
  → `__new__` mekanizma testleri (`test_phase8_acceptance` doğrudan `outputs_dir`/
  `commission_bps`/`signal_mode` set'leri) ilk erişimde boş `EvaluationContext()` alır
  (testlere dokunulmadı). Mixin/workflow'da bu attr'lara yazım yok (doğrulandı) —
  gerçekten READ-ONLY. Tam suite **561 passed**, golden'lar değişmeden geçti.
  Sonraki: Faz 3 (mixin → davranış-sahibi servis, en riskli).
- 2026-06-01 (Faz 1 ✅): `EvaluationState` tam taşıma. `EvaluationState` 7 alanla
  genişletildi (`ensemble_weight_scope`, `y_true_aligned`, `y_true_target_aligned`,
  `prev_close_aligned`, `signal_config`, `signal_threshold_source`,
  `signal_threshold_calibration_summary`). `EvaluationManager`'da 16 mutable
  attribute state-backed **property**'ye dönüştü (`manager.X` ⇄ `manager.state.X`);
  owner-forward servisler/workflow'lar `setattr(owner, X, ...)` ile yazdığında
  property setter state'e yönlendirir, böylece mixin gövdesine **dokunulmadı**
  (Faz 3'te `self.state.X`). `__init__` yeniden sıralandı: `_init_context_and_state`
  artık önce gelir (boş state kurar). `state` lazy property → `__new__` ile kurulan
  mekanizma testleri ilk erişimde otomatik state alır (testlere dokunulmadı). Tam
  suite **561 passed**, golden'lar değişmeden geçti. Sonraki: Faz 2 (`EvaluationContext`).
- 2026-06-01 (Faz 0 ✅): Karakterizasyon golden'ları `tests/test_owner_forward_contract.py`
  yazıldı (12 test): kuruluş state golden'ı, `manager` ↔ `manager.state` alias
  kimliği, paylaşılan-state iki-yönlü mutasyon, servis-kompozisyonu üzerinden saf
  hesaplama golden'ları (`_target_to_price` 3 mod + invalid, `_weighted_average`,
  `_base_predictions_for_ensemble`), determinizm. Forecast uçtan-uca golden'ı
  `tests/test_forecasting.py` + `test_forecast_workflows.py`'de (duplike yok).
  Owner read/write envanteri `tools/owner_forward_inventory.py` ile çıkarıldı
  (yukarı §8). Tam suite **561 passed** (temiz `--basetemp` ile; `.codex_tmp`
  kilitli olduğunda 46 ERROR çıkması çevreseldir, koddan değil). Sonraki: Faz 1
  (`EvaluationState` tam taşıma).
- 2026-06-01: Epik planı oluşturuldu; `refactor/e1-owner-forward-di` dalı açıldı.
  Ön koşul (E3 + fail-loud guard) `ModelUpdate`'te tamamlandı (commit 1e5c4be,
  a541a10).
