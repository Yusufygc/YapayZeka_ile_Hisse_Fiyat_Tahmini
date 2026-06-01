---
title: E1 Owner-Forward Removal Epic
type: plan
status: active
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
kilitlenir, sonra refactor edilir, suite (549 test) her commit'te yeşil kalır.
Kaynak doğruluk sırası kod > test > wiki.

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

- `_OwnerBackedService` ve `_OwnerBackedForecastService` repo'dan silindi.
- Hiçbir serviste `__getattr__`/`__setattr__` owner-forward kalmadı.
- Tüm suite yeşil (≥549) + karakterizasyon golden'lar değişmeden geçer.
- Leakage/determinizm assert'leri korundu; monkeypatch hedefleri çalışıyor.
- Public API imzaları değişmedi.

## 7. Geri alma (rollback)

Her faz ayrı commit. Bir faz golden'ı bozarsa `git revert <faz-commit>` ile
izole geri alınır; önceki fazlar etkilenmez. Dal birleştirilmeden önce tam
suite + `python -m src.cli.forecast` smoke koşulur.

## Durum

- 2026-06-01: Epik planı oluşturuldu; `refactor/e1-owner-forward-di` dalı açıldı.
  Ön koşul (E3 + fail-loud guard) `ModelUpdate`'te tamamlandı (commit 1e5c4be,
  a541a10). Henüz hiçbir faz başlamadı — yeni session Faz 0 (karakterizasyon
  testleri) ile başlamalı.
