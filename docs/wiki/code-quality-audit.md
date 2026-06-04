---
title: Code Quality Audit (2026-05-31)
type: audit
status: active
last_updated: 2026-05-31
owner: llm
source_count: 8
---

# Code Quality Audit — 2026-05-31

Bu sayfa, 8 aşamalı code-review sonrası yapılan kod-kalitesi denetiminin
bulgularını ve docstring/yorum ekleme planını kaydeder. Denetim;
god-object, şişkin dosya/fonksiyon, karmaşıklık ve SOLID/KISS/DRY
ihlallerine odaklanır. İlgili eşikler [Code Quality and
Refactoring](code-quality-and-refactoring.md) sayfasında tanımlıdır
(dosya ≤500 satır, class ≤300 satır).

Ölçüm anı metrikleri (AST taraması, `src/` altı):
- Public fonksiyon/metod docstring kapsamı: **%31 (145/467)**.
- Module-docstring'i eksik modül: **31**.
- 500+ satır dosya: **16** (700+ : 6).
- 100+ satır fonksiyon: **31**.
- 250+ satır class: **16**.

## Bulgular

### B1 — God object / kozmetik servis ayrımı (kritik)

`EvaluationManager` ve `forecasting/workflows.py` aynı deseni paylaşır:
servis sınıfları gerçek sorumluluk ayrımı yapmaz, tek `owner`'a forward eder.

- `src/pipeline/evaluation_services.py` → `_OwnerBackedService.__getattr__` /
  `__setattr__` tüm attribute erişimini `owner` (EvaluationManager) üzerine
  yönlendirir. `PredictionService`, `BacktestService`,
  `SignalCalibrationService`, `MetricsReportingService` = mixin + owner adapter.
- Mixin'ler (`_SignalCalibratorMixin` 898 satır/34 metod, `_PredictionEngineMixin`
  714/10, `_BacktestRunnerMixin` 553/16, `_MetricsReporterMixin`) hâlâ
  **paylaşılan mutable owner state** üzerinde çalışır. SRP/encapsulation
  gerçekte sağlanmaz; kuplaj tasarımla korunur.
- `EvaluationManager` de-facto god object: **48 metod**, `__init__` 123 satır /
  ~40 attribute. Davranışı 4 dosyaya yayılmış ~2300 satır.
- `forecasting/workflows.py`: `_OwnerBackedForecastService` + 7 workflow sınıfı,
  aynı owner-forward iskeleti.

Risk: test izolasyonu zor; attribute kaynağı belirsiz; `__setattr__` typo
sessizce owner'a yazar (sessiz bug yüzeyi).

Aksiyon (ayrı/büyük refactor — bu denetimde uygulanmaz): owner-forward yerine
açık bağımlılık enjeksiyonu (`EvaluationContext` + dar protokoller);
mixin'leri davranış-sahibi servislere dönüştür.

> Güncelleme (2026-06-01): bu aksiyon artık
> [E1 Owner-Forward Removal Epic](e1-owner-forward-epic.md) altında
> `refactor/e1-owner-forward-di` dalında yürüyor. `PredictionService` ve
> `BacktestService` `(ctx, state)` DI'ya çevrildi (owner-forward miras kalktı);
> `SignalCalibrationService`/`MetricsReportingService` ve `forecasting/workflows`
> hâlâ owner-backed. Bu denetim metrikleri ölçüm anının (2026-05-31) snapshot'ıdır.

### B2 — Şişkin dosyalar (16 dosya >500 satır, 6 dosya >700)

| LOC | Dosya | Not |
|----:|-------|-----|
| 997 | `src/pipeline/signal_calibrator.py` | mixin 898/34 metod |
| 814 | `src/forecasting/workflows.py` | 12 pub fn / 1 docstring |
| 791 | `src/pipeline/data_services.py` | `run()` 123 satır |
| 764 | `src/database/stock_model_db.py` | 33 metod, 19 pub fn / 1 doc |
| 758 | `src/pipeline/prediction_engine.py` | DRY ihlali (B4) |
| 739 | `src/analysis/run_leaderboard.py` | iyi belgelenmiş (8/9) |
| 693 | `src/features/macro_pipeline.py` | `MacroPipeline` 29 metod |
| 678 | `src/api/services/analysis_service.py` | `build()` 113 satır |
| 669 | `src/analysis/signal_research.py` | iyi belgelenmiş |
| 644 | `src/backtesting/engine.py` | module docstring yok |

Class-bazlı sınır aşımı (≥250 satır): `_SignalCalibratorMixin` 898,
`_PredictionEngineMixin` 714, `MacroPipeline` 560, `XAIExplainer` 554
(class docstring yok), `_BacktestRunnerMixin` 553, `FeaturePipeline` 545,
`EvaluationManager` 533, `ForecastingPipeline` 511 (docstring yok),
`StockModelDB` 470.

### B3 — Dev fonksiyonlar (31 adet ≥100 satır)

En kritik:

| Satır | Konum |
|----:|-------|
| 202 | `backtesting/signals.py:190` generate_professional_signals |
| 197 | `pipeline/prediction_engine.py:285` _add_walk_forward_ensembles |
| 195 | `cli/signal_research.py:42` main |
| 176 | `pipeline/prediction_engine.py:108` _add_single_split_ensembles |
| 176 | `backtesting/metrics.py:99` summarize_backtest |
| 176 | `backtesting/engine.py:24` run_backtest |
| 156 | `validation/walk_forward.py:110` run |
| 143 | `pipeline/orchestrator.py:389` run_all |
| 143 | `pipeline/confidence_calculator.py:49` compute_confidence |

CLI `main()` fonksiyonları (signal_research 195, batch 150) tek-fonksiyon
god-procedure; alt-fonksiyonlara bölünebilir.

### B4 — DRY ihlalleri

- **Ensemble inşası kopyalanmış:** `prediction_engine.py` `_add_single_split_ensembles`
  (108) vs `_add_walk_forward_ensembles` (285) — ikisi de `optimize_inverse_rmse`
  + sequence-model alt blokları, yakın kopya. Ortak helper'a çıkarılabilir.
- 3× neredeyse-aynı `run()` workflow sınıfı (`evaluation_workflows.py`) +
  7× forecasting workflow — aynı owner-forward iskeleti.
- `reconstruct_prices_*` fallback iki kez tanımlı (import guard) — minor.

### İyi yanlar

- `src/analysis/*` (run_leaderboard, signal_research) %85+ docstring kapsamı.
- Leakage guard'lar net adlandırılmış (`_assert_wf_train_scope`).
- DB repositories zaten 1085→5 modüle bölünmüş (geçmiş refactor).

## Docstring & Yorum Planı

Stil: mevcut konvansiyon korunur — **Türkçe docstring + "Sorumluluklar:" bullet**.
Yorumlar **NEDEN**'i anlatır (ne yaptığını değil). Kod davranışı değişmez.

- **Faz 1 — Modül docstring'leri (31 modül):** her modüle 2-3 satır sorumluluk
  özeti. Mekanik, hızlı kazanım. Öncelik: backtesting/*, db/repositories/*,
  forecasting/(runner, bist_rules, artifacts), utils/reporting_utils.
- **Faz 2 — Public API docstring'leri (322 eksik):** öncelik (1) stage-kritik
  `validation/*`, `signal_calibrator`, `prediction_engine`, backtest
  `engine/signals/metrics`; (2) API kontratı `api/services/*`,
  `forecasting/workflows`; (3) persistence `stock_model_db`, `repositories/*`.
- **Faz 3 — Inline "neden" yorumları:** embargo `max(200,time_steps)`,
  composite-score ağırlıkları, recursive forecast causal chain, BIST band clip,
  confidence degradation eşikleri.
- **Faz 4 — Dev fonksiyon param docs:** 31 fonksiyona Args/Returns/Raises;
  bölünmeye aday noktalar `# TODO(refactor):` ile işaretlenir.

## Durum

- 2026-05-31: Denetim yapıldı, plan oluşturuldu. **Faz 1** uygulandı: docstring'i
  olmayan 31 modülden **18 gerçek modüle** (backtesting engine/metrics/reporting/
  signals + paket init, db/repositories/* + paket init, forecasting artifacts/
  bist_rules/persistence/runner, pipeline/artifacts, utils/reporting_utils)
  sorumluluk özeti eklendi. **13 boş paket-işaretçi `__init__.py`** (data,
  database, pipeline, validation, utils, evaluation, features, experiments,
  model_registry, api ve api/* alt paketleri) açıklanacak içerik olmadığı için
  bilinçli atlandı. Kod davranışı değişmedi; py_compile + smoke/backtest/db/
  forecasting testleri (32) yeşil. God-object refactor (B1) ayrı plan gerektirir.
- 2026-05-31: **Faz 2 (öncelik 1-3) uygulandı.** 77 public docstring eklendi:
  - Öncelik 1 (stage-kritik, 11): validation `walk_forward.run`/`purged_kfold.get_n_splits`,
    backtest `engine.run_backtest`/`signals.generate_long_flat_signals`/
    `metrics.summarize_backtest` + reporting'in 5 kaydedicisi.
  - Öncelik 3 (persistence, 37): `stock_model_db` 18 facade metodu +
    repositories `best_model`(6)/`experiment`(3)/`forecast`(3)/
    `forecast_resolution`(3)/`schema`(4).
  - Öncelik 2 (API kontratı, 29): `api/services` rate_limit(4)/response_cache(5)/
    advisory_audit(1)/analysis_service(1)/data_refresh_service(6) +
    `forecasting/workflows` 11 workflow metodu.
  - Sonuç: public docstring kapsamı **%31 → %47 (145 → 221)**; module-docstring
    eksik 31 → 13 (kalan 13 = boş paket-işaretçi `__init__.py`). py_compile +
    89 ilgili test yeşil. Faz 3 (inline neden-yorumları) ve Faz 4 (dev fonksiyon
    param docs) beklemede.
