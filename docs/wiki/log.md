## [2026-06-03] Faz 2 BITTI | per-symbol OOS aggregation harness

Faz 2 son parca kodlandi + test edildi -> Faz 2 ✅ DONE:
- `src/validation/pooled_oos.py` `evaluate_per_symbol`: her CV fold icin taze
  `model_factory()` fit (fold'lar arasi durum sizmasi yok), test satirlarini
  tahmin, OOS tahminleri SEMBOL bazinda grupla -> per-symbol metrik dagilimi
  (`dir_acc`, `rmse`, `base_rate`, base-rate-uzeri `edge`, `positive_fold_ratio`,
  `reliable`) + per-`(symbol,fold)` detay. log-return target -> isaret tabanli
  yon (price-mode/prev_close coupling yok). final_holdout varsayilan dislanir.
- 6 test + gercek Ridge smoke (3 sembol): AKBNK edge +0.5, EREGL -1.9, TUPRS
  -2.4 -> durust, base-rate civari. Faz 5 serving guven skorunu besler.
- Tam suite 587 yesil. Siradaki: Faz 3 global kosullandirilmis model.

## [2026-06-02] Faz 2 KOD | pooled_loader + pooled_cv (cekirdek)

Tasarimin cekirdegi kodlandi + test edildi:
- `src/validation/pooled_cv.py` PooledPurgedWalkForward: tarih-bazli split
  (capraz-sembol leak yok), purge+embargo (E=h+buffer), target_date ile kesin
  purge, coklu-pencere OOS + final-holdout. 7 leak/yapi testi.
- `src/data/pooled_loader.py` PooledPanelLoader: uzun panel (per-symbol feature,
  h-gun target+target_date, sector/symbol_id/causal liq_log/vol, delisted dahil).
  Stray ham `Kapanış` fiyat-seviyesi feature sizintisi giderildi. 6 test + gercek
  3-sembol smoke (8463 satir, 6+1 fold, gercek veride leak-free).
- Tam suite 581 yesil. Kalan Faz 2: per-symbol OOS aggregation harness (dummy/
  global model -> fold tahmin -> sembol bazli dagilim), sonra Faz 3 model.

## [2026-06-02] Faz 2 TASARIM | pooled loader + grup-purged CV

`docs/wiki/e2-faz2-pooled-cv-design.md` yazildi: pooled panel loader + tarih-bazli
purged coklu-pencere CV detayli tasarimi. Mevcut `walk_forward_splits` satir-indeksli
ve tek-sembol oldugu icin panele uymuyor -> yeni tarih-bazli splitter gerek.
Leakage taksonomisi (capraz-sembol same-date / horizon / feature-lookahead) +
purge+embargo (E=h+buffer) ile her birinin onlenmesi. Modul plani
(`src/data/pooled_loader.py` PooledPanelLoader, `src/validation/pooled_cv.py`
PooledPurgedWalkForward), CV algoritmasi, scaling/conditioning politikasi, per-symbol
OOS metrik aggregation (API kontrati degismez), test plani, acceptance, 5 acik soru.
index.md link eklendi. Henuz kod yok; tasarim onayi bekliyor.

## [2026-06-02] Faz 1 | target_horizon knob + predictive horizon kiyas

E2 Faz 1 (predictive dilim): geri-uyumlu `DataConfig.target_horizon` (default 1 =
davranis sabit) eklendi; `build_target_series`/`prepare_tensors` h-aware
(`y[t]=target(close[t+h])`, `X=features[:-h]`). Test: `tests/test_data_services.py`
(3 yeni), tam suite 568 yesil.

`tools/e2_faz1_horizon_compare.py` ile EREGL/AKBNK/TUPRS/AEFES/SASA, h={1,3,5,10},
Ridge+LightGBM, kronolojik 80/20, predictive-only (backtest YOK):
- Ortalama Dir_Acc h=1 ~%50 -> h=5 ~%52.5 -> h=10 ~%52. Haftalik gunlukten biraz
  daha ongorulebilir.
- Uyari: yuksek h pozitif base-rate'i sisirir; naif "hep yukari"ya edge kucuk
  (~0-3pp, h=5). Step-change degil, bedava alpha yok.
- Karar: h=5 E2 default horizon (daha iyi S/N); asil kaldirac pooling (Faz 2-3).
  Horizon-aware backtest/forecast = Faz 1b (pooling baseline'i gecince). h>1
  backtest/Sharpe yorumlanmamali.

## [2026-06-02] Faz 0.6 | sektor backfill (kosullandirma prerequ)

`tools/backfill_sectors.py` ile bist_universe.csv Sector kolonu yfinance
Ticker.info GICS sektoruyle tek tip dolduruldu: 585 sembol, 580 resolved, 5
Unknown (GMTAS, ISGSY, ISKUR, KZGYO, ULUFA). Dagilim: Industrials 115, Consumer
Cyclical 99, Financial Services 77, Basic Materials 71, Consumer Defensive 61,
Real Estate 56, Technology 36, Utilities 33, Healthcare 15, Communication
Services 12, Energy 5. Sector_Index (macro sektor-getiri alani) dokunulmadi;
schema sabit (Industry yazilmadi). Sektor artik E2 kosullandirma icin kullanilabilir
categorical. Universe commit edilmedi (veri, kural).

## [2026-06-02] Faz 0.5 KOSULDU | universe tam yeniden cekildi

`tools/refetch_universe.py` tam 592-sembol kosusu: ok=585, no-data=7, failed=0.
Re-audit ile dogrulandi: tarih formati ISO'ya birlesti (592/592), fresh-to-
2026-06-02=583 (onceden 1), universe coverage 585/592 (onceden 28), 1,275,614
pooled satir (+178k), dup/zero-price=0. No-data 7 sembol (DOBUR, EFORC, IPEKE,
KOZAA, KOZAL, SNKRN, YGYO) yfinance'den veri donmedi -> delist/aski suphesi;
eski CSV'leri korundu, survivorship karari icin isaretli. Kalan Faz 0 gap'leri:
sektor ~557 hissede bos (backfill), 110 hissede |log_return|>=0.30 gunu (split/
temettu audit), 30 thin (<500) cold-start. Veri+universe commit edilmedi.

## [2026-06-02] Faz 0.5 | E2 universe re-pull araci yazildi

`tools/refetch_universe.py`: Faz 0'in 3 sorununu cozer. Her ticker icin
yfinance'den (`{SYMBOL}.IS`, `auto_adjust=True` invariant) TAM gecmis cekip
`data/{TICKER}.csv`'yi TEK TIP ISO (`%Y-%m-%d`) tarihle SIFIRDAN yazar
(overwrite, append degil) + `data/bist_universe.csv`'yi cekilen her hisse icin
upsert eder (Listed_Date=min tarih, Status freshness'a gore, Sector/Delisted_Date
mevcutsa korunur). Smoke: EREGL/AKBNK/TUPRS -> ~2910 satir, 2026-06-02'ye guncel,
gecici dizine (gercek data + universe bozulmadi). Tam 592 kosusu operator tetigi
(uzun, ag-yogun, tum CSV'leri overwrite eder). `.gitignore` whitelist'e eklendi.

## [2026-06-02] Faz 0 | E2 universe/data audit kosuldu

`tools/e2_faz0_universe_audit.py` (read-only) ile 592 hisse CSV denetlendi.
Bulgular `e2-pooled-global-model-epic.md` "Faz 0 Findings"e islendi.

- Olcek: 592 hisse, **1,097,481** pooled satir (8y+ 363 hisse). Thin <500: 71.
- **Freshness BLOCKER:** 569 hisse 2025-10-24'te donmus (tek seferlik snapshot,
  ~220 gun eski); yalniz 1 hisse guncel. Pooled egitim oncesi universe tek
  guncel kesime cekilmeli.
- **3 tarih formati** (`%Y-%m-%d` 577, `%d/%m/%Y` 14, `%d.%m.%Y` 1) — ingestion
  hepsini tanimali (tek format varsayimi auditor'un ilk kosusunda 578/592'yi
  sessizce dusurdu).
- **Survivorship cozulemedi:** universe yalniz 28 sembol kataloglu, 564 CSV
  disarida; delisting kaynagi Faz 2 CV oncesi gerek.
- Veri kalitesi: 102 hisse |log_return(adj)|>=0.30 (CA/split), 1 zero/neg fiyat,
  1 dup-date, 2 buyuk-gap. Sektor eksik: 564 (kosullandirma gap'i).
- Pre-Faz-2 onkosullar listelendi (re-pull, multi-format ingestion, delisting
  kaynagi, sektor backfill, CA temizlik).
- Cikti raporlari `outputs/` altinda (runtime, commit edilmedi).

## [2026-06-02] Plan | E2 Pooled Global Model epic acildi

EREGL all-models kosusunun analizi sonrasi karar: tek-hisse egitim overfit ediyor +
kanitlanmis alpha yok (AttentionLSTM v2 WF comp 83.81 -> holdout 56.15, dir 38.98%).
Cozum yonu kullaniciyla konusuldu ve E2 epic'i acildi.

- **Karar:** egitim kapsami != sunum kapsami. Urun/`GET /analysis/{symbol}` per-symbol
  kalir (kontrat + confidence policy degismez); egitim ~592 hisseyi havuzlayan tek
  **kosullu global model**'e gecer (overfit/alpha fix). Cold-start ince/IPO hisseleri
  ilk gunden cevaplar. Spekulatif hisseler mevcut confidence hard-block (dir_acc<50) ile
  zaten `low`.
- **Fine-tune** opsiyonel/deneysel (urun tasariminin "later experimental phase"i ile
  tutarli). Cross-sectional ranking/portfoy urune uymadigi icin elendi.
- **Yeni dosya:** `docs/wiki/e2-pooled-global-model-epic.md` (problem, core decision,
  Faz 0-6 plan, acceptance, open questions). `index.md` link + durum bolumu guncellendi.
- **Dal:** `feat/e2-pooled-global-model` (E1 dali `refactor/e1-owner-forward-di` push
  edildi, origin'e tracking kuruldu). Henuz kod yok — yalniz plan.
- Data CSV'leri (592) kasten commit/push edilmedi (runtime/veri, kullanici talebi).

## [2026-06-02] Bugfix | run_id model slug MAX_PATH tasmasi (ALL_MODELS etiketi)

EREGL'i tum modellerle (`--role candidate`, 14 model) egitince batch "1 hata" verdi:
egitim + DB tamam ama son `model_result_exporter.export_walk_forward_results` adimi
`FileNotFoundError [Errno 2]` ile coktu; run_all final-holdout + best-model secimine
gecemeden abort oldu (`best_models` eski RF kaydinda kaldi).

- **Kok neden:** Windows `MAX_PATH` (260). `_model_slug_for_run_id` 4+ model icin
  `models-<A>-<B>-<C>-plusN-<sha8>` uretiyordu; uzun model adlariyla run klasor adi ~88
  karakter -> per-model export yolu (`.../model_results/naive_last_value/metrics_walk_forward.json`)
  267 karaktere ciktigi icin `open()` patladi (dizin 241'de makedirs gecmisti).
- **Fix (`src/pipeline/orchestrator.py`):** (a) secim tum candidate setini kapsiyorsa
  run_id slug = `ALL_MODELS`; (b) 4+ kismi liste -> gorunur isimler atilir,
  `models<N>-<sha8>` kalir. Yeni `_covers_all_candidates` yardimcisi candidate kapsamini
  kontrol eder. Tam yol ~210 (<260).
- **Test:** `tests/test_run_id_naming.py` guncellendi (uzun-slug artik `models6-<hash>`,
  yeni `ALL_MODELS` testi). Tam suite yesil (`.codex_tmp` kilidi disinda; temiz basetemp'le 42/42).
- Davranis: yalniz run klasor *adi* degisti; per-model alt klasor yapisi
  (`runs/<id>/model_results/<slug>/`) ayni.

## [2026-06-02] Bugfix | Macro feature-cache zehirlenmesi + forced --model bos model_path

E1 manuel testinde (EREGL forecast smoke) yuzeye cikan iki bagimsiz, E1-disi bug
duzeltildi. Davranis-koruyan, her biri kendi testiyle kilitli.

- **Bug 1 (macro -> feature-cache poisoning):** `macro_pipeline.py` runtime print'lerindeki
  `→` karakteri Windows cp1252 stdout'a yazilirken `UnicodeEncodeError` firlatiyor;
  `DataIngestionService._fetch_macro` bunu yutup bos df donuyor -> makrosuz frame
  `use_macro=True` cache anahtari altina yaziliyor -> sonraki saglikli kosular cache
  hit'te makrosuz frame aliyor. Fix: (1a) `→` -> `->` (macro prints); (1b)
  `_engineer_features_cached`'e write-guard (makro istendi ama yoksa degraded frame
  cache'lenmez) + read-side self-heal (`_has_macro_features` ile makro-yoksun zehirli
  kayit evict edilir). Test: `tests/test_data_services.py` (2 yeni).
- **Bug 2 (forced --model bos model_path):** `BestModelResolver.resolve(force_model_name)`
  `model_path` dondurmuyordu -> `ProductionTrainingWorkflow.train` bos path'le artifact
  yuklemeye calisip `artifact model file not found` firlatiyordu. Fix: force branch artik
  `latest_member_experiment` ile artifact'i diskte mevcut en guncel deneyi bulup
  `model_path` + egitim config'ini doldurur; artifact yoksa eski best-metadata fallback'i
  korunur (geriye uyumlu, mevcut test gecer). Test: `tests/test_forecast_workflows.py` (1 yeni).
- Canli dogrulama: forced LightGBM forecast green (run_id=108); cache temiz + `--use-macro`
  (PYTHONUTF8 olmadan) RF forecast green (run_id=107) -> macro artik dusmuyor.
- Tam suite **564 passed** (+3 yeni test). Guncellenen wiki: data-pipeline.md (cache guard),
  persistence-and-api.md (forced --model), log.md.

## [2026-06-01] Refactor (E1 Faz 7) | Temizlik & dokumantasyon — E1 EPIGI KAPANDI

- Gecici AST envanter script'i tools/owner_forward_inventory.py silindi (hic commit
  edilmemisti; untracked diskten kaldirildi).
- Kod gercegi dogrulandi (kod > wiki): `_OwnerBackedForecastService` zaten Faz 5'te
  silinmis. `_OwnerBackedService` ise CANLI — 3 eval workflow + 3 training workflow +
  4 DataManager servisi miras aliyor, hepsi paylasilan owner state'i okur+yazar
  (servis<->workflow entegrasyon sozlesmesi).
- Epic'in orijinal "taban silindi" kabul kriteri REVIZE edildi: tabani silmek = bu 10
  sinifi DI'ya cevirmek; epic §1 "big-bang yuksek risk" + §8 training kapsam-disi →
  ayri gelecek epik (E1.x) olarak isaretlendi. Taban bilincli korundu; tum forward
  yazimlar fail-loud (Faz 6).
- Kod davranisi DEGISMEDI (sadece temp script + doc). Tam suite 561 passed, golden sabit.
- Guncellenen wiki: architecture.md (E1 closed + gerekce), refactor-plan.md (E1 KAPANDI),
  e1-owner-forward-epic.md (frontmatter status: done, §6 revize, Durum Faz 7),
  index.md (durum), log.md.

## [2026-06-01] Refactor (E1 Faz 6) | DataManager servisleri fail-loud guard'a alindi

- 4 DataManager owner-forward servisinden (DataIngestionService,
  TensorPreparationService, ValidationSplitService, DataQualityReportingService)
  `_FAIL_LOUD = False` opt-out kaldirildi; artik ortak `_OwnerBackedService`
  varsayilani (`_FAIL_LOUD = True`) gecerli — bilinmeyen owner attribute'una
  forward yazim sessizce yaratmak yerine AttributeError firlatir.
- Hardening: tum mutable runtime state `__init__`'te zaten pre-init; `__new__` ile
  kurulan legacy test objeleri icin `_ensure_config_objects`'e mutable state
  pre-init blogu eklendi (df, feature_names, tensors, wf_splits, selection_df,
  final_holdout_df, dataset_metadata, dataset_hash, *_report; hasattr-guard).
- `_OwnerBackedService` docstring/yorum guncellendi (DataManager artik hardened).
- Davranis degismez. Tam suite 561 passed. Commit d4f9297. Kalan owner-forward:
  yalniz `_OwnerBackedService` base + tukettigi evaluation/training workflow +
  DataManager servis aileleri (taban kaldirma Faz 7'de degerlendirilir).
- Guncellenen wiki: e1-owner-forward-epic.md (Durum), index.md (durum), log.md.

## [2026-06-01] Refactor (E1 Faz 5) | ForecastRunner owner-forward'dan DI'ya (ForecastContext)

- `_OwnerBackedForecastService` base sinifi silindi (read-only __getattr__ forward).
- Yeni ForecastContext dataclass (forecasting/workflows.py): config (project_root,
  db, rules, model_config, persistence) + factory callable'lari (make_model_instance,
  make_prophet, target_to_price) + 5 kardes servis referansi.
- 6 forecast servisi ctor'da (ctx) alir; govde self.X -> self.ctx.X. BestModelResolver
  owner hop'u kendi metoduna indirildi (self.best_trainable_experiment).
- ForecastRunner._init_workflows ctx kurar + kardes referanslari baglar; runner
  test/public yuzeyi korundu. Stub-owner testlerine dokunulmadi.
- Tam suite 561 passed, forecast golden degismedi. Commit 0806c11. Kalan owner-forward:
  yalniz evaluation _OwnerBackedService (Faz 7). Sonraki: Faz 6 (DataManager guard).
- Guncellenen wiki: e1-owner-forward-epic.md (Durum), index.md (durum), log.md.

## [2026-06-01] Refactor (E1 Faz 4) | EvaluationManager ince orkestrator - olu delegasyonlar silindi

- `EvaluationManager`'dan hicbir yerden cagrilmayan 10 olu servis delegasyonu
  silindi (workflow/src/test = 0 referans; davranis degismez): _save_selected_models_plot,
  _diagnostic_numeric, _diagnostic_float, _count_decision, _payload_expected_return,
  _write_signal_gate_diagnostics, _write_shadow_backtest_reports, _signal_calibration_sort_key,
  _write_signal_calibration_reports, _get_signal_calibration_decision_md.
- Mixin ici self.X cagrilari ilgili servisin kendi metoduna cozuluyordu; manager
  delegasyonu kullanilmiyordu. _filter_backtest_inputs_by_folds korundu (manager
  _split_walk_forward_signal_sets dahili kullanir). Workflow forward (wf=1) ve
  test-erisimli delegasyonlara dokunulmadi.
- EvaluationManager 1035 -> 979 satir. Tam suite 561 passed, golden degismedi.
  Commit 6d142ff. Sonraki: Faz 5 (ForecastRunner DI).
- Guncellenen wiki: e1-owner-forward-epic.md (Durum), index.md (durum), log.md.

## [2026-06-01] Refactor (E1 Faz 3.4) | MetricsReportingService owner-forward'dan DI'ya

- `MetricsReportingService` `_OwnerBackedService` mirasindan `(ctx, state)` DI'ya
  cevrildi (Faz 3 servis #4/4 — son servis). `_MetricsReporterMixin` govdesi
  self.ctx.X (config: dataset_metadata, commission/slippage_bps, stock_symbol,
  feature_names, xai_dir, write_xai_tables, write_markdown_reports) / self.state.X
  (mutable: predictions, prediction_targets, quantile_predictions, y_true_aligned,
  ensemble_weights, latest_backtest_results) kullanir.
- `EvaluationContext`'e 2 XAI-yazim flag eklendi (write_xai_tables=False,
  write_markdown_reports=True; default'lar eski getattr fallback'leriyle esit).
  Manager'da 2 yeni context-backed property.
- `tests/test_xai_routing.py` `_StubReporter` DI sekline (ctx/state) guncellendi.
- 4 evaluation servisinin tamami artik DI. `_OwnerBackedService` yalnizca
  workflow + `DataManager` servisleri tarafindan kullaniliyor (Faz 7 temizligi).
- Davranis degismedi: golden (12) sabit, tam suite **561 passed**. Commit `398c82f`.
- Guncellenen sayfalar: `e1-owner-forward-epic.md` (Durum Faz 3.4), `architecture.md`
  (migration status), `index.md` (DI durum maddesi).

## [2026-06-01] Refactor (E1 Faz 3.3) | SignalCalibrationService owner-forward'dan DI'ya

- `SignalCalibrationService` `_OwnerBackedService` mirasindan `(ctx, state)` DI'ya
  cevrildi (Faz 3 servis #3/4). `_SignalCalibratorMixin` govdesi self.ctx.X
  (READ-ONLY config) / self.state.X (mutable runtime) kullanir.
- `EvaluationContext`'e mixin + `signal_calibration/grid.apply_trial_policy`'nin
  okudugu 9 exe_cfg flag eklendi (calibration_scope, require_oos_confirmation,
  min_eval_excess_return/sharpe, objective, profile, sampler, seed, max_trials);
  default'lar eski getattr fallback'leriyle birebir esit. `apply_trial_policy`
  artik `self.ctx` alir. Manager'da 9 yeni context-backed property.
- Davranis degismedi: golden karakterizasyon testleri (12) degismeden, tam suite
  **561 passed**. Commit'lendi. Kalan: Faz 3.4 (MetricsReportingService).
- Guncellenen sayfalar: `e1-owner-forward-epic.md` (Durum Faz 3.3), `architecture.md`
  (migration status), `index.md` (DI durum maddesi).

## [2026-06-01] Wiki bakim | E1 ilerlemesi tum sayfalara yansitildi

- Kullanici istegi uzerine tum wiki tam kontrol edildi; E1 owner-forward
  ilerlemesi (Faz 3.1/3.2 DI'ya gecti) bayat kalan sayfalara islendi.
  `graphify`'a dokunulmadi.
- `architecture.md`: "Evaluation Services: Owner-Forward -> Explicit DI (E1 epic,
  in progress)" alt-bolumu eklendi; eski "owner-backed servisler" anlatimi guncel
  duruma baglandi (Prediction/Backtest DI, Signal/Metrics hala owner-backed).
  `last_updated` 2026-06-01; Related Pages'e epik linki.
- `index.md`: Current Project State'e E1 DI gecis durumu maddesi + epik cross-link;
  `last_updated` 2026-06-01.
- `refactor-plan.md`: "Hala acik" maddesi "Ilerlemede" olarak guncellendi (Faz 0-3.2
  tamam, kalan fazlar listelendi), epik cross-link; `last_updated` 2026-06-01.
- `code-quality-audit.md`: B1 bulgusuna ileri-isaret notu (snapshot korundu,
  aksiyon E1 epiginde yuruyor).
- `e1-owner-forward-epic.md`: temel ilkedeki stale "549 test" -> "epik basinda 549,
  guncelde 561".

## [2026-06-01] Faz 3.2 | E1 owner-forward: BacktestService DI'ya gecti

- **Faz 3 servis #2 tamamlandi** (`refactor/e1-owner-forward-di`). `BacktestService`
  artik `_OwnerBackedService`'ten MIRAS ALMIYOR; ctor `(ctx, state)` DI alir.
- `src/pipeline/backtest_runner.py` (`_BacktestRunnerMixin`): owner-forward erisimleri
  acik hale getirildi — READ-ONLY `self.ctx.X` (commission_bps, slippage_bps,
  initial_capital, backtest_enabled, signal_mode, dataset_metadata, outputs_dir,
  stock_symbol + 6 yeni exe_cfg flag), mutable `self.state.X` (signal_config,
  signal_threshold_source, latest_backtest_results, latest_backtest_metrics). Tum
  defensive `getattr(self, "...")` formlari kaldirildi.
- `src/pipeline/evaluation_services.py`: `EvaluationContext`'e BacktestService'in
  okudugu 6 exe_cfg flag eklendi (READ-ONLY): `write_trade_logs`,
  `signal_calibration_min_trades`, `signal_calibration_reject_behavior`,
  `auto_signal_diagnostics`, `enable_gate_diagnostics`, `enable_shadow_backtests`.
  Default'lar eski getattr fallback'leriyle birebir ayni (davranis korunur).
- `src/pipeline/evaluation_manager.py`: bu 6 flag context-backed property'ye cevrildi
  (`_init_execution_attrs`'taki mevcut atamalar setter uzerinden context'e akar);
  `_init_services` `BacktestService(self.context, self.state)` enjekte ediyor
  (kalan 2 servis hala owner-backed).
- Tam suite **561 passed**, golden'lar degismeden gecti. Sonraki: Faz 3.3
  (SignalCalibrationService).

## [2026-06-01] Faz 3.1 | E1 owner-forward: PredictionService DI'ya gecti

- **Faz 3 servis #1 tamamlandi** (`refactor/e1-owner-forward-di`). `PredictionService`
  artik `_OwnerBackedService`'ten MIRAS ALMIYOR; ctor `(ctx, state)` DI alir
  (`src/pipeline/evaluation_services.py`). `__getattr__`/`__setattr__` owner-forward
  yolu bu servis icin devre disi.
- `src/pipeline/prediction_engine.py` (`_PredictionEngineMixin`): owner-forward
  edilen tum attribute erisimleri acik hale getirildi — READ-ONLY config/identity
  `self.ctx.X` (`dataset_metadata`, `ensemble_enabled`, `selected_models`), mutable
  runtime cikti `self.state.X` (`predictions`, `prediction_targets`,
  `quantile_predictions`, `single_backtest_inputs`, `latest_tensors`,
  `ensemble_weights`, `ensemble_weight_scope`, `y_true_aligned`,
  `y_true_target_aligned`, `prev_close_aligned`). `getattr(self, "...")` defensive
  formlar ve gereksiz `ensemble_weight_scope` hasattr guard'i kaldirildi (state
  alani `default_factory` ile her zaman dict).
- `src/pipeline/evaluation_manager.py`: `_init_services` artik
  `PredictionService(self.context, self.state)` enjekte ediyor (digerleri hala
  owner-backed). Servis ctx/state nesnelerini cache'ler; init sonrasi
  `manager.state`/`manager.context` yeniden atanmadigi dogrulandi (sadece
  `_init_context_and_state`, servislerden once).
- `tests/test_prediction_date_aware.py` yeni DI ctor'a uyarlandi (SimpleNamespace
  owner -> `EvaluationContext`+`EvaluationState`); golden contract testleri
  (`test_owner_forward_contract.py`) DEGISMEDI ve gecti.
- Tam suite **561 passed**. Sonraki: Faz 3.2 (BacktestService).

## [2026-06-01] Faz 2 | E1 owner-forward: EvaluationContext tam tasima

- **Faz 2 tamamlandi** (`refactor/e1-owner-forward-di`). Servislerin owner'dan
  OKUDUGU tum READ-ONLY config/identity attribute'lari artik `EvaluationContext`'te
  yasiyor. `src/pipeline/evaluation_services.py`: `EvaluationContext` tum alanlari
  default'lu hale getirildi (lazy/`__new__` icin) + 9 turetilmis READ-ONLY alan
  eklendi (`ensemble_enabled`, `selected_models`, `backtest_enabled`,
  `commission_bps`, `slippage_bps`, `initial_capital`, `signal_mode`,
  `default_signal_config`, `xai_dir`).
- `src/pipeline/evaluation_manager.py`: 19 config/identity attribute (10 base +
  9 turetilmis) context-backed **property**'ye cevrildi
  (`manager.X` <-> `manager.context.X`). `__init__`'teki duz `self.stock_symbol = ...`
  atamalari kaldirildi; base alanlar context constructor'da, turetilmis alanlar
  `_init_model_attrs`/`_init_execution_attrs`/`_init_signal_calibration_state`/
  `_init_mutable_state` icindeki mevcut atamalar uzerinden (property setter ile)
  context'e yazilir — mixin govdesine yine DOKUNULMADI (Faz 3'te `self.ctx.X`).
- `context` lazy property: `state` ile ayni gerekce; `__new__` ile kurulan
  mekanizma testleri (`test_phase8_acceptance` `manager.outputs_dir = ...`,
  `commission_bps = ...` vb.) ilk eriste bos `EvaluationContext()` alir; testlere
  dokunulmadi.
- Mixin/workflow tarafinda bu attribute'lara YAZIM yok (AST + grep dogrulandi) —
  gercekten READ-ONLY; context'e tasima davranisi degistirmez.
- Davranis korundu: tam suite **561 passed**, karakterizasyon golden'lari
  (`test_owner_forward_contract.py`, ozellikle `test_initial_signal_state_golden`:
  `default_signal_config == signal_config`, `xai_dir == outputs_dir/xai`)
  degismeden gecti. Sonraki: Faz 3 (mixin -> davranis-sahibi servis, en riskli).

## [2026-06-01] Faz 1 | E1 owner-forward: EvaluationState tam tasima

- **Faz 1 tamamlandi** (`refactor/e1-owner-forward-di`). EvaluationManager'in tum
  mutable evaluation state'i artik `EvaluationState`'te yasiyor.
  `src/pipeline/evaluation_services.py`: `EvaluationState` 7 alanla genisletildi
  (`ensemble_weight_scope`, `y_true_aligned`, `y_true_target_aligned`,
  `prev_close_aligned`, `signal_config`, `signal_threshold_source`,
  `signal_threshold_calibration_summary`).
- `src/pipeline/evaluation_manager.py`: 16 mutable attribute state-backed
  **property**'ye cevrildi (`manager.X` <-> `manager.state.X`). Owner-forward
  servisler/workflow'lar `setattr(owner, X, ...)` ile yazdiginda yazim property
  setter'i uzerinden state'e gider — mixin govdesine DOKUNULMADI (Faz 3'te
  `self.state.X`'e gececek). `__init__` yeniden siralandi: `_init_context_and_state`
  artik ONCE calisir (bos `EvaluationState()` kurar), state'e yazan `_init_*`'ler
  sonra. Eski "state init'ten dict referanslari kopyalama" kaldirildi.
- `state` lazy property: `__new__` ile `__init__`'i atlayan mekanizma testleri
  (`test_reporting_metrics`, `test_phase8_acceptance`, ...) icin ilk eriste
  otomatik `EvaluationState()` kurar; gercek `__init__` zaten explicit kurar.
  Bu sayede mekanizma testlerine dokunulmadan suite yesil kaldi.
- Davranis korundu: karakterizasyon golden'lari (`test_owner_forward_contract.py`,
  12 test) degismeden gecti. Tam suite **561 passed**. Sonraki: Faz 2
  (`EvaluationContext` tam tasima — READ-ONLY config/identity attribute'lari).

## [2026-06-01] Faz 0 | E1 owner-forward karakterizasyon + envanter

- **Faz 0 tamamlandi** (`refactor/e1-owner-forward-di`). Karakterizasyon golden'lari
  `tests/test_owner_forward_contract.py` (12 test): davranis-seviyesi, mekanizma
  degil — owner-forward servisler DI'ya cevrilirken DEGISMEZ referans. Kapsam:
  kurulus state golden'i, `manager` <-> `manager.state` alias kimligi, paylasilan
  mutable state iki-yonlu mutasyon, servis-kompozisyonu uzerinden saf hesaplamalar
  (`_target_to_price`/`_weighted_average`/`_base_predictions_for_ensemble`),
  determinizm.
- Owner read/write envanteri `tools/owner_forward_inventory.py` (gecici AST araci,
  Faz 7'de silinir) ile cikarildi; MUTABLE -> `EvaluationState`, READ-ONLY ->
  `EvaluationContext` siniflandirmasi `e1-owner-forward-epic.md` §8'e islendi.
  Faz 1'de state'e tasinacak yeni alanlar: `y_true_aligned`,
  `y_true_target_aligned`, `prev_close_aligned`, `ensemble_weight_scope`,
  `signal_config`, `signal_threshold_source`, `signal_threshold_calibration_summary`.
- Tam suite **561 passed** (temiz `--basetemp`). NOT: `.codex_tmp/pytest` baska
  process'e kilitliyken `tmp_path` kullanan testlerde 46 ERROR (WinError 5) cevresel
  — koddan degil; `--basetemp` override ile temiz gecer.

## [2026-06-01] Ekle | E1 owner-forward kaldirma epik plani + yeni dal

- `docs/wiki/e1-owner-forward-epic.md` olusturuldu: tam owner-forward kaldirma
  (mixin -> EvaluationContext/EvaluationState DI), 2 taban + 3 owner state yuzeyi
  envanteri, korunacak invariantlar (leakage/determinizm/monkeypatch namespace),
  karakterizasyon testi stratejisi (test_owner_forward_contract.py), 7 fazli plan,
  kabul kriterleri + rollback.
- Yeni dal `refactor/e1-owner-forward-di` acildi (yeni session burada Faz 0'dan
  baslar). index.md link tablosu guncellendi.

## [2026-06-01] Refactor | Tier 3 mimari (E3 god ctor + E1 owner-forward guard)

- **E3 god constructor** (commit 1e5c4be): `ForecastingPipeline.__init__` ve
  `EvaluationManager.__init__` attribute-grup yardimcilarina bolundu
  (`_init_config_attrs`/`_init_run_identity`/`_init_collaborators`;
  `_init_model_attrs`/`_init_execution_attrs`/`_init_signal_calibration_state`/
  `_init_mutable_state`/`_init_context_and_state`). Davranis + cagri sirasi
  (signal_threshold_metadata erken servis init dahil) birebir korundu.
- **E1 owner-forward fail-loud** (commit a541a10): `_OwnerBackedService.__setattr__`
  artik var-olan-owner-attribute (veya bildirilmis lazy `ensemble_weight_scope`)
  disindaki yazimda AttributeError firlatir -> B1 sessiz-typo encapsulation acigi
  kapandi. Guard opt-in (`_FAIL_LOUD`): evaluation servisleri + workflow'larda
  aktif; DataManager servis ailesi (4 sinif) permissive birakildi (owner state
  yuzeyi henuz sertlestirilmedi). `ensemble_weight_scope` EvaluationManager'da
  pre-init edildi.
- **Ertelendi:** tam owner-forward kaldirma (mixin -> davranis-sahibi DI servis,
  ~4500 satir, leakage/sozlesme riski) ve `forecasting/workflows`'un ayri
  `_OwnerBackedForecastService` base'i. Bagimsiz buyuk epik; karakterizasyon
  testi sart.
- Test: tum suite yesil (549). Guard fonksiyonel dogrulandi (typo yakalandi).

## [2026-05-31] Refactor | Tier 2 dosya/sorumluluk bolme (5 commit)

- `analysis_service.build`: iki refresh dali `_try_refresh_and_rebuild`'e DRY.
- `api/main.py` (425->~340L): POST /run job tracker yeni `pipeline_jobs.py`'ye
  tasindi; main artik yalniz route katmani.
- `macro_pipeline`: global-gosterge dongusu `_refresh_global_daily_frames`'e.
  Bulgu: 29 metod ama kohezyon yuksek -> zorla file-split yok.
- `data_services`: `run` (123->~30L) iki helper'a bolundu. Bulgu: paket-split
  GUVENSIZ (testler load_data/DataUpdater/FeatureCache monkeypatch ediyor);
  in-place decomposition secildi. prepare_tensors (scaling) ertelendi.
- `signal_calibrator` (997L): bulgu -> zaten yogun decompose (grid logic
  calibration_grid.py'de). Zararli god object degil; esas borc owner-forward
  mixin (B1/E1). Zorla split YAPILMADI; E1/Tier 3'e ertelendi.
- Ilke: buyuk dosya != kotu dosya. Kozmetik LOC-azaltma icin zorla bolme yok.
- Her commit davranis-koruyan; ilgili test suitleri yesil. Sirada Tier 3 (E1/E3).

## [2026-05-31] Refactor | Tier 1 E2 DRY birlestirmeleri

- **Ikiz ensemble builder:** prediction_engine `_add_single_split_ensembles`
  (CXTY 51->17) + `_add_walk_forward_ensembles` (CXTY 58->21); ~140 satir kopya
  cekirdek `_compute_ensemble_blends`'e cikarildi, ortak payload dilimleme
  `_slice_template_payload`'a alindi. Cash-gate length-guard ortaklasti.
- **Tree tune_and_train:** yeni `src/models/_tuning.py` (`run_optuna_study` +
  `stability_adjusted_cv_objective`). XGBoost + Random Forest ortak cekirdegi
  kullanir; ~50 satir x2 kopya silindi.
- **3x run() workflow:** guclu ikiz OLMADIGI tespit edildi; zorla template-method
  KASITLI uygulanmadi (B1 kozmetik-SRP tuzagi). Yalniz metadata-attach ciftleri
  `_attach_score_metadata`/`_attach_guard_metadata`'ya alindi.
- Davranis korundu (leakage/determinizm/Optuna warm-start). py_compile + 114 test
  + tune_and_train fonksiyonel smoke yesil. Sirada Tier 2 (dosya bolme).

## [2026-05-31] Refactor | Tier 0 fonksiyon parcalama (davranis-koruyan)

- `refactor-plan.md` Tier 0 uygulandi: 7 yuksek-karmasiklik fonksiyon helper'lara
  bolundu. CXTY hepsinde hedefe (<12) indi:
  `walk_forward_splits` 27→6, `WalkForwardValidator.run` 28→7 (171→43L),
  `summarize_backtest` 27→10 (193→130L), `compute_market_regime` 33→4,
  `compute_regime_context` 24→7, `compute_confidence` 41→11 (143→84L),
  `batch.main` 37→6 (150→38L).
- Davranis degismedi (leakage/determinizm/sys.exit/warnings semantigi korundu);
  py_compile + 121 ilgili test yesil.
- CLAUDE.md auto-refresh kurali graphify'dan WIKI guncellemesine cevrildi
  (graphify auto-trigger kaldirildi).
- Sirada Tier 1 (E2 DRY birlestirmeleri).

## [2026-05-31] Ekle | Asamali refactor plani (8 asama, davranis-koruyan)

- Yeni sayfa `docs/wiki/refactor-plan.md`: [code-review-stages.md] 8 asamasina
  gore god-object / sismi dosya-fonksiyon / CXTY / SOLID-KISS-DRY bulgulari +
  davranis-koruyan refactor aksiyonlari. `index.md`'ye link eklendi.
- Taze AST taramasi (per-stage LOC/CXTY/god-class). En kritik: CXTY
  `_add_walk_forward_ensembles` 58 / `_add_single_split_ensembles` 51
  (prediction_engine), `compute_confidence` 41, `batch.main` 37,
  `compute_market_regime` 33. En buyuk dosya `signal_calibrator.py` 997L; en
  buyuk class `_SignalCalibratorMixin` 34m/898L.
- Caprez-kesen epikler: E1 owner-forward kaldirma (evaluation_services +
  forecasting/workflows, B1), E2 DRY (ikiz ensemble builder + 3x run() + tree
  tune_and_train), E3 god constructor (orchestrator/evaluation_manager __init__).
- Uygulama sirasi risk-ayarli 4 tier: Tier0 fonksiyon parcalama (dusuk risk) →
  Tier1 DRY → Tier2 dosya bolme → Tier3 mimari (en son). Henuz uygulanmadi.
- Docstring Faz 3-4 refactor sonrasina ertelendi (yapi degisecek).

## [2026-05-31] Ekle | Kod kalitesi denetimi + docstring plani (Faz 1)

- Yeni sayfa `docs/wiki/code-quality-audit.md`: god-object/SOLID/DRY bulgulari +
  fazli docstring/yorum plani. `index.md`'ye link eklendi.
- Ana bulgu (B1): `evaluation_services._OwnerBackedService` ve
  `forecasting/workflows._OwnerBackedForecastService` kozmetik servis ayrimi —
  tum attribute erisimi tek owner'a forward; `EvaluationManager` de-facto god
  object (48 metod, mixin'ler ~2300 satir paylasimli state). Refactor ayri plan.
- Metrikler: public docstring kapsami %31 (145/467); module-docstring eksik 31
  modul; 500+ satir 16 dosya; 100+ satir 31 fonksiyon.
- **Faz 1 uygulandi:** module-docstring'i olmayan modullere sorumluluk ozeti
  eklendi (kod davranisi degismedi).
- **Faz 2 (oncelik 1-3) uygulandi:** 77 public docstring eklendi — validation +
  backtest (11), persistence stock_model_db/repositories (37), api/services +
  forecasting/workflows (29). Public docstring kapsami %31 -> %47 (145 -> 221).
  py_compile + 89 ilgili test yesil. Faz 3-4 beklemede.

## [2026-05-31] Fix | Review acik bulgulari giderildi (3 fix, 549 pass)

- **#1 forecast UTC** — `database/repositories/forecast.py` `run_at` artik
  `datetime.now(tz=timezone.utc)` (Sprint 9 UTC mandate'i; api katmaniyla tutarli).
- **#2 PSI ham kolon** — `data/quality.py::compute_psi` ham OHLCV/fiyat
  (Open/High/Low/Close/Adj_Close/Volume) sutunlarini disliyor; non-stationary
  trend `psi_high`'i yaniltici tetiklemesin.
- **#3 correlation pruning leakage** — `correlation_pruning.prune_correlated_features`
  artik opsiyonel `fit_df` aliyor: korelasyon yalnizca egitim diliminde
  hesaplanir, dusurme tum frame'e uygulanir. `FeaturePipeline.engineer_features`
  `prune_fit_tail` parametresi ekledi; `DataIngestionService` final_holdout_size
  kadar tail'i corr fit'inden cikariyor. (Residual: WF test fold'lari hala fit'te;
  pruning default kapali.)
- Yeni test: `tests/test_stage1_review_fixes.py` (3 test) — PSI ham kolon
  exclusion + pruning fit_df train-only + backcompat.
- Tum suite: **549 passed, 0 failed** (546 + 3 yeni).

## [2026-05-31] Review | 8 asamali code-review tamamlandi (546 pass)

- `code-review-stages.md` rehberindeki 8 asama sirayla review edildi (dl_env).
  Tum suite yesil: **546 passed, 0 failed**.
- Asama sonuclari: S1 Veri/Feature 66, S2 Model/Factory 125, S3 Egitim/
  Validation 73, S4 Backtest/Sinyal 36, S5 Eval/Rapor/XAI 77, S6 Persistence 46,
  S7 Forecast/API 92, S8 CLI/Orkestrasyon 69.
- **3 stale test fix** (kod davranisi hep dogruydu; testler eski varsayim):
  1. `test_data_services.py` — embargo auto-default 200 nedeniyle holdout
     ayrilamadi; `wf_embargo_size=2` eklendi.
  2. `test_forecast_workflows.py` — Sprint 4 `predict_quantiles_target`
     cagrisi; fake Predictor double'a None donen metod eklendi.
  3. `test_scope_backward_compat.py` — Sprint 4 `LightGBM Quantile` candidate
     eklendi; beklenen set+order guncellendi (canonical leftover, sona eklenir).
- **Leakage dogrulamasi (temiz):** scaler train-only fit, kronolojik split +
  embargo, makro release-lag, backtest `positions[t]*realized[t]` kausal (ekstra
  lag yok), sinyal kalibrasyon `_assert_wf_train_scope` (final holdout disi),
  recursive forecast kausal zincir, serve'de retrain yok.
- **Acik bulgular (fix yok, karar bekliyor):**
  1. `correlation_pruning` tum frame uzerinde corr → feature-secim leakage
     (default kapali, dusuk sev).
  2. `quality.compute_psi` ham non-stationary kolonlari da alabilir → psi_high
     yanlis tetikleme riski (caller frame netlestirilmeli).
  3. `database/repositories/forecast.py:39` `datetime.now()` timezone-naive;
     Sprint 9 UTC mandate'ina aykiri (api katmani UTC). Fix: `tz=timezone.utc`.
- Kullanilan interpreter: `anaconda3/envs/dl_env` (PATH'teki Python310'da `ta` yok).

## [2026-05-31] Fix | Stage 1 review — stale test (embargo auto-default)

- `tests/test_data_services.py::test_walk_forward_split_excludes_final_holdout_from_selection`
  fail veriyordu: `wf_embargo_size` set edilmediginden Sprint 0 auto-default
  `_resolve_wf_embargo_size(None|0, time_steps=3)=max(200,3)=200` uygulaniyor,
  40-satirlik fixture'da `min_required=10+200+8+4=222>40` → holdout ayrilamiyor,
  `source_rows=40 != 36`. Kod davranisi DOGRU/kasitli; test Sprint 0 sonrasi
  guncellenmemis. Fix: teste `wf_embargo_size=2` eklendi (monkeypatch'li split
  zaten 2-satir embargo_context kullaniyor; min_required=24<=40). Test yesil.
- Not: `_resolve_wf_embargo_size` 0 ve negatifi de "unset" sayip 200'e cikariyor;
  embargo'yu kucuk pozitif vermek gerekir.
- Stage 1 suite dl_env'de yesil: 66 passed.
- Acik bulgular (henuz fix yok): (1) `correlation_pruning` tum frame uzerinde
  corr hesapliyor → feature-secim leakage'i (default kapali, dusuk sev);
  (2) `quality.compute_psi` Date/Symbol disindaki tum numeric kolonlari aliyor,
  ham non-stationary `Close` gecerse psi_high yanlis tetiklenebilir — caller'in
  hangi frame'i gecirdigi netlestirilmeli.
- Kullanilan interpreter: `anaconda3/envs/dl_env` (Python310 PATH'inde `ta` yok).

## [2026-05-31] Ekle | Aşamalı code-review rehberi

- `docs/wiki/code-review-stages.md` eklendi: proje bağımlılık-sıralı 8 aşamaya
  bölündü (1-Veri/Feature, 2-Model/Factory, 3-Eğitim/Validation, 4-Backtest/
  Sinyal, 5-Eval/Rapor/XAI, 6-Persistence/Seçim, 7-Forecast/API, 8-CLI/
  Orkestrasyon). Her aşama: dosya+amaç tablosu, review checklist, ilgili pytest
  komutu, bağımlılık notu. Ayrıca aşama bağımlılık grafiği + çapraz-kesen
  endişeler (leakage, reproducibility, UTC, CORS/rate-limit) bölümü.
- Amaç: tek seferde review zor olan ~120 dosya / 16 modül / 68 test'i izole,
  sıralı incelenebilir parçalara ayırmak.
- `index.md`: System Knowledge altına yeni sayfa linki eklendi.
- Yalnız dokümantasyon — kod değişmedi.

## [2026-05-26] Feature | Sprint 9 — Advisory audit log + response cache + rate limit + UTC datetime

- `src/api/services/advisory_audit.py` eklendi: `AdvisoryAuditRecord`
  dataclass + `build_record_from_response()` (AnalysisResponse'tan field
  cikartir) + `append_record()` / `append_response()` (CSV append, thread-
  safe). Disk yolu varsayilan `data/advisory_history.csv`. Yazma hatasi
  response'u bozmaz (best-effort).
- `src/api/services/response_cache.py` eklendi: `ResponseCache` 24h TTL
  in-memory cache; uppercase + strip key normalize; lazy eviction;
  `AI_CORE_RESPONSE_CACHE_TTL_SECONDS` ve `AI_CORE_RESPONSE_CACHE_DISABLED`
  env'leri ile yonetim.
- `src/api/services/rate_limit.py` eklendi: `RateLimiter` fixed-window
  IP rate limit (default 60/min/IP); `AI_CORE_RATE_LIMIT_PER_MINUTE` ve
  `AI_CORE_RATE_LIMIT_TRUSTED_IPS` env'leri; `rate_limit_middleware_factory()`
  lazy-import starlette ile FastAPI'ye middleware uretir. Over-limit 429 +
  Retry-After.
- `src/api/routers/analysis.py`: `/analysis` ve `/v1/analysis` icin response
  cache hit kontrolu + audit log append eklendi.
- `src/api/main.py`: rate limit middleware enable koruyucusu + tum
  `datetime.now()` cagrilari `datetime.now(tz=timezone.utc)` ile degistirildi
  (RULES.md timezone-aware).
- Wiki: `persistence-and-api.md` source_count 11->14 (Advisory Audit Log +
  Response Cache + Rate Limit + Timezone-Aware Datetimes bolumleri); log
  ust girisi.
- Tests: `test_advisory_audit.py` (6), `test_response_cache.py` (7),
  `test_rate_limit.py` (6). Sprint 0-9 toplam: 127 pass / 6 skip.

## [2026-05-26] Feature | Sprint 8 — Analysis API Faz 2 doldurma (confidence reasons + ensemble agreement + /v1 alias)

- `src/api/services/analysis_service.py`:
  - `compute_confidence` cagrisi `psi_high` (PSI 30d major_drift),
    `rmse_vs_benchmark`, `ensemble_direction_agreement` parametreleriyle
    beslendi (Faz 2 confidence hesabi).
  - `_build_positive_reasons()` helper eklendi: `dir_acc`, `hit_rate`,
    `rmse_vs_benchmark < 1.0`, `stability_score >= 0.5`, `composite_score`,
    `ensemble_direction_agreement >= 5/7`, `psi_status == "stable"`
    sinyallerinden Turkce kisa neden ifadeleri uretir.
  - `confidence.reasons` medium/high label'larinda dolduruluyor.
  - moderate_drift `data_drift_moderate:` warning string'i ve high->medium
    downgrade'i tutuldu; major_drift compute_confidence path'iyla zaten
    low'a dusurulurken `data_drift_major:` warning'i de eklendi.
- `src/api/routers/analysis.py`: `/v1/analysis/{symbol}` alias rotasi eklendi
  (gelecek breaking change icin yer). Davranis `/analysis/{symbol}` ile birebir
  ayni; sadece path versiyonlu.
- `forecast.ensemble_agreement` (zaten Sprint 4'te persistence + schema
  hazirlanmisti) confidence-and-risk-policy ile akti dogrulandi.
- XAI top reasons (`build_xai_product_summary`) zaten CSV'den okuyup
  yayinliyor; degisiklik gerektirmedi.
- Wiki: `analysis-api-contract.md` source_count 6->7; "Confidence Reasons"
  + "Forecast Ensemble Agreement" + `/v1` alias bolumleri eklendi.
- Tests: `test_analysis_api_faz2.py` (8 test: confidence reasons,
  quantile fields propagation via monkeypatch, ensemble agreement,
  disclaimer, data_quality block, /v1 alias). Sprint 0-8 toplam:
  108 pass / 6 skip (FastAPI/httpx eksik env'lerde alias testleri skip).

## [2026-05-25] Decision | Plan PAUSED at Sprint 8

- Sprint 7 (`71bba60`) tamamlandi (calendar + cross-sectional momentum +
  PSI 30d monitor API). graphify 3351n/5125e/281c (AST-only RULES.md).
- Kullanici Sprint 8 (Analysis API Faz 2 doldurma: confidence.reasons,
  xai.top_*_reasons, forecast.ensemble_agreement, /v1 alias) onayini
  "Hayir, dur" olarak verdi.
- Plan Bolum 10 Live State: Sprint 8-9 pending.
- Devam icin: "Sprint 8'i baslat" denildiginde A8.1'den baslanir.

## [2026-05-25] Feature | Sprint 7 — Calendar + Cross-Sectional Momentum + PSI 30d API

- `src/features/feature_pipeline.py`: `_add_calendar_features(df)` Date'e
  dayali stasyoner takvim sutunlari: `day_of_week`, `day_of_month`,
  `days_to_month_end`, `days_to_quarter_end`, `is_quarter_end_week`,
  `days_to_next_fomc`. FOMC tarihi `data/meta/fomc_calendar.csv` statik
  dosyadan okunur; yoksa 365.0 placeholder.
- `_add_cross_sectional_momentum`: `momentum_60d` (Close.pct_change(60)),
  `market_momentum_60d` (BIST100_Return cumprod), `sector_momentum_60d`
  (sektor index Return cumprod; eksikse market fallback),
  `relative_momentum_60d`, `relative_to_market_60d`. `_merge_macro` icinde
  cagrilir; eksik kolon = sessiz atla.
- FeaturePipeline constructor: `enable_calendar_features`,
  `enable_cross_sectional_momentum`, `fomc_calendar_path` parametreleri
  eklendi (default True).
- `src/api/services/data_quality_monitor.py` eklendi:
  `compute_psi_30d(symbol_csv)` son 30 isgununu onceki 252g ile karsilastirir;
  stasyoner OHLCV-turevli feature'lar (log_return, range_pct,
  volume_log_change), bins=3 (kucuk holdout noise icin).
  Tier: `<0.10 stable`, `0.10-0.25 moderate_drift`, `>=0.25 major_drift`,
  `unavailable` (CSV yok / insufficient history).
- `src/api/schemas/analysis.py`: `DataQualityBlock` eklendi;
  `AnalysisResponse.data_quality` opsiyonel.
- `src/api/services/analysis_service.py`: `data_quality` blogu API'ye
  tasindi. `major_drift` -> confidence `low` + `data_drift_major:`
  warning; `moderate_drift` -> `high`->`medium` downgrade +
  `data_drift_moderate:` warning.
- `data/meta/fomc_calendar.csv` (statik 2022-2027 FOMC takvimi) eklendi.
- Wiki: `data-pipeline.md` (source_count 9->11) Calendar + Cross-Sectional
  Momentum bolumleri; `analysis-api-contract.md` (source_count 5->6)
  data_quality blogu + confidence etkilesimi; `confidence-and-risk-policy.md`
  Live PSI 30d Monitor bolumu.
- Tests: `test_calendar_features.py` (7), `test_cross_sectional_momentum.py`
  (6), `test_psi_monitor.py` (6). Sprint 0-5 + Sprint 7 = 80 pass / 4 skip.

## [2026-05-25] Decision | Plan PAUSED at Sprint 7

- Sprint 5 (`bcf5e62`) tamamlandi, Sprint 6 (TFT) bilincli olarak
  ertelendi (ayri plan ileride).
- Kullanici Sprint 7 (Calendar + Cross-Sectional Momentum + PSI
  Monitor API) onayini "Hayir, dur" olarak verdi.
- Plan Bolum 10 Live State: Sprint 7 ⏸ AWAITING APPROVAL.
- Devam icin: "Sprint 7'yi baslat" denildiginde A7.1'den baslanir.

## [2026-05-25] Feature | Sprint 5 — Recursive Feature Recompute + Macro Forward Projection

- `src/features/feature_pipeline.py`: `recompute_close_dependent(frame)`
  eklendi. Recursive forecast satiri eklendikten sonra close-bagimli
  tum teknik gostergeleri (SMA/EMA/RSI/MACD/Bollinger/ATR/NATR/ADX/
  MFI/CMF/OBV/VWAP/Market_Regime) `_add_*` zinciri ile yeniden uretir.
  Macro/sector/lag sutunlari KORUNUR.
- `src/features/macro_forward_projection.py` eklendi.
  `MacroForwardProjector`: known macro whitelist (USDTRY/BIST100/VIX/
  INTEREST_RATE/CPI/BRENT/GOLD/EURTRY + sektor _Return'leri) icin
  ARIMA(1,1,1) tek-adim forecast (history_window=252). ARIMA fail
  -> son 20 gun ortalama-delta trend extrapolation fallback.
  `project_last_row(frame, target_date)` yalniz son (recursive)
  satiri gunceller; tarihsel rows degismez.
- `src/forecasting/workflows.py`:
  `ForecastPointGenerator._recompute_close_dependent_safe()` ve
  `_apply_macro_forward_projection_safe()` helper'lari eklendi.
  Recursive loop her horizon adimi sonrasi sirasiyla recompute +
  projection cagiriyor; helper'lar exception'da frame'i degistirmez
  (graceful degradation).
- `frozen_exogenous_features` warning enum'u `projected_exogenous_features`
  ile degistirildi (single forecast + ensemble path).
- Tests:
  `tests/test_recompute_close_dependent.py` (5 test) - real `ta`
    real-dep guard, SMA_7_rel/RSI/Market_Regime guncellenir, lag/
    macro sutunlar dokunulmaz.
  `tests/test_macro_forward_projection.py` (8 test) - auto column
    resolve, override, project_last_row macro-only update,
    empty/no-macro/short-series/single-value/empty-series fallback.
  `tests/test_analysis_endpoint.py`: fixture warning enum guncellendi.
- Sprint 0+1+2+3+4+5 toplam: 82/82 pass, 3 skip (lightgbm/tensorflow/
  ta real-dep gerekli).
- Wiki: data-pipeline.md "Recursive Forecast Feature Recompute" +
  "Macro Forward Projection" bolumleri (source_count 8->9).
  analysis-api-contract.md warning enum guncellendi.

## [2026-05-25] Feature | Sprint 4 — Probabilistic Forecasting (Quantile LightGBM + MC Dropout LSTM + Multi-Horizon)

- `src/models/quantile_lightgbm_model.py` eklendi.
  `QuantileLightGBMModel`: her quantile icin ayri LGBMRegressor
  (objective="quantile", alpha=q). Default `(0.1, 0.5, 0.9)`.
  `predict_quantiles(X) -> (N, len(quantiles))` row-wise sort
  (quantile crossing guard). `predict()` median (p50) doner —
  BaseModel sozlesmesi korunur. Registry'de
  `ensemble_eligible=False` (scalar ensemble quantile cikti ile
  uyumsuz). Requires `lightgbm`.
- `src/models/lstm_lite_model.py`: `predict_quantiles(X,
  n_samples=200, quantiles=(0.1,0.5,0.9), seed=20260525)` eklendi.
  MC Dropout (`training=True` inference) ile empirical posterior;
  row-wise sort.
- `src/pipeline/data_services.py`:
  `TensorPreparationService.build_multi_horizon_targets(close,
  horizons=(1,3,5,10))` eklendi. Opt-in multi-horizon target uretimi
  (tek-horizon path geriye uyumlu). Cikti: `{h: np.ndarray}` per
  target_mode (log_return/return/price).
- `src/pipeline/config.py` `DataConfig.target_horizons: Optional[List[int]]`
  opt-in flag eklendi (default `None` -> tek-horizon).
- `src/forecasting/workflows.py`:
  `LatestTargetPredictionWorkflow.predict_quantiles_target()` model
  `predict_quantiles` destekliyorsa `{quantile: target_value}` dict
  doner; scaler_y inverse_transform ile target space. Sequence/tree/
  date-aware modeller icin uygun `latest_*` context'i secer.
  `ForecastPointGenerator.roll_forward_recursive()` her horizon
  adiminda quantile_targets uretir; her q icin `_target_to_price`
  + `bound_forecast_price` -> `p10_close/p50_close/p90_close` +
  `predicted_return_p10/p50/p90` point alanlari + nested
  `quantile_close`/`quantile_returns` dict'leri.
- `src/api/schemas/analysis.py` `ForecastPoint`:
  `p10_close, p50_close, p90_close,
   predicted_return_p10/p50/p90,
   lower_band, upper_band, price_tick` opsiyonel alanlar.
- `src/api/services/analysis_service.py` `_build_forecast_block`:
  yeni quantile/band alanlarini point dict'lerinden okuyup yayinlar.
- `src/models/__init__.py` `QuantileLightGBMModel` lazy export.
- `conftest.py`: joblib + sklearn (preprocessing) stub eklendi
  (data_services/preprocessor chain test ortaminda kirilmasin diye).
  sklearn.metrics BILEREK stub edilmez — financial_metrics try/except
  yedek implementasyona dusebilsin.
- Testler:
  `tests/test_quantile_lightgbm.py` (8 test) — lightgbm real-dep
  guard (conftest MagicMock atlanir), constructor validasyon,
  train/predict/quantile sirali, median == p50, save/load roundtrip.
  `tests/test_mc_dropout_lstm.py` (6 test) — tensorflow real-dep
  guard, quantile shape + row-sort, dropout stokastik.
  `tests/test_multi_horizon_targets.py` (8 test) — h=1 legacy
  uyumu, lengths, h>n empty, h=0 raises, simple_return/price modlari.
  `tests/test_recursive_quantile_path.py` (5 test) —
  `predict_quantiles_target` stub model/scaler ile dict output,
  seq vs tree path, missing latest, scaler inverse.
- Sprint 0+1+2+3+4 toplam: 74/74 gecti, 2 skip (lightgbm/tensorflow
  real-dep gerekli).

## [2026-05-25] Feature | Sprint 3 — PurgedKFold + CPCV + Concat-Sharpe

- `src/validation/purged_kfold.py` eklendi. AFML Ch.7 PurgedKFold:
  test fold'unun etrafinda `purge_window` (onerilen `max(200, time_steps)`)
  + arkasinda `embargo` train ornekleri atilir. Constructor validasyon,
  `split()` iterator (train_idx, test_idx).
- `src/validation/cpcv.py` eklendi. AFML Ch.12 CombinatorialPurgedCV:
  veriyi `n_groups` parcaya boler, `C(n_groups, k_test)` kombinasyon
  uretir; her path icin purge + embargo. Default `n_groups=6, k_test=2`
  → 15 path.
- `src/validation/walk_forward.py`:
  `_compute_strategy_returns()` + `_bootstrap_sharpe_ci()` yardimcilari.
  `WalkForwardValidator.run()` fold strateji getirilerini biriktirir,
  concat eder; `aggregated_metrics` icine `Sharpe_Concat`,
  `Sharpe_CI_95_Low`, `Sharpe_CI_95_High`, `Concat_Returns_N` eklenir.
  Bootstrap 1000 resample, seed 20260525 (deterministic).
  Preprocessor lazy-import (`reconstruct_prices_*`) — joblib yokken
  yardimcilar test edilebilir.
- `src/pipeline/config.py` `ValidationConfig`: opt-in flag'ler
  `use_purged_kfold`, `use_cpcv`, `cpcv_n_groups=6`, `cpcv_k_test=2`.
  Default `False` — production WF akisi degismez.
- Risk-free fail-loud (Sprint 1 A1.1) ile uyumlu: rf yoksa concat-Sharpe
  ve CI alanlari NaN doner; `aggregated_metrics`'da `Sharpe_Warning`
  zinciri korunur.
- Testler:
  `tests/test_purged_kfold.py` (9 test) — kurucu validasyon, partition,
  purge zone, embargo zone, disjoint.
  `tests/test_cpcv.py` (9 test) — `C(N,k)` path sayisi, purge, disjoint.
  `tests/test_concat_sharpe.py` (7 test) — strategy returns log_return /
  sign mismatch / empty, bootstrap CI low/high / NaN-on-no-rf / few samples,
  concat-Sharpe ≠ mean-of-fold-Sharpe.
- Sprint 0+1+2+3 toplam: 61/61 test gecti.
- Plan Bolum 10 Live State guncellendi: Sprint 3 ✅ TAMAM.

## [2026-05-25] Feature | Corporate Action Audit + Survivorship Report + PSI Threshold Tiers

- `tools/audit_corporate_actions.py` eklendi. `data/*.csv` altindaki
  her hisseyi tarayip `|log_return| >= 0.30` olan gunleri tespit eder;
  CSV raporu `outputs/_audits/corporate_action_audit_{ts}.csv` ve
  `corporate_action_audit_latest.csv` olarak yazilir. Severity:
  `high` (>= 0.30) / `extreme` (>= 0.50). `--threshold`, `--universe`,
  `--data-dir`, `--out` argumanlari ile esnek calistirilabilir.
- `src/data/data_updater.py`: `YFinanceProvider.AUTO_ADJUST = True` HARD
  kontrolu (sinif sabiti). Daha onceki `auto_adjust=False` ham seriyi
  donduruyordu; bu split/temettu sizintisini kalici olarak kapatti.
- `src/data/data_loader.py` `_compute_survivorship_report()` ekledi.
  `df.attrs["survivorship_bias_report"]`:
  `{symbol, actual_start, actual_end, span_days, row_count,
    max_gap_days, short_history_warning,
    delisted_or_late_listing_warning, warning}`.
  Triggerlar: span < 2 yil veya max_gap > 10 gun.
- `src/data/quality.py` iki yenilik:
  - `_check_audit_anomaly()`: audit_latest.csv'den son 252 isgunu
    icin symbol anomalisi tarar; eski Adj_Close-bazli bayraga ek
    kaynak. `corporate_action_anomaly` ikinci kaynaktan beslenir.
  - `compute_quality_flags()` cikitsina `survivorship_bias_report`
    payload'i eklendi.
- PSI hesabi (`compute_psi`, `_psi_one_feature`) zaten mevcuttu; Sprint
  2 sadece test+wiki zinciri ekledi. Threshold tiers:
  `< 0.10 stable | 0.10-0.25 moderate_drift | >= 0.25 major_drift`.
- Yeni testler: `tests/test_audit_corporate_actions.py` (5),
  `tests/test_psi.py` (6). Sprint 0+1+2 yeni testler 36/36 gecti.
- Wiki: `data-pipeline.md` "Corporate Action Audit" + "Survivorship
  Bias Report" + "PSI" bolumleri (last_updated 2026-05-25,
  source_count 7->8). `confidence-and-risk-policy.md` PSI tier
  tablosu + audit-bazli corporate_action_anomaly aciklamasi.

Plan referansi:
`~/.claude/plans/sistematik-ad-m-ad-m-yap-lacak-expressive-eich.md`
Sprint 2 (A2.1, A2.2, A2.3, A2.4, A2.5, A2.6).

## [2026-05-25] Decision | Advisory Metric Onceligi + Risk-Free Fail-Loud + Backtest Disclaimer

- `src/utils/risk_free_rate.get_current_risk_free_rate` artik 0.40 sabit
  fallback'i ICERMEZ; macro `INTEREST_RATE.csv` + `RISK_FREE_RATE_ANNUAL`
  env yoksa `None` doner. Cagiran katman (`compute_financial_metrics`,
  `summarize_backtest`) Sharpe/Sortino/BuyHold_Sharpe degerlerini
  `NaN` olarak isaretler ve metric sozlugune
  `Risk_Free_Unavailable=True` + `Sharpe_Warning="risk_free_unavailable"`
  ekler. Bu uyari Sprint 8'de `confidence.warnings` zincirine baglanir.
- `src/evaluation/evaluator.METRICS_REPORT_COLUMNS` advisory-oriented
  siralandi: `Dir_Acc`, `Hit_Rate`, `Composite_Score` ust banda;
  `Calmar`, `Deflated_Sharpe`, `Sharpe` ortada; `Net_Return`,
  `BuyHold_Return` en dipte (dipnot — cost=0 oldugundan yatirimsal
  yorumlanmamali).
- `save_metrics_report` MD raporu basina otomatik disclaimer:
  "Backtest sonuclari islem maliyeti ICERMEZ" + "Kisisel yatirim
  tavsiyesi degildir". Console summary ayni notu basar.
- `src/database/stock_model_db.compute_composite_score` formulu
  yeniden agirliklandirildi (Plan A1.4): `RMSE_vs_benchmark` 0.45->0.30,
  `Dir_Acc` 0.10->0.20, yeni `Hit_Rate` 0.15, `Sharpe` 0.20->0.10.
  `Net_Return` zaten yoktu — advisory icin maliyetsiz getiri yaniltici.
  Sharpe NaN durumunda nötr 50 puan alinir (crash etmez).
- `docs/wiki/confidence-and-risk-policy.md` Soft Degradations tablosuna
  `risk_free_unavailable -> lower one level` satiri eklendi.
- Yeni testler: `tests/test_risk_free_fail_loud.py` (7 test) +
  `tests/test_metrics_priority.py` (9 test). Toplam Sprint 0+1 yeni
  testler 25/25 gecti.

Plan referansi:
`~/.claude/plans/sistematik-ad-m-ad-m-yap-lacak-expressive-eich.md`
Sprint 1 (A1.1, A1.2, A1.3, A1.4, A1.5, A1.6).

## [2026-05-25] Decision | Single-Split Kaldirildi, WF Default + Auto Embargo

- `ValidationConfig.validation_mode` default `walk_forward` oldu
  (`src/pipeline/config.py:45`). Eski default `single_split` idi ve
  ensemble agirliklarinin test seti uzerinde optimize edilmesine yol
  aciyordu (look-ahead leakage).
- `wf_embargo_size` None veya 0 ise `_resolve_wf_embargo_size` otomatik
  `max(200, time_steps)` doner (`src/pipeline/data_manager.py`). Sebep:
  `Market_Regime_SMA200` ve diger 200-bar rolling feature'lar train/test
  arasinda tampon istemeden sizinti uretiyordu.
- CLI'dan validasyon modu prompt'u kaldirildi (`src/cli/interactive.py`).
  Batch CLI'da `--debug-quick` bayragi eklendi; bu mod single_split
  calistirir ama `research_policy="debug_quick_single_split"` +
  `research_metadata.research_only=true` ile damgalanir ve uretim
  leaderboard'una sizmaz.
- `ForecastingPipeline._run_research_single_split()` private metot eklendi;
  research_only flag'i olmayan single_split runlarini RuntimeError ile
  durdurur.
- `_add_single_split_ensembles` icinde leakage flag'i
  `ensemble_weight_scope[name] = "in_sample_test_set_research_only"`
  metadata olarak isaretlendi. Gercek train-tail validation slice fix'i
  Sprint 4 (probabilistic forecasting) ile gelecek.
- `src/cli/db_maintenance.py:25` backtest suffix mapping `latest ->
  walk_forward` olarak guncellendi.
- Yeni testler: `tests/test_walk_forward_default.py`,
  `tests/test_embargo_auto.py`.
- Wiki: `validation-and-backtesting.md` "Removed Single Split" bolumu +
  "Walk-Forward default" guncellemesi. `index.md` Current Project State
  WF default notu.

Plan referansi:
`~/.claude/plans/sistematik-ad-m-ad-m-yap-lacak-expressive-eich.md`
Sprint 0 (A0.1, A0.2, A0.3, A0.4, A0.5, A0.6, A0.7).

## [2026-05-24] Maintenance | Ignored Plan Dosyalari Kaldirildi

- Removed the ignored local `docs/gelistirmeplani1.md` and
  `docs/gelistirmeplani2.md` files from the workspace on request.
- Moved the active Plan 1/Plan 2 closure state into
  `docs/wiki/backtest-signal-improvement-plan.md` so committed documentation
  does not point to local-only plan files.

## [2026-05-24] Feature | Plan 1/2 Kapanis Komutlari

- Fixed run-leaderboard and signal-research history parsing so ISO dates such
  as `2020-02-06` are parsed year-first and ARDYZ remains `mid_history`.
- Added `signal_research ensure-data` and `signal_research run --resume` so
  Plan 1 V0-V4 research runs can restore missing CSVs, skip completed
  model/policy runs, and record policy metadata in `run_manifest.json`.
- Restored the Plan 1 full symbol set CSVs from `data/old` where needed; the
  current preflight has no `missing_data` rows for
  KCHOL, SAHOL, ENKAI, EREGL, TUPRS, SASA, ASELS, LOGO, and ARDYZ.

## [2026-05-24] Decision | Veri Gecmisi Referans Esigi

- Changed the Plan 1/Plan 2 history rule from a 10-year exclusion gate to a
  diagnostic reference threshold.
- Added `history_bucket`, `meets_10y_reference`, and `data_history_warning`
  semantics so ARDYZ-style 5-10 year symbols remain runnable as `mid_history`.
- Added history-effect summaries to run-level diagnostics for reliability,
  incomplete/invalid rate, WF-final gap, and model-family distribution.

## [2026-05-24] Feature | Model Bazli Run Sonuc Klasorleri

- Added `model_results/{model_slug}/` as an inspection-only run output layer for
  train-all and multi-model CLI runs.
- Kept canonical forecast artifacts under `models/` unchanged while exporting
  per-model metrics, predictions, fold metrics, and artifact manifests for
  easier review.

## [2026-05-24] Feature | Dinamik Hisse-Sektor Eslesmesi

- Replaced the hard-coded stock-sector mapping with universe-backed
  `Sector_Index` resolution from `data/bist_universe.csv`.
- Added BIST100 fallback handling plus `sector_mapping` metadata for feature
  engineering, feature caches, and dataset metadata.
- Widened `data/bist_universe.csv` with optional `Sector` and `Sector_Index`
  columns seeded from the previous static mappings.

## [2026-05-24] Feature | Plan 1 ve Plan 2 Kalan Isler

- Extended `run_leaderboard` with multi-symbol, all-symbol, history-class,
  sector, leader-rank, and sector-summary outputs while keeping `latest/` out
  of the analysis source path.
- Fixed date-aware prediction routing for `Prophet-ML/DL Hybrid` in final
  holdout and forward forecast flows, and added manifest-visible
  `final_holdout_status` diagnostics.
- Added `python -m src.cli.signal_research` dry-run commands for universe
  checks, symbol/model/policy run matrices, and completed-run summaries without
  using final holdout for policy selection.

## [2026-05-24] Feature | Run-Level Holdout Leaderboard ve Latest Sync Kilidi

- Added `src.analysis.run_leaderboard` and `python -m src.cli.run_leaderboard`
  to classify run outputs by final-holdout completeness, WF/final return gap,
  benchmark-clone behavior, trade sufficiency, and reliability class.
- Hardened `ForecastingPipeline._sync_latest_output()` with a temp-copy step and
  symbol-local `.latest_sync.lock` so nearby runs do not corrupt `latest/`.
- Added focused tests for leaderboard classifications, semicolon/BOM report
  parsing, and temp-based `latest/` replacement.

## [2026-05-24] Plan | ARDYZ Run-Level Holdout Dayaniklilik Plani

- Created `docs/gelistirmeplani2.md` as the follow-up plan to
  `docs/gelistirmeplani1.md`.
- Captured the ARDYZ short-history technology pilot finding: Random Forest and
  LightGBM led walk-forward results but weakened on final holdout, while DLinear
  was the most defensible but still below buy-and-hold.
- Added run-level source-of-truth, final-holdout completeness, WF-to-holdout gap,
  and single-trade/benchmark-clone rejection rules to the signal improvement
  wiki.

## [2026-05-23] Analysis | Industrial Sector Signal Findings

- Added the EREGL, ERBOS, and FROTO industrial-stock review to the signal
  improvement plan.
- Captured the three distinct industrial profiles found in the outputs:
  trend-capture for EREGL, defensive/selective-entry for ERBOS, and
  drawdown-reduction for FROTO.
- Documented that industrial model selection should score trend capture,
  negative-regime loss reduction, and confirmed trade count separately.

## [2026-05-23] Plan | Cross-Sector Signal Research Plan

- Created `docs/gelistirmeplani1.md` as the first durable development plan for
  sector-based signal improvement research.
- The plan extends the AKBNK, GARAN, and ISCTR bank review to three holding,
  three industrial, and three technology stocks with 10+ year data checks.
- Added planned signal policy variants: soft gates, percentile entries, trade
  band calibration, and sector-relative confirmation.

## [2026-05-23] Decision | Signal Gate Research Scope

- Bank-sector review of AKBNK, GARAN, and ISCTR identified recurring
  `gate_too_strict` and `model_signal_weak` diagnoses in otherwise useful
  final-holdout candidates.
- Documented that transaction-cost modelling remains out of scope for the
  current research phase; zero commission/slippage is acceptable while signal
  behavior is being studied.
- Added a sector-focused signal improvement direction: softer entry gates,
  percentile/rank-based entries, and sector-relative confirmation using only
  walk-forward calibration data.

## [2026-05-21] Feature | Özellik Mühendisliği İyileştirmeleri ve Sektörel Göreli Güç Entegrasyonu

- Durağan olmayan `BIST100_Norm` ve `USDTRY_MA7` özellikleri temizlendi.
- default `correlation_threshold` değeri `0.98`'den `0.88`'e çekilerek daha sıkı özellik seçimi sağlandı.
- `NATR_14` (Normalized Average True Range), `MFI_14` (Money Flow Index), `ADX_14` (Average Directional Index) ve `CMF_20` (Chaikin Money Flow) durağan teknik göstergeleri sisteme entegre edildi.
- Hisse senetleri için sektörel eşleşme tablosu (`_SYMBOL_SECTORS`) kuruldu ve `Sector_Relative_Strength` (Hisse_Return - Sektör_Return) özelliği eklendi (fallback olarak `BIST100_Return` kullanıldı).
- `tests/test_feature_improvements.py` üzerinden yeni özelliklerin doğruluğu test edildi.

## [2026-05-21] Feature | Derin Öğrenme Modellerinin İyileştirilmesi ve Prophet-ML/DL Hybrid Model Entegrasyonu

- `AttentionLSTM v2` ve `LSTM Lite` modellerine L2 regularizer (`l2_rate`), hücre tipi seçimi (`cell_type`: GRU/LSTM), AdamW optimizer seçeneği (`optimizer_type`) ve genişletilmiş Optuna HPO aralığı eklendi.
- Fiyat trendlerini Prophet ile çıkarıp residual'ları (artıkları) makine öğrenmesi veya derin öğrenme modelleriyle tahmin eden hibrit `Prophet-ML/DL Hybrid` modeli (`ProphetHybridModel`) hem `trend_gate` hem de `residual_decomp` modlarıyla geliştirildi ve `model_registry`'ye eklendi.
- `ProphetHybridModel` için model serileştirme/kaydetme desteği joblib ile çoklu dosya (base model + prophet model) olarak kuruldu.
- Geriye dönük model scope uyumluluk testi (`tests/test_scope_backward_compat.py`) yeni hibrit modeli kapsayacak şekilde güncellendi.
- `docs/wiki/model-catalog.md` dokümantasyonu yeni model mimarisi detaylarıyla güncellendi.

## [2026-05-21] Refactor | Kod Kalitesi ve Güvenlik Sertleştirmesi (Hardening)

- Legacy `repositories.py` (1085 satır) kaldırılarak mantıksal modüllerine bölündü (`src/database/repositories/` paketi altında `schema.py`, `experiment.py`, `best_model.py`, `forecast.py`, `forecast_resolution.py`).
- `orchestrator.py` içindeki dosya yazma, raporlama ve manifest hazırlama işlevleri `src/pipeline/artifacts.py` modülüne taşındı.
- Dış sınır giriş parametreleri için regex doğrulama (`^[A-Z0-9]{1,10}$`) analysis_service.py ve data_refresh_service.py dosyalarına eklendi.
- Arka plan thread'lerinin kaynak tüketimini sınırlamak için `ThreadPoolExecutor(max_workers=4)` yapısı kuruldu.
- requirements.txt paket bağımlılıkları mevcut ortama göre sürümleriyle sabitlendi (pinning).
- Boş hata yakalama blokları log uyarılarıyla güncellendi ve timezone-naive datetime kullanımları timezone-aware (UTC) datetime'lara geçirildi.
- `docs/wiki/code-quality-and-refactoring.md` kılavuzu oluşturuldu, `docs/wiki/index.md` güncellendi ve `RULES.md` dosyasına kod kalitesi sınırları eklendi.

## [2026-05-20] Feature | AttentionLSTM v2 Entegrasyonu ve Seq-Attention Ensemble Modelleri

- `AttentionLSTM v2` modeli deep sequence modeli olarak prediction engine'e, `PredictionService`'e ve XAI `SEQ_MODELS` listesine (`src/xai/explainer.py`) kaydedilerek tabular model muamelesi görmesi engellendi.
- Sadece sequence modellerinden (`LSTM`, `LSTM Lite`, `AttentionLSTM v2`) oluşan `Ensemble Seq-Attention Equal` ve `Ensemble Seq-Attention Inverse RMSE` ensemble modelleri eklendi.
- Yeni `Seq-Attention Inverse RMSE` ensemble modeli database repository listesine, model scope'a ve selection guard whitelist'e eklenerek eligible production ensemble leader'ı yapıldı.
- `tests/test_analysis_endpoint.py`, `tests/test_model_scope_production.py` ve `tests/test_reporting_metrics.py` üzerinde yeni testler eklenip tüm testler başarıyla koşuldu.

## [2026-05-20] Bugfix | XAI okuma ve trade metriklerinin kalıcılaştırılması

- XAI ürün özeti artık semicolon/BOM CSV dosyalarını okuyabiliyor ve en iyi
  modelin `run_id` dizinini `latest/` öncesinde kullanıyor.
- Backtest `Trade_Count`, `Signal_Diagnosis`, `Net_Return`, `BuyHold_Return`
  ve `Max_Drawdown` alanları SQLite `experiments`/`best_models` kayıtlarına
  kalıcı metrik olarak taşındı.
- Mevcut run çıktıları için `db_maintenance backfill-run-metrics` bakım komutu
  eklendi; ASELS üzerinde 20 rapordan 50 experiment satırı güncellendi.

## [2026-05-20] Bugfix | Best-model seçiminde son kayıt ezmesini engelle

- ASELS testinde `NLinear` son yazılan final-holdout adayı olduğu için daha
  yüksek skorlu `LSTM` kaydını `best_models` içinde ezdiği görüldü.
- `BestModelRepository` artık mevcut best ile yeni adayı eligibility ve
  `composite_score` üzerinden karşılaştırıyor; aynı eligibility sınıfında yüksek
  skor kazanıyor, eligible kayıt ineligible kayıtla ezilmiyor.
- Schema refresh akışı da latest experiment yerine en yüksek skorlu production
  experiment satırından `best_models` kaydını yeniden kuracak şekilde düzeltildi.

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
