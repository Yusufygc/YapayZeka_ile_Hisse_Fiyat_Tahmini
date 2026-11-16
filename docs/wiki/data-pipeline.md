---
title: Data Pipeline
type: concept
status: active
last_updated: 2026-11-16
owner: llm
source_count: 11
---

# Data Pipeline

## Güncel durum (2026-11-16, Kol-A emekli, makro kaldırıldı)

Ürün yalnız **havuzlu (pooled) panel** üretir; aşağıdaki "Tarihsel" bölümlerin çoğu (tensörler,
bölme/ölçekleme, recursive forecast, makro projeksiyon, momentum, `DataManager`, `src.cli.*`) Kol-A'ya
aittir ve artık kodda YOKTUR.

- **Ham veri:** `data/*.csv` (+ `data/yedek` yedek klasör, `--fallback-dir`). Okuma/temizleme tek yerde
  (`src/data/ohlcv_clean.py`: alan sayısı doğrulama, `dd/mm/yyyy` değer-bazlı tarih, tekrar tarih,
  Adj/Close ile O/H/L ölçekleme, sıfır-hacim atma); Parquet temiz depo `src/data/clean_store.py`.
- **Panel:** `src/data/pooled_loader.py` (`PooledPanelLoader`; sembol-paralel `n_jobs`, `keep_unlabeled`
  ile son h satır skorlama için kalır). Özellikler `FeaturePipeline` ile sembol başına: Return,
  SMA_{7,21,50}_rel, RollStd_{14,21}_norm, NATR_14, RSI_14, MACD_Diff_norm, LogRet_Lag_1 (takvim kapalı);
  çapraz-kesit özellikler/hedef `src/data/cross_sectional.py`. **Makro katman (MacroPipeline, USDTRY/
  BIST100, momentum_60d, Relative_Strength, risk-free rate) ve `macro_df` yolu 2026-11'de silindi**;
  havuz paneli kaldırma öncesi/sonrası 4 gerçek sembolde bit-bit aynı doğrulandı. Özellik önbelleği
  (`pooled_feature_cache`) anahtarı kod parmak izine bağlı olduğundan kaldırma önbelleği kendiliğinden geçersiz kıldı.
- **K8 ölçek-kırığı karantinası** (`quarantine_jumps=True`): `|1g log-getiri| > 0.30` (bölünme / düzeltilmemiş
  seri) etrafında satırlar atılır: hedefi sıçramayı içeren `[p-h, p-1]`, sıçrama günü ve sonraki 63 satır.
  Risk modeli bu panelden kurulduğu için karantinalı sembolde `rv_63` (>= %80 geçerli gün) yeniden dolana dek
  (~63+51 işlem günü) risk tahmini NaN olur (ör. ASELS 2026-06-09 bölünmesi); API uyarısı nedeni söyler,
  bkz. [Risk Modeli](risk-model.md).
- **Tazelik:** `src/data/freshness.py` (sembol bazlı), API tarafı `analysis_freshness.py`
  (CSV son tarihi; havuz `as_of_date` CSV'den > 1 işlem günü geriyse `stale_data`).
- **Takvim:** `data/meta/bist_calendar.csv` (2027-06'ya kadar); bakım yardımcısı `src/forecasting/bist_calendar.py`.
- **Silinen modüller:** `data_loader`, `quality` (eski PSI), `universe_sync`, `database/paths`, `feature_cache`
  (Kol-A), `macro_*`, `interval_calibration`, `utils/risk_free_rate`.

> **Tarihsel (Kol-A dönemi):** bu satırdan sonrası eski mimariyi anlatır ve yalnız referans içindir.
> Kod > test > wiki sırası gereği güncel davranış için yukarıdaki bölüme bakın.

The data layer turns BIST OHLCV CSV files into model-ready tabular and sequence
tensors while enforcing chronological order and leakage controls.

## Raw Inputs

- Stock CSV files live under `data/`.
- `python -m src.cli.interactive` filters CSVs so metadata files such as `bist_universe.csv`
  and `bist_calendar.csv` are not shown as stocks.
- `DataUpdater` can update stock files when `DataConfig.auto_update_data` is enabled.
- Macro data is optional and controlled by `DataConfig.use_macro`. 2026-11-16
  itibarıyla yalnız USDTRY ve BIST100 model özelliği üretir; ayrıntı için
  [Kürate özellik seti](#kürate-özellik-seti-2026-11-16). Makro veri nightly işte
  yenilenmez (bilinen açık).

`DataUpdater.check_and_update()` now returns a structured `DataUpdateResult`
instead of only printing side effects. The result records status, before/after
latest dates, rows added, and error text. It also accepts a `MarketDataProvider`
interface, with `YFinanceProvider` as the default, so refresh behavior is
testable without direct network calls.

### Date Parsing Bug & Fix (2026-06-08)
When target BIST stock CSV files are in YYYY-MM-DD format (e.g., `2026-06-02`), passing `dayfirst=True` to `pd.to_datetime` causes pandas to parse month and day incorrectly if both are $\le 12$ (e.g., `2026-05-12` becomes `2026-12-05`). This resulted in incorrect future dates, negative `diff_days` calculation, and silent skipping of data updates during pipeline execution. The fix introduces format-detection logic in `_parse_last_date`: if the date string starts with a 4-digit year, `dayfirst` is set to `False`, allowing correct date parsing and data updates.


### DataUpdater 2026-11 Hardening

- **Bayatlik:** `diff_days <= 1` kaldirildi. CSV, yalniz son *tamamlanmis* BIST seansina
  (`src/data/session_clock.py::latest_completed_session`, 18:10 Europe/Istanbul sonrasi;
  takvim `data/meta/bist_calendar.csv`, yoksa/kapsam disiysa hafta ici) esit veya ilerideyse
  `up_to_date` sayilir. `check_and_update(..., now=...)` ile zaman enjekte edilebilir.
- **Kismi mum:** `_prepare_provider_rows` son tamamlanmis seanstan sonraki satirlari atar
  (append ve full refetch).
- **Toplu indirme:** `DataUpdater.update_many([(csv, sym), ...])` -> `yf.download(tickers=[...],
  group_by="ticker", threads=True, auto_adjust=AUTO_ADJUST)` 50'lik chunk'lar, bos/basarisiz
  ticker icin ustel backoff'lu retry (`src/data/market_providers.py`). Sembol basina dogrulama/append
  tekli yolla ayni kod.
- **Kilit/atomiklik:** `src/data/csv_safety.py` - `filelock` ile `<csv>.lock`; append ve full
  refetch gecici dosya + `os.replace`. Kilit alininca CSV yeniden okunur.
- **Yedek:** full refetch oncesi `data/yedek/<SYM>.csv.bak_YYYYMMDD`, sembol basina en fazla 3.
- **Tarih formati (2026-11-16):** `src/data/updater_frames.py` tarihleri `src/data/dates.py`
  ayrıştırıcısının değer-bazlı (per-value) versiyonuyla okur; tüm yazıcılar (`DataUpdater`,
  `tools/repair_symbol_csv.py`) ISO (`YYYY-MM-DD`) yazar. Diskte hâlâ `dd/mm/yyyy` olan
  `data/ASELS.csv` ve `data/EREGL.csv` doğru ayrıştırılır; dosyalar henüz yeniden yazılmadı.

**Universe-wide nightly refresh (E2 Faz 8):** there is no single
`data_updater` CLI; the E2 nightly orchestrator `tools/e2_nightly_pipeline.py`
(`refresh_universe`) loops every `data/*.csv` (excluding `bist_universe` /
`advisory_history`) and calls `DataUpdater.check_and_update(path, sym,
interactive=False)`, aggregating updated/up_to_date/skipped/failed counts and
continuing on per-symbol errors. A BIST trading-day gate
(`is_trading_day`, XIST via `pandas-market-calendars`, weekday fallback) skips
weekends/holidays. See [Persistence and API](persistence-and-api.md) and
[E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md).

## Loading and Cleaning

`src/data/data_loader.py` handles raw OHLCV normalization.

Key behaviors:

- Turkish column names are mapped to English (`Date`, `Open`, `High`, `Low`, `Close`, `Adj_Close`, `Volume`).
- Dates are parsed with mixed-format tolerance and sorted chronologically.
- Zero-volume rows can be dropped to remove non-trading days.
- `Adj_Close` is used only when it passes anomaly checks.
- If adjusted close is missing or anomalous, raw `Close` remains the price source and the decision is recorded in dataframe attributes.

## Corporate Action Handling

`load_and_clean()` stores a `corporate_action_report` in `df.attrs`.

Important fields include:

- `adj_close_available`
- `price_source`
- `adjusted_price_trusted`
- `corporate_action_anomaly`
- `max_abs_adj_close_diff_pct`
- `warning`

This metadata is later folded into run metadata by `DataManager`.

### Corporate Action Audit (Sprint 2 — 2026-05-25)

`tools/audit_corporate_actions.py` scans every CSV under `data/` and flags
days where `|log_return| >= 0.30`. Such jumps usually mean an unadjusted
split, a large dividend ex-day, or a data error. Output:

- `outputs/_audits/corporate_action_audit_{timestamp}.csv`
- `outputs/_audits/corporate_action_audit_latest.csv`

Columns: `Symbol, Date, prev_close, close, log_return, abs_log_return,
auto_adjust_active, severity (high/extreme), notes`.

Usage:

```bash
python tools/audit_corporate_actions.py
python tools/audit_corporate_actions.py --universe data/bist_universe.csv
python tools/audit_corporate_actions.py --threshold 0.20 --out outputs/_audits
```

`src/data/quality.py` reads the latest audit and sets
`corporate_action_anomaly=True` whenever the current symbol has an
anomaly within the last `252` business days. This flag is a hard block
in the analysis API confidence chain
([Confidence and Risk Policy](confidence-and-risk-policy.md)).

### yfinance Auto-Adjust Hard Control (Sprint 2 A2.1)

`YFinanceProvider.AUTO_ADJUST = True` is required. yfinance applies the
split/dividend correction at fetch time; without it, a `1:2` split day
produces `log_return ≈ -0.69` and corrupts every trained model.
Production code must not lower this flag.

### Survivorship Bias Report (Sprint 2 A2.3)

`load_and_clean()` also writes a `survivorship_bias_report` to
`df.attrs`. Shape:

```json
{
  "symbol": "TUPRS",
  "actual_start": "2015-05-01",
  "actual_end": "2026-05-23",
  "span_days": 4036,
  "row_count": 2715,
  "max_gap_days": 4,
  "short_history_warning": false,
  "delisted_or_late_listing_warning": false,
  "warning": ""
}
```

Triggers:

- `short_history_warning`: total span < 2 years.
- `delisted_or_late_listing_warning`: a single gap > 10 calendar days
  between consecutive observations (suggests delisting, late listing,
  or data hole).

`compute_quality_flags()` exposes the report under
`survivorship_bias_report` so downstream confidence/warning chains can
relay the rationale to the user.

### Population Stability Index (PSI)

`src/data/quality.compute_psi(train_df, holdout_df)` returns a per-column
PSI dictionary. `_psi_one_feature` builds a shared 10-bin histogram from
the combined sample, then computes:

```
PSI = Σ (f_holdout - f_train) * log(f_holdout / f_train)
```

`compute_quality_flags()` sets `psi_high=True` whenever
`psi_max > 0.25`. This is a hard block. Thresholds (Sprint 7 will add
`psi_status` enum to the analysis API):

| `psi_max` | Status | Action |
|---|---|---|
| < 0.10 | `stable` | no effect |
| 0.10 - 0.25 | `moderate_drift` | soft degradation (lower confidence) |
| ≥ 0.25 | `major_drift` | hard block → `low` confidence |

## Recursive Forecast Feature Recompute (Sprint 5 — 2026-05-25)

`FeaturePipeline.recompute_close_dependent(frame)` rewrites every
close-derived technical indicator after a recursive future-forecast row is
appended. Without this rewrite, SMA/EMA/RSI/MACD/Bollinger/ATR/ADX/MFI/OBV
values from the original last row would be carried forward unchanged and
the model would predict against stale technical context — multi-day
horizons (5+ days) degrade fast.

Order of operations during `ForecastPointGenerator.roll_forward_recursive`:

1. Predict next-day target.
2. Bound predicted close with BIST band rules.
3. Append recursive row (`_append_recursive_row`) — Close and Return shifted
   in; `LogRet_Lag_1` = önceki adımın getirisi (2026-11-16 düzeltmesi: önceden
   yeni satırın kendi tahmini geri besleniyordu).
4. **`recompute_close_dependent(frame)`** — every close-derived feature
   regenerated through the same `_add_*` chain used in training; momentum
   özellikleri (`momentum_60d`, `market_momentum_60d`, `relative_to_market_60d`)
   de her recursive adımda yeniden hesaplanır (2026-11-16). Recursive `feature_frame`
   `Open/High/Low/Volume` kolonlarını da taşır (NATR_14 High/Low gerektirir; aksi halde
   h>=2'de teknik göstergeler son gerçek barda donuyordu) ve `day_of_week` yeni satırın
   tarihinden türetilir.
5. **`MacroForwardProjector.project_last_row`** — macro features projected
   (see below).

Macro and lag features are not touched by step 4. Lag handled in step 3,
macro in step 5. Sektör endeksi özellikleri kaldırıldığı için sektör adımı yoktur.

## Macro Forward Projection (Sprint 5 — 2026-05-25)

`src/features/macro_forward_projection.py:MacroForwardProjector` advances
known macro columns one step. P1 denetim revizyonu (2026-11) ile ARIMA kaldırıldı
(martingale baseline, tahmin iddiası yok): `*_Return` / `*_Change` kolonları
0.0, diğer seviye kolonları son gözlenen değer (random walk / ffill). ARIMA ve
statsmodels artık kullanılmaz.

Macro whitelist (`_KNOWN_MACRO_COLUMNS`, 2026-11-16 itibarıyla):

- `USDTRY`, `USDTRY_Return`, `USDTRY_Volatility7`
- `BIST100`, `BIST100_Return`, `BIST100_MA7`

Makro whitelist, 2026-11-16 özellik setinin daraltılmasıyla yalnız bu iki
sembolü kapsar (CPI, faiz, VIX, altın, petrol, EURTRY, sektör endeksleri artık
özellik üretmez; bkz. [Kürate özellik seti](#kürate-özellik-seti-2026-11-16)).

Projection is applied only to the last (recursive) row. Historical macro
rows are not modified — backtest and training data stay deterministic.

The forecast warnings enum updated from `frozen_exogenous_features` to
`projected_exogenous_features`, signaling macro inputs are advanced
deterministically rather than held constant. See
[Analysis API Contract](analysis-api-contract.md) `forecast_source.warnings`.

Projeksiyon adımı istisna fırlatmaz; forecast döngüsü bu adımda kesilmez.

## Calendar Features (Sprint 7 — 2026-05-25)

`FeaturePipeline._add_calendar_features` derives stationary calendar signals
from the `Date` column. Always recomputed (Date-only) and safe to reapply on
recursive forecast rows.

| Column | Definition |
|---|---|
| `day_of_week` | 0=Monday … 4=Friday |
| `day_of_month` | 1..31 |
| `days_to_month_end` | days remaining in current month |
| `days_to_quarter_end` | days remaining in current quarter |
| `is_quarter_end_week` | 1 if `days_to_quarter_end <= 5` else 0 |
| `days_to_next_fomc` | days until next FOMC date, `365` if not found |

**2026-11-16:** Model özellik setinde yalnız `day_of_week` kalır. Diğer takvim
kolonları (`day_of_month`, `days_to_*`, `is_quarter_end_week`) ve FOMC özelliği
(`days_to_next_fomc`) IC denetiminde gürültü çıktığı için kaldırıldı.

FOMC dates were read from `data/meta/fomc_calendar.csv` (historical; özellik
artık üretilmez). Toggle via `FeaturePipeline(enable_calendar_features=False)`.

## Cross-Sectional Momentum (Sprint 7 — 2026-05-25)

`FeaturePipeline._add_cross_sectional_momentum` runs inside `_merge_macro`
when sector/market returns are available. Compares the symbol's 60-day
momentum against sector index and BIST100.

| Column | Definition |
|---|---|
| `momentum_60d` | `Close.pct_change(60)` |
| `market_momentum_60d` | cumprod-equivalent of `BIST100_Return` over 60d; makro takviminde hesaplanıp tarihe göre birleştirilir (aynı tarihte tüm hisselerde aynı; 2026-11-16, feature cache v6) |
| `sector_momentum_60d` | cumprod-equivalent of sector index return (falls back to market when sector missing) |
| `relative_momentum_60d` | `momentum_60d - sector_momentum_60d` |
| `relative_to_market_60d` | `momentum_60d - market_momentum_60d` |

**2026-11-16:** Model setinde yalnız `momentum_60d`, `market_momentum_60d` ve
`relative_to_market_60d` kalır. `sector_momentum_60d` ve `relative_momentum_60d`
sektör endeksine bağlı olduğu için kaldırıldı (sektör endeksleri artık kullanılmıyor).

Missing macro return columns make the related rows silently absent — no
exception is raised. Toggle via
`FeaturePipeline(enable_cross_sectional_momentum=False)`.

## Feature Engineering

Feature generation is split across:

- `src/data/data_loader.py`: base stationary technical features and lagged log returns.
- `src/features/feature_pipeline.py`: expanded technical feature set.
- `src/features/macro_pipeline.py`: macro context features.
- `src/features/feature_cache.py`: cache by data/config key.
- `src/features/sector_mapping.py`: dynamic stock-to-sector-index mapping from `data/bist_universe.csv`.

### Feature Caching and Self-Healing Cache Eviction
Feature engineering caching uses `FeatureCache` (a disk-based cache under `data/feature_cache`) to avoid redundant preprocessing. 
To prevent transient data-gathering failures (such as EVDS API errors) from poisoning the cache with macro-less (degraded) datasets when `use_macro=True` is requested, `DataIngestionService._engineer_features_cached` implements a self-healing eviction check. If macro features are expected but the cached feature names lack the central policy interest rate feature (`Rate_Level`), the cache entry is considered degraded, gets automatically evicted via `_cache._evict()`, and the features are regenerated. (2026-11-16: politika faizi özelliği `Rate_Level`
artık üretilmediği için bu kontrol eski önbellek kayıtlarının tarihsel temizliği olarak
kalır; güncel model özellik kümesi için bkz. Kürate özellik seti.)

### Kol-B sembol başı Parquet özellik önbelleği (2026-11-15)

`src/features/pooled_feature_cache.py::PooledFeatureCache`, `PooledPanelLoader._engineer_features` içinde
`FeaturePipeline.engineer_features` (macro_df=None, takvim kapalı) sonucunu sembol başına saklar.
`PooledLoaderConfig.feature_cache_dir` (varsayılan `None` = kapalı). Açık olduğu yerler:
`src/serving/nightly.py` (varsayılan KAPALI: veri her gece değiştiği için hep ıska; açmak için `--feature-cache`, dizin `--feature-cache-dir`)
ve `tools/e2_kolb_validity_report.py` (aynı bayraklar). Çapraz-kesit özellikler (`pooled_dataset.py`) önbelleksizdir.

- **Depolama:** `{dir}/{SEMBOL}.parquet` (zstd), sembol başına TEK girdi (üzerine yazılır; dizin sembol sayısıyla sınırlı, TTL yok).
  Anahtar Parquet şema metadata'sında (`tsfl.feature_cache_key`). Atomik yazma (geçici dosya + `os.replace`).
  Bozuk dosya -> uyarı + miss + yeniden yazım. Sembol adı regex ile doğrulanır (path traversal). `.gitignore`: `data/feature_cache/`.
- **Anahtar (sha256):** `_SCHEMA_VERSION` (biçim değişince elle) + kod parmak izi + temizlenmiş girdi çerçevesinin tam içeriği
  (kolon adı, dtype, değer baytları; index hariç) + `feature_mode` + takvim bayrağı (+ takvim açıksa FOMC csv özeti) + sembol.
- **Kod parmak izi** (`src/features/code_fingerprint.py`): kökten AST ile `src.*` importları transitif izlenir; Kol-B için dosyalar:
  `src/features/feature_pipeline.py`, `correlation_pruning.py`, `sector_mapping.py`, `src/xai/feature_dictionary.py`.
  Kaynak baytları (CRLF->LF) + pandas/numpy/ta sürümleri sha256'lanır, süreç başına bir kez hesaplanır.
  Özellik koduna dokunmak önbelleği otomatik geçersiz kılar (elle sürüm artırma gerekmez). Kol-A `FeatureCache` anahtarı da
  aynı yardımcıyı kullanır (kökler: feature_pipeline + macro_pipeline); `_VERSION` yine duruyor ama tek geçersiz kılma yolu değil.
- **Raporlama:** yükleme sonunda tek log satırı; `PooledDataset.loader_meta["feature_cache"] = {enabled, hits, misses, errors}`.
- **Ölçüm** (585 CSV, 581 sembol, bu makine): önbelleksiz 53,7 s; soğuk 65,1 s (581 miss, yazma maliyeti); sıcak 34,5 s (581 hit).
  Sıcakta özellik adımı 57,5 s -> 2,6 s; kalan süre CSV temizleme (~14 s) ve çapraz-kesit adımlar. `full/aug/labeled/feats`
  önbelleksiz yapımla `assert_frame_equal(check_exact=True)` ile BİREBİR eşit. Disk: 581 dosya, ~356 MB.
- **Artımlı (son K satır) hesap bilinçli olarak YOK:** sonlu pencere en fazla 200 (SMA_200), ama EMA/MACD/RSI/ATR/ADX
  özyinelemeli (sonsuz bellek) ve OBV kümülatif. THYAO'da son 400/800/1500 satırla yeniden hesap, tam hesaptan 3.5e-9 / 9.5e-12 / 2.3e-12
  farklı (bit-bit eşit değil) -> yalnız durum (state) saklanarak kesin olur. Bir gün eklenince girdi hash'i değiştiği için
  her sembol yeniden hesaplanır; kazanç tekrarlı koşularda (kapı yeniden denemesi, skorlama/araştırma tekrarı) olur.
- Testler: `tests/test_pooled_feature_cache.py` (isabet, değer/parmak izi değişimi, bozuk dosya, dtype, yükleyici entegrasyonu).

## Kürate özellik seti (2026-11-16)

Varsayılan `feature_mode = "stationary_features"` (`src/features/feature_pipeline.py`)
model için 19 özellik üretir:

`Return`, `SMA_7_rel`, `SMA_21_rel`, `SMA_50_rel`, `RollStd_14_norm`,
`RollStd_21_norm`, `NATR_14`, `RSI_14`, `MACD_Diff_norm`, `LogRet_Lag_1`,
`day_of_week`, `USDTRY_Return`, `USDTRY_Volatility7`, `BIST100_Return`,
`BIST100_MA7`, `Relative_Strength`, `momentum_60d`, `market_momentum_60d`,
`relative_to_market_60d`.

**Neden:** 250 hisse üzerinde IC denetimi (2021-07 .. 2026-06) önceki 63 özelliğin
çoğunun gereksiz olduğunu gösterdi:

- Tam kopyalar: `Return` = `Log_Return`, `Relative_Strength` = `Sector_Relative_Strength` vb.
- Yakın kopyalar (rank korelasyonu > 0.9): SMA/EMA/RSI/VWAP/MACD kümesi; `BB_Width` ≈ `RollStd`.
- Gürültü: ADX, OBV, CMF, MFI, lag 2-5, FOMC/takvim ekleri.
- Sağlam çapraz-kesit sinyal yalnız volatilite özelliklerinde: `NATR_14` IC -0.074
  (t -9.7), `RollStd` / `BB_Width` (düşük volatilite etkisi).

**Makro:** Yalnız USDTRY ve BIST100 kalır. CPI, politika faizi, VIX, DXY, US10Y, altın,
petrol ve EURTRY artık özellik üretmez; indirme kodları silindi. `INTEREST_RATE.csv`
hâlâ yenilenir çünkü risksiz faiz kaynağıdır (`src/utils/risk_free_rate.py`,
`MacroPipeline.refresh_interest_rate_cache`). Sonuç: eğitim geçmişi artık 2021-07'de
kesilmez (CPI başlangıcı + 12 aylık YoY kısıtı kalktı); THYAO 2015-03-27'den başlar.

**Sektör endeksi:** Kaldırıldı. Sektör endeks CSV'leri 2-8 satırlıktı ve her hisse
sessizce BIST100'e düşüyordu. `sector_mapping_report.status = "disabled"`. Kol-B
sektör ETİKETİ (label) kullanılmaya devam eder.

**Diğer kurallar:**

- `Log_Return` üretilmez; `LogRet_Lag_1` doğrudan Close'dan hesaplanır.
  `lag_feature_count` varsayılanı 1.
- `Market_Regime_SMA200` hesaplanmaya devam eder (sinyal/backtest kapıları okur) ama
  model özelliği değildir; `PIPELINE_AUX_COLS` ile (`src/data/panel_columns.py`) özellik
  listesinden hariç tutulur.
- Korelasyon budaması kapalı kalır. `tests/test_curated_feature_set.py` tam kümeyi ve
  hiçbir çiftin |Spearman| > 0.95 olmadığını doğrular.
- Feature cache sürümü 5'e çıkarıldı (`src/features/feature_cache.py::_VERSION`).
- Eski özellik listeli artefaktlar `_validate_artifact_features`
  (`src/forecasting/workflows.py`) ile açık bir "yeniden eğit" hatası verir.
- Bağlantılı: ARIMA/statsmodels, Prophet ve `pandas-datareader` bağımlılıkları da
  kaldırıldı (bkz. [Architecture](architecture.md)).

Eski ifadeler (tarihsel): önceki sürüm ayrıca `Sector_Relative_Strength` (sektör
endeksi, BIST100 fallback'li), korelasyon eşiği `0.98 -> 0.88` ve `MFI_14`, `ADX_14`,
`CMF_20` gibi özellikleri kullanıyordu; bunların hiçbiri artık model girdisi değildir.
Stasyonerlik ilkesi (oran/normalize özellikler, Close'a göre) korunur.

Makro gecikme kontrolleri (faiz/CPI) kaldırılan kolonlar için artık geçerli değildir;
`INTEREST_RATE` yalnız risksiz faiz kaynağı olarak kalır.

`MacroPipeline.get_macro_features()` now acts as an orchestration layer around
separate cache refresh, cache loading, date filtering, monthly release-lag,
daily/global merge, monthly ffill, and final feature-engineering helpers. Public
inputs and output schema are unchanged; the split exists to keep EVDS/FRED/manual
CSV fallback behavior and release-lag leakage controls independently testable.

### Parquet temiz veri deposu (2026-11-15)

`src/data/clean_store.py::CleanStore` (varsayılan kök `data/clean/`, `.gitignore`'da) **CSV'lerden türetilen** temizlenmiş OHLCV deposudur.
**Kaynak doğruluk CSV'de kalır** (DataUpdater, GUI, araçlar CSV yazıp okur; Kol-A `data_loader.py` değişmedi); depo silinirse kendiliğinden yeniden kurulur.
Amaç: panel kurulumunda 585 CSV'yi her koşuda okuyup `clean_ohlcv_csv` ile temizlemenin (~8-14 s) tekrarını önlemek.

- **Düzen:** `panel.parquet` (uzun: `symbol, Date, Open, High, Low, Close, Volume`; zstd; sembol sıralı; `drop_zero_volume=True` hali = panel yolunun kullandığı) +
  `manifest.parquet` (sembol başına `source_path, source_sha256, source_size, source_mtime_ns, n_rows, first_date, last_date, rejected_rows, cleaned_at, schema`)
  + `panel.parquet.lock` (`csv_safety.csv_lock`). Standart dışı kolonlar (Dividends vb.) depolanmaz (panel yolu zaten yalnız standart OHLCV'yi kullanır).
- **`refresh(paths)`:** hızlı yol boyut + `mtime_ns`; uyuşmazsa sha256; sha aynıysa (yalnız mtime değişti) yeniden temizleme YOK, stat manifeste yazılır;
  yalnız sha'sı değişen/yeni semboller yeniden temizlenir, kalkanlar düşürülür. Panel ve manifest geçici dosya + `os.replace` ile yazılır
  (manifest son = commit noktası; ikisini `tsfl.panel_id` metadata'sı bağlar). Temizlenemeyen CSV depoya alınmaz (`RefreshReport.failed`); yükleyici o sembol için doğrudan CSV'ye düşüp hatayı kendisi raporlar.
- **`load(symbols)`:** sembol çerçeveleri `clean_ohlcv_csv(path)` çıktısının standart kolon izdüşümüyle `assert_frame_equal(check_exact=True)` aynıdır
  (RangeIndex, dtype manifest `schema`'sından, `attrs["rejected_rows"]` manifesttan). Panel float64 saklar; tam sayı dtype'lı kolonlar (>2^53 değilse) geri cast edilir.
- **`content_hash(symbols=None)`:** sha256( bicim sürümü + temizleyici kod parmak izi + sıralı (sembol, `source_sha256`) ). Veri veya temizleyici kod değişince değişir; mtime tek başına değiştirmez.
- **Otomatik tam yeniden kurulum (uyarılı):** temizleyici kod parmak izi (`ohlcv_clean.py` + `dates.py` kapanışı + pandas/numpy sürümü, `code_fingerprint`) veya `SCHEMA_VERSION` değişirse; panel/manifest bozuk, `panel_id` uyuşmuyor veya satır sayısı tutmuyorsa.
- **Yükleyici:** `PooledLoaderConfig.clean_store_dir` (varsayılan `None` = eski per-CSV yol; testler etkilenmez). Açıkken `load()` TÜM kaynak CSV'leri (`_symbol_paths()`, `fallback_dirs` önceliği aynen) bir kez `refresh` eder
  (`limit` depoyu daraltmaz), sonra `_engineer_symbol` temiz çerçeveyi depodan alır; sonrası değişmedi. Açık: `src/serving/nightly.py` (`--clean-store-dir`, `--no-clean-store`) ve
  `tools/e2_kolb_validity_report.py` (aynı bayraklar). Bir depo = bir kaynak-klasör yapılandırması (farklı `data_dir`/`fallback_dirs` ile aynı depoyu paylaşmayın: kalkan semboller düşürülür).
  `PooledDataset.loader_meta["clean_store"] = {enabled, content_hash, refresh, failed}`.
- **Tekrarlanabilirlik:** nightly `data_snapshot_hash` eskiden `len(aug)|as_of|ilk 5 sembol` idi (veri hash'i değildi); artık depo açıkken `content_hash()[:16]`, kapalıyken panelin `(symbol, Date, Close, target)` içerik hash'i (`panel_content_hash`). Tam hash + kaynağı PeerStore run config'inde (`data_hash`, `data_hash_source`, `clean_store`).
  Kol-A `dataset_hash` yalnız metadata JSON'unu hashliyordu; artık metadata'ya `data_content_sha256` (özellik çerçevesinin `frame_content_hash`'i) eklendi (şema/kolon değişmedi), aynı konfigürasyonla değişen veri hash'i değiştirir.
- **Kol-A `data_loader.py` depodan OKUMAZ:** kendi Adj_Close anomali kontrolü/`attrs` raporu panel temizleyicisinden farklıdır; birebir eşitlik kolay değil.
- **Ölçüm** (585 CSV, 581 sembol, özellik önbelleği kapalı): `build_pooled_dataset` depo kapalı 48,4 s; depo soğuk 48,5 s (refresh 7,6 s: 585 sha + temizleme + yazma); depo sıcak 41,0 s (refresh 0,02 s, `load` ~0,4 s) -> ~7,4 s kazanç (CSV temizleme maliyeti bu makinede ~7,6 s). 1 CSV değişince refresh 0,41 s. `full/aug/labeled/feats/cat_idx/loader_report` depo kapalıyla `check_exact` BİREBİR eşit (soğuk ve sıcak). Disk: `panel.parquet` ~29 MB.
  Not: gece işinde DataUpdater her CSV'ye yeni gün ekler -> her sembol `changed` olur; depo asıl kazancı aynı veride tekrar koşularda (doğrulama raporu, skorlama tekrarı, araştırma) sağlar.
- Testler: `tests/test_clean_store.py`, `tests/test_data_services.py::test_dataset_hash_reflects_data_content_not_only_metadata`.

### Pooled panel sembol-paralel hesap (2026-11-15)

`PooledLoaderConfig.n_jobs` (varsayılan `1` = eski sekansiyel yol aynen; modül düzeyinde `FeaturePipeline`'ı patch'leyen testler bunu kullanır) ve
`parallel_chunk_size` (0 = otomatik). `n_jobs=-1` -> `min(os.cpu_count()-1, 8)`; `>1` -> o kadar işçi; `0`/`<-1` -> `ValueError`.
Açık: `src/serving/nightly.py` (`--n-jobs`, varsayılan -1) ve `tools/e2_kolb_validity_report.py` (`--n-jobs`, varsayılan -1).

- **Tasarım (`src/data/pooled_parallel.py`):** işçinin yaptığı iş `PooledPanelLoader._engineer_symbol_safe` ile AYNI koddur (sekansiyel döngü de bunu çağırır; ayrı "işçi kopyası" yok).
  `load()` sırasıyla: depo `refresh` (ana süreçte) -> sembol listesi -> delisted-hariç olanlar ana süreçte `delisted_excluded` -> kalanlar `SymbolTask`
  (sembol, sektör, delisting tarihi, CSV yolu, depodan gelen temiz çerçeve) olarak `chunk`'lara bölünür -> `joblib.Parallel(backend="loky")` -> `ChunkResult`
  (sembol sırasında; sembol başına `SymbolResult`: çerçeve, `report` durumu, `rejected_rows`, `quarantined_days`; parça başına önbellek `hits/misses/errors` + işçi WARNING logları).
  Ana süreç sonuçları SEMBOL SIRASINDA tek döngüde birleştirir (report/rejected/quarantine sözlüklerinin ekleme sırası dahil sekansiyelle aynı), önbellek sayaçlarını toplar,
  işçi uyarılarını (`_CaptureHandler` ile toplanır) ana süreç logger'larından yeniden yayınlar (`replay_logs`). Sembol başı hata `error:...` aynen (`_engineer_symbol_safe`).
- **Backend: joblib/loky (süreç).** İş parçacığı GIL'e takılır (pandas/ta). `ProcessPoolExecutor` ile ölçüm eşdeğer (load 7,8-8,0 s vs 6,8-8,1 s; tek seferlik koşuda loky yeniden kullanımı kazandırmaz);
  loky seçildi: cloudpickle, BLAS/OMP iş parçacığı sınırlaması işçide otomatik, `max_nbytes=None` (DataFrame'ler memmap'e yazılmaz), işçi çökmesinde net hata.
- **Çerçeve geçirme:** depo açıkken ana süreç zaten 581 temiz çerçeveyi yükler (~0,4 s, toplam ~29 MB) -> parça başına yalnız o parçanın çerçeveleri pickle'lanır (işçiler Parquet'i yeniden okumaz, tüm panel her işçiye gitmez).
  Depo kapalıyken işçi yolu alır ve CSV'yi KENDİ temizler (temizleme de paralelleşir: load 34,9 s -> 8,6 s). `refresh` her zaman ana süreçte.
- **Parça boyutu:** 0 = `clamp(ceil(N / (işçi*4)), 4, 20)` (işçi başı ~4 görev, yük dengesi). Ölçüm: parça 5/20/40, 8 işçi -> 21,1 / 20,3 / 20,0 s (fark gürültü düzeyi); 581 sembolde otomatik ~18-20.
- **İşçi import maliyeti:** `import src.data.pooled_parallel + pooled_loader + pooled_feature_cache` ~0,65 s (723 modül; `ta`, `pyarrow` var; tensorflow/torch/sklearn/lightgbm/scipy/matplotlib YOK; `engineer_features` çalıştıktan sonra da yok).
  `pooled_parallel` üst düzeyinde yalnız stdlib+pandas import eder, `pooled_loader` işçi fonksiyonunda tembel yüklenir (döngüsel import yok). Soğuk havuz başlatma ~0,7-1 s (ilk `n_jobs=8` yükleme 7,5 s vs sıcak 6,8 s).
- **Özellik önbelleği işçide:** işçi parça başına `PooledFeatureCache(dir, fingerprint=ana süreçten)` kurar (parmak izi tekrar hesaplanmaz); her sembol kendi dosyasını okur/yazar (atomik yazma zaten var); sayaçlar parça başına delta döner, ana süreç toplar -> `feature_cache_stats` sekansiyelle aynı.
- **Ölçüm (581 sembol, 20 mantıksal CPU, nightly varsayılan yükleyici: min_rows=300, karantina açık, depo sıcak, önbellek kapalı):**

  | n_jobs | `load()` | `build_pooled_dataset` toplam |
  |---|---|---|
  | 1 (sekansiyel) | 29,2 s | 42,8 s |
  | 4 | 9,8 s | 23,6 s |
  | 6 | 7,9 s | - |
  | 8 (= `-1` otomatik) | 6,8-7,5 s | 21,0-21,7 s |
  | 12 | 6,8 s | 21,0 s |
  | 16 | 5,2 s | 18,7 s |
  | 19 | 5,1 s | - |

  Depo kapalı: 34,9 s -> 8,6 s (8 işçi). Toplamın ~14 s'si çapraz-kesit adımlar (paralel değil). Sınır 8 seçildi: 8->16 işçi yalnız ~1,5-2 s ek kazanç (toplamın ~%8'i), bellek/CPU gece işinin geri kalanıyla paylaşılıyor; gerekirse `n_jobs=16` elle verilir.
- **Birebirlik:** `n_jobs=1` ile `-1/4/8/12/16` ve parça 5/20/40 için `full`, `aug`, `labeled` `assert_frame_equal(check_exact=True)`; `feats`, `cat_idx`, `loader_report` (sıra dahil), `loader_meta` (rejected_rows, quarantined_days, stale_symbols, feature_cache...) eşit.
- **Dikkat:** Windows'ta betikler `if __name__ == "__main__":` korumalı olmalı (nightly ve doğrulama raporu öyle). `limit>0` (`build_pooled_dataset`) paralel yolda da çalışır; tek sembol/`n_jobs=1` sekansiyele düşer.
- Testler: `tests/test_pooled_parallel.py` (gerçek FeaturePipeline + loky: sekansiyelle birebir, too_short/delisted/rejected/karantina; önbellek sayaç toplama; stub ile hata satırı, sıra, parça bağımsızlığı, delisted_excluded, log yeniden yayını).

### Macro feature-cache poisoning guard (2026-06-02 fix)

The feature-cache key already includes `use_macro`, but a transient macro-fetch
failure used to poison the cache: when macro requested but unavailable, the
engineered frame was still written under the `use_macro=True` key, so later
(healthy) runs served the macro-less frame on a cache hit. A real trigger was a
Windows `cp1252` `UnicodeEncodeError` on the `→` character in `MacroPipeline`
log prints — the exception bubbled into `DataIngestionService._fetch_macro`,
which swallowed it and returned an empty frame, silently dropping macro. Two
guards in `DataIngestionService._engineer_features_cached`
(`src/pipeline/data_services.py`):

- **Write guard:** when `use_macro` is True but the resolved `macro_df` is empty/
  None, the degraded frame is **not** cached (a transient failure cannot poison).
- **Self-heal on read:** a cache hit whose `feature_names` contain no macro
  feature (`_has_macro_features`, checked against
  `MacroPipeline.macro_feature_names`) while macro is expected is evicted and
  recomputed, so already-poisoned entries recover automatically.

The root print crash was also removed (`→` replaced with `->` in
`macro_pipeline.py` runtime prints) so macro no longer drops on cp1252 stdout.

## DataManager Responsibilities

`DataManager` in `src/pipeline/data_manager.py` owns:

- Optional data update.
- Raw data loading.
- Training-window filtering.
- Macro fetch and cache.
- Feature cache lookup/write.
- Feature pruning metadata.
- Dynamic sector mapping from `DataConfig.universe_file` and `sector_mapping`
  metadata propagation.
- Survivorship/listing checks.
- Chronological splitting.
- Tensor preparation.
- Dataset hash and run metadata construction.

## Splitting

Splitting is chronological. No shuffle-based split is valid for this project.

Modes:

- `single_split`: one chronological train/test split using `DataConfig.test_ratio`.
- `walk_forward`: fold generation using `ValidationConfig` and `WalkForwardValidator`.
- `final_holdout`: last reserved block used only after walk-forward selection.

Important walk-forward settings:

- `wf_n_splits`
- `wf_min_train_size`
- `wf_test_size`
- `wf_max_train_size`
- `wf_window_type`: `sliding` or `expanding`
- `wf_embargo_size`: None/0 -> otomatik `target_horizon + 5` (2026-11-15 öncesi
  `max(200, time_steps)` idi). Yalnız etiket örtüşmesi sızdırır; sequence bağlamı
  embargodan bağımsız olarak test başlangıcından önceki satırlardan alınır.
  Açık değer `< target_horizon` ise `ValueError`.
- `final_holdout_size`

Model özellikleri (`DataServices.model_feature_names`) ham `Open/High/Low/Volume`
seviyelerini ve `Adj_Close`'u içermez (durağan olmayan seviyeler). `data_loader`
O/H/L'yi `Adj_Close/Close` oranıyla ölçekler. `DataUpdater` artık son 7 günü
örtüşmeli indirir; örtüşmede |oran-1| > 0.005 (split/temettü yeniden ölçekleme)
ise tüm geçmişi atomik olarak yeniden yazar (`full_refetch`).

## Calendar Maintenance

`src/forecasting/bist_calendar.py` maintains `data/meta/bist_calendar.csv`.
`ensure_bist_calendar()` generates a rolling BIST trading calendar with Turkish
fixed-date closures, weekends, and deterministic session labels. Existing rows
are treated as manual overrides and merged back into the generated range, so
religious holidays or half-day sessions can be added to the CSV without changing
consumer code.

The analysis refresh service calls calendar generation before freshness checks
and forecast generation. `analysis_freshness.compute_freshness()` warns when the
calendar range does not cover the requested freshness window and falls back to a
weekday count.

## Scaling

`src/data/preprocessor.py` enforces train-only fitting.

Supported scaling modes:

| Mode | X scaler | y scaler | Clip |
|---|---|---|---|
| `robust_x_standard_y_clip` | RobustScaler | StandardScaler | yes |
| `robust` | RobustScaler | RobustScaler | yes |
| `standard` | StandardScaler | StandardScaler | no |
| `minmax` | MinMaxScaler | MinMaxScaler | no |

The default modern mode is `robust_x_standard_y_clip`.

Leakage rule:

- `fit_transform` is allowed only on the training slice.
- Test, fold-test, and final-holdout slices must use `transform`.
- Walk-forward fold scalers are not written repeatedly to the same inference path.

## Targets

The modern target mode is `log_return`.

Prediction conversion:

- `reconstruct_prices_from_logret()` converts log-return predictions to prices using previous actual close.
- `reconstruct_prices_from_return()` converts simple-return predictions to prices.
- Clipping protects against pathological out-of-distribution predictions.

## Sequence Tensors (kaldırıldı, 2026-11-16)

Sekans tensörleri (`time_steps`, pencereli X/y üretimi) derin sekans modelleriyle
birlikte kaldırıldı; aşağıdaki açıklamalar tarihseldir. Güncel model girdisi tablodur.

`create_sequences()` creates sequence tensors for sequence-style models such as
LSTM and the lightweight linear sequence baselines:

```text
X: (samples, features)
to
X_seq: (samples - time_steps + 1, time_steps, features)
```

The target aligns with the final row of each sequence window. The intended
semantic is one-step-ahead forecasting with features known at the end of day `t`
and target for the next move.

## Leakage Boundaries

- No random train/test split.
- No scaler fit on test, fold-test, or final holdout.
- Macro features must respect configured lags.
- Aylık makro (faiz, CPI) damgası ayın 1'i olsa da değer ay sonu seviyesidir; yayın
  tarihi `ay sonu + lag` (faiz 1, CPI 5 gün) olarak hesaplanır ve günlük takvime
  `merge_asof(direction="backward")` ile taşınır (`macro_transforms.lag_monthly_features`,
  `merge_ffill_monthly_features`; 2026-11-15 derin denetim P0-3/P0-4).
- ABD seansında kapanan seriler (`US_SESSION_COLUMNS`: VIX, DXY, US10Y, GOLD_USD, OIL_USD; 2026-11-16 itibarıyla bu seriler özellik üretmez, tarihsel)
  1 satır geciktirilir; USDTRY ve BIST endeksleri gecikmesizdir (P0-5).
- Makro kapsamı dışındaki hisse günlerinde `*_Return`/`*_Change` kolonları 0 ile
  doldurulur (tekrar edilmez); makro bitişi hisseden >5 gün gerideyse uyarı loglanır.
- Optuna ayarı walk-forward'da bellek içi ve çağrıya izoledir; kalıcı study yalnız açık
  `study_storage` + benzersiz `study_name` ile (P0-1).
- Signal calibration must not optimize on final holdout.
- Final holdout is for confirmation, not model selection training feedback.

## Related Pages

- [Architecture](architecture.md)
- [Validation and Backtesting](validation-and-backtesting.md)
- [Testing and Quality](testing-and-quality.md)
