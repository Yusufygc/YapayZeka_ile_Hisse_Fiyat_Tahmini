---
title: Architecture
type: concept
status: active
last_updated: 2026-11-16
owner: llm
---

# Architecture

`ts_forecasting_lab` artık **gecelik havuzlu analiz servisidir**. Hisse başına yön modelleri (Kol-A),
`ForecastingPipeline`, backtest motoru, sinyal kalibrasyonu, `stock_models.db` ve onlara özgü
CLI/GUI/endpoint'ler 2026-11'de kaldırıldı (karar "A+C"; eski mimari için git geçmişi, f9ca736 öncesi).

## Akış

```text
data/*.csv (yfinance) -> src/serving/nightly.py -> data/serving_pool.db (PeerStore)
   gate (BIST işlem günü)   refresh (DataUpdater)      |-> src/api  (FastAPI, API v2; salt-okunur)
   pooled panel + Kol-B skor  risk modeli (src/risk)   |-> src/gui  (PySide6 tek tablo; salt-okunur)
                                                       `-> data/nightly_status.json  (/health)
```

Giriş noktaları (erişilebilirlik kökleri): `src.api.main`, `src.gui.main`, `src.serving.nightly`,
`tools/*.py`, `scripts/*.ps1`. Bunlardan erişilemeyen kod ölü sayılır ve silinir.

## Dizinler

| Dizin | Görev |
|---|---|
| `src/data/` | OHLCV indirme/temizleme (`ohlcv_clean`), panel yükleyici (`pooled_loader`), temiz depo (`clean_store`), tazelik, çapraz-kesit özellik/hedef |
| `src/features/` | `feature_pipeline` (sembol başına teknik özellikler; makro katman yok), Parquet özellik önbelleği, kod parmak izi, korelasyon budama |
| `src/models/` | Kol-B: `global_pooled_model` (LightGBM), `torch_mlp_model`, `ensemble_pooled_model` |
| `src/risk/` | Havuzlu oynaklık modeli ([Risk Modeli](risk-model.md)) |
| `src/validation/` | Pooled purged CV, OOS, segment IC, baz çizgileri |
| `src/serving/` | `nightly`, `nightly_scoring`, `nightly_risk`, `peer_store`, `peer_scoring`, `confidence`, `trend_tendency`, `peer_xai`, `run_status`, `paths` |
| `src/xai/` | Özellik sözlüğü, SHAP/permütasyon, grup özeti |
| `src/api/` | FastAPI v2 (`routers/analysis`, `services/*`) |
| `src/gui/` | PySide6 özet tablosu (`core/peer_reader`, `tabs/dashboard_tab`) |
| `src/forecasting/bist_calendar.py` | Takvim CSV üretim/bakım yardımcısı (`data/meta/bist_calendar.csv`; girişlerden çağrılmaz, bakım aracı olarak tutulur) |

## Teknoloji

Python 3.13, pandas/numpy, scikit-learn (risk Ridge), LightGBM + PyTorch (Kol-B), joblib/loky
(sembol-paralel özellik), pyarrow (Parquet), SQLite (WAL; yazar `PeerStore`, okuyucular `mode=ro`),
FastAPI/uvicorn/pydantic, PySide6, yfinance, pandas-market-calendars (XIST işlem günü kapısı),
SHAP (XAI). `requests` kaldırıldı (kullanılmıyordu).

## Veritabanı şeması (`data/serving_pool.db`)

- `global_model_runs`: run_id, created_at (UTC), as_of_date, model_name, data_snapshot_hash,
  n_symbols, n_rows, horizon, ic_mean, icir, pct_ic_positive, `config_json` (risk özeti, uyarılar,
  `risk.missing`, kapı değişimi, karantina/fallback sayıları).
- `peer_scores` (run_id x symbol, UNIQUE): peer_score/percentile/label, segment_liq/vol/sector,
  segment_icir, confidence_label/reasons/warnings, trend_label/prob_up/expected_return/prob_outperform, xai_*,
  `kolb_price_p50/low/high`, `kolb_horizon_days`, `kolb_band_level`, `risk_sigma_5d`,
  `risk_sigma_naive_5d`, `risk_vol_percentile`, `risk_level`, `risk_z`, `risk_band_level`,
  `risk_horizon_days`, `risk_method`. Şema değişiklikleri eklemelidir (`_PEER_MIGRATIONS`, ALTER).
- Yazar `PeerStore(path)` şemayı/WAL'i kurar; okuyucular `PeerStore.open_readonly(path)`
  (`file:...?mode=ro`) kullanır — oluşturma, migrasyon veya journal_mode değişikliği yapmaz;
  eski şemada eksik risk kolonları `None` okunur. Ortak yol yardımcısı: `src/serving/paths.py`.

## Zaman ve hata kuralları

Tarih/saat tz-aware (UTC veya İstanbul UTC+3, DST yok); servis günü İstanbul tarihidir (önbellek
parmak izi dahil). Boş `except` yasak; dosyalar <= 500 satır. Ayrıntı: [Code Quality](code-quality-and-refactoring.md).

Bkz. [Persistence and API](persistence-and-api.md), [Data Pipeline](data-pipeline.md), [Risk Modeli](risk-model.md).
