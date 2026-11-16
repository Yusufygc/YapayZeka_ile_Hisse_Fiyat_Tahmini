---
title: Source Map
type: source-map
status: active
last_updated: 2026-11-16
owner: llm
---

# Source Map

This page maps raw source layers to the wiki pages they inform. Use it before
deep inspection so source reads stay targeted.

## Primary Source Layers

| Source | Role | Wiki Pages |
|---|---|---|
| `src/data/` | OHLCV okuma/temizleme, panel yükleyici, temiz depo, tazelik, çapraz-kesit | [Data Pipeline](data-pipeline.md) |
| `src/features/` | `feature_pipeline` (makrosuz), Parquet özellik önbelleği, kod parmak izi | [Data Pipeline](data-pipeline.md) |
| `src/models/` | Kol-B modelleri (LightGBM, MLP, ensemble) | [Model Catalog](model-catalog.md) |
| `src/risk/` | Havuzlu oynaklık modeli, doğrulama, servis | [Risk Modeli](risk-model.md) |
| `src/validation/` | Pooled purged CV, OOS, segment IC, baz çizgileri | [E2 Epic](e2-pooled-global-model-epic.md), [Risk Modeli](risk-model.md) |
| `src/serving/` | Gece işi, skorlama, `PeerStore`, güven, trend, XAI, durum dosyası | [Persistence and API](persistence-and-api.md), [Architecture](architecture.md) |
| `src/api/` | FastAPI v2 (salt-okunur) | [Persistence and API](persistence-and-api.md) |
| `src/gui/` | PySide6 özet tablosu | [Desktop GUI Application](desktop-gui-application.md) |
| `src/xai/` | Özellik sözlüğü, SHAP/permütasyon, grup özeti | [Architecture](architecture.md) |
| `src/forecasting/bist_calendar.py` | Takvim CSV bakım yardımcısı | [Data Pipeline](data-pipeline.md) |
| `tests/` | Davranış sözleşmeleri | [Testing and Quality](testing-and-quality.md) |

## Entrypoints

| Entrypoint | Purpose |
|---|---|
| `python -m src.serving.nightly` | Gece işi: işlem günü kapısı -> veri tazeleme -> skorlama + risk -> PeerStore |
| `uvicorn src.api.main:app` | HTTP servisi (API v2) |
| `python -m src.gui.main` | Masaüstü özet paneli |
| `tools/*.py` | Veri denetim/onarım, Kol-B doğrulama sarmalayıcıları (`tools/archive/` arşiv) |
| `scripts/nightly_serving.ps1` | Windows Task Scheduler hedefi (her gün 21:00) |
| `scripts/register_nightly_task.ps1` | Tek seferlik görev kaydı |

## Reference Documents

| File/Directory | Use |
|---|---|
| `README.md` | Broad Turkish project overview and phase history |
| `docs/glossary.md` | Domain glossary |
| `docs/MyDocs/` | Planning, research, and report artifacts; currently ignored by git |
| `AGENTS.md` | Agent schema and wiki maintenance rules |
| `RULES.md` | Repository change, wiki-update, and Turkish commit-message rules |

## Data and Metadata

| Path | Meaning |
|---|---|
| `data/*.csv` | OHLCV stock files, plus universe metadata |
| `data/bist_universe.csv` | Stock universe metadata used for survivorship and listing checks |
| `data/meta/bist_calendar.csv` | BIST işlem takvimi (tazelik hesabı) |
| `data/feature_cache/` | Feature cache generated from raw data and config |
| `data/clean/` | Kol-B temizlenmiş OHLCV Parquet deposu (CSV'lerden türetilir, gitignored; bkz. data-pipeline.md) |
| `data/serving_pool.db` | `PeerStore` (global_model_runs + peer_scores); gece işi yazar, API/GUI salt-okunur okur |
| `data/nightly_status.json` | Gece koşusu durumu (`risk` alanı dahil) |
| `logs/nightly_*.log` | E2 nightly job logs (gitignored, 14-day retention) |

## Source-of-Truth Rule

If the wiki and source code disagree, trust source code first, then update the
wiki and add a log entry. If source code and tests disagree, inspect both and
surface the contradiction before editing.

