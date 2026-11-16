---
title: Persistence and API
type: concept
status: active
last_updated: 2026-11-16
owner: llm
---

# Persistence and API (API v2)

> Önceki sürümün (Kol-A: `stock_models.db`, `outputs/` kayıtları, `/best-model`, `/experiments`,
> `/metrics`, `/leaderboard`, `/run/*`, `/refresh/*`, yenileme işleri) içeriği kaldırıldı; git geçmişinde.

## Kalıcılık

- **`data/serving_pool.db` (PeerStore)** — tek ürün veritabanı. Tablolar ve kolonlar:
  [Architecture](architecture.md#veritabanı-şeması-dataserving_pooldb). Yazar: gece işi
  (`PeerStore.write_run`: run başlığı + skorlar TEK transaction). Okuyucular salt-okunur açar.
- **`data/nightly_status.json`** — makine-okunur koşu durumu (`src/serving/run_status.py`, atomik
  yazım): `started_at`, `finished_at`, `completed`, `phase`, `exit_code`, `refresh`, `scoring`,
  `stale_fraction`, **`risk: {status: ok|fallback|failed, method, error}`**.
  Koşu normal bitmezse `completed=false` kalır (`/health` -> `interrupted`).
- `data/clean/` (temiz OHLCV Parquet), `data/feature_cache/pooled/` (isteğe bağlı), `logs/nightly_*.log`.

## Gece işi

`scripts/nightly_serving.ps1` (Task Scheduler, her gün 21:00; `register_nightly_task.ps1` kaydeder):
`python -m src.serving.nightly --db data/serving_pool.db --model ensemble --fallback-dir data/yedek`.
Sıra: BIST işlem günü kapısı -> evren veri tazeleme (bayat-pay kapısı `--max-stale-fraction`, aşılırsa
skorlama atlanır, çıkış 3) -> pooled panel (`quarantine_jumps`) -> Kol-B skor -> risk adımı ->
`write_run`. `--horizon` yalnız 5. Risk kapısı bir önceki geceye göre değişirse log + config uyarısı.

## Okuyucu kuralları

`PeerStore.open_readonly` (API `peer_service`/`symbols_service`/`serving_health`/`response_cache`, GUI
`peer_reader`): DB oluşturulmaz, migrasyon/`journal_mode` değişmez. Test:
`test_readers_never_migrate_or_change_journal_mode_on_old_schema_db`.

## Endpoint'ler

| Yöntem/yol | Açıklama |
|---|---|
| `GET /health` | Servis, serving havuzu tazeliği (`serving_pool`, 7 takvim günü), son gece koşusu (`nightly_run`, `risk` dahil). `status`: `ok` / `degraded` (havuz yok/bayat, koşu kesilmiş **veya risk adımı `failed`**). Yanıtta DB yolu, `sys.executable`, proje kökü YOKTUR. |
| `GET /symbols` | Son koşudaki semboller (DB yok -> boş liste; DB okunamıyor -> 503) |
| `GET /analysis/{symbol}` (+ `/v1/analysis/{symbol}` alias) | Sembol analizi (aşağıda) |

Kaldırılan: `/best-model`, `/experiments`, `/metrics`, `/leaderboard`, `/run/*`, `/refresh/*` (404).
Sembol regex `^[A-Z0-9]{1,10}$` (geçersiz -> 400, dosya yoluna girmeden önce). CORS: `src/api/runtime_config.py`.
Hız sınırı: `AI_CORE_RATE_LIMIT_PER_MINUTE`. **Bağlanma:** `python -m src.api.main` varsayılan
`127.0.0.1:8001` (`AI_CORE_API_HOST`, `AI_CORE_API_PORT`; 8000 makinede başka servisçe kullanıldığı için); `uvicorn src.api.main:app` serbesttir.

## `/analysis/{symbol}` yanıt şekli (`api_version="2"`)

`symbol`, `analysis_status`, `generated_at`, `data` (`last_observed_date`, `last_close`,
`data_freshness`, `staleness_days`), `confidence` (`label` low|medium|high, `reasons`, `warnings`),
`disclaimer`, `serving_run` (`run_id`, `model_name`, `as_of_date`, `age_days`, `stale`),
`data_quality` (PSI 30g drift), `peer` (`PeerBlock`: peer_score/percentile/label, segment_*,
confidence_*, trend_*, `kolb_price_p50/low/high`, `risk_*`, `risk_validation` = koşu `config_json["risk"]`, xai_*).
`model`, `forecast`, `performance`, `xai`, refresh-job alanları KALDIRILDI (kırıcı değişiklik).

### `analysis_status`

| Değer | HTTP | Anlam |
|---|---|---|
| `ok` | 200 | Sembol son koşuda var, veri/havuz taze, güven medium/high |
| `stale_data` | 200 | Piyasa verisi > 1 işlem günü eski, havuz > 7 gün eski **veya `peer.as_of_date` CSV son tarihinden > 1 işlem günü geride** (uyarı: bant/fiyat çapası eski kapanışa ait) |
| `low_confidence` | 200 | Sonuç var, güven `low` |
| `no_data` | 200 | Sembol son koşuda yok |
| `unavailable` | **503** | Serving DB yok / kilitli / bozuk / okunamıyor (log'lanır) |

`no_data` ve `unavailable` yanıtları ÖNBELLEĞE ALINMAZ (bilinmeyen sembol sözlüğü büyümez, geçici hata
1 saat kalmaz). Önbellek: TTL 3600 s, **en fazla 1024 girdi (LRU)** (`AI_CORE_RESPONSE_CACHE_*`);
parmak izi = (son run_id, CSV mtime/boyut, **İstanbul** takvim günü).

### Güven kuralları (`analysis_service._confidence`)

Başlangıç `peer.confidence_label`. `low`'a çeken: bayat veri, bayat havuz, `as_of` geriliği, PSI major.
Bir kademe düşüren: `risk_method == naive_fallback`; **`risk_method` NULL** (uyarı koşu-düzeyi hata /
`quarantine` / `short_history` / `gap` nedenini söyler, bkz. [Risk Modeli](risk-model.md)). PSI orta -> high'dan medium.

### Hatalar

500: `{"detail": "Internal server error"}` (ayrıntı yalnız sunucu logunda). `/symbols` DB hatası: 503
`Serving database unavailable`.

## Test

`tests/test_analysis_api_v2.py`, `tests/test_serving_hardening_2026_11.py`, `tests/test_peer_service.py`,
`tests/test_peer_store.py`, `tests/test_serving_health.py`, `tests/test_response_cache.py`.

### Peer bloğu doğrulama metrikleri (2026-11-16)

`peer` bloğu koşu düzeyindeki örneklem dışı metrikleri de taşır: `ic_mean` (günlük kesitsel IC ortalaması),
`pct_ic_positive` (IC > 0 gün oranı), `icir_overall`, satır düzeyinde `segment_icir` ve
`risk_validation` (`mse_ratio`, `dm_t`, `coverage`, `qlike_*`, gate durumu). İstemci bunları "Performans"
bölümünde Türkçe açıklamalarıyla gösterir.
