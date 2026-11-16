---
title: Testing and Quality
type: concept
status: active
last_updated: 2026-11-16
owner: llm
---

# Testing and Quality

## Komut

```bash
# proje .venv, PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

2026-11-16 durumu: tüm paket yeşil (Kol-A / makro / ölü kod testleri silindikten sonra ~460 test; sayı
için komutun son satırına bakın). Pytest geçici dizinleri `.codex_tmp/` altındadır (git dışı).

## Test alanları

| Dosya(lar) | Kapsam |
|---|---|
| `test_analysis_api_v2.py`, `test_serving_hardening_2026_11.py` | API v2 şekli/durumları, 503 ve `no_data` ayrımı, önbellek (LRU, no_data/unavailable alınmaz), `/health` (risk `failed` -> degraded, iç yol yok), as_of geriliği, risk NULL uyarıları, 500 genel mesaj, bind adresi, **okuyucuların DB'yi değiştirmediği** (eski şema: sütunlar, journal_mode, dosya baytları) |
| `test_peer_service.py`, `test_peer_store.py`, `test_persistence_fixes_2026_11.py`, `test_serving_health.py`, `test_response_cache.py`, `test_rate_limit.py` | PeerStore, salt-okunur erişim, migrasyon, tazelik, önbellek |
| `test_risk_features.py`, `test_risk_validation.py`, `test_risk_serving.py` | Risk özellikleri/hedef sızıntısızlığı, purged WF + kapı, model/band formülü (merkez 0), eski şema migrasyonu |
| `test_nightly_*.py`, `test_p4_nightly_module.py`, `test_nightly_scoring.py` | Gece akışı, durum dosyası, bayat-pay kapısı, `--horizon` reddi, `.ps1` kuralları |
| `test_pooled_*.py`, `test_clean_store.py`, `test_data_*`, `test_cross_sectional.py`, `test_curated_feature_set.py`, `test_calendar_features.py` | Havuz yükleyici (karantina, paralel birebirlik), temiz depo, veri düzeltmeleri, kürate 11 özellik (makrosuz) |
| `test_global_pooled_model.py`, `test_ensemble_pooled_*.py`, `test_torch_mlp_model.py`, `test_peer_*.py`, `test_trend_tendency.py`, `test_xai_strategies.py`, `test_segment_ic.py`, `test_baselines.py` | Kol-B modelleri, skor, güven, trend, XAI, doğrulama |
| `test_gui_smoke.py` | PySide6 offscreen duman testi (okuyucu salt-okunur) |

## Kalite değişmezleri

- Kronolojik doğrulama, purge/embargo `target_date` ile; ölçekleyici/kalibrasyon yalnız geçmişten.
- Okuyucular (API/GUI) asla DB oluşturmaz/migre etmez; eski şema okunabilir kalır.
- Boş `except` yok; tz-aware zaman; dosya <= 500 satır; sembol regex `^[A-Z0-9]{1,10}$`.
- Ağ erişimi gerektiren test yok (sağlayıcılar enjekte/sahte).
- Doğrulama: her modül `src/` ve `tools/` (arşiv hariç) import edilebilir; FastAPI `TestClient` ile
  `/health`, `/symbols`, `/analysis/{symbol}` geçici PeerStore DB'sinde duman testi.

Bkz. [Architecture](architecture.md), [Persistence and API](persistence-and-api.md), [Change Management](change-management.md).
