# Kol-B Pooled Fiyat Bandı — overfit'e dayanıklı mutlak tahmin (ikinci görüş)

## Context

Mutlak fiyat manşeti şu an YALNIZ Kol-A'dan (per-symbol, overfit-eğilimli) geliyor;
Kol-B sadece göreli sıra (rank) veriyor → "hâlâ Kol-A'ya bağlıyız" gerilimi. Ama Kol-B
**zaten** kalibre mutlak sinyal üretiyor: `trend_expected_return` (quintile-mapped ort.
h=5 log-getiri) + `trend_prob_up`. Bunlardan **pooled-temelli, overfit'e dayanıklı bir
fiyat + bant** türetilebilir — Kol-A'ya hiç dokunmadan.

`fiyat_p50 = last_close · exp(trend_expected_return)`,
`bant = last_close · exp(trend_expected_return ∓ z·σ_quintile)`.

Eksik tek parça: **dispersion** — trend şu an yalnız quintile ORTALAMA veriyor, std yok.

**Kullanıcı kararları:**
- Band belirsizliği: **quintile std (parametrik z·σ)** — `TrendCalibration`'a sabit
  `quintile_return_std` default eklenir (mevcut `quintile_expected_return` mean'leri gibi),
  Faz 7 OOS çalışmasından türetilir.
- Konumlandırma: **peer kartına ikinci görüş** — "Pooled 5g tahmin: ₺X (₺Y–₺Z)". Kol-A
  prediction card aynen kalır; iki bağımsız mutlak tahmin yan yana.

## Mevcut yapı (yeniden kullanılacak)

- Trend kalibrasyonu: `src/serving/trend_tendency.py::TrendCalibration`
  (`quintile_expected_return`, `quintile_prob_up`), `trend_from_peer(pct, universe, cfg)`
  → `TrendTendency(label, prob_up, expected_return)`. h=5 (`pooled_loader` target_horizon=5).
- Serving birleştirme: `src/serving/nightly_scoring.py::assemble_peer_table` — `merged` df'i
  kurar, trend alanlarını yazar (satır 125-135). `panel_latest` ham `Close` taşır
  (`pooled_loader._NON_FEATURE`) → **last_close kaynağı** (en güncel tarih satırı).
- z değeri: **`src/forecasting/interval_calibration.py::z_for_level`** (0.8→1.2816) — Kol-A
  interval'iyle ortak, yeniden kullanılır (iki kol aynı belirsizlik dili).
- Depolama: `src/serving/peer_store.py` — `peer_scores` DDL + `_PEER_COLS` +
  `_PEER_MIGRATIONS` (additive ALTER deseni; `trend_expected_return` zaten kolon).
- Serving şema/servis: `src/api/schemas/analysis.py::PeerBlock` +
  `src/api/services/peer_service.py::get_peer_block` (kolon→alan map).
- Desktop: `domain/models/ai_analysis.py::PeerInfo` + `infrastructure/ai/
  ai_core_fastapi_client.py::_parse_peer` + `ui/pages/ai_page/left_panel/peer_card.py`
  (yeni "pooled tahmin" satırı) + `ui/shared/locale_tr.py`.

## Yapılacaklar

### 1. Trend dispersion — `src/serving/trend_tendency.py`
- `TrendCalibration`'a `quintile_return_std: tuple` ekle (5 değer, h=5 getiri std'si).
  Defaultlar Faz 7 OOS çalışmasından (bkz. Adım 7); makul başlangıç ~(0.045, 0.040, 0.038,
  0.037, 0.039) gibi quintile'a göre.
- `TrendTendency`'ye `return_std: Optional[float]` ekle.
- `trend_from_peer`: quintile std'sini de döndür (`return_std = cfg.quintile_return_std[qi]`).

### 2. Pooled fiyat bandı — `src/serving/nightly_scoring.py::assemble_peer_table`
- `panel_latest`'in en güncel tarihinden `last_close` map'i çıkar (`symbol → Close`), `merged`'e join.
- Yeni `PriceBandConfig` (veya `scoring_cfg`/`trend_cfg` alanı): `level: float = 0.8`,
  `horizon_days: int = 5`.
- Her satır için (trend dalında, `t.return_std` varsa ve `last_close` finite ise):
  - `z = z_for_level(level)` (interval_calibration'dan import)
  - `p50 = last_close * exp(expected_return)`
  - `low = last_close * exp(expected_return - z*return_std)`, `high = …+z*return_std`
  - kolonlar: `kolb_price_p50`, `kolb_price_low`, `kolb_price_high`, `kolb_horizon_days`,
    `kolb_band_level`. std yoksa/last_close yoksa → None (geriye uyumlu).

### 3. Depolama — `src/serving/peer_store.py`
- `peer_scores` DDL'e + `_PEER_COLS` + `_PEER_MIGRATIONS`'a ekle: `kolb_price_p50 REAL`,
  `kolb_price_low REAL`, `kolb_price_high REAL`, `kolb_horizon_days INTEGER`,
  `kolb_band_level REAL`. Mevcut `_cell`/insert deseni otomatik yazar.

### 4. Serving şema — `src/api/schemas/analysis.py::PeerBlock`
- Ekle: `kolb_price_p50/low/high: Optional[float]`, `kolb_horizon_days: Optional[int]`,
  `kolb_band_level: Optional[float]`. (additive, default None — geriye uyumlu.)

### 5. Serving servis — `src/api/services/peer_service.py::get_peer_block`
- Yeni DB kolonlarını PeerBlock alanlarına map et (`_optional_float`/int deseni).

### 6. Desktop (ikinci görüş kartı)
- `domain/models/ai_analysis.py::PeerInfo`: `kolb_price_p50/low/high: float|None`,
  `kolb_horizon_days: int|None`.
- `ai_core_fastapi_client.py::_parse_peer`: yeni alanları doldur.
- `peer_card.py`: yeni satır — "Pooled {h}g tahmin: ₺{p50} (₺{low}–₺{high})"; üç değer
  doluysa göster, yoksa gizle (mevcut conditional desen).
- `locale_tr.py`: `PEER_POOLED_TAHMIN_TMPL`.

### 7. Quintile std default'larını türet — `tools/`
- `tools/e2_faz7b_quintile_return_std.py` (YENİ, veya `e2_faz7_confidence_diracc.py`
  genişlet): pooled OOS panelinde her peer-percentile quintile için gerçekleşen h=5
  log-getiri std'sini hesapla → `quintile_return_std` default değerlerini yazdır.
  `pooled_oos`/`segment_ic` mevcut OOS hattını kullanır. Tek seferlik kalibrasyon.

### 8. Testler
- `tests/test_trend_tendency*.py` (yeni/var): `trend_from_peer` `return_std` döner; quintile
  eşleşmesi; band hesabı `z_for_level`'la tutarlı.
- `tests/test_nightly_scoring*` veya yeni: `assemble_peer_table` `kolb_price_p50/low/high`
  üretir (low<p50<high, exp/log tutarlı); std yoksa None.
- `tests/test_peer_store.py`: yeni kolon round-trip.
- `tests/test_peer_service.py`: PeerBlock yeni alanlar dolu/None.
- Desktop `_parse_peer`: kolb fiyat alanları parse + None graceful.

### 9. Wiki
- `model-catalog.md` / `persistence-and-api.md`: Kol-B pooled fiyat bandı (z·σ, h=5),
  peer_scores yeni kolonlar, PeerBlock alanları.
- `validation-and-backtesting.md`: trend kalibrasyonu + dispersion (quintile std).
- `log.md`: tarihli giriş.

## Dokunulacak dosyalar
- AI_Core: `src/serving/trend_tendency.py`, `src/serving/nightly_scoring.py`,
  `src/serving/peer_store.py`, `src/api/schemas/analysis.py`,
  `src/api/services/peer_service.py`; YENİ `tools/e2_faz7b_quintile_return_std.py`;
  testler; wiki.
- Desktop: `src/domain/models/ai_analysis.py`,
  `src/infrastructure/ai/ai_core_fastapi_client.py`,
  `src/ui/pages/ai_page/left_panel/peer_card.py`, `src/ui/shared/locale_tr.py`.
- Yeniden kullanım: `src/forecasting/interval_calibration.py::z_for_level`.

## Doğrulama (uçtan uca)
1. Birim test: `dl_env python -m pytest tests/test_trend_tendency*.py tests/test_peer_store.py tests/test_peer_service.py -v --basetemp=...`
2. Nightly smoke: `dl_env python tools/e2_nightly_pipeline.py --model ensemble --skip-trading-gate --limit 30 --db data/serving_pool_smoke.db`
   → `peer_scores.kolb_price_p50/low/high` dolu (sqlite). Smoke DB sil.
3. API smoke: `uvicorn src.api.main:app --port 8000` → `GET /analysis/<sym>` →
   `peer.kolb_price_p50/low/high` dolu (band low<p50<high). Server kapat.
4. Desktop: peer kartında "Pooled 5g tahmin: ₺X (₺Y–₺Z)" satırı görünür (Fintech env,
   offscreen widget smoke + `_parse_peer` testi).

## Notlar / sınırlar
- Band parametrik (z·σ, normallik); conformal'e (kapsama-garantili) sonradan geçilebilir.
- p50 = pooled kalibre eğilim; quintile-grain (5 kova) olduğundan kaba ama overfit'siz.
- last_close panelin en güncel tarihinden; eksikse band None (geriye uyumlu).
- Kol-A prediction card değişmez; bu ikinci görüş. Manşeti Kol-B'ye taşımak ayrı adım.
- Veri CSV + smoke DB commit edilmez; commit Türkçe, AI atfı yok.
- Branch: AI_Core `development`, desktop `update`.