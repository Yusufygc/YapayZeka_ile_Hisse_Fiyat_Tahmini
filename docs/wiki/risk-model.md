---
title: Risk Modeli (havuzlu oynaklık)
type: concept
status: active
last_updated: 2026-11-16
owner: llm
---

# Risk Modeli (havuzlu oynaklık)

Ürün kararı "A": hisse başına yön modelleri (Kol-A) emekli oldu; yerine tek **havuzlu oynaklık
modeli** her hisse için beklenen **5 günlük oynaklığı** ve kalibre **%80 fiyat aralığını** üretir.
Kod: `src/risk/` (`features.py`, `model.py`, `validation.py`, `serving.py`). Gece işi
(`src/serving/nightly.py` -> `src/serving/nightly_risk.py`) çalıştırır; sonuç `peer_scores.risk_*`
ve `kolb_price_low/high` kolonlarına yazılır.

## Hedef ve özellikler

- Getiri `r_t = log(C_t / C_{t-1})`; fiyat **takvim konumuna** (panelin tüm tarihleri) hizalanır.
  Bir sembolün eksik günü (askı, sıfır hacim, karantina) o günün ve ertesi günün getirisini NaN
  yapar; çok günlük getiri gizlice "günlük" sayılmaz.
- **Hedef** `y = log(sqrt(mean(r_{t+1..t+5}^2)) + 1e-4)` (log günlük oynaklık); `fwd_ret` (5g getiri)
  ve `fwd_rv2` yalnız kapsama/QLIKE içindir. Hedef yalnız `t+1..t+h`'ye bağlıdır.
- **Özellikler** (hepsi geriye pencere, nedensel; `RISK_FEATURE_COLS`): `rv_1`, `rv_5`, `rv_21`,
  `rv_63` (HAR günlük/haftalık/aylık/çeyreklik log RMS), `log_natr_14`, `mkt_rv_21` (sembol-arası
  medyan `rv_21`), `rv_1_down` (aşağı yönlü şok), `liq_log`, `mkt_ret` (eşit ağırlıklı piyasa getirisi).
  `w > 5` pencerelerde en az %80 geçerli gün gerekir (`MIN_WINDOW_FRAC`); `rv_5` için 5/5.
- Makro/BIST100 kolonu gerekmez; panel makro katmansız kurulur.

## Model, sapma düzeltmesi, konformal z

1. **Ridge(alpha=1)** + `StandardScaler` (`PooledVolModel`), hedef log günlük oynaklık.
2. **Bias (Jensen) faktörü** `c = mean(exp(2 * resid))`: log-vol regresyonu medyanı tahmin ettiği için
   `sigma_daily = exp(pred) * sqrt(c)`, `sigma_5d = sqrt(5) * sigma_daily`. `c` yalnız ÖNCEKİ blokların
   OOS artıklarından hesaplanır.
3. **Konformal z:** `z = quantile_0.80(|r_5d - merkez| / sigma_5d)` (son 252 tarih penceresi, havuzlanmış
   satırlar; yine yalnız önceki OOS). Gaussal 1.28 yetersiz kapsar (~%72); gerçek z bundan büyüktür.
4. **Bant:** `Close * exp(merkez -/+ z * sigma_5d)`. Merkez = Kol-B `trend_expected_return`;
   trend `belirsiz` (beklenen getiri sonlu değil) ise **merkez 0** alınır (`p50` o durumda boş kalır,
   bant yine dolu; ayrı sütun yok). Genişlik YALNIZ risk modelinden gelir.
5. `risk_level`: gün evreninin `sigma_5d` tertile'ı (düşük/orta/yüksek); `risk_vol_percentile` 0-100.

## Doğrulama ve kabul kapısı

Purged expanding walk-forward (`src/validation/pooled_cv.py`: 63 günlük test blokları, embargo
h+5, `target_date < test_başı` purge, `min_train_days=504`). Kalibrasyon sızıntısız: blok k için
`c`/`z` yalnız hedefi blok başından önce gerçekleşmiş önceki blokların OOS satırlarından gelir; ilk
blok kalibre edilemediği için QLIKE/kapsama kalibre bloklarda raporlanır.

**Kapı sabitleri (ön-kayıtlı, sessizce değişmez; `src/risk/validation.py`):**

| Sabit | Değer |
|---|---|
| `GATE_MAX_MSE_RATIO` | 0.95 (log-vol MSE oranı model/naive) |
| `GATE_MIN_DM_T` | 2.0 (Diebold-Mariano, Newey-West) |
| `GATE_COVERAGE_MIN / MAX` | 0.75 / 0.85 |

Model sunulur <=> `mse_ratio <= 0.95` VE `dm_t >= 2.0` VE kapsama `[0.75, 0.85]`. Aksi halde
`risk_method="naive_fallback"` (naive persistence sigma + KENDİ konformal z'si); metrikler run
`config_json["risk"]` içine yazılır, sessiz düşüş yoktur.

**Gerçek veri doğrulaması** (OOS 2016-11..2026-06):

| Metrik | Model | Naive |
|---|---|---|
| log-vol MSE oranı | **0.800** | 1.0 |
| Diebold-Mariano t | **17.6 - 18.0** | |
| QLIKE | **0.618** | 0.836 (yanlılık düzeltmeli naive 0.750) |
| %80 kapsama | **0.797** | |
| Yıllara göre kapsama | 0.774 - 0.831 | |
| Likidite beşlisi MSE oranı | 0.844 ... 0.782 | |
| Yarı genişlik (log) | 0.0871 | 0.0936 |

(Gerçek veri çalışmasında `RiskValidationConfig.max_date` ile OOS 2026-06'da kesildi; alan
2026-11'de kullanılmadığı için kaldırıldı. Aynı sayıları yeniden üretmek için paneli bu tarihte
keserek `build_risk_frame` çağırın.)

## Servis davranışı ve durumlar

- `run_risk_step` (`src/serving/nightly_risk.py`): `compute_risk_safe` + durum + kapı değişimi.
- **Durum dosyası:** `nightly_status.json` `risk: {status: ok|fallback|failed, method, error}`.
  `failed` -> `/health` `degraded`; risk_* ve bant boş kalır ama Kol-B çıktısı korunur.
- **Kapı değişimi:** önceki koşunun `config_json["risk"]["method"]` okunur; `har_x_ridge <->
  naive_fallback` değişirse `UYARI risk kapisi degisti` loglanır ve `config_json["warnings"]` /
  `risk.gate_flip` alanına yazılır (kapı mantığı değişmedi).
- **Risk tahmini olmayan semboller** (`risk_method` NULL): `risk.missing = {sembol: neden}`;
  neden `quarantine` | `short_history` | `gap`. `/analysis` Türkçe uyarı verir ve güveni bir kademe düşürür.
  Kök neden (ASELS/ASTOR/BLUME, 153 sembolden 4): K8 ölçek-kırığı karantinası sıçramadan sonra
  63 satırı panelden atar; `rv_63` penceresi (en az %80 geçerli gün) yeniden dolana dek (~63+51 işlem
  günü) özellikler NaN kalır. Eğitim aynı kuralla yapıldığı için bilinçli bırakıldı (bkz. [Data Pipeline](data-pipeline.md)).
  `gap`: tek eksik gün `rv_5`'i 5 gün NaN yapar (ör. ALVES), kendiliğinden düzelir.
- `--horizon` yalnız 5 kabul edilir (risk ve Kol-B çeyreklik tablosu 5 günlüktür).

İlgili: [Persistence and API](persistence-and-api.md), [Architecture](architecture.md), [Model Catalog](model-catalog.md).
