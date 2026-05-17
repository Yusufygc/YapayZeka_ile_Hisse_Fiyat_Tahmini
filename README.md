# ts_forecasting_lab

**BIST (Borsa İstanbul) hisseleri için üretim kalitesinde zaman serisi tahmin pipeline'ı.**

11 farklı model türünü (baseline, ağaç tabanlı, derin öğrenme, deneysel), iki doğrulama protokolünü (tek bölünme / walk-forward), finansal backtesting motorunu ve XAI açıklanabilirlik katmanını tek bir tutarlı mimaride birleştirir. Faz 0'dan Faz 5'e uzanan kapsamlı bir refactoring süreci sonucunda bu proje, deneysel bir prototipen üretime hazır bir araştırma platformuna dönüştürülmüştür.

---

## İçindekiler

1. [Neden Bu Proje?](#1-neden-bu-proje)
2. [Mimari Genel Bakış](#2-mimari-genel-bakış)
3. [Veri Akışı](#3-veri-akışı)
4. [Model Kataloğu](#4-model-kataloğu)
5. [Fintech Altyapısı](#5-fintech-altyapısı)
6. [Geliştirme Fazları — Neyi Neden Yaptık?](#6-geliştirme-fazları--neyi-neden-yaptık)
7. [Kurulum](#7-kurulum)
8. [Kullanım](#8-kullanım)
9. [Yapılandırma Referansı](#9-yapılandırma-referansı)
10. [Çıktı Yapısı](#10-çıktı-yapısı)
11. [Test Paketi](#11-test-paketi)
12. [Geliştirici Araçları](#12-geliştirici-araçları)

---

## 1. Neden Bu Proje?

BIST hisseleri, global piyasalar ile kıyaslandığında birkaç yapısal zorluk barındırır: yüksek TL enflasyonu, makro politika belirsizliği ve ince (thin) likidite yapısı. Bu ortamda basit bir tahmin modeli kolayca yanıltıcı sonuçlar üretebilir; özellikle eğer veri sızıntısı (data leakage), scalerın test seti üzerinde fit edilmesi ya da yanlış sinyal kalibrasyonu varsa.

Bu pipeline aşağıdaki sorulara cevap verir:

- Hangi model tipi (LSTM, XGBoost, Ridge, ARIMAâ€¦) belirli bir BIST hissesi için en iyi çalışır?
- Model seçimi rastgele mi, yoksa istatistiksel olarak anlamlı mı?
- Tahmin sinyalleri gerçek bir ticaret stratejisine dönüştürüldüğünde komisyon ve kayma maliyetleri dahil kârlı mı?
- Hangi özellikler (teknik indikatörler, makro değişkenler) modele en çok katkı sağlıyor?

---

## 2. Mimari Genel Bakış

Pipeline, **Facade + Strategy** tasarım deseni üzerine inşa edilmiştir. Üst katman (`ForecastingPipeline`) dış dünyaya tek bir arayüz sunarken, iş mantığı üç alt yöneticiye devredilmiştir:

```
ForecastingPipeline          â† src/pipeline/orchestrator.py  (Facade)
├── DataManager              â† src/pipeline/data_manager.py
│   ├── DataLoader/Updater   â† src/data/
│   ├── Preprocessor         â† src/data/preprocessor.py
│   ├── FeaturePipeline      â† src/features/feature_pipeline.py
│   ├── MacroPipeline        â† src/features/macro_pipeline.py
│   ├── FeatureCache         â† src/features/feature_cache.py
│   └── DataSplitter         â† src/utils/data_splitter.py
│
├── ModelTrainer             â† src/pipeline/model_trainer.py
│   ├── 11 Model Sınıfı      â† src/models/
│   ├── Ensemble Modeli      â† src/models/ensemble.py
│   └── WalkForwardCV        â† src/validation/walk_forward.py
│
└── EvaluationManager        â† src/pipeline/evaluation_manager.py
    ├── _PredictionEngineMixin  â† src/pipeline/prediction_engine.py
    ├── _BacktestRunnerMixin    â† src/pipeline/backtest_runner.py
    ├── _SignalCalibratorMixin  â† src/pipeline/signal_calibrator.py
    └── _MetricsReporterMixin   â† src/pipeline/metrics_reporter.py
```

**Destekleyici alt sistemler:**

| Dizin | Amaç |
|---|---|
| `src/data/` | BIST ve Makro verilerin indirilmesi, ön işlenmesi ve periyodik güncellenmesi |
| `src/backtesting/` | Sinyal üretimi, backtest motoru, Monte Carlo bootstrap, Kelly pozisyon boyutlandırma |
| `src/evaluation/` | Finansal metrikler, permütasyon önem testi |
| `src/xai/` | SHAP/önem açıklamaları, özellik sözlüğü, HTML/metin raporu |
| `src/database/` | `StockModelDB` — SQLite tabanlı merkezi kayıt |
| `src/experiments/` | `ExperimentTracker` — CSV tabanlı çalışma günlüğü |
| `src/model_registry/` | `ModelRegistry` — `registry.json` versiyonlama |
| `src/utils/` | Veri bölücü, tekrar üretilebilirlik tohumları, dinamik risksiz oran |
| `src/api/` | FastAPI HTTP servis katmanı |

---

## 3. Veri Akışı

Pipelinedaki her adımın neden bu sırayla yapıldığı, veri sızıntısını önlemek açısından kritiktir:

```
1. CSV Yükle
   └─ Türkçe sütun adları â†’ İngilizce eşle
   └─ Sıfır hacimli satırlar düşürülür (işlem görmeyen günler)

2. Özellik Mühendisliği
   ├─ FeaturePipeline: 20+ teknik indikatör
   │   (SMA, EMA, RSI, MACD, Bollinger Bantları, ATR, OBV, vb.)
   └─ MacroPipeline: Makro bağlam özellikleri
       (USD/TRY, EUR/TRY, BIST100, VIX, Altın, Brent, DXY, ABD 10Y Faiz,
        TCMB Faiz, CPI — yfinance + FRED API)

3. Chronological Train/Test Split  â† DataSplitter
   └─ Kesinlikle zaman sırasına göre; karıştırma yok, sızıntı yok

4. Ölçekleme  â† Scaler YALNIZCA eğitim setine fit edilir
   ├─ X: RobustScaler (aykırı değerlere dayanıklı)
   └─ y: StandardScaler + klipleme (log-getiri hedefi için)

5. 3D Diziler Oluştur  â† LSTM/sequence modeller için
   └─ [örnekler, TIME_STEPS=30, özellik_sayısı]

6. Model Eğitimi
   ├─ Seçilen modeller eğitilir
   └─ Walk-forward: her fold bağımsız eğitim + tahmin döngüsü

7. Ters Dönüşüm
   └─ Log-getiri tahminleri â†’ Fiyat tahminleri

8. Değerlendirme
   ├─ Yön Doğruluğu, RMSE, Sharpe yaklaşımı, Hit Rate
   ├─ Backtest: Sinyal â†’ Komisyon+Kayma â†’ P&L simülasyonu
   ├─ Monte Carlo Bootstrap (sinyalin şansa karşı anlamlılığı)
   └─ XAI: Özellik önemi + HTML raporu

9. Kayıt
   ├─ Model dosyası (.pkl / .keras / .pt)
   ├─ stock_models.db (SQLite)
   ├─ registry.json
   └─ CSV deney günlüğü
```

---

## 4. Model Kataloğu

Tüm modeller `BaseModel` arayüzünden türer ve `train()`, `predict()`, `save()`, `load()` metodlarını zorunlu olarak uygular.

### Baseline / Referans Modeller

| Sınıf | Dosya | Açıklama |
|---|---|---|
| `NaiveLastValueModel` | `naive_model.py` | Son gözlemlenen değeri tekrarlar. Minimum referans noktası. |
| `NaiveZeroReturnModel` | `naive_model.py` | Her zaman sıfır getiri tahmin eder. "Hiç hareket olmayacak" hipotezi. |
| `NaiveDriftModel` | `naive_model.py` | Lineer trend ekstrapolasyonu. Basit momentum testi. |
| `ARIMAModel` | `arima_model.py` | Yapılandırılabilir (p,d,q) düzeni; auto_order desteği. |
| `ProphetModel` | `prophet_model.py` | Meta Prophet; yalnızca kapanış fiyatı, regresör desteğiyle. |

**Neden?** Baseline modeller olmadan, karmaşık modellerin gerçekten değer katıp katmadığını anlayamazsınız. RMSE sayısal olarak iyi görünse de eğer bir Naive model aynı sonucu veriyorsa, karmaşık modelin fazladan maliyeti (hesaplama, bakım) gereksizdir.

### Ağaç Tabanlı Modeller

| Sınıf | Dosya | Açıklama |
|---|---|---|
| `XGBoostModel` | `xgboost_model.py` | Gradient boosting; Optuna HPO + SQLite warm-start |
| `RandomForestModel` | `random_forest_model.py` | Bagging ensemble; Optuna HPO + SQLite warm-start |
| `LightGBMReturnModel` | `gradient_boosting_model.py` | Opsiyonel; yoksa sessizce atlanır |

### Lineer / Regularize Modeller

| Sınıf | Dosya | Açıklama |
|---|---|---|
| `RidgeReturnModel` | `linear_model.py` | L2 cezası; log-getiri üzerinde |
| `ElasticNetReturnModel` | `linear_model.py` | L1+L2 karışımı; seyrek özellik seçimi |

**Neden?** Lineer modeller, tahmin alanının dışında kalan özellikler olduğunda sağlam kalır. Ayrıca "ne kadar doğrusal olmayan bir ilişki var?" sorusunun cevabını dolaylı olarak verir.

### Derin Öğrenme Modelleri

| Sınıf | Dosya | Açıklama |
|---|---|---|
| `LSTMModel` / `AttentionLSTMModel` | `lstm_model.py` | Çift yönlü LSTM + dikkat mekanizması (Keras/TensorFlow) |

**Neden LSTM'de `clipnorm=1.0`?** BIST verisi, yüksek volatilite dönemlerinde (seçimler, kur krizleri) ani gradient patlamalarına yol açabilir. `Adam(clipnorm=1.0)` ile gradients normalize edilerek eğitim kararsızlığı önlenir.


### Topluluk (Ensemble) Modelleri

| Sınıf | Dosya | Açıklama |
|---|---|---|
| `EnsembleModel` | `ensemble.py` | L2 (Ridge) regresyonu kullanarak diğer modellerin tahminlerini birleştirir |

**Neden Ensemble?** Tek bir modelin (örneğin XGBoost) iyi çalıştığı piyasa koşulları ile LSTM'in iyi çalıştığı koşullar farklı olabilir. `EnsembleModel`, alt modellerin geçmiş tahmin performanslarına göre (meta-öğrenme) dinamik ağırlıklar belirler. Tek bir modele bağımlı kalma riskini (model risk) azaltır ve getiriyi pürüzsüzleştirir.

### Deneysel Sequence Baseline'ları

| Sınıf | Dosya | Açıklama |
|---|---|---|
| `DLinearSequenceModel` | `linear_sequence_model.py` | 3D diziler üzerinde hafif lineer |
| `NLinearSequenceModel` | `linear_sequence_model.py` | Normalize edilmiş lineer (son değer çıkarılır) |

---

## 5. Fintech Altyapısı

### 5.1 Sinyal Üretimi

Üç sinyal modu desteklenir:

**`simple` (varsayılan):** Maliyetsiz long/flat AL/SAT/TUT modu. Beklenen getiri
`buy_threshold` üstündeyse ve pozisyon yoksa AL, beklenen getiri
`-sell_threshold` altındaysa ve pozisyon varsa SAT, diğer durumlarda TUT
üretilir. Varsayılan eşikler `0.0` ve komisyon/slippage `0.0`'dır.

**`professional` (opt-in araştırma modu):** Yön doğruluğu, kalite kapıları,
volatilite, holding period, take-profit ve stop-loss mantığı devreye girer.

**`legacy`:** Tarihsel direction-only long/flat karşılaştırma modu.

### 5.2 Backtest Motoru

`src/backtesting/engine.py` — `run_backtest()` fonksiyonu:

- Sinyalden long/flat pozisyon durumunu hesaplar
- Varsayılan basit modda komisyon (`commission_bps`) ve kayma (`slippage_bps`) 0 kalır
- İstenirse non-zero komisyon/kayma parametreleriyle maliyetli senaryo çalıştırabilir
- Günlük P&L, drawdown ve kümülatif getiri dizisi üretir

**Neden BPS (baz puan)?** BIST'te işlem komisyonları küçük görünse de yüksek frekanslı stratejilerde kümülatif maliyetler getiriyi ciddi erozya yaratır. BPS cinsinden parametrik tanım, gerçekçi simülasyon sağlar.

### 5.3 Gelişmiş Backtest Metrikleri

Standart Sharpe Ratio ve Max Drawdown'ın ötesinde:

| Metrik | Açıklama | Neden Önemli? |
|---|---|---|
| **Omega Ratio** | Kazanç/kayıp oranı eşik üzerinde | Sharpe'dan farklı olarak getiri dağılımının şeklini dikkate alır |
| **Recovery Factor** | Net getiri / Max drawdown | Kayıpları ne kadar hızlı telafi ettiğini gösterir |
| **Max Consecutive Loss** | Üst üste maksimum kayıp sayısı | Psikolojik dayanma kapasitesi için kritik |
| **Information Ratio** | Aktif getiri / Takip hatası (annualize) | Benchmark'a karşı tutarlı üstünlük ölçüsü |

### 5.4 Kaldirilan Arastirma Yardimcilari

Monte Carlo bootstrap, Kelly pozisyon boyutlandirma ve bagimsiz permutation importance helper modulleri aktif urun kapsamindan cikarildi. Varsayilan sistem kaldiracsiz long/flat AL/SAT/TUT sinyali uretir; pozisyon boyutlandirma ve arastirma simulatörleri runtime pipeline parçası değildir.

### 5.5 Dinamik Risksiz Oran

`src/utils/risk_free_rate.py` — `get_current_risk_free_rate()`:

Öncelik sırası:
1. `RISK_FREE_RATE_ANNUAL` ortam değişkeni
2. `data/macro/INTEREST_RATE.csv` son satırı (gerçek TCMB faizi)
3. Fallback: `0.40` (%40 — yüksek faiz dönemine uygun güvenli varsayılan)

**Neden?** Sharpe Ratio hesabında kullanılan risksiz oran, Türkiye bağlamında son derece önemlidir. ABD'de %5 olan risksiz oran, Türkiye'de %50'yi aşabilmektedir. Sabit bir değer yerine gerçek TCMB verisini okumak, Sharpe hesabını anlamlı kılar.

### 5.7 Sinyal Kalibrasyon Kilit Mekanizması

`calibration_scope = "wf_train"` (değiştirilemez):

Sinyal eşikleri yalnızca walk-forward fold eğitim verisi üzerinde kalibre edilir. Final holdout verisi kalibrasyon sürecine **asla** dahil edilemez. Bu kural `_assert_wf_train_scope()` metoduyla çalışma zamanında zorlanır; ihlal halinde `RuntimeError` fırlatılır.

**Neden bu kadar katı?** Sinyal eşiklerini test setine göre ayarlamak "look-ahead bias" (gelecek bilgisi sızması) yaratır ve gerçek dünya performansını abartır. Bu, akademik literatürde sık rastlanan ve fark edilmesi zor bir yanılgıdır.

---

## 6. Geliştirme Fazları — Neyi Neden Yaptık?

Bu proje, deneysel bir prototipten başlayarak altı faz boyunca sistematik olarak geliştirilmiştir. Her faz, belirli bir teknik borcu ya da eksikliği gidermeye odaklanmıştır.

### Faz 0 — Temel Altyapı Sağlamlaştırma

**Problem:** Pipeline büyüdükçe `DataSplitter`'dan kaçan veri sızıntısı riskleri, scaler'ın test seti üzerinde fit edilmesi ve tekrar üretilebilir sonuçların garanti edilmemesi gibi temel hatalar ortaya çıktı.

**Yapılanlar:**
- `DataSplitter`: Kronolojik bölünme garantisi; shuffle yok, stratified split yok.
- `reproducibility.py`: Python, NumPy, TensorFlow ve PyTorch tohumları tek yerden ayarlanır.
- `ExperimentTracker`: Her çalışmanın parametreleri ve metrikleri CSV'ye kaydedilir; deney takibi mümkün hale gelir.

**Bu işe ne yarar?** Aynı konfigürasyonu iki kez çalıştırdığınızda aynı sonucu alırsınız. Karşılaştırmalı analizler güvenilir olur.

### Faz 1 — Model Ekosistemi Genişletme

**Problem:** Proje başlangıçta yalnızca birkaç model destekliyordu. "Hangi model en iyi?" sorusuna yanıt verebilmek için geniş bir model ekosistemi gerekiyordu.

**Yapılanlar:**
- Lineer modeller eklendi: `RidgeReturnModel`, `ElasticNetReturnModel`
- Deneysel sequence baseline'ları: `DLinearSequenceModel`, `NLinearSequenceModel`
- Tüm modeller `BaseModel` arayüzüne bağlandı; herhangi bir model eklenebilir/çıkarılabilir.
- Opsiyonel bağımlılıklar sessizce atlanır (LightGBM, Prophet, TF, PyTorch olmadığında pipeline çalışmaya devam eder).

**Bu işe ne yarar?** Model seçimi veri güdümlü hale gelir. Hisse başına en iyi model otomatik olarak belirlenir.

### Faz 2 — God Object Yıkımı ve Mimari Temizlik

**Problem:** `EvaluationManager` tek bir dosyada 1500+ satıra ulaştı. Test edilemez, bakımı zor, yeni özellik eklemek her seferinde tüm sınıfı riske atıyordu.

**Yapılanlar:**
- `EvaluationManager` dört mixin'e ayrıldı:
  - `_PredictionEngineMixin` — tahmin üretimi
  - `_BacktestRunnerMixin` — backtest orchestration
  - `_SignalCalibratorMixin` — sinyal kalibrasyon mantığı
  - `_MetricsReporterMixin` — metrik raporlama ve kayıt
- `TypedDict` dönüş tipleri (`SingleSplitResult`, `WalkForwardResult`, `FinalHoldoutResult`) eklendi.
- `PipelineConfig` dataclass hiyerarşisi: `DataConfig`, `ValidationConfig`, `ModelConfig`, `ExecutionConfig`.

**Bu işe ne yarar?** Her mixin bağımsız test edilebilir. Yeni bir özellik eklemek tek bir mixin'i etkiler, diğerlerini değil. Tip güvenliği sayesinde IDE otomatik tamamlama çalışır, hata ayıklama kolaylaşır.

### Faz 3 — ML Çekirdeği Güçlendirme

**Problem:** Hiperparametre optimizasyonu her seferinde sıfırdan başlıyordu, LSTM gradientleri patlamalar yaşıyordu ve makro özellikler sınırlıydı.

**Yapılanlar:**

**Optuna Warm-Start (XGBoost ve RandomForest):**
- Her hisse icin ayri SQLite Optuna veritabani (`data/optuna/optuna_studies_{SYMBOL}.db`)
- `load_if_exists=True` ile önceki denemeler birikir; her çalışmada daha iyi başlangıç noktasından devam edilir.
- **Neden?** XGBoost için 100 deneme standart olsa da gerçek projede zaman kısıtları vardır. Warm-start ile 10 deneme bile önceki 100 denemenin üzerine inşa eder.

**LSTM Gradient Klipleme:**
- `Adam(learning_rate=lr, clipnorm=1.0)` her iki LSTM varyantında da.
- **Neden?** BIST'te TL değer kaybı dönemlerinde log-getiri serisi ani sıçramalar yapar. Bu sıçramalar gradient vektörünü patlatabilir ve eğitimi çöküşe uğratır.

**Makro Özellik Genişletmesi:**
- Yeni tickers: EUR/TRY, VIX (korku endeksi), Altın (USD), Brent Petrol, DXY (USD Endeksi), ABD 10Y Getirisi
- Türev özellikler: `EURTRY_Return`, `VIX_Level`, `VIX_Change`, `Gold_TRY_Return`, `Oil_USD_Return`, `DXY_Return`, `US10Y_Level`, `US10Y_Change`
- **Neden?** BIST küresel risk iştahından yoğun şekilde etkilenir. VIX yükseldiğinde BIST genellikle düşer. DXY güçlendiğinde TL baskı altında kalır, bu da ihracatçı hisseler için önemlidir.

**Feature Cache:**
- Pickle tabanlı, MD5 anahtarlı, 24 saatlik TTL
- **Neden?** Makro veri çekme + özellik mühendisliği birlikte 30-60 saniye sürebilir. Cache ile tekrarlı çalışmalar 2-3 saniyeye iner.

### Faz 4 — Fintech Altyapısı

**Problem:** "Model iyi tahmin ediyor" ile "strateji gerçekten para kazandırıyor" arasındaki boşluğu kapatmak gerekiyordu. Sharpe Ratio tek başına yeterli değildi.

**Yapılanlar:**
- Dinamik Risksiz Oran (`src/utils/risk_free_rate.py`)
- Gelişmiş Backtest Metrikleri: Omega Ratio, Recovery Factor, Max Consecutive Loss, Information Ratio
- Sinyal kalibrasyon kilit mekanizması (`calibration_scope`, `_assert_wf_train_scope`)

**Bu işe ne yarar?** Bir stratejiyi değerlendirirken artık şu soruların tamamına yanıt alınabilir:
- Åansın ötesinde performans var mı? (Monte Carlo p-değeri)
- Ne kadar sermaye riske girmeli? (Kelly)
- Kayıp dönemlerinden ne kadar hızlı çıkılıyor? (Recovery Factor)
- Hangi özellikler gerçekten katkı sağlıyor? (Permütasyon önemi)

### Faz 5 — Üretim Hazırlığı

**Problem:** Proje tek hisse, interaktif CLI ile sınırlıydı. Çok hisseli otomasyona ve dış uygulama entegrasyonuna ihtiyaç duyuldu.

**Yapılanlar:**

**Linter ve Kod Kalitesi (`pyproject.toml`, `.flake8`, `.pre-commit-config.yaml`):**
- Black (format), isort (import sıralaması), flake8 (stil denetimi)
- Pre-commit hook'ları: her commit öncesinde otomatik kontrol
- **Neden?** Büyüyen bir kod tabanında tutarsız stil, kod incelemelerini zorlaştırır ve merge çakışmalarına yol açar. Otomatik formatlama bu tartışmayı ortadan kaldırır.

**GitHub Actions CI (`.github/workflows/ci.yml`):**
- Python 3.10 + 3.11 matrix
- Kritik flake8 hataları CI'ı durdurur; stil uyarıları raporlanır ama durdurmaz
- Smoke testleri her push'ta çalışır; tam test paketi çalışır ama başarısız olsa CI geçer (`|| true`)
- FastAPI import kontrolü
- **Neden?** Her push'ta otomatik doğrulama, bozuk kodu ana dala karıştırmaktan korur. Python 3.10/3.11 matrix, ileriye uyumluluğu garanti eder.

**Multi-Stock Batch Modu (`python -m src.cli.batch`):**
- `--stocks TUPRS,ASELS,THYAO` veya `--universe data/bist_universe.csv`
- `--workers N` ile `ProcessPoolExecutor` paralel çalışma
- `--dry-run` modu: veri varlığını kontrol eder, pipeline çalıştırmaz
- `batch_summary_{timestamp}.csv + .json` çıktı özeti
- **Neden?** BIST'te 50+ hisseyi tek tek çalıştırmak pratik değildir. Batch mod ile tüm universe bir gecede çalıştırılabilir.

**FastAPI Servis Katmanı (`src/api/main.py`):**
- `GET /best-model/{symbol}` — Hisse için en iyi model
- `GET /leaderboard` — Tüm hisseler lider tablosu
- `GET /metrics/{symbol}` — Model karşılaştırma
- `POST /run/{symbol}` — Pipeline'ı arka planda tetikle
- `GET /run/status/{job_id}` — İş durumu sorgulama
- **Neden?** `Merge_PortfoySim` gibi portföy simülasyon uygulamaları, tahmin sonuçlarını HTTP üzerinden sorgulayabilir. Böylece bu pipeline, daha büyük bir sistemin bağımsız mikroservisi haline gelir.

### Faz 6 — Modülerizasyon ve Temiz Mimari (İleri Mimariler)

**Problem:** Veri yükleme, ön işleme gibi araçların kök dizine yayılması kod okunabilirliğini zorlaştırıyordu.

**Yapılanlar:**
- **Veri Araçları Ayrıştırıldı:** `data_loader.py` ve `preprocessor.py` gibi araçlar `src/data/` modülüne taşınarak sorumluluklar netleştirildi.
- **Ensemble (Topluluk) Modeli:** En iyi modellerin tahminlerini Ridge regresyonu ile dinamik ağırlıklandıran Ensemble yapısı `src/models/ensemble.py` olarak standart modele dahil edildi.

**Bu işe ne yarar?** Sistemin bakım maliyeti (maintenance cost) ciddi oranda düştü.

---

## 7. Kurulum

### Gereksinimler

- Python 3.10+
- (Opsiyonel) CUDA 12.8 uyumlu GPU (LSTM hızlandırma için)

### Temel Kurulum

```bash
# Bağımlılıkları kur
pip install -r requirements.txt

# Opsiyonel: LightGBM
pip install lightgbm

# FastAPI servisi için:
pip install fastapi uvicorn
```

### Conda (dl_env önerilen)

```bash
conda activate dl_env
pip install -r requirements.txt
pip install fastapi uvicorn
```

### Pre-commit Hook'larını Etkinleştir

```bash
pip install pre-commit
pre-commit install
```

---

## 8. Kullanım

### 8.1 Tek Hisse — İnteraktif CLI

```bash
python -m src.cli.interactive
```

Pipeline başladığında sırasıyla sorar:
1. Hisse kodu (TUPRS, ASELS, THYAO, vb.)
2. Doğrulama modu: `single_split` veya `walk_forward`
3. Model seçimi: tümü veya belirli modeller

### 8.2 Multi-Stock Batch Modu

```bash
# Üç hisse, walk-forward, 2 paralel worker:
python -m src.cli.batch \
    --stocks TUPRS,ASELS,THYAO \
    --mode walk_forward \
    --workers 2

# Tüm BIST universe, 4 worker:
python -m src.cli.batch \
    --universe data/bist_universe.csv \
    --mode walk_forward \
    --workers 4

# Sadece belirli modeller:
python -m src.cli.batch \
    --stocks TUPRS,EREGL \
    --models XGBoost,Ridge,LSTM

# Kuru çalışma (veri kontrol et, pipeline çalıştırma):
python -m src.cli.batch \
    --universe data/bist_universe.csv \
    --dry-run
```

### 8.3 FastAPI Servisi

```bash
# Proje kökünden çalıştır:
uvicorn src.api.main:app --reload --port 8000

# Etkileşimli API dökümantasyonu:
# http://localhost:8000/docs       (Swagger UI)
# http://localhost:8000/redoc      (ReDoc)
```

Örnek sorgular:

```bash
# En iyi model:
curl http://localhost:8000/best-model/TUPRS

# Lider tablosu:
curl http://localhost:8000/leaderboard

# Pipeline tetikle:
curl -X POST http://localhost:8000/run/ASELS \
     -H "Content-Type: application/json" \
     -d '{"mode": "walk_forward", "models": ["XGBoost", "Ridge"]}'

# İş durumu sorgula:
curl http://localhost:8000/run/status/{job_id}
```

### 8.4 Testler

```bash
# Tüm testler:
python -m pytest tests/

# Smoke testleri (hızlı):
python -m pytest tests/test_smoke.py -v

# Belirli test modülü:
python -m pytest tests/test_leakage_guards.py -v
```

### 8.5 Operasyonel Süreçler: Veri Güncelleme

Günlük piyasa kapanışlarından sonra modelin güncel verilerle çalışması için veri setinin güncellenmesi gerekir. Bu işlem `src/data/data_updater.py` kullanılarak veya `python -m src.cli.batch` üzerinden yapılabilir.

```bash
# Sadece veri setini günceller (Model çalıştırmaz)
python -m src.data.data_updater --symbols TUPRS,ASELS

# Tüm BIST evreninin verisini günceller
python -m src.data.data_updater --universe data/bist_universe.csv

# Batch mod ile hem veriyi güncelle hem de tahmin üret
# (Veri eksikse veya eskiyle otomatik olarak Yahoo Finance üzerinden tamamlanır)
python -m src.cli.batch --universe data/bist_universe.csv --mode single_split
```

> **Not:** Makro veriler (BIST100, USDTRY vb.) pipeline çalıştığında `MacroPipeline` veya `data_updater` aracılığıyla otomatik olarak FRED ve Yahoo Finance üzerinden güncellenir.

---

## 9. Yapılandırma Referansı

Tüm konfigürasyon `PipelineConfig` dataclass hiyerarşisi üzerinden yönetilir (`src/pipeline/config.py`):

### DataConfig — Veri ve Özellik Ayarları

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `data_file` | — | Hisse CSV dosyası yolu |
| `test_ratio` | `0.20` | Test seti oranı (kronolojik) |
| `time_steps` | `30` | LSTM için dizi uzunluğu |
| `target_mode` | `log_return` | Hedef: log-getiri (tercih) veya fiyat |
| `feature_mode` | `stationary_features` | Özellik seti tipi |
| `scaling_mode` | `robust_x_standard_y_clip` | Ölçekleme stratejisi |
| `use_macro` | `True` | Makro özellikler dahil edilsin mi? |
| `training_window_years` | `5` | Pencere seçimi için aday yıllar |

### ValidationConfig — Doğrulama Protokolü

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `validation_mode` | `single_split` | `single_split` veya `walk_forward` |
| `wf_n_splits` | `12` | Walk-forward fold sayısı |
| `wf_test_size` | `21` | Her fold test boyutu (gün/bar) |
| `wf_max_train_size` | `756` | Kayan pencere maksimum eğitim boyutu |
| `final_holdout_size` | `60` | Son dokunulmamış test seti boyutu |

### ModelConfig — Model Seçimi

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `selected_models` | `None` (tümü) | Çalıştırılacak modeller listesi |
| `ensemble_enabled` | `True` | Ensemble modeller oluşturulsun mu? |

### ExecutionConfig — Backtest ve Sinyal

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `backtest_enabled` | `True` | Backtest çalıştırılsın mı? |
| `initial_capital` | `100,000` | Başlangıç sermayesi (TL) |
| `commission_bps` | `0.0` | Varsayılan basit modda komisyon kapalı |
| `slippage_bps` | `0.0` | Varsayılan basit modda kayma maliyeti kapalı |
| `signal_mode` | `simple` | `simple`, `professional` veya `legacy` |
| `calibration_scope` | `wf_train` | **Değiştirme:** Sızıntı koruma kilidi |

---

## 10. Çıktı Yapısı

```
outputs/{SYMBOL}/
├── models/
│   ├── xgboost_model.pkl
│   ├── lstm_model.keras
│   └── {model}_final_holdout_model.{ext}
├── experiments/
│   └── experiment_log_{timestamp}.csv
├── xai/
│   ├── xai_report_{model}.html
│   └── xai_report_{model}.txt
└── registry.json                         â† Model versiyonlama metadata

data/stock_models.db                     â† Merkezi SQLite
data/optuna/optuna_studies_{SYMBOL}.db   â† Optuna warm-start (gitignore)
data/feature_cache/                       â† Pickle cache (gitignore)
data/macro/*.csv                          â† İndirilen makro veri (gitignore)
outputs/batch_summaries/*.csv/.json      â† Batch calisma ozeti (gitignore)
```

---

## 11. Test Paketi

| Dosya | Kapsam |
|---|---|
| `test_smoke.py` | Temel import ve başlatma kontrolleri |
| `test_leakage_guards.py` | Veri sızıntısı önleme doğrulaması |
| `test_phase4_models.py` | Baseline + yeni model sınıfları |
| `test_reporting_metrics.py` | Metrik hesaplama doğruluğu |
| `test_validation_protocol.py` | Walk-forward sıralama değişmezleri |
| `test_phase5_data_quality.py` | Veri kalite kontrolleri |
| `test_phase6_backtest_standard.py` | Backtest motor standartları |
| `test_phase7_acceptance.py` | Sistem kabul testleri |
| `test_phase8_acceptance.py` | Üretim kabulü |

---

## 12. Geliştirici Araçları

### Kod Kalitesi

```bash
# Formatlama (Black):
black src/ tests/ --line-length 100

# Import sıralaması (isort):
isort src/ tests/

# Stil denetimi (flake8):
flake8 src/ tests/ --max-line-length 100

# Tip denetimi (mypy):
mypy src/
```

### CI/CD

GitHub Actions otomatik olarak her `push` ve `pull_request`'te çalışır:
- **Python 3.10 + 3.11** — çapraz versiyon uyumluluğu
- **Kritik flake8 hataları** (E9, F63, F7, F82) CI'ı durdurur
- **Smoke testleri** — her push'ta hızlı doğrulama
- **FastAPI import kontrolü** — API katmanı bozulmuşsa erken uyarı

### Ortam Değişkenleri

| Değişken | Açıklama |
|---|---|
| `RISK_FREE_RATE_ANNUAL` | Sharpe hesabı için risksiz oran geçersiz kıl (örn: `0.45`) |

---

## 13. Geliştirici Rehberi: Yeni Model Ekleme

Sisteme kendi algoritmanızı (örneğin yeni bir PyTorch tabanlı Transformer veya özel bir istatistiksel model) eklemek için aşağıdaki 3 adımı izlemeniz yeterlidir:

1. **`BaseModel`'den Türetin:** `src/models/` dizini altında yeni bir dosya oluşturun ve sınıfınızı `BaseModel` arayüzünden (`src.models.base_model.BaseModel`) türetin.
2. **Metodları İmplemente Edin:** Zorunlu olan `train()`, `predict()`, `save()`, ve `load()` metodlarını doldurun.
3. **Kayıt İşlemi:** Modelinizi `src/model_registry/model_registry.py` içindeki sözlüğe veya kullanacağınız `ModelConfig` sınıfına tanıtın.

---

## 14. Sorun Giderme (Troubleshooting)

| Sorun | Neden ve Çözüm |
|---|---|
| **yfinance 429 Too Many Requests** | *Neden:* Çok fazla hisse verisi aynı anda istendi. *Çözüm:* `python -m src.cli.batch` içindeki `--workers` sayısını düşürün veya farklı bir IP adresi/VPN kullanın. |
| **CUDA OOM (Out of Memory)** | *Neden:* LSTM eğitilirken ekran kartı belleği doldu. *Çözüm:* `PipelineConfig` altındaki `time_steps` değerini küçültün veya model batch size değerini azaltın. |
| **Missing Macro Data (FRED/Yahoo)** | *Neden:* İnternet bağlantısı koptu veya API değişti. *Çözüm:* `src/data/data_updater.py`'yi tekrar çalıştırın; veriler önbellekten okunmak yerine sıfırdan indirilir. |
| **SQLite database is locked** | *Neden:* Optuna warm-start kullanırken çok fazla worker aynı DB dosyasına yazmaya çalıştı. *Çözüm:* Worker sayısını düşürün (`--workers 2` veya `1`). |

---

## Lisans

Bu proje, `Merge_PortfoySim` portföy simülasyon sisteminin araştırma ve geliştirme bileşenidir.


