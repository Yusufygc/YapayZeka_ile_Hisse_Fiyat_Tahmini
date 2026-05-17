# ts_forecasting_lab

**BIST (Borsa Ä°stanbul) hisseleri iÃ§in Ã¼retim kalitesinde zaman serisi tahmin pipeline'Ä±.**

11 farklÄ± model tÃ¼rÃ¼nÃ¼ (baseline, aÄŸaÃ§ tabanlÄ±, derin Ã¶ÄŸrenme, deneysel), iki doÄŸrulama protokolÃ¼nÃ¼ (tek bÃ¶lÃ¼nme / walk-forward), finansal backtesting motorunu ve XAI aÃ§Ä±klanabilirlik katmanÄ±nÄ± tek bir tutarlÄ± mimaride birleÅŸtirir. Faz 0'dan Faz 5'e uzanan kapsamlÄ± bir refactoring sÃ¼reci sonucunda bu proje, deneysel bir prototipen Ã¼retime hazÄ±r bir araÅŸtÄ±rma platformuna dÃ¶nÃ¼ÅŸtÃ¼rÃ¼lmÃ¼ÅŸtÃ¼r.

---

## Ä°Ã§indekiler

1. [Neden Bu Proje?](#1-neden-bu-proje)
2. [Mimari Genel BakÄ±ÅŸ](#2-mimari-genel-bakÄ±ÅŸ)
3. [Veri AkÄ±ÅŸÄ±](#3-veri-akÄ±ÅŸÄ±)
4. [Model KataloÄŸu](#4-model-kataloÄŸu)
5. [Fintech AltyapÄ±sÄ±](#5-fintech-altyapÄ±sÄ±)
6. [GeliÅŸtirme FazlarÄ± â€” Neyi Neden YaptÄ±k?](#6-geliÅŸtirme-fazlarÄ±--neyi-neden-yaptÄ±k)
7. [Kurulum](#7-kurulum)
8. [KullanÄ±m](#8-kullanÄ±m)
9. [YapÄ±landÄ±rma ReferansÄ±](#9-yapÄ±landÄ±rma-referansÄ±)
10. [Ã‡Ä±ktÄ± YapÄ±sÄ±](#10-Ã§Ä±ktÄ±-yapÄ±sÄ±)
11. [Test Paketi](#11-test-paketi)
12. [GeliÅŸtirici AraÃ§larÄ±](#12-geliÅŸtirici-araÃ§larÄ±)

---

## 1. Neden Bu Proje?

BIST hisseleri, global piyasalar ile kÄ±yaslandÄ±ÄŸÄ±nda birkaÃ§ yapÄ±sal zorluk barÄ±ndÄ±rÄ±r: yÃ¼ksek TL enflasyonu, makro politika belirsizliÄŸi ve ince (thin) likidite yapÄ±sÄ±. Bu ortamda basit bir tahmin modeli kolayca yanÄ±ltÄ±cÄ± sonuÃ§lar Ã¼retebilir; Ã¶zellikle eÄŸer veri sÄ±zÄ±ntÄ±sÄ± (data leakage), scalerÄ±n test seti Ã¼zerinde fit edilmesi ya da yanlÄ±ÅŸ sinyal kalibrasyonu varsa.

Bu pipeline aÅŸaÄŸÄ±daki sorulara cevap verir:

- Hangi model tipi (LSTM, XGBoost, Ridge, ARIMAâ€¦) belirli bir BIST hissesi iÃ§in en iyi Ã§alÄ±ÅŸÄ±r?
- Model seÃ§imi rastgele mi, yoksa istatistiksel olarak anlamlÄ± mÄ±?
- Tahmin sinyalleri gerÃ§ek bir ticaret stratejisine dÃ¶nÃ¼ÅŸtÃ¼rÃ¼ldÃ¼ÄŸÃ¼nde komisyon ve kayma maliyetleri dahil kÃ¢rlÄ± mÄ±?
- Hangi Ã¶zellikler (teknik indikatÃ¶rler, makro deÄŸiÅŸkenler) modele en Ã§ok katkÄ± saÄŸlÄ±yor?

---

## 2. Mimari Genel BakÄ±ÅŸ

Pipeline, **Facade + Strategy** tasarÄ±m deseni Ã¼zerine inÅŸa edilmiÅŸtir. Ãœst katman (`ForecastingPipeline`) dÄ±ÅŸ dÃ¼nyaya tek bir arayÃ¼z sunarken, iÅŸ mantÄ±ÄŸÄ± Ã¼Ã§ alt yÃ¶neticiye devredilmiÅŸtir:

```
ForecastingPipeline          â† src/pipeline/orchestrator.py  (Facade)
â”œâ”€â”€ DataManager              â† src/pipeline/data_manager.py
â”‚   â”œâ”€â”€ DataLoader/Updater   â† src/data/
â”‚   â”œâ”€â”€ Preprocessor         â† src/data/preprocessor.py
â”‚   â”œâ”€â”€ FeaturePipeline      â† src/features/feature_pipeline.py
â”‚   â”œâ”€â”€ MacroPipeline        â† src/features/macro_pipeline.py
â”‚   â”œâ”€â”€ FeatureCache         â† src/features/feature_cache.py
â”‚   â””â”€â”€ DataSplitter         â† src/utils/data_splitter.py
â”‚
â”œâ”€â”€ ModelTrainer             â† src/pipeline/model_trainer.py
â”‚   â”œâ”€â”€ 11 Model SÄ±nÄ±fÄ±      â† src/models/
â”‚   â”œâ”€â”€ TFT v2 Mimarisi      â† src/models/tft_v2/
â”‚   â”œâ”€â”€ Ensemble Modeli      â† src/models/ensemble.py
â”‚   â””â”€â”€ WalkForwardCV        â† src/validation/walk_forward.py
â”‚
â””â”€â”€ EvaluationManager        â† src/pipeline/evaluation_manager.py
    â”œâ”€â”€ _PredictionEngineMixin  â† src/pipeline/prediction_engine.py
    â”œâ”€â”€ _BacktestRunnerMixin    â† src/pipeline/backtest_runner.py
    â”œâ”€â”€ _SignalCalibratorMixin  â† src/pipeline/signal_calibrator.py
    â””â”€â”€ _MetricsReporterMixin   â† src/pipeline/metrics_reporter.py
```

**Destekleyici alt sistemler:**

| Dizin | AmaÃ§ |
|---|---|
| `src/data/` | BIST ve Makro verilerin indirilmesi, Ã¶n iÅŸlenmesi ve periyodik gÃ¼ncellenmesi |
| `src/backtesting/` | Sinyal Ã¼retimi, backtest motoru, Monte Carlo bootstrap, Kelly pozisyon boyutlandÄ±rma |
| `src/evaluation/` | Finansal metrikler, permÃ¼tasyon Ã¶nem testi |
| `src/xai/` | SHAP/Ã¶nem aÃ§Ä±klamalarÄ±, Ã¶zellik sÃ¶zlÃ¼ÄŸÃ¼, HTML/metin raporu |
| `src/database/` | `StockModelDB` â€” SQLite tabanlÄ± merkezi kayÄ±t |
| `src/experiments/` | `ExperimentTracker` â€” CSV tabanlÄ± Ã§alÄ±ÅŸma gÃ¼nlÃ¼ÄŸÃ¼ |
| `src/model_registry/` | `ModelRegistry` â€” `registry.json` versiyonlama |
| `src/utils/` | Veri bÃ¶lÃ¼cÃ¼, tekrar Ã¼retilebilirlik tohumlarÄ±, dinamik risksiz oran |
| `src/api/` | FastAPI HTTP servis katmanÄ± |

---

## 3. Veri AkÄ±ÅŸÄ±

Pipelinedaki her adÄ±mÄ±n neden bu sÄ±rayla yapÄ±ldÄ±ÄŸÄ±, veri sÄ±zÄ±ntÄ±sÄ±nÄ± Ã¶nlemek aÃ§Ä±sÄ±ndan kritiktir:

```
1. CSV YÃ¼kle
   â””â”€ TÃ¼rkÃ§e sÃ¼tun adlarÄ± â†’ Ä°ngilizce eÅŸle
   â””â”€ SÄ±fÄ±r hacimli satÄ±rlar dÃ¼ÅŸÃ¼rÃ¼lÃ¼r (iÅŸlem gÃ¶rmeyen gÃ¼nler)

2. Ã–zellik MÃ¼hendisliÄŸi
   â”œâ”€ FeaturePipeline: 20+ teknik indikatÃ¶r
   â”‚   (SMA, EMA, RSI, MACD, Bollinger BantlarÄ±, ATR, OBV, vb.)
   â””â”€ MacroPipeline: Makro baÄŸlam Ã¶zellikleri
       (USD/TRY, EUR/TRY, BIST100, VIX, AltÄ±n, Brent, DXY, ABD 10Y Faiz,
        TCMB Faiz, CPI â€” yfinance + FRED API)

3. Chronological Train/Test Split  â† DataSplitter
   â””â”€ Kesinlikle zaman sÄ±rasÄ±na gÃ¶re; karÄ±ÅŸtÄ±rma yok, sÄ±zÄ±ntÄ± yok

4. Ã–lÃ§ekleme  â† Scaler YALNIZCA eÄŸitim setine fit edilir
   â”œâ”€ X: RobustScaler (aykÄ±rÄ± deÄŸerlere dayanÄ±klÄ±)
   â””â”€ y: StandardScaler + klipleme (log-getiri hedefi iÃ§in)

5. 3D Diziler OluÅŸtur  â† LSTM/TFT/sequence modeller iÃ§in
   â””â”€ [Ã¶rnekler, TIME_STEPS=30, Ã¶zellik_sayÄ±sÄ±]

6. Model EÄŸitimi
   â”œâ”€ SeÃ§ilen modeller eÄŸitilir
   â””â”€ Walk-forward: her fold baÄŸÄ±msÄ±z eÄŸitim + tahmin dÃ¶ngÃ¼sÃ¼

7. Ters DÃ¶nÃ¼ÅŸÃ¼m
   â””â”€ Log-getiri tahminleri â†’ Fiyat tahminleri

8. DeÄŸerlendirme
   â”œâ”€ YÃ¶n DoÄŸruluÄŸu, RMSE, Sharpe yaklaÅŸÄ±mÄ±, Hit Rate
   â”œâ”€ Backtest: Sinyal â†’ Komisyon+Kayma â†’ P&L simÃ¼lasyonu
   â”œâ”€ Monte Carlo Bootstrap (sinyalin ÅŸansa karÅŸÄ± anlamlÄ±lÄ±ÄŸÄ±)
   â””â”€ XAI: Ã–zellik Ã¶nemi + HTML raporu

9. KayÄ±t
   â”œâ”€ Model dosyasÄ± (.pkl / .keras / .pt)
   â”œâ”€ stock_models.db (SQLite)
   â”œâ”€ registry.json
   â””â”€ CSV deney gÃ¼nlÃ¼ÄŸÃ¼
```

---

## 4. Model KataloÄŸu

TÃ¼m modeller `BaseModel` arayÃ¼zÃ¼nden tÃ¼rer ve `train()`, `predict()`, `save()`, `load()` metodlarÄ±nÄ± zorunlu olarak uygular.

### Baseline / Referans Modeller

| SÄ±nÄ±f | Dosya | AÃ§Ä±klama |
|---|---|---|
| `NaiveLastValueModel` | `naive_model.py` | Son gÃ¶zlemlenen deÄŸeri tekrarlar. Minimum referans noktasÄ±. |
| `NaiveZeroReturnModel` | `naive_model.py` | Her zaman sÄ±fÄ±r getiri tahmin eder. "HiÃ§ hareket olmayacak" hipotezi. |
| `NaiveDriftModel` | `naive_model.py` | Lineer trend ekstrapolasyonu. Basit momentum testi. |
| `ARIMAModel` | `arima_model.py` | YapÄ±landÄ±rÄ±labilir (p,d,q) dÃ¼zeni; auto_order desteÄŸi. |
| `ProphetModel` | `prophet_model.py` | Meta Prophet; yalnÄ±zca kapanÄ±ÅŸ fiyatÄ±, regresÃ¶r desteÄŸiyle. |

**Neden?** Baseline modeller olmadan, karmaÅŸÄ±k modellerin gerÃ§ekten deÄŸer katÄ±p katmadÄ±ÄŸÄ±nÄ± anlayamazsÄ±nÄ±z. RMSE sayÄ±sal olarak iyi gÃ¶rÃ¼nse de eÄŸer bir Naive model aynÄ± sonucu veriyorsa, karmaÅŸÄ±k modelin fazladan maliyeti (hesaplama, bakÄ±m) gereksizdir.

### AÄŸaÃ§ TabanlÄ± Modeller

| SÄ±nÄ±f | Dosya | AÃ§Ä±klama |
|---|---|---|
| `XGBoostModel` | `xgboost_model.py` | Gradient boosting; Optuna HPO + SQLite warm-start |
| `RandomForestModel` | `random_forest_model.py` | Bagging ensemble; Optuna HPO + SQLite warm-start |
| `LightGBMReturnModel` | `gradient_boosting_model.py` | Opsiyonel; yoksa sessizce atlanÄ±r |

### Lineer / Regularize Modeller

| SÄ±nÄ±f | Dosya | AÃ§Ä±klama |
|---|---|---|
| `RidgeReturnModel` | `linear_model.py` | L2 cezasÄ±; log-getiri Ã¼zerinde |
| `ElasticNetReturnModel` | `linear_model.py` | L1+L2 karÄ±ÅŸÄ±mÄ±; seyrek Ã¶zellik seÃ§imi |

**Neden?** Lineer modeller, tahmin alanÄ±nÄ±n dÄ±ÅŸÄ±nda kalan Ã¶zellikler olduÄŸunda saÄŸlam kalÄ±r. AyrÄ±ca "ne kadar doÄŸrusal olmayan bir iliÅŸki var?" sorusunun cevabÄ±nÄ± dolaylÄ± olarak verir.

### Derin Ã–ÄŸrenme Modelleri

| SÄ±nÄ±f | Dosya | AÃ§Ä±klama |
|---|---|---|
| `LSTMModel` / `AttentionLSTMModel` | `lstm_model.py` | Ã‡ift yÃ¶nlÃ¼ LSTM + dikkat mekanizmasÄ± (Keras/TensorFlow) |
| `TFTModel` | `tft_model.py` | Temporal Fusion Transformer; kantil tahmini (PyTorch) |
| `TFTModelV2` | `tft_v2/model.py` | ModÃ¼ler blok mimarisine sahip yeni nesil Temporal Fusion Transformer |

**Neden LSTM'de `clipnorm=1.0`?** BIST verisi, yÃ¼ksek volatilite dÃ¶nemlerinde (seÃ§imler, kur krizleri) ani gradient patlamalarÄ±na yol aÃ§abilir. `Adam(clipnorm=1.0)` ile gradients normalize edilerek eÄŸitim kararsÄ±zlÄ±ÄŸÄ± Ã¶nlenir.

**Neden TFT ve TFT v2?** TFT, nokta tahmini deÄŸil kantil tahmini (P10/P50/P90) Ã¼retir. Bu sayede "fiyat ne olacak?" sorusuna ek olarak "belirsizlik aralÄ±ÄŸÄ± nedir?" sorusunu da yanÄ±tlar. Yeni eklenen **TFT v2** mimarisi ise statik eÅŸdeÄŸiÅŸken kodlayÄ±cÄ±lar, deÄŸiÅŸken seÃ§im aÄŸlarÄ± ve Gated Residual Network (GRN) bloklarÄ±nÄ± birbirinden ayÄ±rarak Ã§ok daha esnek, bakÄ±mÄ± kolay ve geniÅŸletilebilir bir altyapÄ± sunar.

### Topluluk (Ensemble) Modelleri

| SÄ±nÄ±f | Dosya | AÃ§Ä±klama |
|---|---|---|
| `EnsembleModel` | `ensemble.py` | L2 (Ridge) regresyonu kullanarak diÄŸer modellerin tahminlerini birleÅŸtirir |

**Neden Ensemble?** Tek bir modelin (Ã¶rneÄŸin XGBoost) iyi Ã§alÄ±ÅŸtÄ±ÄŸÄ± piyasa koÅŸullarÄ± ile LSTM'in iyi Ã§alÄ±ÅŸtÄ±ÄŸÄ± koÅŸullar farklÄ± olabilir. `EnsembleModel`, alt modellerin geÃ§miÅŸ tahmin performanslarÄ±na gÃ¶re (meta-Ã¶ÄŸrenme) dinamik aÄŸÄ±rlÄ±klar belirler. Tek bir modele baÄŸÄ±mlÄ± kalma riskini (model risk) azaltÄ±r ve getiriyi pÃ¼rÃ¼zsÃ¼zleÅŸtirir.

### Deneysel Sequence Baseline'larÄ±

| SÄ±nÄ±f | Dosya | AÃ§Ä±klama |
|---|---|---|
| `DLinearSequenceModel` | `linear_sequence_model.py` | 3D diziler Ã¼zerinde hafif lineer |
| `NLinearSequenceModel` | `linear_sequence_model.py` | Normalize edilmiÅŸ lineer (son deÄŸer Ã§Ä±karÄ±lÄ±r) |

---

## 5. Fintech AltyapÄ±sÄ±

### 5.1 Sinyal Ãœretimi

ÃœÃ§ sinyal modu desteklenir:

**`simple` (varsayÄ±lan):** Maliyetsiz long/flat AL/SAT/TUT modu. Beklenen getiri
`buy_threshold` Ã¼stÃ¼ndeyse ve pozisyon yoksa AL, beklenen getiri
`-sell_threshold` altÄ±ndaysa ve pozisyon varsa SAT, diÄŸer durumlarda TUT
Ã¼retilir. VarsayÄ±lan eÅŸikler `0.0` ve komisyon/slippage `0.0`'dÄ±r.

**`professional` (opt-in araÅŸtÄ±rma modu):** YÃ¶n doÄŸruluÄŸu, kalite kapÄ±larÄ±,
volatilite, holding period, take-profit ve stop-loss mantÄ±ÄŸÄ± devreye girer.

**`legacy`:** Tarihsel direction-only long/flat karÅŸÄ±laÅŸtÄ±rma modu.

### 5.2 Backtest Motoru

`src/backtesting/engine.py` â€” `run_backtest()` fonksiyonu:

- Sinyalden long/flat pozisyon durumunu hesaplar
- VarsayÄ±lan basit modda komisyon (`commission_bps`) ve kayma (`slippage_bps`) 0 kalÄ±r
- Ä°stenirse non-zero komisyon/kayma parametreleriyle maliyetli senaryo Ã§alÄ±ÅŸtÄ±rabilir
- GÃ¼nlÃ¼k P&L, drawdown ve kÃ¼mÃ¼latif getiri dizisi Ã¼retir

**Neden BPS (baz puan)?** BIST'te iÅŸlem komisyonlarÄ± kÃ¼Ã§Ã¼k gÃ¶rÃ¼nse de yÃ¼ksek frekanslÄ± stratejilerde kÃ¼mÃ¼latif maliyetler getiriyi ciddi erozya yaratÄ±r. BPS cinsinden parametrik tanÄ±m, gerÃ§ekÃ§i simÃ¼lasyon saÄŸlar.

### 5.3 GeliÅŸmiÅŸ Backtest Metrikleri

Standart Sharpe Ratio ve Max Drawdown'Ä±n Ã¶tesinde:

| Metrik | AÃ§Ä±klama | Neden Ã–nemli? |
|---|---|---|
| **Omega Ratio** | KazanÃ§/kayÄ±p oranÄ± eÅŸik Ã¼zerinde | Sharpe'dan farklÄ± olarak getiri daÄŸÄ±lÄ±mÄ±nÄ±n ÅŸeklini dikkate alÄ±r |
| **Recovery Factor** | Net getiri / Max drawdown | KayÄ±plarÄ± ne kadar hÄ±zlÄ± telafi ettiÄŸini gÃ¶sterir |
| **Max Consecutive Loss** | Ãœst Ã¼ste maksimum kayÄ±p sayÄ±sÄ± | Psikolojik dayanma kapasitesi iÃ§in kritik |
| **Information Ratio** | Aktif getiri / Takip hatasÄ± (annualize) | Benchmark'a karÅŸÄ± tutarlÄ± Ã¼stÃ¼nlÃ¼k Ã¶lÃ§Ã¼sÃ¼ |

### 5.4 Kaldirilan Arastirma Yardimcilari

Monte Carlo bootstrap, Kelly pozisyon boyutlandirma ve bagimsiz permutation importance helper modulleri aktif urun kapsamindan cikarildi. Varsayilan sistem kaldiracsiz long/flat AL/SAT/TUT sinyali uretir; pozisyon boyutlandirma ve arastirma simulatörleri runtime pipeline parçası değildir.

### 5.5 Dinamik Risksiz Oran

`src/utils/risk_free_rate.py` â€” `get_current_risk_free_rate()`:

Ã–ncelik sÄ±rasÄ±:
1. `RISK_FREE_RATE_ANNUAL` ortam deÄŸiÅŸkeni
2. `data/macro/INTEREST_RATE.csv` son satÄ±rÄ± (gerÃ§ek TCMB faizi)
3. Fallback: `0.40` (%40 â€” yÃ¼ksek faiz dÃ¶nemine uygun gÃ¼venli varsayÄ±lan)

**Neden?** Sharpe Ratio hesabÄ±nda kullanÄ±lan risksiz oran, TÃ¼rkiye baÄŸlamÄ±nda son derece Ã¶nemlidir. ABD'de %5 olan risksiz oran, TÃ¼rkiye'de %50'yi aÅŸabilmektedir. Sabit bir deÄŸer yerine gerÃ§ek TCMB verisini okumak, Sharpe hesabÄ±nÄ± anlamlÄ± kÄ±lar.

### 5.7 Sinyal Kalibrasyon Kilit MekanizmasÄ±

`calibration_scope = "wf_train"` (deÄŸiÅŸtirilemez):

Sinyal eÅŸikleri yalnÄ±zca walk-forward fold eÄŸitim verisi Ã¼zerinde kalibre edilir. Final holdout verisi kalibrasyon sÃ¼recine **asla** dahil edilemez. Bu kural `_assert_wf_train_scope()` metoduyla Ã§alÄ±ÅŸma zamanÄ±nda zorlanÄ±r; ihlal halinde `RuntimeError` fÄ±rlatÄ±lÄ±r.

**Neden bu kadar katÄ±?** Sinyal eÅŸiklerini test setine gÃ¶re ayarlamak "look-ahead bias" (gelecek bilgisi sÄ±zmasÄ±) yaratÄ±r ve gerÃ§ek dÃ¼nya performansÄ±nÄ± abartÄ±r. Bu, akademik literatÃ¼rde sÄ±k rastlanan ve fark edilmesi zor bir yanÄ±lgÄ±dÄ±r.

---

## 6. GeliÅŸtirme FazlarÄ± â€” Neyi Neden YaptÄ±k?

Bu proje, deneysel bir prototipten baÅŸlayarak altÄ± faz boyunca sistematik olarak geliÅŸtirilmiÅŸtir. Her faz, belirli bir teknik borcu ya da eksikliÄŸi gidermeye odaklanmÄ±ÅŸtÄ±r.

### Faz 0 â€” Temel AltyapÄ± SaÄŸlamlaÅŸtÄ±rma

**Problem:** Pipeline bÃ¼yÃ¼dÃ¼kÃ§e `DataSplitter`'dan kaÃ§an veri sÄ±zÄ±ntÄ±sÄ± riskleri, scaler'Ä±n test seti Ã¼zerinde fit edilmesi ve tekrar Ã¼retilebilir sonuÃ§larÄ±n garanti edilmemesi gibi temel hatalar ortaya Ã§Ä±ktÄ±.

**YapÄ±lanlar:**
- `DataSplitter`: Kronolojik bÃ¶lÃ¼nme garantisi; shuffle yok, stratified split yok.
- `reproducibility.py`: Python, NumPy, TensorFlow ve PyTorch tohumlarÄ± tek yerden ayarlanÄ±r.
- `ExperimentTracker`: Her Ã§alÄ±ÅŸmanÄ±n parametreleri ve metrikleri CSV'ye kaydedilir; deney takibi mÃ¼mkÃ¼n hale gelir.

**Bu iÅŸe ne yarar?** AynÄ± konfigÃ¼rasyonu iki kez Ã§alÄ±ÅŸtÄ±rdÄ±ÄŸÄ±nÄ±zda aynÄ± sonucu alÄ±rsÄ±nÄ±z. KarÅŸÄ±laÅŸtÄ±rmalÄ± analizler gÃ¼venilir olur.

### Faz 1 â€” Model Ekosistemi GeniÅŸletme

**Problem:** Proje baÅŸlangÄ±Ã§ta yalnÄ±zca birkaÃ§ model destekliyordu. "Hangi model en iyi?" sorusuna yanÄ±t verebilmek iÃ§in geniÅŸ bir model ekosistemi gerekiyordu.

**YapÄ±lanlar:**
- Lineer modeller eklendi: `RidgeReturnModel`, `ElasticNetReturnModel`
- Deneysel sequence baseline'larÄ±: `DLinearSequenceModel`, `NLinearSequenceModel`
- TÃ¼m modeller `BaseModel` arayÃ¼zÃ¼ne baÄŸlandÄ±; herhangi bir model eklenebilir/Ã§Ä±karÄ±labilir.
- Opsiyonel baÄŸÄ±mlÄ±lÄ±klar sessizce atlanÄ±r (LightGBM, Prophet, TF, PyTorch olmadÄ±ÄŸÄ±nda pipeline Ã§alÄ±ÅŸmaya devam eder).

**Bu iÅŸe ne yarar?** Model seÃ§imi veri gÃ¼dÃ¼mlÃ¼ hale gelir. Hisse baÅŸÄ±na en iyi model otomatik olarak belirlenir.

### Faz 2 â€” God Object YÄ±kÄ±mÄ± ve Mimari Temizlik

**Problem:** `EvaluationManager` tek bir dosyada 1500+ satÄ±ra ulaÅŸtÄ±. Test edilemez, bakÄ±mÄ± zor, yeni Ã¶zellik eklemek her seferinde tÃ¼m sÄ±nÄ±fÄ± riske atÄ±yordu.

**YapÄ±lanlar:**
- `EvaluationManager` dÃ¶rt mixin'e ayrÄ±ldÄ±:
  - `_PredictionEngineMixin` â€” tahmin Ã¼retimi
  - `_BacktestRunnerMixin` â€” backtest orchestration
  - `_SignalCalibratorMixin` â€” sinyal kalibrasyon mantÄ±ÄŸÄ±
  - `_MetricsReporterMixin` â€” metrik raporlama ve kayÄ±t
- `TypedDict` dÃ¶nÃ¼ÅŸ tipleri (`SingleSplitResult`, `WalkForwardResult`, `FinalHoldoutResult`) eklendi.
- `PipelineConfig` dataclass hiyerarÅŸisi: `DataConfig`, `ValidationConfig`, `ModelConfig`, `ExecutionConfig`.

**Bu iÅŸe ne yarar?** Her mixin baÄŸÄ±msÄ±z test edilebilir. Yeni bir Ã¶zellik eklemek tek bir mixin'i etkiler, diÄŸerlerini deÄŸil. Tip gÃ¼venliÄŸi sayesinde IDE otomatik tamamlama Ã§alÄ±ÅŸÄ±r, hata ayÄ±klama kolaylaÅŸÄ±r.

### Faz 3 â€” ML Ã‡ekirdeÄŸi GÃ¼Ã§lendirme

**Problem:** Hiperparametre optimizasyonu her seferinde sÄ±fÄ±rdan baÅŸlÄ±yordu, LSTM gradientleri patlamalar yaÅŸÄ±yordu ve makro Ã¶zellikler sÄ±nÄ±rlÄ±ydÄ±.

**YapÄ±lanlar:**

**Optuna Warm-Start (XGBoost ve RandomForest):**
- Her hisse icin ayri SQLite Optuna veritabani (`data/optuna/optuna_studies_{SYMBOL}.db`)
- `load_if_exists=True` ile Ã¶nceki denemeler birikir; her Ã§alÄ±ÅŸmada daha iyi baÅŸlangÄ±Ã§ noktasÄ±ndan devam edilir.
- **Neden?** XGBoost iÃ§in 100 deneme standart olsa da gerÃ§ek projede zaman kÄ±sÄ±tlarÄ± vardÄ±r. Warm-start ile 10 deneme bile Ã¶nceki 100 denemenin Ã¼zerine inÅŸa eder.

**LSTM Gradient Klipleme:**
- `Adam(learning_rate=lr, clipnorm=1.0)` her iki LSTM varyantÄ±nda da.
- **Neden?** BIST'te TL deÄŸer kaybÄ± dÃ¶nemlerinde log-getiri serisi ani sÄ±Ã§ramalar yapar. Bu sÄ±Ã§ramalar gradient vektÃ¶rÃ¼nÃ¼ patlatabilir ve eÄŸitimi Ã§Ã¶kÃ¼ÅŸe uÄŸratÄ±r.

**Makro Ã–zellik GeniÅŸletmesi:**
- Yeni tickers: EUR/TRY, VIX (korku endeksi), AltÄ±n (USD), Brent Petrol, DXY (USD Endeksi), ABD 10Y Getirisi
- TÃ¼rev Ã¶zellikler: `EURTRY_Return`, `VIX_Level`, `VIX_Change`, `Gold_TRY_Return`, `Oil_USD_Return`, `DXY_Return`, `US10Y_Level`, `US10Y_Change`
- **Neden?** BIST kÃ¼resel risk iÅŸtahÄ±ndan yoÄŸun ÅŸekilde etkilenir. VIX yÃ¼kseldiÄŸinde BIST genellikle dÃ¼ÅŸer. DXY gÃ¼Ã§lendiÄŸinde TL baskÄ± altÄ±nda kalÄ±r, bu da ihracatÃ§Ä± hisseler iÃ§in Ã¶nemlidir.

**Feature Cache:**
- Pickle tabanlÄ±, MD5 anahtarlÄ±, 24 saatlik TTL
- **Neden?** Makro veri Ã§ekme + Ã¶zellik mÃ¼hendisliÄŸi birlikte 30-60 saniye sÃ¼rebilir. Cache ile tekrarlÄ± Ã§alÄ±ÅŸmalar 2-3 saniyeye iner.

### Faz 4 â€” Fintech AltyapÄ±sÄ±

**Problem:** "Model iyi tahmin ediyor" ile "strateji gerÃ§ekten para kazandÄ±rÄ±yor" arasÄ±ndaki boÅŸluÄŸu kapatmak gerekiyordu. Sharpe Ratio tek baÅŸÄ±na yeterli deÄŸildi.

**YapÄ±lanlar:**
- Dinamik Risksiz Oran (`src/utils/risk_free_rate.py`)
- GeliÅŸmiÅŸ Backtest Metrikleri: Omega Ratio, Recovery Factor, Max Consecutive Loss, Information Ratio
- Sinyal kalibrasyon kilit mekanizmasÄ± (`calibration_scope`, `_assert_wf_train_scope`)

**Bu iÅŸe ne yarar?** Bir stratejiyi deÄŸerlendirirken artÄ±k ÅŸu sorularÄ±n tamamÄ±na yanÄ±t alÄ±nabilir:
- ÅansÄ±n Ã¶tesinde performans var mÄ±? (Monte Carlo p-deÄŸeri)
- Ne kadar sermaye riske girmeli? (Kelly)
- KayÄ±p dÃ¶nemlerinden ne kadar hÄ±zlÄ± Ã§Ä±kÄ±lÄ±yor? (Recovery Factor)
- Hangi Ã¶zellikler gerÃ§ekten katkÄ± saÄŸlÄ±yor? (PermÃ¼tasyon Ã¶nemi)

### Faz 5 â€” Ãœretim HazÄ±rlÄ±ÄŸÄ±

**Problem:** Proje tek hisse, interaktif CLI ile sÄ±nÄ±rlÄ±ydÄ±. Ã‡ok hisseli otomasyona ve dÄ±ÅŸ uygulama entegrasyonuna ihtiyaÃ§ duyuldu.

**YapÄ±lanlar:**

**Linter ve Kod Kalitesi (`pyproject.toml`, `.flake8`, `.pre-commit-config.yaml`):**
- Black (format), isort (import sÄ±ralamasÄ±), flake8 (stil denetimi)
- Pre-commit hook'larÄ±: her commit Ã¶ncesinde otomatik kontrol
- **Neden?** BÃ¼yÃ¼yen bir kod tabanÄ±nda tutarsÄ±z stil, kod incelemelerini zorlaÅŸtÄ±rÄ±r ve merge Ã§akÄ±ÅŸmalarÄ±na yol aÃ§ar. Otomatik formatlama bu tartÄ±ÅŸmayÄ± ortadan kaldÄ±rÄ±r.

**GitHub Actions CI (`.github/workflows/ci.yml`):**
- Python 3.10 + 3.11 matrix
- Kritik flake8 hatalarÄ± CI'Ä± durdurur; stil uyarÄ±larÄ± raporlanÄ±r ama durdurmaz
- Smoke testleri her push'ta Ã§alÄ±ÅŸÄ±r; tam test paketi Ã§alÄ±ÅŸÄ±r ama baÅŸarÄ±sÄ±z olsa CI geÃ§er (`|| true`)
- FastAPI import kontrolÃ¼
- **Neden?** Her push'ta otomatik doÄŸrulama, bozuk kodu ana dala karÄ±ÅŸtÄ±rmaktan korur. Python 3.10/3.11 matrix, ileriye uyumluluÄŸu garanti eder.

**Multi-Stock Batch Modu (`python -m src.cli.batch`):**
- `--stocks TUPRS,ASELS,THYAO` veya `--universe data/bist_universe.csv`
- `--workers N` ile `ProcessPoolExecutor` paralel Ã§alÄ±ÅŸma
- `--dry-run` modu: veri varlÄ±ÄŸÄ±nÄ± kontrol eder, pipeline Ã§alÄ±ÅŸtÄ±rmaz
- `batch_summary_{timestamp}.csv + .json` Ã§Ä±ktÄ± Ã¶zeti
- **Neden?** BIST'te 50+ hisseyi tek tek Ã§alÄ±ÅŸtÄ±rmak pratik deÄŸildir. Batch mod ile tÃ¼m universe bir gecede Ã§alÄ±ÅŸtÄ±rÄ±labilir.

**FastAPI Servis KatmanÄ± (`src/api/main.py`):**
- `GET /best-model/{symbol}` â€” Hisse iÃ§in en iyi model
- `GET /leaderboard` â€” TÃ¼m hisseler lider tablosu
- `GET /metrics/{symbol}` â€” Model karÅŸÄ±laÅŸtÄ±rma
- `POST /run/{symbol}` â€” Pipeline'Ä± arka planda tetikle
- `GET /run/status/{job_id}` â€” Ä°ÅŸ durumu sorgulama
- **Neden?** `Merge_PortfoySim` gibi portfÃ¶y simÃ¼lasyon uygulamalarÄ±, tahmin sonuÃ§larÄ±nÄ± HTTP Ã¼zerinden sorgulayabilir. BÃ¶ylece bu pipeline, daha bÃ¼yÃ¼k bir sistemin baÄŸÄ±msÄ±z mikroservisi haline gelir.

### Faz 6 â€” ModÃ¼lerizasyon ve Temiz Mimari (Ä°leri Mimariler)

**Problem:** Veri yÃ¼kleme, Ã¶n iÅŸleme gibi araÃ§larÄ±n kÃ¶k dizine yayÄ±lmasÄ± kod okunabilirliÄŸini zorlaÅŸtÄ±rÄ±yor ve derin Ã¶ÄŸrenme modelleri (TFT) yeni Ã¶zellik eklemek iÃ§in Ã§ok monolitik (tek parÃ§a) kalÄ±yordu. 

**YapÄ±lanlar:**
- **Veri AraÃ§larÄ± AyrÄ±ÅŸtÄ±rÄ±ldÄ±:** `data_loader.py` ve `preprocessor.py` gibi araÃ§lar `src/data/` modÃ¼lÃ¼ne taÅŸÄ±narak sorumluluklar netleÅŸtirildi.
- **TFT v2 Mimarisi:** PyTorch tabanlÄ± Temporal Fusion Transformer modeli, bloklarÄ±na (Encoder, Decoder, Multi-Head Attention, GLU, GRN) ayrÄ±ÅŸtÄ±rÄ±larak "modÃ¼ler" hale getirildi. ArtÄ±k mimarinin iÃ§ine yeni bir bileÅŸen takmak Ã§ok daha kolay.
- **Ensemble (Topluluk) Modeli:** En iyi modellerin tahminlerini Ridge regresyonu ile dinamik aÄŸÄ±rlÄ±klandÄ±ran Ensemble yapÄ±sÄ± `src/models/ensemble.py` olarak standart modele dahil edildi.

**Bu iÅŸe ne yarar?** Sistemin bakÄ±m maliyeti (maintenance cost) ciddi oranda dÃ¼ÅŸtÃ¼. Ä°leride makine Ã¶ÄŸrenimi mÃ¼hendisleri, TFT'nin sadece Gated Residual Network bloklarÄ±nda veya veri yÃ¼kleyicinin sadece tek bir API Ã§aÄŸrÄ±sÄ±nda deÄŸiÅŸiklik yaparak tÃ¼m sistemi bozmadan Ã§alÄ±ÅŸabilecek.

---

## 7. Kurulum

### Gereksinimler

- Python 3.10+
- (Opsiyonel) CUDA 12.8 uyumlu GPU (TFT/LSTM hÄ±zlandÄ±rma iÃ§in)

### Temel Kurulum

```bash
# BaÄŸÄ±mlÄ±lÄ±klarÄ± kur
pip install -r requirements.txt

# PyTorch (TFT iÃ§in) â€” CPU:
pip install torch

# PyTorch â€” CUDA 12.8 (RTX serileri):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Opsiyonel: LightGBM
pip install lightgbm

# FastAPI servisi iÃ§in:
pip install fastapi uvicorn
```

### Conda (dl_env Ã¶nerilen)

```bash
conda activate dl_env
pip install -r requirements.txt
pip install fastapi uvicorn
```

### Pre-commit Hook'larÄ±nÄ± EtkinleÅŸtir

```bash
pip install pre-commit
pre-commit install
```

---

## 8. KullanÄ±m

### 8.1 Tek Hisse â€” Ä°nteraktif CLI

```bash
python -m src.cli.interactive
```

Pipeline baÅŸladÄ±ÄŸÄ±nda sÄ±rasÄ±yla sorar:
1. Hisse kodu (TUPRS, ASELS, THYAO, vb.)
2. DoÄŸrulama modu: `single_split` veya `walk_forward`
3. Model seÃ§imi: tÃ¼mÃ¼ veya belirli modeller

### 8.2 Multi-Stock Batch Modu

```bash
# ÃœÃ§ hisse, walk-forward, 2 paralel worker:
python -m src.cli.batch \
    --stocks TUPRS,ASELS,THYAO \
    --mode walk_forward \
    --workers 2

# TÃ¼m BIST universe, 4 worker:
python -m src.cli.batch \
    --universe data/bist_universe.csv \
    --mode walk_forward \
    --workers 4

# Sadece belirli modeller:
python -m src.cli.batch \
    --stocks TUPRS,EREGL \
    --models XGBoost,Ridge,LSTM

# Kuru Ã§alÄ±ÅŸma (veri kontrol et, pipeline Ã§alÄ±ÅŸtÄ±rma):
python -m src.cli.batch \
    --universe data/bist_universe.csv \
    --dry-run
```

### 8.3 FastAPI Servisi

```bash
# Proje kÃ¶kÃ¼nden Ã§alÄ±ÅŸtÄ±r:
uvicorn src.api.main:app --reload --port 8000

# EtkileÅŸimli API dÃ¶kÃ¼mantasyonu:
# http://localhost:8000/docs       (Swagger UI)
# http://localhost:8000/redoc      (ReDoc)
```

Ã–rnek sorgular:

```bash
# En iyi model:
curl http://localhost:8000/best-model/TUPRS

# Lider tablosu:
curl http://localhost:8000/leaderboard

# Pipeline tetikle:
curl -X POST http://localhost:8000/run/ASELS \
     -H "Content-Type: application/json" \
     -d '{"mode": "walk_forward", "models": ["XGBoost", "Ridge"]}'

# Ä°ÅŸ durumu sorgula:
curl http://localhost:8000/run/status/{job_id}
```

### 8.4 Testler

```bash
# TÃ¼m testler:
python -m pytest tests/

# Smoke testleri (hÄ±zlÄ±):
python -m pytest tests/test_smoke.py -v

# Belirli test modÃ¼lÃ¼:
python -m pytest tests/test_leakage_guards.py -v
```

### 8.5 Operasyonel SÃ¼reÃ§ler: Veri GÃ¼ncelleme

GÃ¼nlÃ¼k piyasa kapanÄ±ÅŸlarÄ±ndan sonra modelin gÃ¼ncel verilerle Ã§alÄ±ÅŸmasÄ± iÃ§in veri setinin gÃ¼ncellenmesi gerekir. Bu iÅŸlem `src/data/data_updater.py` kullanÄ±larak veya `python -m src.cli.batch` Ã¼zerinden yapÄ±labilir.

```bash
# Sadece veri setini gÃ¼nceller (Model Ã§alÄ±ÅŸtÄ±rmaz)
python -m src.data.data_updater --symbols TUPRS,ASELS

# TÃ¼m BIST evreninin verisini gÃ¼nceller
python -m src.data.data_updater --universe data/bist_universe.csv

# Batch mod ile hem veriyi gÃ¼ncelle hem de tahmin Ã¼ret
# (Veri eksikse veya eskiyle otomatik olarak Yahoo Finance Ã¼zerinden tamamlanÄ±r)
python -m src.cli.batch --universe data/bist_universe.csv --mode single_split
```

> **Not:** Makro veriler (BIST100, USDTRY vb.) pipeline Ã§alÄ±ÅŸtÄ±ÄŸÄ±nda `MacroPipeline` veya `data_updater` aracÄ±lÄ±ÄŸÄ±yla otomatik olarak FRED ve Yahoo Finance Ã¼zerinden gÃ¼ncellenir.

---

## 9. YapÄ±landÄ±rma ReferansÄ±

TÃ¼m konfigÃ¼rasyon `PipelineConfig` dataclass hiyerarÅŸisi Ã¼zerinden yÃ¶netilir (`src/pipeline/config.py`):

### DataConfig â€” Veri ve Ã–zellik AyarlarÄ±

| Parametre | VarsayÄ±lan | AÃ§Ä±klama |
|---|---|---|
| `data_file` | â€” | Hisse CSV dosyasÄ± yolu |
| `test_ratio` | `0.20` | Test seti oranÄ± (kronolojik) |
| `time_steps` | `30` | LSTM/TFT iÃ§in dizi uzunluÄŸu |
| `target_mode` | `log_return` | Hedef: log-getiri (tercih) veya fiyat |
| `feature_mode` | `stationary_features` | Ã–zellik seti tipi |
| `scaling_mode` | `robust_x_standard_y_clip` | Ã–lÃ§ekleme stratejisi |
| `use_macro` | `True` | Makro Ã¶zellikler dahil edilsin mi? |
| `training_window_years` | `5` | Pencere seÃ§imi iÃ§in aday yÄ±llar |

### ValidationConfig â€” DoÄŸrulama ProtokolÃ¼

| Parametre | VarsayÄ±lan | AÃ§Ä±klama |
|---|---|---|
| `validation_mode` | `single_split` | `single_split` veya `walk_forward` |
| `wf_n_splits` | `12` | Walk-forward fold sayÄ±sÄ± |
| `wf_test_size` | `21` | Her fold test boyutu (gÃ¼n/bar) |
| `wf_max_train_size` | `756` | Kayan pencere maksimum eÄŸitim boyutu |
| `final_holdout_size` | `60` | Son dokunulmamÄ±ÅŸ test seti boyutu |

### ModelConfig â€” Model SeÃ§imi

| Parametre | VarsayÄ±lan | AÃ§Ä±klama |
|---|---|---|
| `selected_models` | `None` (tÃ¼mÃ¼) | Ã‡alÄ±ÅŸtÄ±rÄ±lacak modeller listesi |
| `ensemble_enabled` | `True` | Ensemble modeller oluÅŸturulsun mu? |

### ExecutionConfig â€” Backtest ve Sinyal

| Parametre | VarsayÄ±lan | AÃ§Ä±klama |
|---|---|---|
| `backtest_enabled` | `True` | Backtest Ã§alÄ±ÅŸtÄ±rÄ±lsÄ±n mÄ±? |
| `initial_capital` | `100,000` | BaÅŸlangÄ±Ã§ sermayesi (TL) |
| `commission_bps` | `0.0` | VarsayÄ±lan basit modda komisyon kapalÄ± |
| `slippage_bps` | `0.0` | VarsayÄ±lan basit modda kayma maliyeti kapalÄ± |
| `signal_mode` | `simple` | `simple`, `professional` veya `legacy` |
| `calibration_scope` | `wf_train` | **DeÄŸiÅŸtirme:** SÄ±zÄ±ntÄ± koruma kilidi |

---

## 10. Ã‡Ä±ktÄ± YapÄ±sÄ±

```
outputs/{SYMBOL}/
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ xgboost_model.pkl
â”‚   â”œâ”€â”€ lstm_model.keras
â”‚   â”œâ”€â”€ tft_model.pt
â”‚   â””â”€â”€ {model}_final_holdout_model.{ext}
â”œâ”€â”€ experiments/
â”‚   â””â”€â”€ experiment_log_{timestamp}.csv
â”œâ”€â”€ xai/
â”‚   â”œâ”€â”€ xai_report_{model}.html
â”‚   â””â”€â”€ xai_report_{model}.txt
â””â”€â”€ registry.json                         â† Model versiyonlama metadata

data/stock_models.db                     â† Merkezi SQLite
data/optuna/optuna_studies_{SYMBOL}.db   â† Optuna warm-start (gitignore)
data/feature_cache/                       â† Pickle cache (gitignore)
data/macro/*.csv                          â† Ä°ndirilen makro veri (gitignore)
outputs/batch_summaries/*.csv/.json      â† Batch calisma ozeti (gitignore)
```

---

## 11. Test Paketi

| Dosya | Kapsam |
|---|---|
| `test_smoke.py` | Temel import ve baÅŸlatma kontrolleri |
| `test_leakage_guards.py` | Veri sÄ±zÄ±ntÄ±sÄ± Ã¶nleme doÄŸrulamasÄ± |
| `test_phase4_models.py` | Baseline + yeni model sÄ±nÄ±flarÄ± |
| `test_reporting_metrics.py` | Metrik hesaplama doÄŸruluÄŸu |
| `test_validation_protocol.py` | Walk-forward sÄ±ralama deÄŸiÅŸmezleri |
| `test_phase5_data_quality.py` | Veri kalite kontrolleri |
| `test_phase6_backtest_standard.py` | Backtest motor standartlarÄ± |
| `test_phase7_acceptance.py` | Sistem kabul testleri |
| `test_phase8_acceptance.py` | Ãœretim kabulÃ¼ |

---

## 12. GeliÅŸtirici AraÃ§larÄ±

### Kod Kalitesi

```bash
# Formatlama (Black):
black src/ tests/ --line-length 100

# Import sÄ±ralamasÄ± (isort):
isort src/ tests/

# Stil denetimi (flake8):
flake8 src/ tests/ --max-line-length 100

# Tip denetimi (mypy):
mypy src/
```

### CI/CD

GitHub Actions otomatik olarak her `push` ve `pull_request`'te Ã§alÄ±ÅŸÄ±r:
- **Python 3.10 + 3.11** â€” Ã§apraz versiyon uyumluluÄŸu
- **Kritik flake8 hatalarÄ±** (E9, F63, F7, F82) CI'Ä± durdurur
- **Smoke testleri** â€” her push'ta hÄ±zlÄ± doÄŸrulama
- **FastAPI import kontrolÃ¼** â€” API katmanÄ± bozulmuÅŸsa erken uyarÄ±

### Ortam DeÄŸiÅŸkenleri

| DeÄŸiÅŸken | AÃ§Ä±klama |
|---|---|
| `RISK_FREE_RATE_ANNUAL` | Sharpe hesabÄ± iÃ§in risksiz oran geÃ§ersiz kÄ±l (Ã¶rn: `0.45`) |

---

## 13. GeliÅŸtirici Rehberi: Yeni Model Ekleme

Sisteme kendi algoritmanÄ±zÄ± (Ã¶rneÄŸin yeni bir PyTorch tabanlÄ± Transformer veya Ã¶zel bir istatistiksel model) eklemek iÃ§in aÅŸaÄŸÄ±daki 3 adÄ±mÄ± izlemeniz yeterlidir:

1. **`BaseModel`'den TÃ¼retin:** `src/models/` dizini altÄ±nda yeni bir dosya oluÅŸturun ve sÄ±nÄ±fÄ±nÄ±zÄ± `BaseModel` arayÃ¼zÃ¼nden (`src.models.base_model.BaseModel`) tÃ¼retin.
2. **MetodlarÄ± Ä°mplemente Edin:** Zorunlu olan `train()`, `predict()`, `save()`, ve `load()` metodlarÄ±nÄ± doldurun.
3. **KayÄ±t Ä°ÅŸlemi:** Modelinizi `src/model_registry/model_registry.py` iÃ§indeki sÃ¶zlÃ¼ÄŸe veya kullanacaÄŸÄ±nÄ±z `ModelConfig` sÄ±nÄ±fÄ±na tanÄ±tÄ±n.

---

## 14. Sorun Giderme (Troubleshooting)

| Sorun | Neden ve Ã‡Ã¶zÃ¼m |
|---|---|
| **yfinance 429 Too Many Requests** | *Neden:* Ã‡ok fazla hisse verisi aynÄ± anda istendi. *Ã‡Ã¶zÃ¼m:* `python -m src.cli.batch` iÃ§indeki `--workers` sayÄ±sÄ±nÄ± dÃ¼ÅŸÃ¼rÃ¼n veya farklÄ± bir IP adresi/VPN kullanÄ±n. |
| **CUDA OOM (Out of Memory)** | *Neden:* TFT v2 veya LSTM eÄŸitilirken ekran kartÄ± belleÄŸi doldu. *Ã‡Ã¶zÃ¼m:* `PipelineConfig` altÄ±ndaki `time_steps` deÄŸerini kÃ¼Ã§Ã¼ltÃ¼n veya model batch size deÄŸerini azaltÄ±n. |
| **Missing Macro Data (FRED/Yahoo)** | *Neden:* Ä°nternet baÄŸlantÄ±sÄ± koptu veya API deÄŸiÅŸti. *Ã‡Ã¶zÃ¼m:* `src/data/data_updater.py`'yi tekrar Ã§alÄ±ÅŸtÄ±rÄ±n; veriler Ã¶nbellekten okunmak yerine sÄ±fÄ±rdan indirilir. |
| **SQLite database is locked** | *Neden:* Optuna warm-start kullanÄ±rken Ã§ok fazla worker aynÄ± DB dosyasÄ±na yazmaya Ã§alÄ±ÅŸtÄ±. *Ã‡Ã¶zÃ¼m:* Worker sayÄ±sÄ±nÄ± dÃ¼ÅŸÃ¼rÃ¼n (`--workers 2` veya `1`). |

---

## Lisans

Bu proje, `Merge_PortfoySim` portfÃ¶y simÃ¼lasyon sisteminin araÅŸtÄ±rma ve geliÅŸtirme bileÅŸenidir.


