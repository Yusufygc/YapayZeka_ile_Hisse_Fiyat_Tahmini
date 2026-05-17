# Proje Sozlugu

Bu dokuman, `ts_forecasting_lab` projesine yeni katilan gelistiriciler icin hazirlanmis bir kavram sozlugudur. Terimler alfabetik degil, pipeline'i anlama sirasina gore gruplandirilmistir.

Her satir su mantikla okunmalidir: terim nedir, ne ise yarar, bu projede nerede karsilik bulur ve hangi noktaya dikkat edilmelidir.

## Mimari ve Akis

| Terim | Kategori | Kisa tanim | Ne ise yarar? | Projede karsiligi | Dikkat edilmesi gerekenler |
|---|---|---|---|---|---|
| `ForecastingPipeline` | Mimari | Tum egitim, tahmin, degerlendirme ve kayit akisinin ust seviye sinifi. | Dis dunyaya tek bir calistirma arayuzu sunar. | `src/pipeline/orchestrator.py` icindeki ana facade sinifi. | Is akisi burada orkestre edilir; detay is mantigi alt yoneticilerde tutulmalidir. |
| Facade | Mimari desen | Karmasik alt sistemleri tek ve sade bir arayuz arkasinda toplama deseni. | Pipeline kullanan kodun `DataManager`, `ModelTrainer` ve `EvaluationManager` detaylarini bilmesini engeller. | `ForecastingPipeline` bu rolu ustlenir. | Facade icine model veya veri donusumu detayi tasinmamali. |
| Strategy | Mimari desen | Ayni arayuzu izleyen farkli algoritmalari degistirilebilir hale getirme yaklasimi. | Model tipleri ve validasyon protokolleri ayni akis icinde calistirilabilir. | `BaseModel` sozlesmesi ve farkli model siniflari. | Yeni model eklenirken `train`, `predict`, `save`, `load` sozlesmesi korunmali. |
| `DataManager` | Mimari | Veri yukleme, ozellik uretimi, bolme, olcekleme ve tensor hazirlamadan sorumlu yonetici. | Model egitimine hazir veri setleri uretir. | `src/pipeline/data_manager.py`. | Veri sizintisi onlemleri burada kritik oldugu icin split ve scaler sirasi bozulmamali. |
| `ModelTrainer` | Mimari | Secili modelleri egiten ve kayit surecine hazirlayan yonetici. | Model katalogunu tek bir egitim akisina baglar. | `src/pipeline/model_trainer.py`. | Opsiyonel modellerin bagimlilikleri yoksa pipeline'in tamamen durmamasi beklenir. |
| `EvaluationManager` | Mimari | Tahminleri, metrikleri, backtestleri, XAI ve raporlari yoneten katman. | Model sonucunu yalniz hata metrikleriyle degil finansal uygulanabilirlikle de olcer. | `src/pipeline/evaluation_manager.py` ve mixin dosyalari. | Final holdout veya test verisi sinyal kalibrasyonu icin yanlis kullanilmamali. |
| `PipelineConfig` | Konfigurasyon | Tum pipeline ayarlarini gruplandiran kok dataclass. | Parametre kalabaligini azaltir ve calistirma ayarlarini tasir. | `src/pipeline/config.py`. | Yeni ayar eklenirken dogru alt config sinifina konulmali. |
| `DataConfig` | Konfigurasyon | Veri dosyasi, hedef modu, ozellik modu, makro veri ve kalite ayarlarini tasir. | Veri hazirlama davranisini tek noktadan belirler. | `data_file`, `target_mode`, `feature_mode`, `scaling_mode`, `use_macro`. | `target_mode` degisirse metrik ve inverse transform davranisi da etkilenir. |
| `ValidationConfig` | Konfigurasyon | `single_split` ve `walk_forward` protokol ayarlarini tasir. | Zaman serisi validasyonunun pencere ve holdout kurallarini belirler. | `wf_n_splits`, `wf_test_size`, `wf_window_type`, `final_holdout_size`. | Zaman sirasini bozan herhangi bir degisiklik veri sizintisi dogurur. |
| `ModelConfig` | Konfigurasyon | Secili modelleri, registry versiyonunu, ensemble ve model hiperparametrelerini tasir. | Hangi modellerin calisacagini ve nasil egitilecegini belirler. | `selected_models`, `ensemble_enabled`, `model_settings`. | Model adlari `src/pipeline/model_scope.py` ile uyumlu olmali. |
| `ExecutionConfig` | Konfigurasyon | Backtest, maliyet, sinyal ve raporlama ayarlarini tasir. | Tahminlerin islem sinyaline ve rapora nasil donusecegini belirler. | `commission_bps`, `slippage_bps`, `signal_mode`, `signal_config`. | Maliyetleri sifirlamak backtest sonuclarini gercekci olmaktan cikarabilir. |
| `BaseModel` | Model arayuzu | Tum model siniflarinin uymasi gereken soyut sozlesme. | Egitim, tahmin, kaydetme ve yukleme davranisini standartlastirir. | `src/models/base_model.py`. | Yeni model bu arayuzu tam uygulamadan pipeline'a baglanmamali. |

## Veri ve Piyasa Terimleri

| Terim | Kategori | Kisa tanim | Ne ise yarar? | Projede karsiligi | Dikkat edilmesi gerekenler |
|---|---|---|---|---|---|
| BIST | Finans | Borsa Istanbul piyasasi. | Projenin hedef hisse evrenini tanimlar. | `data/*.csv` altindaki BIST hisse dosyalari. | BIST verisi tatil, hacim ve likidite farklari nedeniyle global piyasa verisinden farkli davranabilir. |
| BIST100 | Finans | Borsa Istanbul'un ana piyasa endeksi. | Hissenin genel piyasaya gore gucunu olcmek icin referans olur. | Makro ozelliklerde `BIST100` ve `Relative_Strength`. | Endeks verisi hisse takvimiyle hizalanirken eksik gunler dikkatle doldurulmali. |
| OHLCV | Veri | Open, High, Low, Close ve Volume kolonlarindan olusan piyasa verisi. | Teknik indikatorlerin temel girdisidir. | Hisse CSV dosyalari ve `load_data()` cikisi. | Tarih sirasi ve kolon adlari normalize edilmeden modelleme yapilmamali. |
| `Open` | Veri | Gunluk acilis fiyati. | Gun baslangic fiyat seviyesini temsil eder. | Ham veri kolonu ve teknik ozellik girdisi. | Tek basina hedef degil, ozellik olarak degerlendirilir. |
| `High` | Veri | Gun icinde gorulen en yuksek fiyat. | Volatilite ve tipik fiyat hesaplarinda kullanilir. | `VWAP_20_rel` ve bazi legacy indikator hesaplari. | Aykiri veri varsa volatilite sinyallerini bozabilir. |
| `Low` | Veri | Gun icinde gorulen en dusuk fiyat. | Gun ici araligi ve risk sinyalleri icin kullanilir. | `VWAP_20_rel` ve legacy volatilite hesaplari. | Eksik veya hatali dusuk fiyatlar drawdown yorumunu bozabilir. |
| `Close` | Veri | Gunluk kapanis fiyati. | Hedef, getiri ve teknik indikatorlerin ana kaynagidir. | `FeaturePipeline` icinde hareketli ortalamalar, getiri ve momentum hesaplari. | Birden fazla donusumde kullanildigi icin split oncesi ve sonrasi hizalama korunmali. |
| `Volume` | Veri | Gunluk islem hacmi. | Likidite ve hacim destekli sinyalleri yakalar. | `OBV_Norm_20`, `VWAP_20_rel`. | Sifir hacimli gunler islem gormeyen gunler olarak temizlenir. |
| Makro veri | Veri | Kur, faiz, enflasyon, emtia ve endeks gibi hisse disi piyasa degiskenleri. | Hisse hareketini etkileyebilecek dis kosullari modele ekler. | `MacroPipeline`, `use_macro=True`, `data/macro`. | Makro aciklanma gecikmeleri lag uygulanmadan kullanilirsa leakage olusabilir. |
| CSV veri dosyasi | Veri | Her hisse icin tarihsel fiyat ve hacim verisini tasiyan kaynak dosya. | Pipeline'in ham veri girdisidir. | `data/AKSA.csv`, `data/TUPRS.csv` gibi dosyalar. | Meta CSV dosyalari hisse verisi gibi secilmemeli. |
| Chronological split | Validasyon | Veriyi zaman sirasini bozmadan egitim ve test bolumlerine ayirma. | Gelecek bilgisinin gecmise sizmasini onler. | `TimeSeriesSplitter` ve `DataManager`. | Zaman serilerinde shuffle kullanilmaz. |
| Data leakage | Risk | Modelin egitim sirasinda gelecege ait bilgi gormesi. | Gercekte calismayacak kadar iyi metrikleri engellemek icin onlenir. | Scaler'in yalniz train setine fit edilmesi, holdout'un secimde kullanilmamasi. | En kritik proje invarianti budur; rapor, scaler ve kalibrasyon adimlari da buna uymali. |
| Training window | Validasyon | Egitimde kullanilan geriye donuk tarih araligi. | Eski rejimlerin modele fazla agirlik vermesini sinirlar. | `training_window_years`, `window_candidates`. | Cok kisa pencere veri azligi, cok uzun pencere rejim kaymasi riski tasir. |
| New listing | Veri kalitesi | Borsada yeni islem gormeye baslamis ve gecmisi kisitli hisse. | Minimum veri gereksinimini kontrol eder. | `new_listing_min_days`, `min_history_days`. | Yeni hisseler icin walk-forward veya derin ogrenme yeterli ornek bulamayabilir. |
| Corporate action | Veri kalitesi | Bedelli, bedelsiz, temettu veya bolunme gibi fiyat serisini etkileyen olay. | Fiyat serisindeki yapay sicramalari yorumlamak icin kullanilir. | `corporate_action_report` metadata alani. | Duzeltilmemis fiyatlar getiri ve metrikleri bozabilir. |
| Cache | Performans | Daha once uretilen veriyi tekrar kullanmak icin saklama mekanizmasi. | Makro ve ozellik uretimini hizlandirir. | `FeatureCache`, `data/feature_cache`, `data/macro`. | Cache anahtari config ile uyumlu olmali; eski cache yanlis ozellik seti dondurebilir. |

## Ozellik Muhendisligi

| Terim | Kategori | Kisa tanim | Ne ise yarar? | Projede karsiligi | Dikkat edilmesi gerekenler |
|---|---|---|---|---|---|
| Feature | Ozellik | Modelin tahmin yaparken kullandigi girdi degiskeni. | Ham fiyati modelin ogrenebilecegi sinyallere donusturur. | `feature_names` listesi. | Hedef kolonu veya gelecek bilgi feature listesine girmemeli. |
| `FeaturePipeline` | Ozellik | Ham OHLCV ve makro veriden model hazir ozellik matrisi ureten sinif. | Teknik, hacim, lag ve makro ozellikleri tek akista uretir. | `src/features/feature_pipeline.py`. | Yeni ozellik eklenirse `feature_group` ve XAI aciklamasi da guncellenmeli. |
| Technical indicator | Finans | Fiyat ve hacimden turetilen teknik gosterge. | Trend, momentum, volatilite veya hacim davranisini yakalar. | SMA, EMA, RSI, MACD, Bollinger, OBV, VWAP. | Indikator pencereleri NaN uretir; drop veya hizalama dogru yapilmali. |
| SMA | Ozellik | Simple Moving Average, basit hareketli ortalama. | Fiyatin ortalama trend seviyesine gore konumunu gosterir. | `SMA_7_rel`, `SMA_14_rel`, `SMA_21_rel`, `SMA_50_rel`. | Projede mutlak SMA yerine Close'a gore oran tercih edilir. |
| EMA | Ozellik | Exponential Moving Average, son gunlere daha fazla agirlik veren ortalama. | Yeni fiyat hareketlerine SMA'dan daha hizli tepki verir. | `EMA_7_rel`, `EMA_14_rel`, `EMA_21_rel`, `EMA_50_rel`. | EMA da stasyoner kalmasi icin Close'a gore normalize edilir. |
| RSI | Ozellik | Relative Strength Index, momentum ve asiri alim satim gostergesi. | Fiyatin asiri alim veya asiri satim bolgesine yaklasip yaklasmadigini gosterir. | `RSI_14`. | 0-100 araliginda oldugu icin ek fiyat normalizasyonu gerektirmez. |
| MACD | Ozellik | Hareketli ortalama farkina dayali momentum gostergesi. | Momentumun guclenip zayifladigini izler. | `MACD_norm`, `MACD_Signal_norm`, `MACD_Diff_norm`. | Fiyat birimli MACD yerine Close'a bolunmus normalize kolonlar kullanilir. |
| Bollinger Band | Ozellik | Hareketli ortalama etrafinda volatiliteye gore genisleyen fiyat bandi. | Fiyat araliginin genisleyip daraldigini gosterir. | `BB_Width_14`, `BB_Width_21`. | Projede bant genisligi oran olarak kullanilir; mutlak ust alt bantlar legacy modda kalabilir. |
| ATR | Ozellik | Average True Range, fiyat araligi bazli volatilite gostergesi. | Risk ve volatiliteyi fiyat araligi uzerinden olcer. | README ve legacy loader tarafinda `ATR_norm` olarak anilir. | Aktif ozellik yolunda hangi feature modunun kullanildigi kontrol edilmeli. |
| OBV | Ozellik | On-Balance Volume, fiyat yonu ile hacim akisini birlestiren gosterge. | Hacmin fiyat hareketini destekleyip desteklemedigini anlamaya yardim eder. | `OBV_Norm_20`. | Hacim verisi sifir veya eksikse sinyal zayiflar. |
| VWAP | Ozellik | Volume Weighted Average Price, hacim agirlikli ortalama fiyat. | Fiyatin hacim agirlikli seviyeye gore nerede oldugunu gosterir. | `VWAP_20_rel`. | Tipik fiyat ve hacim toplamlarina dayandigi icin hacim kalitesi onemlidir. |
| `Relative_Strength` | Ozellik | Hisse getirisinin BIST100 getirisine gore farki. | Hissenin piyasaya gore guclu veya zayif ayrisip ayrismadigini gosterir. | Makro merge sonrasi uretilen piyasa goreli ozellik. | BIST100 verisi eksikse forward fill ve hizalama kontrol edilmeli. |
| `Market_Regime_SMA200` | Ozellik | Close fiyatinin SMA-200 ustunde veya altinda olmasina dayali rejim etiketi. | Boga, ayi veya notr piyasa kosulunu sinyale katar. | `1`, `-1`, `0` degerleriyle uretilir. | Ilk 200 gun yeterli veri olmadigi icin notr deger gorulebilir. |
| Lag feature | Ozellik | Gecmis gunlere ait hedef veya getiri degerlerinin feature olarak eklenmesi. | Kisa vadeli otokorelasyon veya momentum etkisini yakalar. | `LogRet_Lag_1` ile `LogRet_Lag_N`. | Lag sayisi artarsa NaN kaybi ve overfitting riski artar. |
| Stationary feature | Ozellik | Zaman icinde seviyesi trendle buyumeyen, oran veya getiri gibi daha kararli ozellik. | Scaler'in test doneminde asiri disari tasmasini azaltir. | `stationary_features` ve `hybrid` modlari. | Mutlak fiyat seviyeleri yerine oranlar tercih edilir. |
| Correlated feature pruning | Ozellik secimi | Birbirine cok benzeyen ozelliklerden bazilarini dusurme islemi. | Gereksiz tekrar sinyallerini azaltir. | `prune_correlated_features`, `correlation_threshold`. | Pruning raporu XAI yorumunu etkileyebilir; dusen feature'lar raporlanmali. |
| Feature group | XAI | Feature'lari teknik, makro, hacim, volatilite gibi gruplara ayirma. | XAI raporlarini daha okunabilir yapar. | `src/xai/feature_dictionary.py` icindeki `feature_group()`. | Yeni feature eklenirse dogru gruba map edilmezse rapor `other` gosterir. |

## Hedef, Donusum ve Olcekleme

| Terim | Kategori | Kisa tanim | Ne ise yarar? | Projede karsiligi | Dikkat edilmesi gerekenler |
|---|---|---|---|---|---|
| `target_mode` | Hedef | Modelin neyi tahmin edecegini belirleyen mod. | Fiyat, basit getiri veya log getiri hedefini secer. | `price`, `return`, `log_return`. | Metrik, inverse transform ve sinyal uretimi bu moda baglidir. |
| `log_return` | Hedef | `log(Close_t / Close_t-1)` ile hesaplanan logaritmik getiri. | Fiyat seviyesini daha stasyoner bir hedefe cevirir. | Varsayilan `DataConfig.target_mode`. | Fiyata geri donuste onceki kapanis gerekir. |
| `return` | Hedef | Basit yuzdesel getiri. | Fiyat degisimini dogrudan oran olarak modelletir. | `target_mode="return"`. | Buyuk negatif getirilerde fiyata geri donus mantigi dikkat ister. |
| `price` | Hedef | Dogrudan fiyat seviyesi tahmini. | Kullaniciya fiyat seviyesi olarak sezgisel sonuc verir. | `target_mode="price"`. | Trend ve seviye degisimleri scaler disina cikma riskini artirabilir. |
| Scaler | Olcekleme | Sayisal verileri modelin daha rahat ogrenebilecegi araliga donusturen nesne. | Model egitimini kararlilastirir. | `scale_data()` ve kaydedilen scaler dosyalari. | Scaler yalniz egitim setine fit edilmelidir. |
| `RobustScaler` | Olcekleme | Medyan ve ceyrekliklere dayali olcekleme yontemi. | Aykiri fiyat veya hacim hareketlerine daha dayaniklidir. | `scaling_mode="robust_x_standard_y_clip"` icinde X icin kullanilir. | Aykiri degeri yok etmez, sadece etkisini azaltir. |
| `StandardScaler` | Olcekleme | Ortalama ve standart sapmaya dayali olcekleme yontemi. | Hedef degerleri ortalama sifir ve birim varyans civarina getirir. | Y hedefi icin kullanilir. | Train disinda fit edilirse leakage olusur. |
| Clip | Olcekleme | Asiri uc degerleri belirli sinirlara kisma islemi. | Modelin ekstrem hedeflerden bozulmasini azaltir. | `robust_x_standard_y_clip` modu. | Fazla agresif clipping gercek piyasa kuyruk riskini bastirabilir. |
| Inverse transform | Donusum | Olceklenmis tahmini eski birimine geri cevirme islemi. | Metrikleri fiyat veya getiri uzayinda anlamli hesaplamayi saglar. | `scaler_y.inverse_transform()` ve fiyat rekonstruksiyon fonksiyonlari. | Tahmin, hedef modu ve onceki kapanis dizisi ayni hizaya getirilmeli. |
| Sequence | Veri sekli | Zaman adimlari iceren 3 boyutlu model girdisi. | LSTM ve TFT gibi sequence modellerinin gecmis pencereyi gormesini saglar. | `[samples, time_steps, features]`. | `time_steps` arttikca ilk gozlemler kaybolur ve minimum ornek ihtiyaci artar. |
| `time_steps` | Veri sekli | Sequence modellerinin kac gecmis gunu girdi olarak alacagini belirler. | Modelin gecmis baglam uzunlugunu ayarlar. | `DataConfig.time_steps`, varsayilan 30. | Walk-forward embargo varsayilani bu degerle iliskilidir. |

## Validasyon ve Test Protokolleri

| Terim | Kategori | Kisa tanim | Ne ise yarar? | Projede karsiligi | Dikkat edilmesi gerekenler |
|---|---|---|---|---|---|
| `single_split` | Validasyon | Veriyi bir kez train ve test olarak bolen hizli protokol. | Hizli deneme ve temel karsilastirma yapar. | `ValidationConfig.validation_mode="single_split"`. | Tek doneme bagimli oldugu icin piyasa rejimi yanliligi olabilir. |
| `walk_forward` | Validasyon | Zaman boyunca kayan veya genisleyen pencerelerle tekrarli egitim test protokolu. | Modelin farkli zaman donemlerinde dayanikliligini olcer. | `WalkForwardValidator` ve `wf_*` config alanlari. | Her fold kendi train scaler'i ve train bilgisiyle calismali. |
| Fold | Validasyon | Walk-forward icindeki tek train test penceresi. | Her donemde bagimsiz performans olcumu saglar. | `split_idx`, `window_results`. | Fold sonuclari ortalanirken tarih hizasi korunmali. |
| Sliding window | Validasyon | Egitim penceresinin sabit uzunlukta ilerlemesi. | Eski piyasa rejimlerini disarida birakarak guncel veriye odaklanir. | `wf_window_type="sliding"`. | Cok kisa pencere modelin genellemesini zayiflatabilir. |
| Expanding window | Validasyon | Egitim penceresinin her fold'da buyumesi. | Tum gecmis bilgiyi koruyarak veri miktarini artirir. | `wf_window_type="expanding"`. | Eski rejimler yeni kosullari bastirabilir. |
| Embargo | Validasyon | Train ile test arasina bilerek bosluk koyma uygulamasi. | Sequence pencere veya gecikmeli bilgi sizintisini azaltir. | `wf_embargo_size`, varsayilan olarak `time_steps`. | Embargo cok kucukse pencere overlap leakage riski artar. |
| Final holdout | Validasyon | Model secimi ve kalibrasyondan ayrilan son test bolumu. | Nihai, dokunulmamis performans kontrolu saglar. | `final_holdout_size`. | Model secimi veya sinyal kalibrasyonu icin kullanilmamali. |
| Out-of-sample | Validasyon | Modelin egitimde gormedigi veri uzerindeki performansi. | Gercek hayata daha yakin performans olcer. | Test fold'lari ve final holdout. | OOS sonucuna gore hiperparametreyi tekrar tekrar ayarlamak gizli overfitting yaratir. |
| Benchmark model | Karsilastirma | Karmasik modellerin gecmesi beklenen basit referans model. | Modelin gercekten deger katip katmadigini gosterir. | `Naive Last Value`, `Naive Zero Return`, `Naive Drift`. | Lider seciminde yalniz RMSE degil finansal metrikler de incelenmeli. |

## Model Katalogu

| Terim | Kategori | Kisa tanim | Ne ise yarar? | Projede karsiligi | Dikkat edilmesi gerekenler |
|---|---|---|---|---|---|
| Naive model | Model | Basit kural tabanli referans model ailesi. | Karmasik modeller icin minimum performans cizgisi olusturur. | `Naive Last Value`, `Naive Zero Return`, `Naive Drift`. | Baseline'i gecemeyen karmasik model pratikte deger katmiyor olabilir. |
| ARIMA | Model | Otoregresif, fark alma ve hareketli ortalama bilesenlerinden olusan klasik zaman serisi modeli. | Istatistiksel baseline ve trend kontrolu saglar. | `ARIMAModel`. | Parametre secimi veri frekansi ve hedef moduna duyarlidir. |
| Prophet | Model | Meta tarafindan gelistirilen trend ve mevsimsellik odakli model. | Kapanis fiyatindaki yapisal trendleri yakalamaya calisir. | `ProphetModel`. | Opsiyonel bagimlilik yoksa atlanabilir; regresor kullanimi config'e baglidir. |
| XGBoost | Model | Gradient boosting tabanli guclu agac modeli. | Tabular teknik ve makro ozelliklerde nonlineer iliskileri yakalar. | `XGBoostModel`, Optuna HPO. | Hiperparametre aramasi maliyetli olabilir; leakage olmadan fold bazli calismali. |
| RandomForest | Model | Cok sayida karar agacini bagging ile birlestiren model. | Aykiri ve nonlineer tabular iliskilerde saglam baseline saglar. | `RandomForestModel`, Optuna HPO. | Zaman sirasi dogrudan model icinde bilinmez; feature'lar dogru hazirlanmali. |
| LightGBM | Model | Histogram tabanli hizli gradient boosting modeli. | XGBoost'a alternatif hizli boosting baseline'i sunar. | `LightGBMReturnModel`. | Bagimlilik opsiyoneldir; kurulu degilse pipeline durmamalidir. |
| Ridge | Model | L2 regularizasyonlu lineer regresyon. | Fazla karmasik olmayan, stabil bir return modeli saglar. | `RidgeReturnModel`. | Nonlineer iliskileri yakalamaz ama guclu baseline olabilir. |
| ElasticNet | Model | L1 ve L2 regularizasyonu birlestiren lineer model. | Hem stabilite hem seyrek ozellik secimi etkisi saglar. | `ElasticNetReturnModel`. | L1 etkisi bazi ozellik katsayilarini sifirlayabilir. |
| LSTM | Model | Uzun kisa sureli hafiza hucreleriyle sequence ogrenebilen derin ogrenme modeli. | Gecmis zaman penceresindeki ardisik yapilari yakalar. | `LSTMModel` ve attention destekli varyantlar. | Veri azsa veya volatility yuksekse overfitting ve egitim kararsizligi riski vardir. |
| TFT | Model | Temporal Fusion Transformer, sequence ve kantil tahmini odakli derin model. | Belirsizlik araligi ve zaman baglamli tahmin uretir. | `TFTModel`, `TFTModelV2`. | Kantil ciktilari metrik ve rapor tarafinda ayrica ele alinmali. |
| DLinear | Model | Sequence girdileri uzerinde hafif lineer baseline. | Derin modele gore daha basit ama sequence-aware karsilastirma saglar. | `DLinearSequenceModel`. | Deneysel baseline olarak degerlendirilmeli. |
| NLinear | Model | Son degeri normalize ederek calisan lineer sequence baseline. | Kisa vadeli normalize hareketleri yakalamaya calisir. | `NLinearSequenceModel`. | Normalizasyon varsayimi piyasa rejimine gore degisebilir. |
| Ensemble | Model | Birden fazla model tahminini birlestiren topluluk yaklasimi. | Tek model riskini azaltir ve tahmini yumusatabilir. | `EnsembleModel`, equal ve inverse RMSE agirliklari. | Ensemble sadece uyumlu uzunluk ve hedef semantigi olan tahminlerle kurulmali. |
| Candidate model | Model secimi | Uretim veya lider secimi icin aday kabul edilen model. | Raporlarda hangi modellerin asil yarista oldugunu ayirir. | `CANDIDATE_MODELS`, `DEFAULT_CANDIDATE_MODELS`. | Benchmark modeller candidate gibi secilmemeli. |
| Optional dependency | Bagimlilik | Kurulu degilse ilgili modelin atlanabildigi paket. | Hafif ortamda testlerin ve diger modellerin calismasini saglar. | Prophet, LightGBM, TensorFlow, PyTorch gibi paketler. | Import hatalari tum pipeline'i bozmayacak sekilde izole edilmeli. |

## Metrikler ve Model Degerlendirme

| Terim | Kategori | Kisa tanim | Ne ise yarar? | Projede karsiligi | Dikkat edilmesi gerekenler |
|---|---|---|---|---|---|
| MAE | Metrik | Ortalama mutlak hata. | Tahminin ortalama mutlak sapmasini gosterir. | `compute_financial_metrics()` ve rapor kolonlari. | Fiyat birimindeyse farkli hisseler arasi dogrudan karsilastirma zor olabilir. |
| RMSE | Metrik | Hatalarin kare ortalamasinin karekoku. | Buyuk hatalari daha agir cezalandirir. | Lider siralama ve benchmark oranlari. | Volatil hisselerde yuksek cikmasi normal olabilir. |
| MAPE | Metrik | Mutlak yuzdesel hata ortalamasi. | Hatanin fiyata gore oranini verir. | `MAPE` rapor kolonu. | Fiyat cok kucukse veya sifira yakinsa dengesizlesebilir. |
| `Return_MAE` | Metrik | Getiri uzayindaki mutlak hata. | Fiyat seviyesi yerine hedef getirideki hatayi olcer. | `compute_financial_metrics()`. | `target_mode` ile ayni semantikte yorumlanmali. |
| `Return_RMSE` | Metrik | Getiri uzayindaki RMSE. | Finansal hedefte buyuk sapmalari olcer. | `METRICS_REPORT_COLUMNS`. | Fiyat RMSE'siyle karistirilmamali. |
| Directional Accuracy | Metrik | Gercek ve tahmin edilen yonun ayni olma orani. | Fiyat seviyesinden cok yukari asagi kararinin kalitesini olcer. | `Dir_Acc`. | Notr veya sifira yakin getirilerde yorum dikkat ister. |
| Sharpe | Finansal metrik | Riskten arindirilmis getiri orani. | Strateji getirisinin oynakliga gore kalitesini olcer. | `Sharpe`, dinamik veya fallback risksiz faizle hesaplanir. | Yillik risksiz faiz varsayimi sonucu etkiler. |
| BuyHold Sharpe | Finansal metrik | Al ve tut stratejisinin Sharpe degeri. | Aktif stratejiyi pasif referansla karsilastirir. | `BuyHold_Sharpe`. | Aktif strateji buy-hold'u gecmiyorsa islem maliyetleri anlamsiz risk yaratabilir. |
| Hit Rate | Finansal metrik | Aktif sinyal verilen gunlerde pozitif getiri yakalama orani. | Isleme girildiginde ne kadar sik dogru kazanildigini gosterir. | `Hit_Rate`. | Cok az islem varsa yuksek oran yaniltici olabilir. |
| Neutral Rate | Finansal metrik | Sinyalin notr veya islemsiz kaldigi oran. | Modelin ne kadar sik pozisyon almaktan kactigini gosterir. | `Neutral_Rate`. | Yuksek notr oran dusuk risk veya yetersiz sinyal anlamina gelebilir. |
| Composite Score | Metrik | Birden fazla performans olcutunu birlestiren skor. | Lider secimini tek metrik bagimliligindan kurtarir. | `Composite_Score` rapor kolonu. | Formul degisirse eski kosularla karsilastirma dikkat ister. |
| `RMSE_vs_benchmark` | Metrik | Model RMSE'sinin benchmark RMSE'sine orani. | Modelin basit referansa gore ne kadar iyi oldugunu gosterir. | `enrich_with_benchmark_metrics()`. | 1'in alti daha iyi RMSE anlamina gelir. |
| Pinball Loss | Kantil metrik | Kantil tahmin hatasini olcen kayip fonksiyonu. | TFT gibi belirsizlik tahmini yapan modelleri degerlendirir. | `compute_quantile_metrics()`. | Nokta tahmini RMSE'siyle ayni sey degildir. |
| P10/P90 Coverage | Kantil metrik | Gercek degerin alt ve ust kantil bandina dusme orani. | Tahmin araliginin gercegi ne kadar kapsadigini olcer. | `P10_P90_Coverage`, `Interval_Coverage`. | Cok genis bant coverage'i artirirken kullanisliligi azaltabilir. |
| Winkler Score | Kantil metrik | Aralik genisligi ve kapsama hatasini birlikte cezalandiran skor. | Tahmin bandinin hem dogrulugunu hem darligini olcer. | `Winkler_Score`. | Daha dusuk skor genelde daha iyi aralik anlamina gelir. |

## Backtesting ve Sinyal Uretimi

| Terim | Kategori | Kisa tanim | Ne ise yarar? | Projede karsiligi | Dikkat edilmesi gerekenler |
|---|---|---|---|---|---|
| Backtest | Finans | Gecmis veride strateji calistirma simulasyonu. | Tahminlerin islem kararina donusunce ne kadar anlamli oldugunu olcer. | `src/backtesting/engine.py`, `run_backtest()`. | Backtest gercek piyasa garantisi degildir; maliyet ve kayma dahil edilmelidir. |
| Long/flat signal | Sinyal | Ya pozisyonda olma ya da nakitte kalma stratejisi. | Short pozisyon riskini disarida birakarak basit islem karari uretir. | `generate_long_flat_signals()`, professional sinyal modu. | Short desteklemez; dusus beklentisi genellikle pozisyon acmama seklinde davranir. |
| `BUY` | Sinyal | Yeni long pozisyon acma karari. | Beklenen getiri esigi asarsa isleme girer. | `generate_professional_signals()` cikisi. | Komisyon, kayma ve volatilite esigi hesaba katilir. |
| `HOLD` | Sinyal | Mevcut pozisyonu koruma karari. | Sinyal zayiflamadikca veya risk bariyeri tetiklenmedikce pozisyonu surdurur. | `Decision` kolonu. | Minimum ve maksimum elde tutma sureleri davranisi etkiler. |
| `EXIT` | Sinyal | Mevcut pozisyondan cikma karari. | Zayif sinyal, kar al, zarar kes veya sure sinirinda pozisyon kapatir. | `Risk_State` ve `Signal_Reason` ile raporlanir. | Cikis maliyetleri de hesaba katilmalidir. |
| `NO_TRADE` | Sinyal | Yeni pozisyon acmama karari. | Yetersiz sinyal veya cooldown doneminde riski azaltir. | Professional signal cikisi. | Cok fazla `NO_TRADE`, modelin trade edilebilir sinyal uretmedigini gosterebilir. |
| `SignalConfig` | Konfigurasyon | Professional sinyal uretiminin esik ve risk ayarlari. | Giris cikis, volatilite, kalite ve bekleme kurallarini belirler. | `src/backtesting/signals.py`. | Degerler validasyon guard'larindan gecmeli. |
| Quality gate | Sinyal filtresi | Dusuk kaliteli modellerin sinyal uretmesini kisitlayan filtre. | Kotu tahminlerin otomatik isleme donusmesini engeller. | `quality_gate_mode`, `min_directional_accuracy`, `max_rmse_vs_benchmark`. | `hard`, `soft`, `off` modlari farkli risk profili verir. |
| Commission bps | Maliyet | Komisyon maliyetinin baz puan cinsinden ifadesi. | Gercekci islem maliyeti simule eder. | `commission_bps`, varsayilan 10. | 1 bps yuzde 0.01 demektir. |
| Slippage bps | Maliyet | Emir fiyatiyla gerceklesen fiyat arasindaki kayma maliyeti. | Likidite ve uygulama farkini simule eder. | `slippage_bps`, varsayilan 5. | Ince likiditede kayma maliyeti yukselebilir. |
| Drawdown | Risk | Sermayenin zirveden dip noktaya dusus orani. | Stratejinin yasattigi ara kaybi olcer. | `Max_Drawdown`. | Sadece nihai getiriye bakmak drawdown riskini gizler. |
| CAGR | Getiri | Yillik bilesik getiri orani. | Farkli uzunluktaki backtestleri karsilastirir. | `CAGR`, `Annualized_Return`. | Kisa testlerde yilliklandirma abartili gorunebilir. |
| Sortino | Risk metrik | Sharpe benzeri ama yalniz asagi yonlu oynakligi cezalandiran metrik. | Kotu volatiliteye odaklanir. | `Sortino`. | Pozitif getirilerdeki oynakligi cezalandirmaz. |
| VaR | Risk metrik | Belirli guven seviyesinde beklenen kayip esigi. | Kuyruk riskini ozetler. | `VaR_95`, `BuyHold_VaR_95`. | Normal kosullari ozetler; asiri krizlerde yetersiz kalabilir. |
| CVaR | Risk metrik | VaR esiginin otesindeki ortalama kayip. | Daha kotu kuyruk senaryolarini olcer. | `CVaR_95`, `BuyHold_CVaR_95`. | Veri azsa kuyruk tahmini dengesiz olabilir. |
| Omega Ratio | Risk metrik | Esik ustu kazanc toplaminin esik alti kayip toplamina orani. | Getiri dagiliminin simetrik olmayan yapisini Sharpe'dan farkli yakalar. | `Omega_Ratio`. | Kayip yoksa sonsuz deger donebilir; raporda dikkatle yorumlanmali. |
| Recovery Factor | Risk metrik | Net getiri ile maksimum drawdown iliskisi. | Kayiplarin ne kadar verimli telafi edildigini gosterir. | `Recovery_Factor`. | Drawdown sifira yakin oldugunda oran asiri buyuyebilir. |
| Exposure | Backtest | Stratejinin piyasada pozisyonda oldugu zaman orani. | Riskin ne kadar sure tasindigini gosterir. | `Exposure`, `Days_In_Market`. | Dusuk exposure ile yuksek getiri iyi sinyal seciciligi gosterebilir. |
| Turnover | Backtest | Pozisyon degisim sikligi. | Islem maliyeti baskisini anlamaya yarar. | `Turnover`, `Trade_Count`. | Yuksek turnover komisyon ve kayma maliyetini buyutur. |

## XAI, Raporlama ve Kayit

| Terim | Kategori | Kisa tanim | Ne ise yarar? | Projede karsiligi | Dikkat edilmesi gerekenler |
|---|---|---|---|---|---|
| XAI | Aciklanabilirlik | Explainable AI, model tahminlerini insan tarafindan yorumlanabilir hale getirme alani. | Hangi ozelliklerin tahmine katkisini anlamayi saglar. | `src/xai/`. | Aciklama nedensellik kaniti degildir. |
| Feature importance | Aciklanabilirlik | Bir ozelligin tahmin performansina katkisini ozetleyen olcu. | Modelin hangi sinyallere dayandigini gosterir. | Permutasyon testi ve XAI raporlari. | Importance train veya test kapsamina gore farkli yorumlanir. |
| SHAP | Aciklanabilirlik | Tahminleri ozellik katkilarina ayiran model aciklama yontemi. | Tekil tahminlerin hangi feature'lardan etkilendigini gosterir. | README'de XAI katmani kapsaminda anilir. | SHAP degerleri model tipine ve arka plan verisine duyarlidir. |
| Permutation importance | Aciklanabilirlik | Bir ozelligi bozup performans dususunu olcen model-agnostik yontem. | Her model icin tutarli onem testi saglar. | XAI strategy fallbacks under `src/xai/`. | Korelasyonlu feature'larda onem paylasilabilir. |
| `describe_feature()` | Aciklanabilirlik | Feature adini sade Turkce aciklamaya ceviren yardimci fonksiyon. | Raporlari teknik kolon adlarindan kurtarir. | `src/xai/feature_dictionary.py`. | Yeni feature icin aciklama yoksa genel fallback metni doner. |
| Narrative report | Raporlama | Sayisal sinyal ve XAI sonucunu Turkce cumleye donusturen rapor. | Teknik olmayan okuyucuya tahmin nedenini anlatir. | `src/xai/narrative.py`, `src/xai/report_writer.py`. | Cumleler aciklama amaclidir, yatirim tavsiyesi gibi yazilmamali. |
| `registry.json` | Kayit | Model versiyon metadata'sini saklayan JSON kayit dosyasi. | Kaydedilen modelin ozellik, metrik ve dosya bilgilerini izler. | `ModelRegistry` uyumluluk katmani. | Guncel uretim kaydi SQLite tarafina kaymis olabilir; ikisi karistirilmamali. |
| `StockModelDB` | Kayit | SQLite tabanli merkezi model ve deney veritabani. | Model yasam dongusunu ve deneyleri sorgulanabilir hale getirir. | `src/database/stock_model_db.py`, `data/stock_models.db`. | Sema degisiklikleri geriye uyumluluk dusunulerek yapilmali. |
| `ExperimentTracker` | Kayit | Kosu parametreleri ve metrikleri CSV olarak izleyen katman. | Deney karsilastirmasini kolaylastirir. | `src/experiments/experiment_tracker.py`. | Run ID ve output path tutarliligi korunmali. |
| `run_id` | Raporlama | Her calistirmayi benzersiz adlandiran kimlik. | Ciktilari tarih, hisse, validasyon ve model setiyle ayirir. | `YYYYMMDD_HHMMSS_SYMBOL_validation_models...`. | Cok uzun model listelerinde hash kisaltmasi kullanilir. |
| `outputs/{SYMBOL}` | Cikti | Hisse bazli model, rapor ve deney ciktisi kok dizini. | Kosulari hisse bazinda ayirir. | `outputs/AKSA`, `outputs/TUPRS` gibi. | Generated ciktilar repo kaynak kodu gibi duzenlenmemeli. |
| `latest` | Cikti | En son kosuya kolay erisim veren sembolik veya kopya hedef dizin. | Kullanici ve API tarafinda son sonucu bulmayi kolaylastirir. | `outputs/{SYMBOL}/latest`. | Eski kosularla karsilastirma icin `runs/{run_id}` esas alinmali. |
| Model artifact | Cikti | Egitilmis modelin diske kaydedilmis dosyasi. | Daha sonra inference veya analiz icin modeli tekrar kullanmayi saglar. | `.pkl`, `.keras`, `.pt` dosyalari. | Artifact, onu uretilen scaler ve feature listesiyle birlikte anlamlidir. |
| Metrics report | Raporlama | Model performans metriklerini tablo halinde yazan rapor. | Model karsilastirma ve lider secimi icin kullanilir. | `metrics_reporter`, `evaluation/evaluator.py`. | Hangi validasyon protokolunden geldigi raporda belirtilmeli. |

## Gelistirici Icin Hizli Okuma Rehberi

Yeni bir gelistirici once `ForecastingPipeline` akisinin hangi alt yoneticilere ayrildigini anlamalidir. Sonra `DataManager` icindeki split, scaler ve sequence hazirlama kurallarini okumali; bu proje icin en kritik invariant veri sizintisi olmamasidir. Model ekleme veya degistirme yaparken `BaseModel` sozlesmesi ve `src/pipeline/model_scope.py` icindeki model adlari kontrol edilmelidir. Son olarak tahmin kalitesini yalniz RMSE ile degil, directional accuracy, backtest, maliyet ve risk metrikleriyle birlikte yorumlamak gerekir.


