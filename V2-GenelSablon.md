# 🔍 KAPSAMLI SİSTEM DENETİM RAPORU TALEBİ

## 👤 ROL VE BAKIŞ AÇISI
Sen kıdemli bir denetim ekibisin. Aşağıdaki boyutları analiz ederken her bölümde hangi persona ile baktığını belirt:
- **Bölüm 1-2:** Kıdemli Yazılım Mimarı bakışı
- **Bölüm 3:** DevSecOps Uzmanı bakışı
- **Bölüm 4:** QA / Test Mühendisi bakışı
- **Bölüm 5:** SRE / DevOps bakışı

Tavrın eleştirel, çözüm odaklı ve **somut** olmalı. Genel geçer tavsiyelerden kaçın; her bulgu kanıta dayansın.

---

## 📂 TARAMA STRATEJİSİ

### Adımlar
1. Önce **dizin ağacını** çıkar (klasör yapısı + dosya sayısı).
2. **Giriş noktalarını** (main.*, __init__.*, index.*, app.*) önce incele.
3. Sonra **core/business logic** modüllerini analiz et.
4. Büyük dosyaları (>300 satır) ayrı incele.
5. Tarayamadığın dosyaları raporun başında **"İncelenmedi"** listesinde belirt; nedenini yaz (boyut, encoding, vb.).

### Hariç Tutulacaklar
````
.venv/, venv/, env/, __pycache__/, .pyc, .pyo
node_modules/, dist/, build/, out/, target/
.git/, .idea/, .vscode/, .DS_Store
*.lock dosyaları (sadece içerik analizi için, varlığını raporla)
*_ui.py, *_rc.py, resources_rc.py  (otomatik üretilen UI dosyaları)
*.min.js, *.min.css, bundle.*, vendor.*
migrations/ (sadece sayısını raporla)
*.log, logs/, tmp/, .cache/
````

---

## 📏 ÖLÇÜM EŞİKLERİ (Bunları aşan her yapıyı raporla)

| Metrik | Eşik |
|--------|------|
| Cyclomatic complexity | > 10 |
| Kognitif karmaşıklık | > 15 |
| Fonksiyon uzunluğu | > 50 satır |
| Sınıf uzunluğu | > 300 satır |
| Dosya uzunluğu | > 500 satır |
| Parametre sayısı | > 5 |
| İç içe blok derinliği | > 4 |
| Sınıf metodu sayısı | > 20 |
| Bir modülün import sayısı | > 25 |

---

## 🎯 ÖNCELİK TANIMLARI

- 🔴 **KRİTİK:** Veri kaybı, güvenlik açığı, production çökme riski, mali/yasal risk. → **Bugün düzelt.**
- 🟠 **YÜKSEK:** Performans/güvenilirlik ciddi etkisi, refactor'u zorlaştıran teknik borç. → **Bu sprint.**
- 🟡 **ORTA:** Bakım maliyeti yüksek, okunabilirlik düşük. → **Sonraki sprint.**
- 🟢 **DÜŞÜK:** Stil, isimlendirme, kozmetik. → **Fırsat oldukça.**

---

## 📋 HER BULGU İÇİN ZORUNLU ŞABLON

````markdown
### [🔴/🟠/🟡/🟢] Bulgu Başlığı
- **Dosya/Konum:** `path/to/file.ext:line-line`
- **Kategori:** Kod Kalitesi / Mimari / Güvenlik / Test / Production
- **Sorun:** (2-3 cümle teknik açıklama)
- **Etki:** (Somut senaryo: bu sorun ne zaman, nasıl patlar?)
- **Kanıt:**
```dil
  // mevcut sorunlu kod
```
- **Çözüm:**
```dil
  // önerilen düzeltme
```
- **Düzeltme Eforu:** S (≤1sa) / M (1-4sa) / L (1-3 gün) / XL (haftalar)
- **Bağımlılık:** (Bu düzeltmenin başka neyi etkileyebileceği)
````

---

## 🔬 ANALİZ BOYUTLARI

### 1. YAZILIM KALİTESİ VE KOD ANALİZİ
- **Karmaşıklık:** Eşikleri aşan fonksiyonlar/sınıflar.
- **Anti-Pattern Taraması:** God Object, Spaghetti Code, Magic Numbers, Long Method, Feature Envy, Shotgun Surgery, Primitive Obsession.
- **Ölü Kod:** Kullanılmayan değişkenler, çağrılmayan fonksiyonlar, gereksiz import'lar, ulaşılamayan branch'ler.
- **Okunabilirlik ve Standartlar:** İsimlendirme tutarlılığı, formatlamayı, anlaşılmayan kısaltmalar.
- **Bellek ve Performans:** Memory leak riskleri, gereksiz kopya, Big O kötü algoritmalar, N+1 pattern, blocking I/O.
- **Dokümantasyon:** README yeterliliği, public API docstring durumu, kurulum talimatlarının doğruluğu.
- **Yerel Bağlam Kontrolleri:**
  - UTF-8 / Türkçe karakter handling (dosya I/O, JSON, DB)
  - Locale-sensitive operasyonlar (`.lower()`/`.upper()` Türkçe 'i' sorunu)
  - Tarih/saat: timezone-aware mı, naive mi?
  - Para/sayı formatı: ondalık ayırıcı tutarlılığı

### 2. MİMARİ VE TASARIM
- **Prensip Uyumu:** SOLID, DRY, KISS, YAGNI ihlalleri (her birine somut örnek ver).
- **Bağımlılık Yönetimi:** Sıkı bağlılık (tight coupling), Dependency Injection eksikleri, döngüsel bağımlılık (circular imports).
- **Katmanlama:** Katmanlar arası sızıntı var mı? (UI'ın DB'ye direkt erişmesi, business logic'in UI framework'üne bağımlı olması)
- **Tasarım Desenleri:** Doğru uygulanmış mı, yanlış yerde mi (over-engineering), eksik mi?
- **Genişletilebilirlik:** Yeni özellik eklemek için kaç dosya değişmesi gerekir? Open/Closed prensibi tutuyor mu?

### 3. GÜVENLİK (DevSecOps)
- **Sır Yönetimi:**
  - Hardcoded şifre/API key/token var mı?
  - `.env` kullanılıyor mu, `.gitignore`'da mı, `.env.example` var mı?
  - Git history'de sızmış sır şüphesi
  - Konfigürasyon kaynağı tek mi, dağınık mı?
- **Girdi Doğrulama:** Kullanıcı/dış kaynak verisi sanitize ediliyor mu? (Injection, XSS, Path Traversal, SSRF, Deserialization riskleri)
- **Erişim Kontrolü:** Yetkilendirme mekanizmaları, least privilege, default-deny pattern.
- **Kriptografi:** Düz metin parola depolama, zayıf hash (MD5/SHA1), kendi yazılmış kripto.
- **Bağımlılık Güvenliği:**
  - `requirements.txt` / `package.json`'da pinlenmemiş paketler
  - Bilinen CVE'li sürümler
  - Lisans uyumsuzluğu (GPL paket kapalı projede vb.)
  - Kullanılmayan paketler

### 4. TEST VE KAPSAM
- **Mevcut Durum:** Test dosyalarını listele, tahmini coverage oranı (test/kaynak), test piramidi dengesi (birim/entegrasyon/e2e).
- **Test Edilebilirlik:** Mock/Stub kullanımını engelleyen mimari sorunlar (global state, singleton'lar, hard-coded bağımlılıklar).
- **Flaky Test Belirtileri:** `time.sleep`, gerçek API çağrısı, gerçek file/DB I/O, random olmadan random davranan testler.
- **Edge Case'ler:** Boş/null/aşırı uzun girdi, sıfıra bölme, race condition, concurrent erişim, network kesintisi, disk dolu, yetki yokluğu.
- **Hata Yönetimi:**
  - Silent fail (sessizce yutulan exception'lar)
  - Çıplak `except:` veya `except Exception:` kullanımı
  - Custom exception sınıfı eksikliği
  - Hata loglaması var mı, anlamlı mı?
- **En Kritik 5 Fonksiyon İçin:** Hangi test case'leri yazılmalı önerisi.

### 5. PRODUCTION VE DAĞITIM
- **Çevre Yapılandırması:**
  - dev / staging / prod ayrımı net mi?
  - Environment variable kullanımı tutarlı mı?
  - `DEBUG=True` production'a sızabilir mi?
- **Loglama ve İzlenebilirlik:**
  - `print()` mi, gerçek logger mı?
  - Log seviyeleri doğru mu (DEBUG/INFO/WARNING/ERROR/CRITICAL)?
  - Yapılandırılmış log (JSON) mu, düz metin mi?
  - Korelasyon ID / request ID var mı?
  - Hassas veri log'a sızıyor mu?
- **Performans Darboğazları (statik göstergeler):**
  - N+1 query pattern (loop içinde DB/HTTP çağrısı)
  - Senkron blocking I/O (UI thread'de file/network)
  - O(n²) veya kötü algoritma
  - Cache yokluğu (tekrarlayan pahalı hesaplama)
  - Connection pool / resource leak riski
- **Dağıtım/Build:**
  - Build adımları tek komutla çalışıyor mu?
  - Reproducible build mi (lock file var mı)?
  - CI/CD yapılandırması var mı?
  - Rollback stratejisi var mı?

---

## 📤 RAPOR ÇIKTI YAPISI

Raporu **tam olarak şu sırayla** üret:

### Bölüm A — Yönetici Özeti (1 sayfa)
Tek paragraflık genel değerlendirme + aşağıdaki tablo:

| # | Bulgu | Öncelik | Boyut | Efor | Etki |
|---|-------|---------|-------|------|------|
| 1 | ... | 🔴 | Güvenlik | M | Yüksek |

### Bölüm B — Korunması Gereken İyi Pratikler (Pozitif Bulgular)
Projede doğru yapılmış 3-5 mimari/kalite kararını belirt. Refactor sırasında "neye dokunmayalım" rehberi olarak.

### Bölüm C — İncelenmedi Listesi
Tarayamadığın dosyalar ve nedenleri.

### Bölüm D — Detaylı Bulgular
Yukarıdaki şablona göre, **öncelik sırasıyla** (Kritik → Düşük). Her bulguda ilgili persona'yı belirt.

### Bölüm E — Kriz Yönetimi / Aksiyon Planı

#### 🚨 24 SAAT İÇİNDE (Yangın Söndürme)
En kritik 3 madde. Her biri için:
- Eylem
- Beklenen sonuç
- Bağımlılık
- Risk (bu adım neyi bozabilir)

#### 📅 BU HAFTA (Stabilizasyon)
5 madde, aynı format.

#### 🗓 BU AY (Sağlamlaştırma)
5 madde, aynı format.

### Bölüm F — Sonraki Adımlar
Bana şu soruyu yönelt:
> "Hangi bulgudan başlamak istersin? Seçtiğin maddenin detaylı refactor planını, etkilenecek dosyaları ve örnek 'before/after' kodunu üretebilirim."

---

## 📐 ÇIKTI HEDEFLERİ

- **Toplam uzunluk:** ~2000-4000 kelime
- **Bulgu sayısı:** En kritik 15-25 bulgu (her şeyi listeleme; sinyal/gürültü oranı yüksek olsun)
- **Dil:** Türkçe (kod örnekleri orijinal dilinde)
- **Ton:** Eleştirel ama yapıcı, somut, kanıta dayalı
- **Yasaklı:** "Belki", "olabilir", "genelde" gibi bulanık ifadeler. Bulguya dönüştüremiyorsan rapora ekleme.

---

**Şimdi başla.** Önce dizin ağacını çıkararak tarama planını paylaş, sonra rapora geç.