---
title: Code Quality and Refactoring
type: concept
status: active
last_updated: 2026-05-21
owner: llm
source_count: 3
---

# Code Quality and Refactoring Guidelines

Bu sayfa, `ts_forecasting_lab` platformundaki kod kalitesi sınırlarını, girdi doğrulamalarını, hata yönetim prensiplerini ve refaktör yönergelerini tanımlar. Projede teknik borcun (technical debt) birikmesini ve devasa karmaşık sınıfların ("God Object") oluşmasını önlemek hedeflenmiştir.

## Kod Boyutu ve Yapı Sınırları

Karmaşıklığı yönetilebilir düzeyde tutmak amacıyla kod dosyaları ve sınıfları için aşağıdaki üst sınırlar getirilmiştir:
- **Dosya Sınırı:** Herhangi bir `.py` kaynak kod dosyası en fazla **500 satır** içerebilir.
- **Sınıf Sınırı:** Herhangi bir sınıf (class) en fazla **300 satır** uzunluğunda olabilir.
- **Bölme Yöntemi:** Bu limitleri aşmaya başlayan dosyalar, mantıksal alt sorumluluklara göre bölünerek modüler paketler veya yardımcı modüller (helpers) haline getirilmelidir.
  - *Örnek (Database Repositories):* Bloat (1085 satırlık) durumdaki `repositories.py` kaldırılmış ve mantıksal olarak `schema.py`, `experiment.py`, `best_model.py`, `forecast.py`, `forecast_resolution.py` modüllerine ayrılarak `src/database/repositories/` paketine dönüştürülmüştür.
  - *Örnek (Pipeline Orchestrator):* `orchestrator.py` içindeki rapor oluşturma ve manifest dosyası yazma gibi I/O ve dosya formatlama sorumlulukları `src/pipeline/artifacts.py` dosyasına taşınmıştır.

## Giriş Doğrulama (Input Validation) ve Güvenlik

Sisteme dışarıdan veya üst katmanlardan gelen verilerin doğrulama süzgecinden geçirilmesi zorunludur:
- **Sembol Doğrulamaları:** Hisse senedi sembolleri gibi metin tabanlı giriş parametreleri için regex kontrolü uygulanır. Kabul edilen format: `^[A-Z0-9]{1,10}$` (örn. `TUPRS`, `KCHOL`, `ASELS`).
- **Path Traversal ve SQL Injection:** Dışarıdan alınan yol (path) ve parametrelerin SQL Injection veya Path Traversal ataklarına sebep olmaması için parametrik sorgular kullanılmalı, doğrudan string eklemelerinden kaçınılmalı ve dosya yolları (`os.path.abspath`) doğrulanmalıdır.
- **İş Parçacığı Sınırlandırması (Thread Bounding):** Arka planda çalışan işlemler kontrolsüz kaynak tüketimini (thread exhaustion) önlemek adına sınırlandırılmış havuzlar (`ThreadPoolExecutor(max_workers=4)`) üzerinden yürütülmelidir.

## Hata Yönetim (Exception Handling) Prensipleri

Sistemdeki kritik akışlarda hataların sessizce yutulması kesinlikle yasaktır:
- Boş hata yakalama blokları (`except Exception: pass`) kullanılmamalıdır.
- Yakalanan istisnalar en azından `logger.warning` veya `logger.exception` çağrıları ile izlenebilir hale getirilmeli veya uygun bir hata iletisiyle yukarı fırlatılmalıdır.
- API katmanlarında, doğrulama hataları yakalanarak uygun HTTP durum kodları (örn. 400 Bad Request) istemciye dönülmelidir.

## Zaman Dilimi ve Tarih Disiplini

Sistem genelinde zaman karşılaştırmalarının tutarlı olması ve sunucu farklarından etkilenmemesi için:
- Timezone-naive (zaman dilimsiz) `datetime.now()` kullanımı yerine timezone-aware (zaman dilimine duyarlı) datetime nesneleri kullanılmalıdır.
- Varsayılan olarak UTC veya İstanbul zaman dilimleri tercih edilmelidir:
  ```python
  from datetime import timezone
  datetime.now(timezone.utc).isoformat(timespec="seconds")
  ```
