---
title: Change Management
type: workflow
status: active
last_updated: 2026-05-09
owner: llm
source_count: 1
---

# Değişiklik Yönetimi

Bu sayfa, depoda yapılan değişikliklerin wikiye ve commit geçmişine nasıl
yansıtılacağını tanımlar. Ana kural dosyası `RULES.md` dosyasıdır.

## Temel Kural

Sistemde yapılan her anlamlı değişiklikte ilgili wiki dosyaları da güncellenir.
Bu kapsam şunları içerir:

- Dosya ekleme
- Dosya silme
- Dosya yeniden adlandırma
- Kod davranışı değişikliği
- Mimari karar
- Hata çözümü
- Yeni özellik planı
- Kalıcı çalışma veya bakım kuralı değişikliği

Gerekirse `docs/wiki/` altında yeni bir Markdown dosyası oluşturulur. Yeni sayfa
oluşturulduğunda `docs/wiki/index.md` güncellenerek sayfa wiki haritasına
eklenir.

## Commit Kuralı

Değişiklikler Türkçe karakterlere dikkat edilerek, açık ve anlaşılır Türkçe
açıklamalarla commit edilir.

İyi commit mesajı:

- Ne değiştiğini söyler.
- Neden değiştiğini kısa biçimde açıklar.
- Gereksiz teknik gürültü içermez.
- Türkçe karakterleri doğru kullanır.

Örnek:

```text
Wiki bakım kurallarını ekle

RULES.md dosyası oluşturuldu ve değişikliklerde wiki güncelleme zorunluluğu
AGENTS.md ile docs/wiki sayfalarına işlendi.
```

## Ajan İş Akışı

1. Değişikliğe başlamadan önce `docs/wiki/index.md` okunur.
2. İlgili wiki sayfaları ve kaynak dosyalar incelenir.
3. Kod, doküman veya yapılandırma değişikliği yapılır.
4. Değişiklik kalıcı bilgi içeriyorsa ilgili wiki sayfası güncellenir.
5. Yeni bir wiki sayfası gerekiyorsa oluşturulur ve `index.md` içine eklenir.
6. `docs/wiki/log.md` dosyasının en üstüne yeni kayıt yazılır.
7. Değişiklikler Türkçe ve açıklayıcı bir commit mesajıyla commit edilir.

## İlgili Sayfalar

- [Wiki Guidelines](wiki-guidelines.md)
- [Source Map](source-map.md)
- [Testing and Quality](testing-and-quality.md)
