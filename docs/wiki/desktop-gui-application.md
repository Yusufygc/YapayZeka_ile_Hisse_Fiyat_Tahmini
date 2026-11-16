---
title: Desktop GUI Application
type: architecture-guide
status: active
last_updated: 2026-11-16
owner: llm
---

# Desktop GUI Application

`python -m src.gui.main` — PySide6 masaüstü paneli, **tek tablo**: son gecelik havuz koşusunun sembol
bazlı özeti. Eğitim/tahmin/deney sekmeleri (Kol-A) kaldırıldı.

- `src/gui/core/peer_reader.py` — `PeerReader.load()` -> `PeerSnapshot(run, rows, nightly, nightly_error, error)`.
  `data/serving_pool.db`'yi **salt-okunur** (`PeerStore.open_readonly`) okur; dosya yoksa/okunamıyorsa
  boş görüntü + hata metni; eski şemada risk kolonları `None`. `data/nightly_status.json` durum çubuğunda.
- `src/gui/tabs/dashboard_tab.py` — sıralanabilir/filtrelenebilir tablo: Hisse, Sektör, Akran Etiketi,
  Akran Yüzdeliği, Trend, Yükseliş Olasılığı, Risk Seviyesi, Fiyat Bandı Alt/Üst + özet kartları
  (`widgets/cards.py`, `MetricCard`).
- `src/gui/styles/` — modüler QSS (`base`, `widgets`, `custom`) + `manager.py`.
- Test: `tests/test_gui_smoke.py` (offscreen).

Bkz. [Persistence and API](persistence-and-api.md), [Architecture](architecture.md).
