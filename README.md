# ts_forecasting_lab

BIST (Borsa İstanbul) hisseleri için **gecelik havuz (pooled) analiz servisi**.

Tek bir pooled model (Kol-B: LightGBM + MLP ensemble) tüm evreni birlikte öğrenir; her gece
hisseleri birbirine göre sıralar (akran skoru), kalibre trend eğilimi, risk ve fiyat aralığı
üretir. Sonuçlar `data/serving_pool.db` içine yazılır; API ve masaüstü GUI yalnızca bu veritabanını
okur. İstek sırasında model eğitilmez.

> **Karar "C":** Hisse başına yön modelleri (Kol-A: XGBoost/Ridge/LightGBM, LSTM vb.) 815
> walk-forward deneyin hiçbirinde naive tahmini geçemedi ve ürün kapsamından çıkarıldı. Eğitim
> yığını, backtest motoru, sinyal kalibrasyonu, deney kaydı (`stock_models.db`) ve onlara özgü
> CLI/GUI/endpoint'ler silindi. Eski sonuçlar için git geçmişine bakın.

## Mimari

```
data/*.csv (yfinance)  ->  src/serving/nightly.py  ->  data/serving_pool.db (PeerStore)
                            (işlem günü kapısı, veri        |
                             tazeleme, pooled skorlama,     +--> src/api   (FastAPI, salt okunur)
                             risk modeli)                   +--> src/gui   (PySide6 özet tablosu)
                                                            +--> data/nightly_status.json
```

| Dizin | Görev |
|---|---|
| `src/data/` | OHLCV indirme/temizleme, panel yükleyici, tazelik, temiz depo (Parquet) |
| `src/features/` | Teknik özellik üretimi (makro katman kaldırıldı), Parquet özellik önbelleği |
| `src/risk/` | Havuzlu oynaklık (risk) modeli: HAR-X Ridge, konformal %80 bant, kabul kapısı |
| `src/models/` | Kol-B modelleri: `global_pooled_model`, `torch_mlp_model`, `ensemble_pooled_model` |
| `src/validation/` | Pooled purged CV, OOS değerlendirme, segment IC, baz çizgileri |
| `src/serving/` | Gece işi, skorlama, `PeerStore` (SQLite), güven etiketi, trend eğilimi, XAI |
| `src/xai/` | Özellik sözlüğü, SHAP/permütasyon stratejisi, grup özeti (Kol-B XAI) |
| `src/api/` | FastAPI servisi (API v2) |
| `src/gui/` | PySide6 masaüstü özet paneli |
| `tools/` | Veri denetim/onarım ve Kol-B doğrulama araçları (`tools/archive/` arşiv) |

## Kurulum

```bash
pip install -r requirements.txt
# torch CPU wheel: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Kullanım

**Gece işi** (Windows Task Scheduler her gün 21:00; `scripts/register_nightly_task.ps1`):

```bash
python -m src.serving.nightly --db data/serving_pool.db --model ensemble
python -m src.serving.nightly --help
```

Kayıt: `data/serving_pool.db` (`global_model_runs`, `peer_scores`) ve `data/nightly_status.json`.

**API:**

```bash
python -m src.api.main                  # 127.0.0.1:8001, /docs: Swagger
```

| Endpoint | Açıklama |
|---|---|
| `GET /health` | Serving havuzu tazeliği (`serving_pool`) ve son gece işi durumu (`nightly_run`) |
| `GET /symbols` | Son gece koşusundaki semboller |
| `GET /analysis/{symbol}` (`/v1/analysis/{symbol}`) | Sembol analizi (API v2) |

`/analysis` yanıtı (`api_version: "2"`): `symbol`, `analysis_status` (`ok` · `stale_data` ·
`low_confidence` · `no_data`), `generated_at`, `data` (son gözlem/kapanış/tazelik), `confidence`
(Kol-B güven etiketi; bayat veri, yedek risk yöntemi veya veri kayması düşürür), `serving_run`,
`data_quality` (PSI), `peer` (akran skoru, trend, risk ve fiyat aralığı alanları, XAI) ve
`disclaimer`. v1'deki `model`, `forecast`, `performance`, `xai`, `forecast_source` ve refresh-job
alanları ile `/best-model`, `/experiments`, `/metrics`, `/leaderboard`, `/run/*`, `/refresh/*`
endpoint'leri kaldırıldı (kırıcı değişiklik).

**GUI:**

```bash
python -m src.gui.main
```

Son koşunun sembol tablosu (sıralanabilir/aranabilir): sektör, akran etiketi/yüzdeliği, trend,
risk seviyesi, fiyat bandı; üstte veri tarihi ve gece işi durumu.

## Test

```bash
python -m pytest tests -q
```

Bu yazılım yatırım tavsiyesi değildir.
