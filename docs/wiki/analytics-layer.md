---
title: DuckDB Analiz Katmanı
type: architecture
status: historical
last_updated: 2026-11-16
owner: llm
---

> **TARİHSEL (2026-11-16):** DuckDB analitik katmanı kodda yok. Güncel durum için [Architecture](architecture.md), [Persistence and API](persistence-and-api.md) ve [Risk Modeli](risk-model.md) sayfalarına bakın. Bu sayfa yalnız referans içindir; kod > test > wiki.

# DuckDB Analiz Katmanı

Salt-okunur, bellek-içi DuckDB katmanı: **çapraz-koşu / çapraz-kaynak analitik** içindir (koşular arası kıyas, forecast doğruluğu, IC geçmişi, veri kapsaması). **Servis (OLTP) yolu değildir**; SQLite (`data/*.db`) servis ve kayıt için aynen kalır.

## Neden SQLite kalıyor?

Gerçek veride ölçüm: SQLite leaderboard sorgusu 1-4 ms, DuckDB 9-121 ms (DuckDB tek-satır/küçük sorguda sütunlu motor yükü öder; API gecikme bütçesi için yanlış araç). DuckDB'nin değeri, birden çok kaynağı (CSV koşu raporları + SQLite + Parquet) tek SQL'de birleştirmesidir. Koşu çıktıları CSV/JSON kalır (biçim göçü yok).

## Ağdan eklenti yok

`sqlite_scanner` eklentisi pakete gömülü DEĞİL, çalışma anında ağdan indirme gerektirir; bu yüzden `INSTALL/LOAD` kullanılmaz. SQLite, Python `sqlite3` ile `file:...?mode=ro` URI'siyle (`src/database/connection.py::ClosingConnection`) açılır, tablolar pandas'a okunup `register()` edilir (DB'ler küçük). Yalnız gömülü yetenekler: `read_csv`, `read_parquet`, JSON fonksiyonları. Kaynaklara hiç yazılmaz (yazma girişimi `OperationalError: readonly`; testlerle dosya sha256'sı değişmez doğrulanır).

## Modüller

- `src/analysis/analytics_db.py`: `AnalyticsDB(project_root, outputs_dir, main_db, serving_db, clean_store_dir)`; lazy bağlantı, `.views()`, `.sql(query)` (tek SELECT/WITH; `extract_statements` ile ifade türü doğrulanır, DML/DDL/çoklu ifade reddedilir), `.query(name, symbol=None)`.
- `src/analysis/analytics_sources.py`: koşu taraması, salt-okunur SQLite köprüsü, CSV view üretimi.
- `src/analysis/analytics_queries.py`: adlandırılmış sorgular (`QUERIES`).
- `tools/analytics.py`: CLI (`.gitignore` istisnası `!tools/analytics.py`).
- `duckdb==2.0.0` (`requirements.txt`).

## View'lar (kaynak varsa kurulur)

| View | Kaynak | Not |
|---|---|---|
| `runs` | `outputs/{SYM}/runs/{RUN_ID}/` | Python taraması (manifestler heterojen). Kolonlar: symbol, run_id, status, has_* bayrakları, generated_at/started_at/finished_at (yoksa NULL), mode (`validation_protocol`), hash'ler, random_seed, git_commit, research_*, final_holdout_status, `model_list`, n_models. |
| `run_model_metrics` | `csv/backtest_report_{wf,final_holdout}.csv` | Uzun biçim: symbol, run_id, `split` (`wf`/`final_holdout`), `model` + ham kolonlar. |
| `run_experiments` | `experiments/experiment_log.csv` | Koşu başına deney günlüğü (Dir_Acc, RMSE_vs_benchmark, Composite_Score...). |
| `experiments`, `best_models`, `forecast_runs`, `forecast_points` | `data/stock_models.db` | |
| `peer_runs` (=`global_model_runs`), `peer_scores` | `data/serving_pool.db` | Aynı dosyada yaşıyorsa ana DB'den de alınır. |
| `panel`, `panel_manifest` | `data/clean/*.parquet` | CleanStore çıktısı; yoksa kurulmaz. |

CSV okuma: `all_varchar` + dinamik `TRY_CAST(replace(',', '.') AS DOUBLE)` (dosyalar arası tip tutarsızlığı ve ondalık virgül toleransı; `NaN` -> NULL; `inf` korunur), ayırıcı dosya başlığından koklanır (`;` rapor, `,` günlük), `union_by_name`. `run_model_metrics` ve `run_experiments` küçük olduğundan **tabloya materyalize** edilir (açılış anının anlık görüntüsü; her sorguda yüzlerce CSV tekrar okunmaz). Sembol dizini `^[A-Z0-9]{1,10}$` değilse atlanır.

### `runs.status` türetimi

Manifest'te `status` alanı yoksa (bugünkü manifestlerde yok): manifest yok + son 2 saatte değişmiş -> `running`; manifest yok + eski -> `failed`; manifest bozuk -> `failed`; manifest var ama `REQUIRED_RUN_FILES` (wf, final_holdout, validation raporları) eksik -> `incomplete`; hepsi var -> `completed`. Manifest'te açık `status` varsa o kullanılır.

## Adlandırılmış sorgular

| Ad | Açıklama |
|---|---|
| `leaderboard` | Sembol başına **son `completed` koşunun** ticaret lideri. Kural `run_leaderboard._rank_leaders`: `final_excess*1000 + final_net` (NULL=-1e9), aday = final_holdout satırı olan ve `model_list`'te bulunan modeller; `leader_reliability_class` `_reliability_class` ile aynı eşikler (işlem bandı 8-20, klon toleransı 0,001, kararsızlık farkı 0,25). Ek: `prediction_leader` (wf_net_return). |
| `forecast_accuracy` | Çözümlenmiş `forecast_points`: model/sembol/ufuk adımı başına n, `direction_accuracy`, `mae_return`, `interval_coverage` (p10-p90), `mean_interval_width`. |
| `ic_history` | `peer_runs` IC/ICIR zaman serisi + `icir_change`, config JSON'dan `boost`. Sembol kolonu yok. |
| `data_coverage` | `panel_manifest` (yoksa `panel`) -> sembol başına first/last date, n_rows, rejected_rows, `lag_days` (en güncel son tarihe göre). |
| `run_status` | Sembol başına completed/incomplete/running/failed sayıları, son ve son tamamlanan run_id. |
| `model_stability` | Sembol+model+doğrulama modu başına koşu sayısı, Dir_Acc ve RMSE_vs_benchmark ortalama/std (+Sharpe, Net_Return, Composite). |

`has_symbol` olan sorgular `--symbol` ile süzülür (regex doğrulama + parametreli sorgu).

## CLI

```bash
python tools/analytics.py list
python tools/analytics.py views
python tools/analytics.py query leaderboard --format md
python tools/analytics.py query run_status --symbol THYAO
python tools/analytics.py query forecast_accuracy --format csv --out outputs/analytics/acc.csv
python tools/analytics.py sql "SELECT symbol, count(*) n FROM runs GROUP BY 1"
```

`sql` yalnız tek SELECT/WITH kabul eder; aksi çıkış kodu 2 + `HATA:` mesajı. Not: SELECT, `read_csv('...')` gibi tablo fonksiyonlarıyla yerel dosya okuyabilir; bağlantı bellek-içi ve kaynaklar salt-okunur olduğundan yazma riski yoktur, ama araç güvenilmeyen kullanıcıya açılmamalıdır.

## Ölçümler (gerçek veri, 2026-11-15)

Açılış ~2,2 s (178 koşu taraması, 357 rapor + 162 günlük CSV okuma, 2 SQLite). View satırları: runs 178, run_model_metrics 957, run_experiments 904, experiments 964, best_models 16, forecast_runs 44, forecast_points 220, peer_runs 3, peer_scores 1722 (`data/clean/` henüz yok -> panel view'ları yok). Sorgu medyanı: leaderboard 32 ms, model_stability 17 ms, forecast_accuracy 5 ms, run_status 4 ms, ic_history 3 ms.

**Parite:** `leaderboard` sorgusu `build_multi_symbol_leaderboard` ile (varsayılan CLI argümanları) 14/14 sembolde aynı lider, aynı run_id, aynı metrikler (<1e-9), aynı güvenilirlik sınıfı ve bayraklar. Sınırlar: referans tüm koşuların birlikte sıralamasını yapar, bizimki son tamamlanan koşu içinde lider seçer (talep edilen tanım); eşit skorda deterministik olarak model adı sıralanır; referanstaki geçmiş/sektör metadata kolonları (history_*, sector) yoktur.

## Testler

`tests/test_analytics_db.py` (26 test): view'ların yalnız kaynak varken kurulması, durum türetimi, her adlandırılmış sorgu, `run_leaderboard` paritesi, SELECT dışı reddi, SQLite salt-okunur + dosya değişmezliği, CLI.
