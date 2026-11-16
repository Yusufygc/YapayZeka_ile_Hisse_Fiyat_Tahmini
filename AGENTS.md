# Project and Wiki Guidelines

You are the lead developer of this project and the sole maintainer of the knowledge base (Wiki) located in the `docs/wiki/` directory.

## Core Rules
1. **Consult the Wiki:** Whenever a question is asked or a new feature is requested, you must read the `docs/wiki/index.md` file before touching any code. Understand the architectural decisions and the current state from there.
2. **Self-Update (Ingest):** If we make a new technology decision, find a solution to a bug, or plan a new feature during our conversation, do not just keep it in your memory. Update the relevant markdown file within `docs/wiki/` or create a new one.
3. **Keep a Log (Log):** Append every significant architectural change or wiki update to the very top of the `docs/wiki/log.md` file with a timestamp (using the `## [YYYY-MM-DD] Action | Topic` format).

## Wiki Directory (`docs/wiki/`)
- `index.md`: The map and cross-references of all wiki pages.
- `log.md`: The chronological record of all updates made.
- `architecture.md`: Technologies used, database schema, and overall structure.

# AGENTS.md

This file is the schema layer for the repository's LLM-maintained wiki. The
wiki follows the "LLM Wiki" pattern: raw project sources are compiled into a
persistent, cross-linked markdown knowledge base that improves over time.
Repository-level change and commit rules live in `RULES.md`.

## LLM Wiki Operating Model

There are three layers:

1. **Raw sources:** Source code, tests, README, existing docs, data metadata, and user-provided decisions. Source code and tests are the primary truth for behavior.
2. **Wiki:** `docs/wiki/*.md`. The LLM owns this layer: summaries, cross-links, decisions, contradiction notes, and maintenance logs.
3. **Schema:** This `AGENTS.md` file. It defines how agents should operate on the wiki and codebase.
4. **Rules:** `RULES.md`. It defines how system changes, wiki updates, and Turkish commit messages are handled.

When the wiki and source code disagree, trust the source code, update the wiki,
and log the correction.

## Mandatory Wiki Workflow

Before code or architecture work:

1. Read `docs/wiki/index.md`.
2. Read the relevant linked wiki pages.
3. Inspect source code or tests when exact behavior matters.

When a decision, bug solution, feature plan, or durable explanation appears:

1. Update the relevant wiki page or create a new one.
2. Update `docs/wiki/index.md` if page navigation changes.
3. Add a new top entry to `docs/wiki/log.md`.
4. Follow `RULES.md` for wiki updates and Turkish commit messages.

For detailed wiki conventions, use `docs/wiki/wiki-guidelines.md`.

## Wiki Pages

- `docs/wiki/index.md`: Wiki map and current project state.
- `docs/wiki/wiki-guidelines.md`: LLM Wiki rules, frontmatter, ingest/query/lint workflows.
- `docs/wiki/change-management.md`: Change, wiki-update, and Turkish commit-message workflow.
- `docs/wiki/source-map.md`: Raw source map and where facts should be filed.
- `docs/wiki/architecture.md`: System architecture, technology stack, config, outputs, database schema.
- `docs/wiki/data-pipeline.md`: Data loading, features, scaling, targets, leakage boundaries.
- `docs/wiki/model-catalog.md`: Kol-B pooled model, risk model, retired Kol-A.
- `docs/wiki/risk-model.md`: Pooled volatility risk model (target, features, calibration, gate, validation numbers).
- `docs/wiki/validation-and-backtesting.md`: Historical Kol-A validation notes; Kol-B/risk validation pointers.
- `docs/wiki/persistence-and-api.md`: PeerStore, nightly status, FastAPI API v2.
- `docs/wiki/testing-and-quality.md`: Test areas, commands, quality invariants.
- `docs/wiki/log.md`: Append-first chronological update log.

## Project Overview

`ts_forecasting_lab` is a nightly pooled-model analysis service for BIST equities
(decision support, not a trading bot). One pooled model (Kol-B: LightGBM + MLP
ensemble) learns the whole universe; every night it ranks stocks against each other
(peer score, calibrated trend tendency, confidence), and a pooled volatility risk
model (`src/risk/`, HAR-X Ridge) sets the 5-day risk level and the 80% price band
(`kolb_price_low/high`). Results go to `data/serving_pool.db` (`PeerStore`:
`global_model_runs`, `peer_scores`) plus `data/nightly_status.json`. The API and the
desktop GUI only READ that database (read-only `PeerStore.open_readonly`); no model
is trained at request time.

The per-symbol direction models (Kol-A), `ForecastingPipeline`, the backtest engine,
`src.cli.*`, `model_scope.py` and `stock_model_db.py` were removed (decision "C";
see git history and `docs/wiki/model-catalog.md`).

```text
data/*.csv -> src/serving/nightly.py -> data/serving_pool.db (PeerStore)
              (trading-day gate, refresh,      |-> src/api  (FastAPI, API v2, read-only)
               pooled scoring, risk model)     |-> src/gui  (PySide6 single table)
                                               `-> data/nightly_status.json (/health)
```

## Main Commands

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Nightly job (Windows Task Scheduler runs scripts/nightly_serving.ps1 at 21:00;
# register with scripts/register_nightly_task.ps1). --horizon only accepts 5.
python -m src.serving.nightly --db data/serving_pool.db --model ensemble --fallback-dir data/yedek
python -m src.serving.nightly --help

# API (binds 127.0.0.1; override with AI_CORE_API_HOST / AI_CORE_API_PORT)
python -m src.api.main   # 127.0.0.1:8001 (AI_CORE_API_PORT ile degisir)

# Desktop GUI
python -m src.gui.main

# Tests (use the project .venv; PYTHONIOENCODING=utf-8)
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

## Architecture Rules

- Preserve chronological ordering in all validation logic; purge/embargo by `target_date`.
- Fit scalers/calibrations only on data available before the evaluated slice (risk bias
  `c` and conformal `z` come from PREVIOUS OOS blocks only). Never tune on the final holdout.
- The product is served ONLY from the nightly output. API/GUI readers must open the DB
  read-only (`PeerStore.open_readonly`); they never create, migrate or change journal mode.
- Writers (`src/serving/nightly.py`) use `PeerStore(path)`; schema changes are additive
  (`_PEER_MIGRATIONS`) and old DBs must stay readable (missing columns -> None).
- Risk model and Kol-B quintile table are 5-day; do not add other horizons without
  re-validating `src/risk/validation.py` (acceptance gate constants are pre-registered).
- API: DB missing/locked/corrupt -> `analysis_status="unavailable"` (HTTP 503), symbol absent
  -> `no_data`; neither is cached. 500 responses never echo exception text; `/health` exposes
  no internal paths.
- No empty `except`; timezone-aware datetimes (Istanbul = UTC+3); files <= 500 lines.
- Keep generated runtime artifacts (`data/*.db`, `data/clean/`, caches) out of source control.

## Editing Guidance

- Follow existing module boundaries; reachability roots are `src.api.main`, `src.gui.main`,
  `src.serving.nightly`, `tools/*.py` and `scripts/*.ps1`. Code reachable from none of them is dead.
- Data behavior: add tests around leakage, chronological splits, quarantine and target conversion.
- Risk/serving behavior: test the gate (`mse_ratio`, DM t, coverage), `risk_method` fallback
  and the API status mapping (`ok`/`stale_data`/`low_confidence`/`no_data`/`unavailable`).
- API/GUI changes: use a temporary `PeerStore` DB; assert readers leave the DB unchanged.
- When a commit is requested, write the commit message in clear Turkish with correct Turkish
  characters and no AI attribution lines (`RULES.md`).

## Documentation Discipline

Use the wiki as the durable memory of the project. Chat history is temporary;
the wiki is the maintained knowledge base.

## Detaylı Wiki Güncelleme ve Kontrol Kuralı (Auto-Refresh)

Aşağıdaki durumlarda **kullanıcı hatırlatmasa bile** detaylı wiki (`docs/wiki/`) güncelleme ve kontrol kuralı otomatik işletilir:

1. **Her commit sonrası** — `git commit` başarılı olduğu anda değişiklikler detaylıca incelenip wiki'ye işlenir. Commit'in dokunduğu mimari ve davranış değişiklikleri ilgili sayfalara yansıtılır.
2. **Anlamlı kod/yapı değişikliğinden sonra** — yeni dosya, silinen dosya, fonksiyon/sınıf imzası değişikliği, mimari taşıma, model registry değişikliği yapıldığında commit beklenmeden wiki güncellenir.
3. **Yeni kararlar alındığında** — tasarım kararları, epic bitişleri veya teknoloji seçimleri anında ilgili wiki sayfalarına eklenir veya yeni sayfa oluşturulur.

İşletim:
- Wiki güncellendiğinde mutlaka `docs/wiki/log.md` dosyasına en üste timestamp ile (`## [YYYY-MM-DD] Action | Topic`) not düşülür.
- Projenin ana navigasyonu veya ilişkili dosyaları değiştiyse `docs/wiki/index.md` çapraz referansları da güncellenir.
- Kaynak doğruluk sırası: **kod > test > wiki**. Uyumsuzluk tespit edildiğinde öncelikle kod baz alınır ve wiki düzeltilerek loglanır.
- Bu kural CAVEMAN modunda da geçerlidir. Pasif kalıp "wiki güncelleneyim mi?" diye sormak yok — değişiklik yapıldığı anda otomatik olarak güncellenir ve kısa bir sonuç raporlanır.
