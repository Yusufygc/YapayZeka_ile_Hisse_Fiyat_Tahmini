# -*- coding: utf-8 -*-
"""Salt-okunur DuckDB analiz katmanı (çapraz-koşu / çapraz-kaynak analitik).

DuckDB burada SERVİS için değil analitik içindir: SQLite (data/*.db) servis
yolunda kalır. Bellek-içi bir DuckDB bağlantısı, kaynaklar VARSA şu view'ları kurar:

    runs, run_model_metrics, run_experiments        <- outputs/{SYM}/runs/{RUN_ID}/
    experiments, best_models, forecast_runs,
    forecast_points                                 <- data/stock_models.db (salt-okunur köprü)
    peer_runs, peer_scores                          <- data/serving_pool.db (salt-okunur köprü)
    panel, panel_manifest                           <- data/clean/*.parquet

Ağdan eklenti indirilmez (sqlite_scanner YOK): SQLite, Python `sqlite3` ile
`mode=ro` açılıp pandas üzerinden `register()` edilir. Kaynaklara hiç yazılmaz.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import duckdb
import pandas as pd

from src.analysis import analytics_sources as src
from src.database.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

SYMBOL_RE = src.SYMBOL_RE
_MAIN_TABLES = {
    "experiments": "experiments",
    "best_models": "best_models",
    "forecast_runs": "forecast_runs",
    "forecast_points": "forecast_points",
}
_SERVING_TABLES = {"global_model_runs": "peer_runs", "peer_scores": "peer_scores"}
_SELECT_TYPES = ("SELECT",)


class AnalyticsError(ValueError):
    """Geçersiz sorgu / eksik kaynak gibi kullanıcıya bildirilecek hatalar."""


def validate_select(con: Any, query: str) -> str:
    """Tek bir SELECT/WITH..SELECT ifadesine izin verir; aksi AnalyticsError."""
    text = str(query or "").strip().rstrip(";").strip()
    if not text:
        raise AnalyticsError("Boş sorgu.")
    try:
        statements = con.extract_statements(text)
    except duckdb.Error as exc:  # ParserException vb.
        raise AnalyticsError(f"SQL ayrıştırılamadı: {exc}") from exc
    if len(statements) != 1:
        raise AnalyticsError("Yalnızca tek bir SELECT ifadesine izin verilir.")
    kind = str(statements[0].type).rsplit(".", 1)[-1].upper()
    if kind not in _SELECT_TYPES:
        raise AnalyticsError(f"Yalnızca SELECT/WITH sorgularına izin verilir (bulunan: {kind}).")
    return text


class AnalyticsDB:
    """Bellek-içi DuckDB + kaynaklar mevcutsa kurulan view'lar."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        outputs_dir: str | Path | None = None,
        main_db: str | Path | None = None,
        serving_db: str | Path | None = None,
        clean_store_dir: str | Path | None = None,
        running_window_hours: float = src.RUNNING_WINDOW_HOURS,
    ) -> None:
        root = Path(project_root) if project_root is not None else Path(PROJECT_ROOT)
        self.project_root = root
        self.outputs_dir = Path(outputs_dir) if outputs_dir is not None else root / "outputs"
        self.main_db = Path(main_db) if main_db is not None else root / "data" / "stock_models.db"
        self.serving_db = (
            Path(serving_db) if serving_db is not None else root / "data" / "serving_pool.db"
        )
        self.clean_store_dir = (
            Path(clean_store_dir) if clean_store_dir is not None else root / "data" / "clean"
        )
        self.running_window_hours = running_window_hours
        self._con: Any = None
        self._views: List[str] = []
        self._frames: Dict[str, pd.DataFrame] = {}  # register() nesnelerini canlı tutar

    # ------------------------------------------------------------------ lifecycle
    @property
    def con(self) -> Any:
        if self._con is None:
            self._con = duckdb.connect(":memory:")
            self._build_views()
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
            self._views = []
            self._frames = {}

    def __enter__(self) -> "AnalyticsDB":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ public API
    def views(self) -> List[str]:
        """Kurulan (kaynağı olan) kullanıcı view'ları; `_` önekli yardımcılar hariç."""
        _ = self.con
        return list(self._views)

    def sql(self, query: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        """Tek SELECT/WITH sorgusu çalıştırır, DataFrame döner."""
        text = validate_select(self.con, query)
        return self.con.execute(text, list(params) if params else None).df()

    def query(self, name: str, symbol: Optional[str] = None) -> pd.DataFrame:
        """Adlandırılmış sorgu; kaynak view'ı yoksa AnalyticsError."""
        from src.analysis.analytics_queries import QUERIES, resolve_sql

        if name not in QUERIES:
            raise AnalyticsError(f"Bilinmeyen sorgu: {name!r}. Seçenekler: {sorted(QUERIES)}")
        spec = QUERIES[name]
        body = resolve_sql(spec, set(self.views()))
        if body is None:
            raise AnalyticsError(
                f"{name!r} için gereken kaynak yok (view'lar: {self.views()}). "
                f"Gerekli: {[list(v[0]) for v in spec.variants]}"
            )
        if symbol is None:
            return self.sql(body)
        if not spec.has_symbol:
            raise AnalyticsError(f"{name!r} sorgusu sembol kolonu içermiyor (--symbol kullanılamaz).")
        sym = str(symbol).strip().upper()
        if not SYMBOL_RE.match(sym):
            raise AnalyticsError(f"Geçersiz sembol: {symbol!r}")
        return self.sql(f"SELECT * FROM ({body}) AS q WHERE symbol = ?", [sym])

    # ------------------------------------------------------------------ view building
    def _register(self, name: str, frame: pd.DataFrame, expose: bool = True) -> None:
        self._frames[name] = frame
        self._con.register(f"_t_{name}", frame)
        self._con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM _t_{name}")
        if expose and name not in self._views:
            self._views.append(name)

    def _build_views(self) -> None:
        self._build_run_views()
        self._build_sqlite_views()
        self._build_panel_views()

    def _build_run_views(self) -> None:
        scan = src.scan_runs(self.outputs_dir, self.running_window_hours)
        if not scan.rows:
            return
        self._register("_runs_raw", src.runs_frame(scan), expose=False)
        self._con.execute(
            """CREATE OR REPLACE VIEW runs AS SELECT
                 CAST(symbol AS VARCHAR) AS symbol, CAST(run_id AS VARCHAR) AS run_id,
                 CAST(run_dir AS VARCHAR) AS run_dir, CAST(status AS VARCHAR) AS status,
                 CAST(has_manifest AS BOOLEAN) AS has_manifest,
                 CAST(manifest_valid AS BOOLEAN) AS manifest_valid,
                 CAST(has_wf_report AS BOOLEAN) AS has_wf_report,
                 CAST(has_final_report AS BOOLEAN) AS has_final_report,
                 CAST(has_validation_report AS BOOLEAN) AS has_validation_report,
                 CAST(has_experiment_log AS BOOLEAN) AS has_experiment_log,
                 CAST(generated_at AS VARCHAR) AS generated_at,
                 CAST(started_at AS VARCHAR) AS started_at,
                 CAST(finished_at AS VARCHAR) AS finished_at,
                 CAST(mode AS VARCHAR) AS mode, CAST(data_hash AS VARCHAR) AS data_hash,
                 CAST(feature_pipeline_version AS VARCHAR) AS feature_pipeline_version,
                 CAST(model_config_hash AS VARCHAR) AS model_config_hash,
                 CAST(signal_config_hash AS VARCHAR) AS signal_config_hash,
                 CAST(random_seed AS BIGINT) AS random_seed,
                 CAST(git_commit AS VARCHAR) AS git_commit,
                 CAST(research_policy AS VARCHAR) AS research_policy,
                 CAST(research_phase AS VARCHAR) AS research_phase,
                 CAST(uses_final_holdout_for_selection AS BOOLEAN) AS uses_final_holdout_for_selection,
                 CAST(final_holdout_status AS VARCHAR) AS final_holdout_status,
                 CAST(model_list AS VARCHAR[]) AS model_list,
                 CAST(n_models AS BIGINT) AS n_models
               FROM _runs_raw"""
        )
        self._views.append("runs")
        if scan.report_files:
            src.create_csv_view(self._con, "run_model_metrics", scan.report_files, "report")
            self._views.append("run_model_metrics")
        if scan.experiment_files:
            src.create_csv_view(self._con, "run_experiments", scan.experiment_files, "experiment")
            self._views.append("run_experiments")

    def _build_sqlite_views(self) -> None:
        sources: Dict[Path, Dict[str, str]] = {}
        sources.setdefault(self.main_db, {}).update(_MAIN_TABLES)
        sources.setdefault(self.serving_db, {}).update(_SERVING_TABLES)
        for path, tables in sources.items():
            for name, frame in src.read_sqlite_tables(path, tables).items():
                self._register(name, frame)
        # Servis tabloları ana DB ile aynı dosyada yaşıyor olabilir (serving_db yoksa).
        missing = {t: v for t, v in _SERVING_TABLES.items() if v not in self._views}
        if missing and self.serving_db != self.main_db:
            for name, frame in src.read_sqlite_tables(self.main_db, missing).items():
                self._register(name, frame)

    def _build_panel_views(self) -> None:
        for view, filename in (("panel", "panel.parquet"), ("panel_manifest", "manifest.parquet")):
            path = self.clean_store_dir / filename
            if path.is_file():
                self._con.execute(
                    f"CREATE OR REPLACE VIEW {view} AS "
                    f"SELECT * FROM read_parquet({src.sql_quote(src.posix(path))})"
                )
                self._views.append(view)

