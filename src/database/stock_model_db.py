# -*- coding: utf-8 -*-
"""
stock_model_db.py — SQLite Tabanlı Model Kayıt ve Yönetim Sistemi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Her hisse senedi için tüm model denemelerini ve "en iyi modeli"
merkezi bir SQLite veritabanında tutar.

Tablolar:
  - experiments  : Her eğitim çalışmasının tam metrik kaydı
  - best_models  : Her hisse için anlık en iyi model (otomatik güncellenir)

Bileşik Skor Formülü (0–100 arası):
  composite = benchmark-relative hata, yön, Sharpe ve aktiflik bileşenlerinden üretilir.

Kullanım:
    db = StockModelDB("path/to/stock_models.db")
    db.log_experiment(stock_symbol, model_name, metrics, model_path, ...)
    best = db.get_best_model("TUPRS")   # ileriye dönük otomatik seçim için
"""

import hashlib
import math
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from src.database.connection import open_db

from src.evaluation.scoring import compute_composite_score  # noqa: F401  (re-export; mantik scoring.py'de)
from src.pipeline.model_scope import BENCHMARK_MODELS


from src.database.schema_ddl import (  # noqa: F401  (re-export; SchemaRepository `schema._CREATE_*` okur)
    _CREATE_EXPERIMENTS,
    _CREATE_BEST_MODELS,
    _CREATE_IDX_SYMBOL,
    _CREATE_IDX_SCORE,
    _CREATE_FORECAST_RUNS,
    _CREATE_FORECAST_POINTS,
    _CREATE_FORECAST_ACCURACY,
    _CREATE_IDX_FORECAST_SYMBOL,
    _CREATE_IDX_FORECAST_RUN_KEY,
    _CREATE_IDX_FORECAST_POINTS_DATE,
    _CREATE_ANALYSIS_REFRESH_JOBS,
    _CREATE_IDX_REFRESH_JOBS_SYMBOL,
    REQUIRED_SCHEMA_TABLES,
)


# Süreç içi şema-hazır kaydı: yol -> dosya kimliği (dev, ino). Şema kurulumu +
# migrasyonlar + best_models yeniden inşası yazma işlemidir; her `StockModelDB(...)`
# (örn. her API isteği) çağrısında tekrarlanmamalı. Dosya silinip yeniden
# oluşturulursa kimlik değişir ve şema yeniden kurulur.
_SCHEMA_READY: Dict[str, Tuple[int, int]] = {}
_SCHEMA_LOCK = threading.RLock()
_DB_CACHE: Dict[str, "StockModelDB"] = {}
_DB_CACHE_LOCK = threading.Lock()


def _db_key(db_path: str) -> str:
    return os.path.normcase(os.path.abspath(db_path))


def _file_identity(db_path: str) -> Optional[Tuple[int, int]]:
    try:
        st = os.stat(db_path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def get_db(db_path: str) -> "StockModelDB":
    """Yol başına tek (thread-safe, önbellekli) `StockModelDB` döner.

    `StockModelDB` durumsuzdur (her işlemde yeni bağlantı açar), bu yüzden
    paylaşımı güvenlidir. Şema başlatma bu yol için süreçte bir kez çalışır.
    """
    key = _db_key(db_path)
    with _DB_CACHE_LOCK:
        db = _DB_CACHE.get(key)
        if db is None:
            db = StockModelDB(db_path)
            _DB_CACHE[key] = db
        return db


class StockModelDB:
    """
    SQLite tabanlı hisse-model kayıt yöneticisi.

    Args:
        db_path: Veritabanı dosyasının tam yolu.
                 Yoksa otomatik oluşturulur.
        init_schema: False ise şema kurulumu atlanır (çağıran `ensure_schema()`
                 çağırmalıdır). Varsayılan True; ancak aynı süreçte bu yol için
                 şema zaten kurulduysa tekrar çalıştırılmaz.
    """

    def __init__(self, db_path: str, init_schema: bool = True):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_repositories()
        if init_schema:
            self.ensure_schema()

    def ensure_schema(self, force: bool = False) -> None:
        """Şema/migrasyon/best_models yeniden inşasını süreçte yol başına bir kez çalıştırır.

        `force=True` her koşulda yeniden çalıştırır (bakım araçları için).
        """
        key = _db_key(self.db_path)
        with _SCHEMA_LOCK:
            identity = _file_identity(self.db_path)
            if not force and identity is not None and _SCHEMA_READY.get(key) == identity:
                return
            self._init_db()
            _SCHEMA_READY[key] = _file_identity(self.db_path)

    def _init_repositories(self) -> None:
        from src.database.repositories import (
            BestModelRepository,
            ExperimentRepository,
            ForecastRepository,
            ForecastResolutionRepository,
            SchemaRepository,
        )

        self.schema_repository = SchemaRepository(self)
        self.best_model_repository = BestModelRepository(self)
        self.experiment_repository = ExperimentRepository(self, self.best_model_repository)
        self.forecast_repository = ForecastRepository(self)
        self.forecast_resolution_repository = ForecastResolutionRepository(self)

    def _ensure_repositories(self) -> None:
        if not all(
            hasattr(self, attr)
            for attr in (
                "schema_repository",
                "best_model_repository",
                "experiment_repository",
                "forecast_repository",
                "forecast_resolution_repository",
            )
        ):
            self._init_repositories()

    def _connect(self) -> sqlite3.Connection:
        # WAL + busy_timeout + foreign_keys: src.database.connection.open_db.
        return open_db(self.db_path)

    def _init_db(self) -> None:
        self._ensure_repositories()
        return self.schema_repository.initialize()

    def log_experiment(
        self,
        stock_symbol: str,
        model_name: str,
        metrics: Dict[str, float],
        model_path: str = "",
        features: List[str] = None,
        dataset_hash: str = "N/A",
        validation_mode: str = "single_split",
        dataset_metadata: Optional[Dict[str, Any]] = None,
        is_production_candidate: bool = False,
        selection_source: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """Bir model değerlendirme deneyini kaydeder (ExperimentRepository'e delege).

        `is_production_candidate` + `validation_mode` best-model seçim guard'ını
        besler.

        Returns:
            Eklenen deney kaydının id'si.
        """
        self._ensure_repositories()
        return self.experiment_repository.log_experiment(
            stock_symbol=stock_symbol,
            model_name=model_name,
            metrics=metrics,
            model_path=model_path,
            features=features,
            dataset_hash=dataset_hash,
            validation_mode=validation_mode,
            dataset_metadata=dataset_metadata,
            is_production_candidate=is_production_candidate,
            selection_source=selection_source,
            run_id=run_id,
        )

    def get_best_model(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        """Sembolün üretim 'best model' kaydını döner (BestModelRepository'e delege)."""
        self._ensure_repositories()
        return self.best_model_repository.get_best_model(stock_symbol)

    def get_leaderboard(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """En iyi modellerin lider tablosunu döner (BestModelRepository'e delege)."""
        self._ensure_repositories()
        return self.best_model_repository.get_leaderboard(top_n)

    def get_experiments(
        self,
        stock_symbol: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Deney kayıtlarını filtreleyerek döner (ExperimentRepository'e delege)."""
        self._ensure_repositories()
        return self.experiment_repository.get_experiments(
            stock_symbol=stock_symbol,
            model_name=model_name,
            limit=limit,
        )

    def get_model_comparison(self, stock_symbol: str) -> List[Dict[str, Any]]:
        """Sembol için model karşılaştırma satırlarını döner (ExperimentRepository'e delege)."""
        self._ensure_repositories()
        return self.experiment_repository.get_model_comparison(stock_symbol)

    def get_cross_run_leaderboard(
        self, stock_symbol: str, n_runs: int = 5
    ) -> List[Dict[str, Any]]:
        """Çoklu run boyunca lider tabloyu döner (ExperimentRepository'e delege)."""
        self._ensure_repositories()
        return self.experiment_repository.get_cross_run_leaderboard(stock_symbol, n_runs=n_runs)

    def log_forecast_run(
        self,
        *,
        stock_symbol: str,
        model_name: str,
        source_experiment_id: Optional[int],
        last_observed_date: str,
        last_close: float,
        horizon_days: int,
        trend_label: str,
        weekly_expected_return: float,
        trend_threshold: float,
        rules_version: str,
        points: List[Dict[str, Any]],
        status: str = "pending",
        run_at: Optional[str] = None,
        ensemble_direction_agreement: Optional[float] = None,
        forecast_strategy: Optional[str] = None,
        artifact_mode: Optional[str] = None,
        forecast_warnings: Optional[List[str]] = None,
        ensemble_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Forecast çalıştırması ve nokta tahminlerini kaydeder (ForecastRepository'e delege).

        Returns:
            Eklenen forecast run kaydının id'si.
        """
        self._ensure_repositories()
        return self.forecast_repository.log_forecast_run(
            stock_symbol=stock_symbol,
            model_name=model_name,
            source_experiment_id=source_experiment_id,
            last_observed_date=last_observed_date,
            last_close=last_close,
            horizon_days=horizon_days,
            trend_label=trend_label,
            weekly_expected_return=weekly_expected_return,
            trend_threshold=trend_threshold,
            rules_version=rules_version,
            points=points,
            status=status,
            run_at=run_at,
            ensemble_direction_agreement=ensemble_direction_agreement,
            forecast_strategy=forecast_strategy,
            artifact_mode=artifact_mode,
            forecast_warnings=forecast_warnings,
            ensemble_metadata=ensemble_metadata,
        )

    def get_latest_forecast(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        """En son forecast çalıştırmasını döner (ForecastRepository'e delege)."""
        self._ensure_repositories()
        return self.forecast_repository.get_latest_forecast(stock_symbol)

    def get_forecast_history(self, stock_symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Forecast geçmişini döner (ForecastRepository'e delege)."""
        self._ensure_repositories()
        return self.forecast_repository.get_forecast_history(stock_symbol, limit=limit)

    def resolve_forecasts(self, stock_symbol: str, actual_prices: Dict[str, float]) -> int:
        """Gerçekleşen fiyatlarla forecast noktalarını eşleştirir (ForecastResolutionRepository'e delege)."""
        self._ensure_repositories()
        return self.forecast_resolution_repository.resolve_forecasts(stock_symbol, actual_prices)

    def resolve_forecasts_from_csv(self, stock_symbol: str, csv_path: str) -> int:
        """CSV'deki gerçekleşen fiyatlarla forecast'leri çözer (ForecastResolutionRepository'e delege)."""
        self._ensure_repositories()
        return self.forecast_resolution_repository.resolve_forecasts_from_csv(stock_symbol, csv_path)

    def get_rolling_resolution_accuracy(
        self, stock_symbol: str, days: int = 60
    ) -> Dict[str, Any]:
        """Son `days` günlük yuvarlanan çözümleme doğruluğunu döner (ForecastResolutionRepository'e delege)."""
        self._ensure_repositories()
        return self.forecast_resolution_repository.get_rolling_resolution_accuracy(
            stock_symbol, days=days
        )

    def create_or_get_refresh_job(
        self,
        *,
        symbol: str,
        reason: str,
        job_type: str = "analysis_refresh",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sembol+reason için açık (queued/running) refresh job'ı bulur veya yeni oluşturur.

        Idempotent: aynı sembol/reason için aktif bir job varsa onu döner, yoksa
        yeni `queued` job ekler.

        Returns:
            Mevcut veya yeni oluşturulan refresh job satırı (dict).
        """
        import json as _json
        import uuid as _uuid
        from datetime import datetime as _datetime

        symbol = symbol.upper()
        payload_json = _json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
        now = _datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT *
                FROM analysis_refresh_jobs
                WHERE symbol = ?
                  AND reason = ?
                  AND status IN ('queued', 'running')
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (symbol, reason),
            ).fetchone()
            if existing is not None:
                return dict(existing)

            job_id = _uuid.uuid4().hex[:12]
            conn.execute(
                """
                INSERT INTO analysis_refresh_jobs
                    (job_id, symbol, job_type, status, reason, started_at, payload_json)
                VALUES (?, ?, ?, 'queued', ?, ?, ?)
                """,
                (job_id, symbol, job_type, reason, now, payload_json),
            )
            return dict(conn.execute(
                "SELECT * FROM analysis_refresh_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone())

    def update_refresh_job(
        self,
        job_id: str,
        *,
        status: str,
        error: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        finish: bool = False,
    ) -> None:
        """Refresh job durumunu günceller; `finish=True` ise bitiş zamanı yazılır.

        `error` ve `payload` yalnızca verildiğinde (COALESCE ile) güncellenir.
        """
        import json as _json
        from datetime import datetime as _datetime

        finished_at = _datetime.now().isoformat(timespec="seconds") if finish else None
        payload_json = None if payload is None else _json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE analysis_refresh_jobs
                SET status = ?,
                    error = COALESCE(?, error),
                    payload_json = COALESCE(?, payload_json),
                    finished_at = COALESCE(?, finished_at)
                WHERE job_id = ?
                """,
                (status, error, payload_json, finished_at, job_id),
            )

    def get_refresh_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """`job_id`'ye karşılık gelen refresh job satırını döner (yoksa None)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_refresh_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_latest_refresh_job(self) -> Optional[Dict[str, Any]]:
        """En son başlatılan refresh job satırını döner (yoksa None)."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM analysis_refresh_jobs
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    def get_table_counts(self, tables: Optional[List[str]] = None) -> Dict[str, Optional[int]]:
        """Tablo başına satır sayısını döner; erişilemeyen tablo için None.

        `tables` verilmezse `REQUIRED_SCHEMA_TABLES` kullanılır (yalnızca dahili
        sabit tablo adları beklenir).
        """
        tables = list(tables or REQUIRED_SCHEMA_TABLES)
        counts: Dict[str, Optional[int]] = {}
        with self._connect() as conn:
            for table in tables:
                try:
                    counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                except sqlite3.Error:
                    counts[table] = None
        return counts

    def get_schema_status(self) -> Dict[str, Any]:
        """Şema sağlığını döner: zorunlu tablolar, eksik tablolar ve satır sayıları."""
        with self._connect() as conn:
            existing = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        missing = [table for table in REQUIRED_SCHEMA_TABLES if table not in existing]
        return {
            "ok": not missing,
            "required_tables": list(REQUIRED_SCHEMA_TABLES),
            "missing_tables": missing,
            "table_counts": self.get_table_counts(REQUIRED_SCHEMA_TABLES),
        }

    @staticmethod
    def _forecast_run_key(
        stock_symbol: str,
        model_name: str,
        source_experiment_id: Optional[int],
        last_observed_date: str,
        horizon_days: int,
        rules_version: str,
    ) -> str:
        raw = "|".join([
            stock_symbol.upper(),
            model_name,
            str(source_experiment_id or ""),
            str(last_observed_date)[:10],
            str(int(horizon_days)),
            rules_version,
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _sign(value: float) -> int:
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    def __repr__(self) -> str:
        return f"<StockModelDB path={self.db_path!r}>"
