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

import sqlite3
import hashlib
import math
from typing import Any, Dict, List, Optional

from src.pipeline.model_scope import BENCHMARK_MODELS


_CREATE_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS experiments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_symbol     TEXT    NOT NULL,
    model_name       TEXT    NOT NULL,
    validation_mode  TEXT    NOT NULL DEFAULT 'single_split',
    target_mode      TEXT    NOT NULL DEFAULT 'price',
    feature_mode     TEXT    NOT NULL DEFAULT 'legacy_price_features',
    scaling_mode     TEXT    NOT NULL DEFAULT 'minmax',
    mae              REAL,
    rmse             REAL,
    mape             REAL,
    dir_acc          REAL,
    sharpe           REAL,
    hit_rate         REAL,
    composite_score  REAL,
    model_path       TEXT,
    features         TEXT,
    dataset_hash     TEXT,
    is_production_candidate INTEGER NOT NULL DEFAULT 0,
    selection_source TEXT,
    run_id           TEXT,
    trained_at       TEXT    NOT NULL
);
"""

_CREATE_BEST_MODELS = """
CREATE TABLE IF NOT EXISTS best_models (
    stock_symbol    TEXT    PRIMARY KEY,
    model_name      TEXT    NOT NULL,
    experiment_id   INTEGER NOT NULL,
    composite_score REAL    NOT NULL,
    target_mode     TEXT    NOT NULL DEFAULT 'price',
    feature_mode    TEXT    NOT NULL DEFAULT 'legacy_price_features',
    scaling_mode    TEXT    NOT NULL DEFAULT 'minmax',
    validation_mode TEXT    NOT NULL DEFAULT 'final_holdout',
    dataset_hash    TEXT,
    run_id          TEXT,
    selection_source TEXT,
    mae             REAL,
    rmse            REAL,
    mape            REAL,
    dir_acc         REAL,
    sharpe          REAL,
    hit_rate        REAL,
    model_path      TEXT,
    updated_at      TEXT    NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
"""

_CREATE_IDX_SYMBOL = """
CREATE INDEX IF NOT EXISTS idx_experiments_symbol
    ON experiments (stock_symbol);
"""

_CREATE_IDX_SCORE = """
CREATE INDEX IF NOT EXISTS idx_experiments_score
    ON experiments (stock_symbol, composite_score DESC);
"""

_CREATE_FORECAST_RUNS = """
CREATE TABLE IF NOT EXISTS forecast_runs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_key                TEXT    NOT NULL UNIQUE,
    stock_symbol           TEXT    NOT NULL,
    model_name             TEXT    NOT NULL,
    source_experiment_id   INTEGER,
    run_at                 TEXT    NOT NULL,
    last_observed_date     TEXT    NOT NULL,
    last_close             REAL    NOT NULL,
    horizon_days           INTEGER NOT NULL,
    trend_label            TEXT    NOT NULL,
    weekly_expected_return REAL,
    trend_threshold        REAL,
    rules_version          TEXT    NOT NULL,
    status                 TEXT    NOT NULL DEFAULT 'pending',
    FOREIGN KEY (source_experiment_id) REFERENCES experiments(id)
);
"""

_CREATE_FORECAST_POINTS = """
CREATE TABLE IF NOT EXISTS forecast_points (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                   INTEGER NOT NULL,
    target_date              TEXT    NOT NULL,
    horizon_index            INTEGER NOT NULL,
    raw_predicted_close      REAL,
    bounded_predicted_close  REAL,
    predicted_return         REAL,
    lower_band               REAL,
    upper_band               REAL,
    price_tick               REAL,
    actual_close             REAL,
    actual_return            REAL,
    abs_error                REAL,
    direction_correct        INTEGER,
    resolved_at              TEXT,
    UNIQUE(run_id, target_date),
    FOREIGN KEY (run_id) REFERENCES forecast_runs(id) ON DELETE CASCADE
);
"""

_CREATE_FORECAST_ACCURACY = """
CREATE TABLE IF NOT EXISTS forecast_accuracy_summary (
    run_id                   INTEGER PRIMARY KEY,
    stock_symbol             TEXT    NOT NULL,
    model_name               TEXT    NOT NULL,
    rmse                     REAL,
    mae                      REAL,
    mape                     REAL,
    dir_acc                  REAL,
    weekly_direction_correct INTEGER,
    resolved_points          INTEGER NOT NULL,
    updated_at               TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES forecast_runs(id) ON DELETE CASCADE
);
"""

_CREATE_IDX_FORECAST_SYMBOL = """
CREATE INDEX IF NOT EXISTS idx_forecast_runs_symbol
    ON forecast_runs (stock_symbol, run_at DESC);
"""

_CREATE_IDX_FORECAST_RUN_KEY = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_forecast_runs_run_key
    ON forecast_runs (run_key);
"""

_CREATE_IDX_FORECAST_POINTS_DATE = """
CREATE INDEX IF NOT EXISTS idx_forecast_points_target_date
    ON forecast_points (target_date);
"""


def compute_composite_score(metrics: Dict[str, float]) -> float:
    """
    Farklı ölçeklerdeki metrikleri 0-100 aralığına normalize edip
    ağırlıklı bileşik skor üretir.

    Yeni mantıkta benchmark-relative alanlar da hesaba katılır:
      - RMSE_vs_benchmark <= 1 ise model seçilen benchmark'ı en azından hata açısından yakalamıştır
      - DirAcc_vs_benchmark ve Sharpe_excess_vs_buy_hold pozitifse göreli üstünlük vardır
      - Neutral_Rate çok yükselirse strateji pasifleştiği için ceza uygulanır

    Model seçilen benchmark'tan daha kötü RMSE üretiyorsa skor sert biçimde aşağı çekilir;
    böylece benchmark'ı geçemeyen model lider olamaz.
    """
    import math

    dir_acc = float(metrics.get("Dir_Acc", 0.0))
    # Relative alanlar artık seçilen benchmark modeline göre hesaplanır.
    rmse_vs_benchmark = max(float(metrics.get("RMSE_vs_benchmark", 1.0)), 1e-8)
    diracc_vs_benchmark = float(metrics.get("DirAcc_vs_benchmark", 0.0))
    sharpe_excess = float(metrics.get("Sharpe_excess_vs_buy_hold", 0.0))
    neutral_rate = float(metrics.get("Neutral_Rate", 0.0))

    # RMSE oranı: 1.0 benchmark ile başa baş, daha düşükse daha iyi
    rmse_score = min(100.0, 100.0 / rmse_vs_benchmark)

    # Relative directional skill: ±12.5 puan farkı yaklaşık 0-100 bandına yay
    diracc_relative_score = min(100.0, max(0.0, 50.0 + diracc_vs_benchmark * 4.0))

    # Buy-and-hold üstü Sharpe'ı ödüllendir
    sharpe_relative_score = (math.tanh(sharpe_excess / 1.5) + 1.0) * 50.0

    neutral_penalty = min(15.0, neutral_rate * 0.15)

    composite = (
        rmse_score * 0.45 +
        diracc_relative_score * 0.25 +
        sharpe_relative_score * 0.20 +
        dir_acc * 0.10
    )
    composite -= neutral_penalty

    if rmse_vs_benchmark > 1.0:
        composite = min(composite, 49.0)
    if metrics.get("Eligible_For_Leader") is False:
        composite = min(composite, 49.0)

    return round(max(0.0, composite), 4)


def _compute_composite_score(metrics: Dict[str, float]) -> float:
    return compute_composite_score(metrics)


class StockModelDB:
    """
    SQLite tabanlı hisse-model kayıt yöneticisi.

    Args:
        db_path: Veritabanı dosyasının tam yolu.
                 Yoksa otomatik oluşturulur.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        import os as _os
        _os.makedirs(_os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_repositories()
        self._init_db()

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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        self._ensure_repositories()
        return self.schema_repository.initialize()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        from src.database.repositories import SchemaRepository

        return SchemaRepository.ensure_column(conn, table, column, ddl)

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

    def _update_production_best_model(
        self,
        *,
        stock_symbol: str,
        model_name: str,
        experiment_id: int,
        composite_score: float,
        metrics: Dict[str, float],
        model_path: str,
        trained_at: str,
        target_mode: str,
        feature_mode: str,
        scaling_mode: str,
        validation_mode: str,
        dataset_hash: str,
        is_production_candidate: bool,
        selection_source: Optional[str],
        run_id: Optional[str],
    ) -> None:
        self._ensure_repositories()
        return self.best_model_repository.update_production_best_model(
            stock_symbol=stock_symbol,
            model_name=model_name,
            experiment_id=experiment_id,
            composite_score=composite_score,
            metrics=metrics,
            model_path=model_path,
            trained_at=trained_at,
            target_mode=target_mode,
            feature_mode=feature_mode,
            scaling_mode=scaling_mode,
            validation_mode=validation_mode,
            dataset_hash=dataset_hash,
            is_production_candidate=is_production_candidate,
            selection_source=selection_source,
            run_id=run_id,
        )

    @staticmethod
    def _upsert_best_from_values(
        *,
        conn: sqlite3.Connection,
        stock_symbol: str,
        model_name: str,
        experiment_id: int,
        composite_score: float,
        metrics: Dict[str, Any],
        model_path: str,
        updated_at: str,
        target_mode: str,
        feature_mode: str,
        scaling_mode: str,
        validation_mode: str,
        dataset_hash: Optional[str],
        run_id: Optional[str],
        selection_source: Optional[str],
    ) -> None:
        conn.execute(
            """
            INSERT INTO best_models
                (stock_symbol, model_name, experiment_id, composite_score,
                 target_mode, feature_mode, scaling_mode, validation_mode,
                 dataset_hash, run_id, selection_source,
                 mae, rmse, mape, dir_acc, sharpe, hit_rate,
                 model_path, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stock_symbol) DO UPDATE SET
                model_name       = excluded.model_name,
                experiment_id    = excluded.experiment_id,
                composite_score  = excluded.composite_score,
                target_mode      = excluded.target_mode,
                feature_mode     = excluded.feature_mode,
                scaling_mode     = excluded.scaling_mode,
                validation_mode  = excluded.validation_mode,
                dataset_hash     = excluded.dataset_hash,
                run_id           = excluded.run_id,
                selection_source = excluded.selection_source,
                mae              = excluded.mae,
                rmse             = excluded.rmse,
                mape             = excluded.mape,
                dir_acc          = excluded.dir_acc,
                sharpe           = excluded.sharpe,
                hit_rate         = excluded.hit_rate,
                model_path       = excluded.model_path,
                updated_at       = excluded.updated_at
            """,
            (
                stock_symbol,
                model_name,
                experiment_id,
                composite_score,
                target_mode,
                feature_mode,
                scaling_mode,
                validation_mode,
                dataset_hash,
                run_id,
                selection_source,
                metrics.get("MAE"),
                metrics.get("RMSE"),
                metrics.get("MAPE"),
                metrics.get("Dir_Acc"),
                metrics.get("Sharpe"),
                metrics.get("Hit_Rate"),
                model_path,
                updated_at,
            ),
        )

    def _migrate_legacy_production_candidates(self, conn: sqlite3.Connection) -> None:
        self._ensure_repositories()
        return self.schema_repository.migrate_legacy_production_candidates(conn)

    def _refresh_best_models_from_production_experiments(self, conn: sqlite3.Connection) -> None:
        self._ensure_repositories()
        return self.schema_repository.refresh_best_models_from_production_experiments(conn)

    def get_best_model(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        self._ensure_repositories()
        return self.best_model_repository.get_best_model(stock_symbol)

    def get_leaderboard(self, top_n: int = 20) -> List[Dict[str, Any]]:
        self._ensure_repositories()
        return self.best_model_repository.get_leaderboard(top_n)

    def get_experiments(
        self,
        stock_symbol: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        self._ensure_repositories()
        return self.experiment_repository.get_experiments(
            stock_symbol=stock_symbol,
            model_name=model_name,
            limit=limit,
        )

    def get_model_comparison(self, stock_symbol: str) -> List[Dict[str, Any]]:
        self._ensure_repositories()
        return self.experiment_repository.get_model_comparison(stock_symbol)

    def get_cross_run_leaderboard(
        self, stock_symbol: str, n_runs: int = 5
    ) -> List[Dict[str, Any]]:
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
    ) -> int:
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
        )

    def get_latest_forecast(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        self._ensure_repositories()
        return self.forecast_repository.get_latest_forecast(stock_symbol)

    def get_forecast_history(self, stock_symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        self._ensure_repositories()
        return self.forecast_repository.get_forecast_history(stock_symbol, limit=limit)

    def resolve_forecasts(self, stock_symbol: str, actual_prices: Dict[str, float]) -> int:
        self._ensure_repositories()
        return self.forecast_resolution_repository.resolve_forecasts(stock_symbol, actual_prices)

    def resolve_forecasts_from_csv(self, stock_symbol: str, csv_path: str) -> int:
        self._ensure_repositories()
        return self.forecast_resolution_repository.resolve_forecasts_from_csv(stock_symbol, csv_path)

    def _refresh_forecast_accuracy(self, conn: sqlite3.Connection, run_id: int) -> None:
        self._ensure_repositories()
        return self.forecast_resolution_repository.refresh_forecast_accuracy(conn, run_id)

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
