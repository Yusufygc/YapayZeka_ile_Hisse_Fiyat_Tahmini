# -*- coding: utf-8 -*-
"""Internal SQLite repositories backing ``StockModelDB``.

These classes keep the public ``StockModelDB`` API stable while moving table
schema, experiment, best-model, forecast, and resolution responsibilities out
of the facade.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.database import stock_model_db as schema


class SchemaRepository:
    def __init__(self, db) -> None:
        self.db = db

    def initialize(self) -> None:
        with self.db._connect() as conn:
            conn.execute(schema._CREATE_EXPERIMENTS)
            conn.execute(schema._CREATE_BEST_MODELS)
            conn.execute(schema._CREATE_IDX_SYMBOL)
            conn.execute(schema._CREATE_IDX_SCORE)
            conn.execute(schema._CREATE_FORECAST_RUNS)
            conn.execute(schema._CREATE_FORECAST_POINTS)
            conn.execute(schema._CREATE_FORECAST_ACCURACY)
            self.ensure_column(conn, "forecast_runs", "run_key", "TEXT")
            conn.execute(schema._CREATE_IDX_FORECAST_SYMBOL)
            conn.execute(schema._CREATE_IDX_FORECAST_RUN_KEY)
            conn.execute(schema._CREATE_IDX_FORECAST_POINTS_DATE)
            self._ensure_experiment_columns(conn)
            self._ensure_best_model_columns(conn)
            self._ensure_forecast_run_columns(conn)
            self.migrate_legacy_production_candidates(conn)
            self.refresh_best_models_from_production_experiments(conn)

    @staticmethod
    def ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _ensure_experiment_columns(self, conn: sqlite3.Connection) -> None:
        self.ensure_column(conn, "experiments", "target_mode", "TEXT NOT NULL DEFAULT 'price'")
        self.ensure_column(conn, "experiments", "feature_mode", "TEXT NOT NULL DEFAULT 'legacy_price_features'")
        self.ensure_column(conn, "experiments", "scaling_mode", "TEXT NOT NULL DEFAULT 'minmax'")
        self.ensure_column(conn, "experiments", "is_production_candidate", "INTEGER NOT NULL DEFAULT 0")
        self.ensure_column(conn, "experiments", "selection_source", "TEXT")
        self.ensure_column(conn, "experiments", "run_id", "TEXT")
        self.ensure_column(conn, "experiments", "stability_score", "REAL")

    def _ensure_forecast_run_columns(self, conn: sqlite3.Connection) -> None:
        self.ensure_column(conn, "forecast_runs", "ensemble_direction_agreement", "REAL")

    def _ensure_best_model_columns(self, conn: sqlite3.Connection) -> None:
        self.ensure_column(conn, "best_models", "target_mode", "TEXT NOT NULL DEFAULT 'price'")
        self.ensure_column(conn, "best_models", "feature_mode", "TEXT NOT NULL DEFAULT 'legacy_price_features'")
        self.ensure_column(conn, "best_models", "scaling_mode", "TEXT NOT NULL DEFAULT 'minmax'")
        self.ensure_column(conn, "best_models", "validation_mode", "TEXT NOT NULL DEFAULT 'final_holdout'")
        self.ensure_column(conn, "best_models", "dataset_hash", "TEXT")
        self.ensure_column(conn, "best_models", "run_id", "TEXT")
        self.ensure_column(conn, "best_models", "selection_source", "TEXT")
        self.ensure_column(conn, "best_models", "eligibility_status", "TEXT NOT NULL DEFAULT 'eligible'")
        self.ensure_column(conn, "best_models", "eligibility_reason", "TEXT NOT NULL DEFAULT ''")

    def migrate_legacy_production_candidates(self, conn: sqlite3.Connection) -> None:
        placeholders = ",".join("?" for _ in schema.BENCHMARK_MODELS)
        conn.execute(
            f"""
            UPDATE experiments
            SET is_production_candidate = 1,
                selection_source = COALESCE(selection_source, 'legacy_final_holdout_migration')
            WHERE validation_mode = 'final_holdout'
              AND COALESCE(is_production_candidate, 0) = 0
              AND model_name NOT LIKE 'Ensemble %'
              AND model_name NOT IN ({placeholders})
            """,
            tuple(schema.BENCHMARK_MODELS),
        )

    def refresh_best_models_from_production_experiments(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            DELETE FROM best_models
            WHERE experiment_id NOT IN (
                SELECT id
                FROM experiments
                WHERE validation_mode = 'final_holdout'
                  AND COALESCE(is_production_candidate, 0) = 1
            )
            """
        )
        stocks = conn.execute(
            """
            SELECT DISTINCT stock_symbol
            FROM experiments
            WHERE validation_mode = 'final_holdout'
              AND COALESCE(is_production_candidate, 0) = 1
            """
        ).fetchall()
        for stock_row in stocks:
            row = self._latest_production_experiment(conn, stock_row["stock_symbol"])
            if row is not None:
                BestModelRepository.upsert_best_from_row(conn, row)

    @staticmethod
    def _latest_production_experiment(conn: sqlite3.Connection, stock_symbol: str):
        return conn.execute(
            """
            SELECT *
            FROM experiments
            WHERE stock_symbol = ?
              AND validation_mode = 'final_holdout'
              AND COALESCE(is_production_candidate, 0) = 1
            ORDER BY trained_at DESC, id DESC
            LIMIT 1
            """,
            (stock_symbol,),
        ).fetchone()


class ExperimentRepository:
    def __init__(self, db, best_models: "BestModelRepository") -> None:
        self.db = db
        self.best_models = best_models

    def log_experiment(
        self,
        stock_symbol: str,
        model_name: str,
        metrics: Dict[str, float],
        model_path: str = "",
        features: List[str] | None = None,
        dataset_hash: str = "N/A",
        validation_mode: str = "single_split",
        dataset_metadata: Optional[Dict[str, Any]] = None,
        is_production_candidate: bool = False,
        selection_source: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> int:
        composite = schema.compute_composite_score(metrics)
        trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dataset_metadata = dataset_metadata or {}
        run_id = run_id or dataset_metadata.get("run_id")
        experiment_id = self._insert_experiment(
            stock_symbol, model_name, metrics, model_path, features or [],
            dataset_hash, validation_mode, dataset_metadata,
            is_production_candidate, selection_source, run_id, composite, trained_at,
        )
        self.best_models.update_production_best_model(
            stock_symbol=stock_symbol,
            model_name=model_name,
            experiment_id=experiment_id,
            composite_score=composite,
            metrics=metrics,
            model_path=model_path,
            trained_at=trained_at,
            target_mode=dataset_metadata.get("target_mode", "price"),
            feature_mode=dataset_metadata.get("feature_mode", "legacy_price_features"),
            scaling_mode=dataset_metadata.get("scaling_mode", "minmax"),
            validation_mode=validation_mode,
            dataset_hash=dataset_hash,
            is_production_candidate=bool(is_production_candidate),
            selection_source=selection_source,
            run_id=run_id,
        )
        print(
            f"  [DB] {stock_symbol} | {model_name:15s} -> "
            f"composite={composite:.2f}  "
            f"dir_acc={metrics.get('Dir_Acc', 0):.1f}%  "
            f"sharpe={metrics.get('Sharpe', 0):.3f}"
        )
        return experiment_id

    def _insert_experiment(
        self,
        stock_symbol: str,
        model_name: str,
        metrics: Dict[str, float],
        model_path: str,
        features: List[str],
        dataset_hash: str,
        validation_mode: str,
        dataset_metadata: Dict[str, Any],
        is_production_candidate: bool,
        selection_source: Optional[str],
        run_id: Optional[str],
        composite: float,
        trained_at: str,
    ) -> int:
        stability = self.db._optional_float(metrics.get("Stability_Score"))
        with self.db._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments
                    (stock_symbol, model_name, validation_mode, target_mode, feature_mode, scaling_mode,
                     mae, rmse, mape, dir_acc, sharpe, hit_rate,
                     composite_score, model_path, features, dataset_hash,
                     is_production_candidate, selection_source, run_id, trained_at,
                     stability_score)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stock_symbol, model_name, validation_mode,
                    dataset_metadata.get("target_mode", "price"),
                    dataset_metadata.get("feature_mode", "legacy_price_features"),
                    dataset_metadata.get("scaling_mode", "minmax"),
                    metrics.get("MAE"), metrics.get("RMSE"),
                    metrics.get("MAPE"), metrics.get("Dir_Acc"),
                    metrics.get("Sharpe"), metrics.get("Hit_Rate"),
                    composite, model_path, json.dumps(features, ensure_ascii=False),
                    dataset_hash, int(bool(is_production_candidate)),
                    selection_source, run_id, trained_at,
                    stability,
                ),
            )
            return int(cursor.lastrowid)

    def get_experiments(
        self,
        stock_symbol: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if stock_symbol:
            clauses.append("stock_symbol = ?")
            params.append(stock_symbol)
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self.db._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM experiments
                {where}
                ORDER BY trained_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_cross_run_leaderboard(
        self, stock_symbol: str, n_runs: int = 5
    ) -> List[Dict[str, Any]]:
        """Son N run üzerinden model bazlı ortalama ve istikrar istatistikleri döner."""
        stock_symbol = stock_symbol.upper()
        with self.db._connect() as conn:
            rows = conn.execute(
                """
                WITH recent_runs AS (
                    SELECT DISTINCT run_id
                    FROM experiments
                    WHERE stock_symbol = ?
                      AND run_id IS NOT NULL
                    ORDER BY trained_at DESC
                    LIMIT ?
                )
                SELECT
                    e.model_name,
                    COUNT(*)                                   AS run_count,
                    AVG(e.composite_score)                     AS avg_composite,
                    AVG(e.dir_acc)                             AS avg_dir_acc,
                    AVG(e.sharpe)                              AS avg_sharpe,
                    AVG(e.stability_score)                     AS avg_stability_score,
                    MIN(e.stability_score)                     AS min_stability_score,
                    SUM(CASE WHEN e.stability_score >= 0 THEN 1 ELSE 0 END) AS positive_stability_runs
                FROM experiments e
                JOIN recent_runs r ON e.run_id = r.run_id
                WHERE e.stock_symbol = ?
                GROUP BY e.model_name
                ORDER BY avg_composite DESC
                """,
                (stock_symbol, int(n_runs), stock_symbol),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_model_comparison(self, stock_symbol: str) -> List[Dict[str, Any]]:
        with self.db._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    model_name,
                    COUNT(*)             AS run_count,
                    AVG(composite_score) AS avg_composite,
                    AVG(dir_acc)         AS avg_dir_acc,
                    AVG(sharpe)          AS avg_sharpe,
                    AVG(mae)             AS avg_mae,
                    AVG(rmse)            AS avg_rmse,
                    MAX(composite_score) AS best_composite
                FROM experiments
                WHERE stock_symbol = ?
                GROUP BY model_name
                ORDER BY avg_composite DESC
                """,
                (stock_symbol,),
            ).fetchall()
        return [dict(row) for row in rows]


class BestModelRepository:
    def __init__(self, db) -> None:
        self.db = db

    def update_production_best_model(
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
        if validation_mode != "final_holdout" or not is_production_candidate:
            return
        from src.pipeline.selection_guard import evaluate_best_model_eligibility

        eligibility_row = {
            "model_name": model_name,
            "is_production_candidate": int(is_production_candidate),
            "Trade_Count": metrics.get("Trade_Count", 0),
        }
        eligibility_status, eligibility_reason = evaluate_best_model_eligibility(eligibility_row)

        with self.db._connect() as conn:
            existing = conn.execute(
                "SELECT model_name, experiment_id FROM best_models WHERE stock_symbol = ?",
                (stock_symbol,),
            ).fetchone()
            self.upsert_best_from_values(
                conn=conn,
                stock_symbol=stock_symbol,
                model_name=model_name,
                experiment_id=experiment_id,
                composite_score=composite_score,
                metrics=metrics,
                model_path=model_path,
                updated_at=trained_at,
                target_mode=target_mode,
                feature_mode=feature_mode,
                scaling_mode=scaling_mode,
                validation_mode=validation_mode,
                dataset_hash=dataset_hash,
                run_id=run_id,
                selection_source=selection_source,
                eligibility_status=eligibility_status,
                eligibility_reason=eligibility_reason,
            )
            self._print_best_update(stock_symbol, model_name, experiment_id, existing)

    @staticmethod
    def _print_best_update(stock_symbol: str, model_name: str, experiment_id: int, existing) -> None:
        if existing is None:
            print(f"  [DB] OK {stock_symbol} production best: {model_name}")
        elif int(existing["experiment_id"]) != int(experiment_id):
            print(
                f"  [DB] OK {stock_symbol} production best guncellendi: "
                f"{existing['model_name']} -> {model_name}"
            )

    @staticmethod
    def upsert_best_from_row(conn: sqlite3.Connection, row) -> None:
        BestModelRepository.upsert_best_from_values(
            conn=conn,
            stock_symbol=row["stock_symbol"],
            model_name=row["model_name"],
            experiment_id=int(row["id"]),
            composite_score=float(row["composite_score"] or 0.0),
            metrics={
                "MAE": row["mae"],
                "RMSE": row["rmse"],
                "MAPE": row["mape"],
                "Dir_Acc": row["dir_acc"],
                "Sharpe": row["sharpe"],
                "Hit_Rate": row["hit_rate"],
            },
            model_path=row["model_path"] or "",
            updated_at=row["trained_at"],
            target_mode=row["target_mode"],
            feature_mode=row["feature_mode"],
            scaling_mode=row["scaling_mode"],
            validation_mode=row["validation_mode"],
            dataset_hash=row["dataset_hash"],
            run_id=row["run_id"],
            selection_source=row["selection_source"],
        )

    @staticmethod
    def upsert_best_from_values(
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
        eligibility_status: str = "eligible",
        eligibility_reason: str = "",
    ) -> None:
        conn.execute(
            """
            INSERT INTO best_models
                (stock_symbol, model_name, experiment_id, composite_score,
                 target_mode, feature_mode, scaling_mode, validation_mode,
                 dataset_hash, run_id, selection_source,
                 mae, rmse, mape, dir_acc, sharpe, hit_rate,
                 model_path, updated_at,
                 eligibility_status, eligibility_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stock_symbol) DO UPDATE SET
                model_name          = excluded.model_name,
                experiment_id       = excluded.experiment_id,
                composite_score     = excluded.composite_score,
                target_mode         = excluded.target_mode,
                feature_mode        = excluded.feature_mode,
                scaling_mode        = excluded.scaling_mode,
                validation_mode     = excluded.validation_mode,
                dataset_hash        = excluded.dataset_hash,
                run_id              = excluded.run_id,
                selection_source    = excluded.selection_source,
                mae                 = excluded.mae,
                rmse                = excluded.rmse,
                mape                = excluded.mape,
                dir_acc             = excluded.dir_acc,
                sharpe              = excluded.sharpe,
                hit_rate            = excluded.hit_rate,
                model_path          = excluded.model_path,
                updated_at          = excluded.updated_at,
                eligibility_status  = excluded.eligibility_status,
                eligibility_reason  = excluded.eligibility_reason
            """,
            (
                stock_symbol, model_name, experiment_id, composite_score,
                target_mode, feature_mode, scaling_mode, validation_mode,
                dataset_hash, run_id, selection_source,
                metrics.get("MAE"), metrics.get("RMSE"), metrics.get("MAPE"),
                metrics.get("Dir_Acc"), metrics.get("Sharpe"), metrics.get("Hit_Rate"),
                model_path, updated_at,
                eligibility_status, eligibility_reason,
            ),
        )

    def get_best_model(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM best_models WHERE stock_symbol = ?",
                (stock_symbol,),
            ).fetchone()
        return dict(row) if row else None

    def get_leaderboard(self, top_n: int = 20) -> List[Dict[str, Any]]:
        with self.db._connect() as conn:
            rows = conn.execute(
                """
                SELECT stock_symbol, model_name, composite_score,
                       dir_acc, sharpe, mae, rmse, updated_at
                FROM best_models
                ORDER BY composite_score DESC
                LIMIT ?
                """,
                (top_n,),
            ).fetchall()
        return [dict(row) for row in rows]


class ForecastRepository:
    def __init__(self, db) -> None:
        self.db = db

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
    ) -> int:
        stock_symbol = stock_symbol.upper()
        run_at = run_at or datetime.now().isoformat(timespec="seconds")
        run_key = self.db._forecast_run_key(
            stock_symbol, model_name, source_experiment_id,
            last_observed_date, horizon_days, rules_version,
        )
        with self.db._connect() as conn:
            self._upsert_forecast_run(
                conn, run_key, stock_symbol, model_name, source_experiment_id,
                run_at, last_observed_date, last_close, horizon_days,
                trend_label, weekly_expected_return, trend_threshold, rules_version, status,
                ensemble_direction_agreement=ensemble_direction_agreement,
            )
            run_id = int(conn.execute(
                "SELECT id FROM forecast_runs WHERE run_key = ?",
                (run_key,),
            ).fetchone()["id"])
            for point in points:
                self._upsert_forecast_point(conn, run_id, point)
        return run_id

    @staticmethod
    def _upsert_forecast_run(
        conn, run_key, stock_symbol, model_name, source_experiment_id, run_at,
        last_observed_date, last_close, horizon_days, trend_label,
        weekly_expected_return, trend_threshold, rules_version, status,
        ensemble_direction_agreement=None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO forecast_runs
                (run_key, stock_symbol, model_name, source_experiment_id,
                 run_at, last_observed_date, last_close, horizon_days,
                 trend_label, weekly_expected_return, trend_threshold,
                 rules_version, status, ensemble_direction_agreement)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_key) DO UPDATE SET
                run_at                       = excluded.run_at,
                last_close                   = excluded.last_close,
                trend_label                  = excluded.trend_label,
                weekly_expected_return       = excluded.weekly_expected_return,
                trend_threshold              = excluded.trend_threshold,
                status                       = excluded.status,
                ensemble_direction_agreement = excluded.ensemble_direction_agreement
            """,
            (
                run_key, stock_symbol, model_name, source_experiment_id, run_at,
                last_observed_date, float(last_close), int(horizon_days),
                trend_label, float(weekly_expected_return), float(trend_threshold),
                rules_version, status,
                float(ensemble_direction_agreement) if ensemble_direction_agreement is not None else None,
            ),
        )

    def _upsert_forecast_point(self, conn, run_id: int, point: Dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO forecast_points
                (run_id, target_date, horizon_index, raw_predicted_close,
                 bounded_predicted_close, predicted_return, lower_band,
                 upper_band, price_tick)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, target_date) DO UPDATE SET
                horizon_index           = excluded.horizon_index,
                raw_predicted_close     = excluded.raw_predicted_close,
                bounded_predicted_close = excluded.bounded_predicted_close,
                predicted_return        = excluded.predicted_return,
                lower_band              = excluded.lower_band,
                upper_band              = excluded.upper_band,
                price_tick              = excluded.price_tick,
                actual_close            = NULL,
                actual_return           = NULL,
                abs_error               = NULL,
                direction_correct       = NULL,
                resolved_at             = NULL
            """,
            (
                run_id,
                str(point["target_date"])[:10],
                int(point["horizon_index"]),
                self.db._optional_float(point.get("raw_predicted_close")),
                self.db._optional_float(point.get("bounded_predicted_close")),
                self.db._optional_float(point.get("predicted_return")),
                self.db._optional_float(point.get("lower_band")),
                self.db._optional_float(point.get("upper_band")),
                self.db._optional_float(point.get("price_tick")),
            ),
        )

    def get_latest_forecast(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        rows = self.get_forecast_history(stock_symbol, limit=1)
        return rows[0] if rows else None

    def get_forecast_history(self, stock_symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        stock_symbol = stock_symbol.upper()
        with self.db._connect() as conn:
            runs = conn.execute(
                """
                SELECT *
                FROM forecast_runs
                WHERE stock_symbol = ?
                ORDER BY run_at DESC, id DESC
                LIMIT ?
                """,
                (stock_symbol, int(limit)),
            ).fetchall()
            return [self._run_with_points(conn, dict(run)) for run in runs]

    @staticmethod
    def _run_with_points(conn, run_dict: Dict[str, Any]) -> Dict[str, Any]:
        points = conn.execute(
            """
            SELECT *
            FROM forecast_points
            WHERE run_id = ?
            ORDER BY horizon_index ASC
            """,
            (run_dict["id"],),
        ).fetchall()
        summary = conn.execute(
            "SELECT * FROM forecast_accuracy_summary WHERE run_id = ?",
            (run_dict["id"],),
        ).fetchone()
        run_dict["points"] = [dict(point) for point in points]
        run_dict["accuracy_summary"] = dict(summary) if summary else None
        return run_dict


class ForecastResolutionRepository:
    def __init__(self, db) -> None:
        self.db = db

    def resolve_forecasts(self, stock_symbol: str, actual_prices: Dict[str, float]) -> int:
        stock_symbol = stock_symbol.upper()
        normalized_actuals = {
            str(key)[:10]: float(value)
            for key, value in actual_prices.items()
            if value is not None
        }
        if not normalized_actuals:
            return 0

        resolved = 0
        now = datetime.now().isoformat(timespec="seconds")
        with self.db._connect() as conn:
            runs = conn.execute(
                "SELECT * FROM forecast_runs WHERE stock_symbol = ?",
                (stock_symbol,),
            ).fetchall()
            for run in runs:
                resolved += self._resolve_run(conn, run, normalized_actuals, now)
                self.refresh_forecast_accuracy(conn, int(run["id"]))
        return resolved

    def _resolve_run(self, conn, run, actuals: Dict[str, float], now: str) -> int:
        points = conn.execute(
            """
            SELECT *
            FROM forecast_points
            WHERE run_id = ?
            ORDER BY horizon_index ASC
            """,
            (int(run["id"]),),
        ).fetchall()
        previous_actual = float(run["last_close"])
        contiguous = True
        resolved = 0
        for point in points:
            target_date = str(point["target_date"])[:10]
            actual_close = actuals.get(target_date)
            if actual_close is None:
                contiguous = False
                continue
            if not contiguous:
                continue
            self._update_point_resolution(conn, point, actual_close, previous_actual, now)
            previous_actual = actual_close
            resolved += 1
        return resolved

    def _update_point_resolution(self, conn, point, actual_close: float, previous_actual: float, now: str) -> None:
        predicted = self.db._optional_float(point["bounded_predicted_close"])
        predicted_return = self.db._optional_float(point["predicted_return"])
        actual_return = (actual_close / previous_actual) - 1.0 if previous_actual else None
        abs_error = abs(actual_close - predicted) if predicted is not None else None
        direction_correct = None
        if predicted_return is not None and actual_return is not None:
            direction_correct = int(self.db._sign(predicted_return) == self.db._sign(actual_return))
        conn.execute(
            """
            UPDATE forecast_points
            SET actual_close = ?,
                actual_return = ?,
                abs_error = ?,
                direction_correct = ?,
                resolved_at = ?
            WHERE id = ?
            """,
            (actual_close, actual_return, abs_error, direction_correct, now, int(point["id"])),
        )

    def refresh_forecast_accuracy(self, conn: sqlite3.Connection, run_id: int) -> None:
        run = conn.execute("SELECT * FROM forecast_runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            return
        points = conn.execute(
            """
            SELECT *
            FROM forecast_points
            WHERE run_id = ? AND actual_close IS NOT NULL
            ORDER BY horizon_index ASC
            """,
            (run_id,),
        ).fetchall()
        if not points:
            conn.execute("DELETE FROM forecast_accuracy_summary WHERE run_id = ?", (run_id,))
            return
        self._upsert_accuracy_summary(conn, run, points)

    def _upsert_accuracy_summary(self, conn, run, points) -> None:
        actual = [float(point["actual_close"]) for point in points]
        pred = [float(point["bounded_predicted_close"]) for point in points]
        errors = [actual_value - pred_value for actual_value, pred_value in zip(actual, pred)]
        abs_errors = [abs(err) for err in errors]
        mape_values = [abs((actual_value - pred_value) / actual_value) for actual_value, pred_value in zip(actual, pred) if actual_value]
        direction_values = [int(point["direction_correct"]) for point in points if point["direction_correct"] is not None]
        weekly_direction_correct = self._weekly_direction_correct(run, points)
        conn.execute(
            """
            INSERT INTO forecast_accuracy_summary
                (run_id, stock_symbol, model_name, rmse, mae, mape, dir_acc,
                 weekly_direction_correct, resolved_points, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                rmse                     = excluded.rmse,
                mae                      = excluded.mae,
                mape                     = excluded.mape,
                dir_acc                  = excluded.dir_acc,
                weekly_direction_correct = excluded.weekly_direction_correct,
                resolved_points          = excluded.resolved_points,
                updated_at               = excluded.updated_at
            """,
            (
                int(run["id"]),
                run["stock_symbol"],
                run["model_name"],
                math.sqrt(sum(err * err for err in errors) / len(errors)),
                sum(abs_errors) / len(abs_errors),
                (sum(mape_values) / len(mape_values) * 100.0) if mape_values else None,
                (sum(direction_values) / len(direction_values) * 100.0) if direction_values else None,
                weekly_direction_correct,
                len(points),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

    def _weekly_direction_correct(self, run, points) -> Optional[int]:
        if len(points) < int(run["horizon_days"]):
            return None
        weekly_actual_return = (float(points[-1]["actual_close"]) / float(run["last_close"])) - 1.0
        return int(self.db._sign(float(run["weekly_expected_return"])) == self.db._sign(weekly_actual_return))

    def resolve_forecasts_from_csv(self, stock_symbol: str, csv_path: str) -> int:
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "Date" not in df.columns or "Close" not in df.columns:
            raise ValueError("CSV must include Date and Close columns.")
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        actuals = {
            row["Date"]: float(row["Close"])
            for _, row in df.dropna(subset=["Date", "Close"]).iterrows()
        }
        return self.resolve_forecasts(stock_symbol, actuals)
