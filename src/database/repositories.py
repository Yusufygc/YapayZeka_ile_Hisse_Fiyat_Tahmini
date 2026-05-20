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

PRODUCTION_ENSEMBLE_METHODS = {"Inverse RMSE", "Cash-Gated"}


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in (None, ""):
        return None
    try:
        return json.loads(str(value).replace("'", '"'))
    except Exception:
        return value


def _optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> Optional[int]:
    float_value = _optional_float(value)
    return None if float_value is None else int(float_value)


def _optional_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _ensemble_metadata_for(
    model_name: str,
    metrics: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not str(model_name).startswith("Ensemble "):
        return None
    method = str(metrics.get("Ensemble_Method") or str(model_name).replace("Ensemble ", ""))
    weights = _parse_jsonish(metrics.get("Ensemble_Weights"))
    source_ids = _parse_jsonish(metrics.get("Ensemble_Source_Experiment_IDs"))
    return {
        "type": "ensemble",
        "method": method,
        "production_method": method in PRODUCTION_ENSEMBLE_METHODS,
        "members": list(weights.keys()) if isinstance(weights, dict) else [],
        "weights": weights if isinstance(weights, dict) else {},
        "source_experiment_ids": source_ids if isinstance(source_ids, list) else [],
        "source_run_ids": _parse_jsonish(metrics.get("Ensemble_Source_Run_IDs")) or [],
        "run_id": dataset_metadata.get("run_id"),
    }


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
            conn.execute(schema._CREATE_ANALYSIS_REFRESH_JOBS)
            self.ensure_column(conn, "forecast_runs", "run_key", "TEXT")
            conn.execute(schema._CREATE_IDX_FORECAST_SYMBOL)
            conn.execute(schema._CREATE_IDX_FORECAST_RUN_KEY)
            conn.execute(schema._CREATE_IDX_FORECAST_POINTS_DATE)
            conn.execute(schema._CREATE_IDX_REFRESH_JOBS_SYMBOL)
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
        self.ensure_column(conn, "experiments", "ensemble_metadata_json", "TEXT")
        self.ensure_column(conn, "experiments", "rmse_vs_benchmark", "REAL")
        self.ensure_column(conn, "experiments", "net_return", "REAL")
        self.ensure_column(conn, "experiments", "buyhold_return", "REAL")
        self.ensure_column(conn, "experiments", "max_drawdown", "REAL")
        self.ensure_column(conn, "experiments", "trade_count", "INTEGER")
        self.ensure_column(conn, "experiments", "signal_diagnosis", "TEXT")

    def _ensure_forecast_run_columns(self, conn: sqlite3.Connection) -> None:
        self.ensure_column(conn, "forecast_runs", "ensemble_direction_agreement", "REAL")
        self.ensure_column(conn, "forecast_runs", "live_status", "TEXT NOT NULL DEFAULT 'healthy'")
        self.ensure_column(conn, "forecast_runs", "forecast_strategy", "TEXT")
        self.ensure_column(conn, "forecast_runs", "artifact_mode", "TEXT")
        self.ensure_column(conn, "forecast_runs", "forecast_warnings_json", "TEXT")
        self.ensure_column(conn, "forecast_runs", "ensemble_metadata_json", "TEXT")

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
        self.ensure_column(conn, "best_models", "ensemble_metadata_json", "TEXT")
        self.ensure_column(conn, "best_models", "rmse_vs_benchmark", "REAL")
        self.ensure_column(conn, "best_models", "net_return", "REAL")
        self.ensure_column(conn, "best_models", "buyhold_return", "REAL")
        self.ensure_column(conn, "best_models", "max_drawdown", "REAL")
        self.ensure_column(conn, "best_models", "trade_count", "INTEGER")
        self.ensure_column(conn, "best_models", "signal_diagnosis", "TEXT")

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
                WHERE (
                    validation_mode = 'final_holdout'
                    OR model_name IN ('Ensemble Inverse RMSE', 'Ensemble Cash-Gated')
                )
                  AND COALESCE(is_production_candidate, 0) = 1
            )
            """
        )
        stocks = conn.execute(
            """
            SELECT DISTINCT stock_symbol
            FROM experiments
            WHERE (
                validation_mode = 'final_holdout'
                OR model_name IN ('Ensemble Inverse RMSE', 'Ensemble Cash-Gated')
            )
              AND COALESCE(is_production_candidate, 0) = 1
            """
        ).fetchall()
        for stock_row in stocks:
            row = self._best_production_experiment(conn, stock_row["stock_symbol"])
            if row is not None:
                BestModelRepository.upsert_best_from_row(conn, row)

    @staticmethod
    def _best_production_experiment(conn: sqlite3.Connection, stock_symbol: str):
        rows = conn.execute(
            """
            SELECT *
            FROM experiments
            WHERE stock_symbol = ?
              AND (
                  validation_mode = 'final_holdout'
                  OR model_name IN ('Ensemble Inverse RMSE', 'Ensemble Cash-Gated')
              )
              AND COALESCE(is_production_candidate, 0) = 1
            ORDER BY composite_score DESC, trained_at DESC, id DESC
            """,
            (stock_symbol,),
        ).fetchall()
        if not rows:
            return None

        def _rank(row) -> tuple[int, float, str, int]:
            status, _ = BestModelRepository.eligibility_from_experiment_row(row)
            return (
                1 if status == "eligible" else 0,
                float(row["composite_score"] or 0.0),
                str(row["trained_at"] or ""),
                int(row["id"] or 0),
            )

        return max(rows, key=_rank)


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
        ensemble_metadata = _ensemble_metadata_for(model_name, metrics, dataset_metadata)
        experiment_id = self._insert_experiment(
            stock_symbol, model_name, metrics, model_path, features or [],
            dataset_hash, validation_mode, dataset_metadata,
            is_production_candidate, selection_source, run_id, composite, trained_at,
            ensemble_metadata,
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
            ensemble_metadata=ensemble_metadata,
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
        ensemble_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        stability = self.db._optional_float(metrics.get("Stability_Score"))
        with self.db._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments
                    (stock_symbol, model_name, validation_mode, target_mode, feature_mode, scaling_mode,
                     mae, rmse, mape, dir_acc, sharpe, hit_rate,
                     rmse_vs_benchmark, net_return, buyhold_return, max_drawdown, trade_count, signal_diagnosis,
                     composite_score, model_path, features, dataset_hash,
                     is_production_candidate, selection_source, run_id, trained_at,
                     stability_score, ensemble_metadata_json)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stock_symbol, model_name, validation_mode,
                    dataset_metadata.get("target_mode", "price"),
                    dataset_metadata.get("feature_mode", "legacy_price_features"),
                    dataset_metadata.get("scaling_mode", "minmax"),
                    metrics.get("MAE"), metrics.get("RMSE"),
                    metrics.get("MAPE"), metrics.get("Dir_Acc"),
                    metrics.get("Sharpe"), metrics.get("Hit_Rate"),
                    _optional_float(metrics.get("RMSE_vs_benchmark")),
                    _optional_float(metrics.get("Net_Return")),
                    _optional_float(metrics.get("BuyHold_Return")),
                    _optional_float(metrics.get("Max_Drawdown")),
                    _optional_int(metrics.get("Trade_Count")),
                    _optional_text(metrics.get("Signal_Diagnosis")),
                    composite, model_path, json.dumps(features, ensure_ascii=False),
                    dataset_hash, int(bool(is_production_candidate)),
                    selection_source, run_id, trained_at,
                    stability,
                    json.dumps(ensemble_metadata, ensure_ascii=False) if ensemble_metadata else None,
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
        ensemble_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        is_prod_ensemble = (
            str(model_name).startswith("Ensemble ")
            and ensemble_metadata
            and bool(ensemble_metadata.get("production_method"))
        )
        if (validation_mode != "final_holdout" and not is_prod_ensemble) or not is_production_candidate:
            return
        from src.pipeline.selection_guard import evaluate_best_model_eligibility

        eligibility_row = {
            "model_name": model_name,
            "is_production_candidate": int(is_production_candidate),
            "Trade_Count": metrics.get("Trade_Count", 0),
            "RMSE_vs_benchmark": metrics.get("RMSE_vs_benchmark"),
            "Validation_Mode": validation_mode,
            "ensemble_metadata": ensemble_metadata,
        }
        eligibility_status, eligibility_reason = evaluate_best_model_eligibility(eligibility_row)

        with self.db._connect() as conn:
            existing = conn.execute(
                """
                SELECT model_name, experiment_id, composite_score, eligibility_status
                FROM best_models
                WHERE stock_symbol = ?
                """,
                (stock_symbol,),
            ).fetchone()
            if not self._should_replace_existing_best(
                existing,
                new_score=float(composite_score or 0.0),
                new_eligibility=eligibility_status,
            ):
                return
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
                ensemble_metadata=ensemble_metadata,
            )
            self._print_best_update(stock_symbol, model_name, experiment_id, existing)

    @staticmethod
    def _should_replace_existing_best(
        existing,
        *,
        new_score: float,
        new_eligibility: str,
    ) -> bool:
        if existing is None:
            return True
        existing_eligibility = str(existing["eligibility_status"] or "")
        existing_score = float(existing["composite_score"] or 0.0)
        if existing_eligibility == "eligible" and new_eligibility != "eligible":
            return False
        if existing_eligibility != "eligible" and new_eligibility == "eligible":
            return True
        return new_score > existing_score

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
    def eligibility_from_experiment_row(row) -> tuple[str, str]:
        from src.pipeline.selection_guard import evaluate_best_model_eligibility

        ensemble_metadata = _parse_jsonish(row["ensemble_metadata_json"])
        return evaluate_best_model_eligibility({
            "model_name": row["model_name"],
            "is_production_candidate": row["is_production_candidate"],
            "Trade_Count": row["trade_count"] if "trade_count" in row.keys() else 0,
            "RMSE_vs_benchmark": row["rmse_vs_benchmark"] if "rmse_vs_benchmark" in row.keys() else None,
            "ensemble_metadata": ensemble_metadata,
        })

    @staticmethod
    def upsert_best_from_row(conn: sqlite3.Connection, row) -> None:
        ensemble_metadata = _parse_jsonish(row["ensemble_metadata_json"])
        eligibility_status, eligibility_reason = BestModelRepository.eligibility_from_experiment_row(row)
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
                "RMSE_vs_benchmark": row["rmse_vs_benchmark"] if "rmse_vs_benchmark" in row.keys() else None,
                "Net_Return": row["net_return"] if "net_return" in row.keys() else None,
                "BuyHold_Return": row["buyhold_return"] if "buyhold_return" in row.keys() else None,
                "Max_Drawdown": row["max_drawdown"] if "max_drawdown" in row.keys() else None,
                "Trade_Count": row["trade_count"] if "trade_count" in row.keys() else None,
                "Signal_Diagnosis": row["signal_diagnosis"] if "signal_diagnosis" in row.keys() else None,
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
            eligibility_status=eligibility_status,
            eligibility_reason=eligibility_reason,
            ensemble_metadata=ensemble_metadata,
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
        ensemble_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ensemble_metadata_json = (
            json.dumps(ensemble_metadata, ensure_ascii=False) if ensemble_metadata else None
        )
        conn.execute(
            """
            INSERT INTO best_models
                (stock_symbol, model_name, experiment_id, composite_score,
                 target_mode, feature_mode, scaling_mode, validation_mode,
                 dataset_hash, run_id, selection_source,
                 mae, rmse, mape, dir_acc, sharpe, hit_rate,
                 rmse_vs_benchmark, net_return, buyhold_return, max_drawdown, trade_count, signal_diagnosis,
                 model_path, updated_at,
                 eligibility_status, eligibility_reason, ensemble_metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                rmse_vs_benchmark   = excluded.rmse_vs_benchmark,
                net_return          = excluded.net_return,
                buyhold_return      = excluded.buyhold_return,
                max_drawdown        = excluded.max_drawdown,
                trade_count         = excluded.trade_count,
                signal_diagnosis    = excluded.signal_diagnosis,
                model_path          = excluded.model_path,
                updated_at          = excluded.updated_at,
                eligibility_status  = excluded.eligibility_status,
                eligibility_reason  = excluded.eligibility_reason,
                ensemble_metadata_json = excluded.ensemble_metadata_json
            """,
            (
                stock_symbol, model_name, experiment_id, composite_score,
                target_mode, feature_mode, scaling_mode, validation_mode,
                dataset_hash, run_id, selection_source,
                metrics.get("MAE"), metrics.get("RMSE"), metrics.get("MAPE"),
                metrics.get("Dir_Acc"), metrics.get("Sharpe"), metrics.get("Hit_Rate"),
                _optional_float(metrics.get("RMSE_vs_benchmark")),
                _optional_float(metrics.get("Net_Return")),
                _optional_float(metrics.get("BuyHold_Return")),
                _optional_float(metrics.get("Max_Drawdown")),
                _optional_int(metrics.get("Trade_Count")),
                _optional_text(metrics.get("Signal_Diagnosis")),
                model_path, updated_at,
                eligibility_status, eligibility_reason, ensemble_metadata_json,
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
        forecast_strategy: Optional[str] = None,
        artifact_mode: Optional[str] = None,
        forecast_warnings: Optional[List[str]] = None,
        ensemble_metadata: Optional[Dict[str, Any]] = None,
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
                forecast_strategy=forecast_strategy,
                artifact_mode=artifact_mode,
                forecast_warnings=forecast_warnings,
                ensemble_metadata=ensemble_metadata,
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
        forecast_strategy=None,
        artifact_mode=None,
        forecast_warnings=None,
        ensemble_metadata=None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO forecast_runs
                (run_key, stock_symbol, model_name, source_experiment_id,
                 run_at, last_observed_date, last_close, horizon_days,
                 trend_label, weekly_expected_return, trend_threshold,
                 rules_version, status, ensemble_direction_agreement,
                 forecast_strategy, artifact_mode, forecast_warnings_json,
                 ensemble_metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_key) DO UPDATE SET
                run_at                       = excluded.run_at,
                last_close                   = excluded.last_close,
                trend_label                  = excluded.trend_label,
                weekly_expected_return       = excluded.weekly_expected_return,
                trend_threshold              = excluded.trend_threshold,
                status                       = excluded.status,
                ensemble_direction_agreement = excluded.ensemble_direction_agreement,
                forecast_strategy            = excluded.forecast_strategy,
                artifact_mode                 = excluded.artifact_mode,
                forecast_warnings_json       = excluded.forecast_warnings_json,
                ensemble_metadata_json       = excluded.ensemble_metadata_json
            """,
            (
                run_key, stock_symbol, model_name, source_experiment_id, run_at,
                last_observed_date, float(last_close), int(horizon_days),
                trend_label, float(weekly_expected_return), float(trend_threshold),
                rules_version, status,
                float(ensemble_direction_agreement) if ensemble_direction_agreement is not None else None,
                forecast_strategy,
                artifact_mode,
                json.dumps(forecast_warnings or [], ensure_ascii=False),
                json.dumps(ensemble_metadata, ensure_ascii=False) if ensemble_metadata else None,
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
        date_col = _find_column(df.columns, ["Date", "Tarih"])
        close_col = _find_column(df.columns, ["Close", "Kapanış", "Kapanis", "Adj Close", "Düzeltilmiş_Kapanış"])
        if date_col is None or close_col is None:
            raise ValueError("CSV must include Date/Tarih and Close/Kapanış columns.")
        df["Date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
        df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
        actuals = {
            row["Date"]: float(row["Close"])
            for _, row in df.dropna(subset=["Date", "Close"]).iterrows()
        }
        return self.resolve_forecasts(stock_symbol, actuals)

    def get_rolling_resolution_accuracy(
        self,
        stock_symbol: str,
        days: int = 60,
    ) -> Dict[str, Any]:
        """Son N günlük gerçekleşmiş forecast'lar üzerinde rolling dir_acc ve MAE.

        Returns
        -------
        dict:
            rolling_dir_acc (float | None), rolling_mae (float | None),
            resolved_count (int), model_status ('healthy' | 'degraded')
        """
        stock_symbol = stock_symbol.upper()
        cutoff_date = (
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        )
        from datetime import timedelta
        cutoff_iso = (cutoff_date - timedelta(days=days)).strftime("%Y-%m-%d")

        with self.db._connect() as conn:
            rows = conn.execute(
                """
                SELECT fp.direction_correct, fp.abs_error, fp.resolved_at
                FROM forecast_points fp
                JOIN forecast_runs fr ON fr.id = fp.run_id
                WHERE fr.stock_symbol = ?
                  AND fp.resolved_at IS NOT NULL
                  AND fp.resolved_at >= ?
                ORDER BY fp.resolved_at DESC
                """,
                (stock_symbol, cutoff_iso),
            ).fetchall()

        if not rows:
            return {
                "rolling_dir_acc": None,
                "rolling_mae": None,
                "resolved_count": 0,
                "model_status": "healthy",
            }

        dir_vals = [int(r["direction_correct"]) for r in rows if r["direction_correct"] is not None]
        mae_vals = [float(r["abs_error"]) for r in rows if r["abs_error"] is not None]

        rolling_dir_acc = (sum(dir_vals) / len(dir_vals) * 100.0) if dir_vals else None
        rolling_mae = (sum(mae_vals) / len(mae_vals)) if mae_vals else None
        model_status = "degraded" if (rolling_dir_acc is not None and rolling_dir_acc < 50.0) else "healthy"

        if model_status == "degraded":
            conn_update = self.db._connect()
            with conn_update as conn:
                conn.execute(
                    """
                    UPDATE forecast_runs
                    SET live_status = 'degraded'
                    WHERE stock_symbol = ?
                      AND id = (
                          SELECT id FROM forecast_runs
                          WHERE stock_symbol = ?
                          ORDER BY run_at DESC, id DESC
                          LIMIT 1
                      )
                    """,
                    (stock_symbol, stock_symbol),
                )

        return {
            "rolling_dir_acc": rolling_dir_acc,
            "rolling_mae": rolling_mae,
            "resolved_count": len(rows),
            "model_status": model_status,
        }


def _find_column(columns, candidates: list[str]) -> Optional[str]:
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None
