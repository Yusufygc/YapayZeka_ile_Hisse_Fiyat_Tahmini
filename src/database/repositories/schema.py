# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from src.database import stock_model_db as schema
from src.database.repositories.best_model import BestModelRepository


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
                    OR model_name IN ('Ensemble Inverse RMSE', 'Ensemble Cash-Gated', 'Ensemble Seq-Attention Inverse RMSE')
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
                OR model_name IN ('Ensemble Inverse RMSE', 'Ensemble Cash-Gated', 'Ensemble Seq-Attention Inverse RMSE')
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
                  OR model_name IN ('Ensemble Inverse RMSE', 'Ensemble Cash-Gated', 'Ensemble Seq-Attention Inverse RMSE')
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
