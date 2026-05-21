# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.database import stock_model_db as schema
from src.database.repositories.helpers import (
    _ensemble_metadata_for,
    _optional_float,
    _optional_int,
    _optional_text,
)
from src.database.repositories.best_model import BestModelRepository


class ExperimentRepository:
    def __init__(self, db, best_models: BestModelRepository) -> None:
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
