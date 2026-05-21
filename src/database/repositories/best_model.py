# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from src.database.repositories.helpers import (
    _parse_jsonish,
    _optional_float,
    _optional_int,
    _optional_text,
)


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
