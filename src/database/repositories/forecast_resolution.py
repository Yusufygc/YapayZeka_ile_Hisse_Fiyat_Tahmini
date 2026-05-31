# -*- coding: utf-8 -*-
"""Forecast doğruluk çözümleme deposu (SQLite).

Sorumluluklar:
  - ForecastResolutionRepository: gerçekleşen fiyatları kayıtlı forecast
    noktalarıyla eşleştirir ve doğruluk metriklerini günceller.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.database.repositories.helpers import _optional_float


class ForecastResolutionRepository:
    def __init__(self, db) -> None:
        self.db = db

    def resolve_forecasts(self, stock_symbol: str, actual_prices: Dict[str, float]) -> int:
        """Gerçekleşen fiyatları forecast noktalarıyla eşleştirir, doğruluğu günceller.

        Args:
            actual_prices: tarih(str) -> gerçekleşen kapanış eşlemesi.

        Returns:
            Çözümlenen (eşleşen) forecast noktası sayısı.
        """
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
        """Verilen forecast run'ı için toplu doğruluk metriklerini yeniden hesaplar."""
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
        """CSV'den gerçekleşen fiyatları okuyup `resolve_forecasts`'a yönlendirir.

        Tarih ve kapanış kolonlarını TR/EN başlık varyantlarından otomatik bulur.

        Returns:
            Çözümlenen forecast noktası sayısı.
        """
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
