# -*- coding: utf-8 -*-
"""
src.gui.core.db_helper - SQLite database querying utility.
stock_models.db verilerini okur ve GUI tablo modelleri için hazırlar.
"""

import os
import sqlite3

from src.database.connection import ClosingConnection
import glob
from typing import Any, Dict, List, Optional


class DBHelper:
    """
    stock_models.db SQLite veritabanına erişim sağlayan yardımcı sınıf.
    """
    def __init__(self, db_path: Optional[str] = None):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.db_path = db_path or os.path.join(project_root, "data", "stock_models.db")
        self.data_dir = os.path.join(project_root, "data")

    def _get_connection(self):
        """
        SQLite bağlantısı oluşturur ve satırları dict benzeri döndürecek row_factory kurar.
        """
        conn = sqlite3.connect(self.db_path, factory=ClosingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    def get_best_models_summary(self) -> List[Dict[str, Any]]:
        """
        best_models tablosundaki tüm kayıtları getirir.
        """
        if not os.path.exists(self.db_path):
            return []

        query = """
            SELECT
                stock_symbol,
                model_name,
                composite_score,
                dir_acc,
                hit_rate,
                sharpe,
                rmse,
                mae,
                updated_at
            FROM best_models
            ORDER BY stock_symbol ASC
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"[DBHelper] get_best_models_summary hatası: {e}")
            return []

    def get_forecasts_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Belirli bir hisse için en son başarılı forecast_runs ve onun altındaki forecast_points verilerini getirir.
        """
        if not os.path.exists(self.db_path):
            return []

        query_run = """
            SELECT id, run_key, model_name, run_at, last_observed_date, last_close, trend_label, weekly_expected_return
            FROM forecast_runs
            WHERE stock_symbol = ? AND status = 'completed'
            ORDER BY run_at DESC
            LIMIT 1
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query_run, (symbol.upper(),))
                run = cursor.fetchone()
                if not run:
                    return []

                run_id = run["id"]

                query_points = """
                    SELECT
                        target_date,
                        horizon_index,
                        raw_predicted_close,
                        bounded_predicted_close,
                        predicted_return,
                        lower_band,
                        upper_band,
                        interval_method
                    FROM forecast_points
                    WHERE run_id = ?
                    ORDER BY horizon_index ASC
                """
                cursor.execute(query_points, (run_id,))
                points = [dict(row) for row in cursor.fetchall()]

                return {
                    "run": dict(run),
                    "points": points
                }
        except Exception as e:
            print(f"[DBHelper] get_forecasts_by_symbol hatası: {e}")
            return []

    def get_available_stocks(self) -> List[str]:
        """
        data/ klasörü altında .csv uzantılı hisse dosyalarını bulur ve listeler.
        bist_universe, bist_calendar gibi dosyaları eler.
        """
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        meta_stems = {"bist_universe", "bist_calendar", "batch_summary"}

        stocks = []
        for file in csv_files:
            stem = os.path.splitext(os.path.basename(file))[0]
            if stem.lower() in meta_stems or "batch_summary" in stem:
                continue
            stocks.append(stem.upper())

        return sorted(stocks)
