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

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


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
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_EXPERIMENTS)
            conn.execute(_CREATE_BEST_MODELS)
            conn.execute(_CREATE_IDX_SYMBOL)
            conn.execute(_CREATE_IDX_SCORE)
            self._ensure_column(conn, "experiments", "target_mode", "TEXT NOT NULL DEFAULT 'price'")
            self._ensure_column(conn, "experiments", "feature_mode", "TEXT NOT NULL DEFAULT 'legacy_price_features'")
            self._ensure_column(conn, "experiments", "scaling_mode", "TEXT NOT NULL DEFAULT 'minmax'")
            self._ensure_column(conn, "best_models", "target_mode", "TEXT NOT NULL DEFAULT 'price'")
            self._ensure_column(conn, "best_models", "feature_mode", "TEXT NOT NULL DEFAULT 'legacy_price_features'")
            self._ensure_column(conn, "best_models", "scaling_mode", "TEXT NOT NULL DEFAULT 'minmax'")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

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
    ) -> int:
        """
        Bir model eğitim çalışmasını kaydeder ve bileşik skoru hesaplar.

        Returns:
            Yeni eklenen satırın birincil anahtarı (experiment_id).
        """
        composite = compute_composite_score(metrics)
        trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        features_json = json.dumps(features or [], ensure_ascii=False)
        dataset_metadata = dataset_metadata or {}
        target_mode = dataset_metadata.get("target_mode", "price")
        feature_mode = dataset_metadata.get("feature_mode", "legacy_price_features")
        scaling_mode = dataset_metadata.get("scaling_mode", "minmax")

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments
                    (stock_symbol, model_name, validation_mode, target_mode, feature_mode, scaling_mode,
                     mae, rmse, mape, dir_acc, sharpe, hit_rate,
                     composite_score, model_path, features, dataset_hash, trained_at)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stock_symbol, model_name, validation_mode,
                    target_mode, feature_mode, scaling_mode,
                    metrics.get("MAE"), metrics.get("RMSE"),
                    metrics.get("MAPE"), metrics.get("Dir_Acc"),
                    metrics.get("Sharpe"), metrics.get("Hit_Rate"),
                    composite, model_path, features_json,
                    dataset_hash, trained_at,
                ),
            )
            experiment_id = cursor.lastrowid

        self._update_best_model(
            stock_symbol, model_name, experiment_id,
            composite, metrics, model_path, trained_at,
            target_mode, feature_mode, scaling_mode,
        )

        print(
            f"  [DB] {stock_symbol} | {model_name:15s} → "
            f"composite={composite:.2f}  "
            f"dir_acc={metrics.get('Dir_Acc', 0):.1f}%  "
            f"sharpe={metrics.get('Sharpe', 0):.3f}"
        )
        return experiment_id

    def _update_best_model(
        self,
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
    ) -> None:
        """
        Bu hisse için en iyi modeli günceller.
        Mevcut kayıt yoksa ekler, varsa yalnızca skor daha iyiyse üzerine yazar.
        """
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT composite_score FROM best_models WHERE stock_symbol = ?",
                (stock_symbol,),
            ).fetchone()

            if existing is None or composite_score > existing["composite_score"]:
                conn.execute(
                    """
                    INSERT INTO best_models
                        (stock_symbol, model_name, experiment_id, composite_score,
                         target_mode, feature_mode, scaling_mode,
                         mae, rmse, mape, dir_acc, sharpe, hit_rate,
                         model_path, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(stock_symbol) DO UPDATE SET
                        model_name      = excluded.model_name,
                        experiment_id   = excluded.experiment_id,
                        composite_score = excluded.composite_score,
                        target_mode     = excluded.target_mode,
                        feature_mode    = excluded.feature_mode,
                        scaling_mode    = excluded.scaling_mode,
                        mae             = excluded.mae,
                        rmse            = excluded.rmse,
                        mape            = excluded.mape,
                        dir_acc         = excluded.dir_acc,
                        sharpe          = excluded.sharpe,
                        hit_rate        = excluded.hit_rate,
                        model_path      = excluded.model_path,
                        updated_at      = excluded.updated_at
                    """,
                    (
                        stock_symbol, model_name, experiment_id, composite_score,
                        target_mode, feature_mode, scaling_mode,
                        metrics.get("MAE"), metrics.get("RMSE"),
                        metrics.get("MAPE"), metrics.get("Dir_Acc"),
                        metrics.get("Sharpe"), metrics.get("Hit_Rate"),
                        model_path, trained_at,
                    ),
                )
                if existing is None:
                    print(f"  [DB] ✓ {stock_symbol} → ilk kayıt: {model_name}")
                else:
                    print(
                        f"  [DB] ✓ {stock_symbol} → yeni en iyi: {model_name} "
                        f"(skor {existing['composite_score']:.2f} → {composite_score:.2f})"
                    )

    def get_best_model(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Belirtilen hisse için en iyi modelin tüm bilgilerini döndürür.

        Returns:
            dict veya None (kayıt yoksa).

        Kullanım örneği (ileride otomatik seçim için):
            best = db.get_best_model("TUPRS")
            if best:
                print(best["model_name"], best["model_path"])
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM best_models WHERE stock_symbol = ?",
                (stock_symbol,),
            ).fetchone()
        return dict(row) if row else None

    def get_leaderboard(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Tüm hisseler arasında composite_score'a göre sıralı lider tablosu.

        Returns:
            Her hissenin en iyi modelini içeren liste.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT stock_symbol, model_name, composite_score,
                       dir_acc, sharpe, mae, rmse, updated_at
                FROM   best_models
                ORDER  BY composite_score DESC
                LIMIT  ?
                """,
                (top_n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_experiments(
        self,
        stock_symbol: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Deney geçmişini filtreli sorgular.

        Args:
            stock_symbol : Yalnızca bu hissenin deneyleri (None → hepsi).
            model_name   : Yalnızca bu modelin deneyleri (None → hepsi).
            limit        : Maksimum satır sayısı.
        """
        clauses, params = [], []
        if stock_symbol:
            clauses.append("stock_symbol = ?")
            params.append(stock_symbol)
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM experiments
                {where}
                ORDER BY trained_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_model_comparison(self, stock_symbol: str) -> List[Dict[str, Any]]:
        """
        Belirli bir hisse için tüm modellerin ortalama metriklerini karşılaştırır.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    model_name,
                    COUNT(*)          AS run_count,
                    AVG(composite_score) AS avg_composite,
                    AVG(dir_acc)      AS avg_dir_acc,
                    AVG(sharpe)       AS avg_sharpe,
                    AVG(mae)          AS avg_mae,
                    AVG(rmse)         AS avg_rmse,
                    MAX(composite_score) AS best_composite
                FROM   experiments
                WHERE  stock_symbol = ?
                GROUP  BY model_name
                ORDER  BY avg_composite DESC
                """,
                (stock_symbol,),
            ).fetchall()
        return [dict(r) for r in rows]

    def __repr__(self) -> str:
        return f"<StockModelDB path={self.db_path!r}>"
