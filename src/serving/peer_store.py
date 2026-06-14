# -*- coding: utf-8 -*-
"""Isolated SQLite store for pooled-model serving outputs — E2 Faz 5.

Karar (kilitli): mevcut `best_models` semasina DOKUNMA. Cross-sectional ciktilar
ayri tablolarda:
    global_model_runs : 1 satir/egitim kosusu (artifact ad, data-snapshot hash,
                        IC ozeti, config, as_of_date).
    peer_scores       : kosu x sembol (peer_score, percentile, label, segment,
                        segment_icir, confidence label/reasons/warnings).

Ayni DB dosyasinda yasayabilir (yalniz kendi CREATE TABLE IF NOT EXISTS'lerini
calistirir) ya da ayri bir dosyada. Per-query egitim yok: nightly batch yazar,
API okur.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS global_model_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    as_of_date TEXT,
    model_name TEXT,
    data_snapshot_hash TEXT,
    n_symbols INTEGER,
    n_rows INTEGER,
    horizon INTEGER,
    ic_mean REAL,
    icir REAL,
    pct_ic_positive REAL,
    config_json TEXT
);
CREATE TABLE IF NOT EXISTS peer_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    as_of_date TEXT,
    peer_score REAL,
    peer_percentile REAL,
    peer_label TEXT,
    raw_pred REAL,
    universe_size INTEGER,
    segment_liq TEXT,
    segment_vol TEXT,
    segment_sector TEXT,
    segment_icir REAL,
    confidence_label TEXT,
    confidence_reasons TEXT,
    confidence_warnings TEXT,
    trend_label TEXT,
    trend_prob_up REAL,
    trend_expected_return REAL,
    xai_top_features TEXT,
    xai_method TEXT,
    xai_approximate INTEGER,
    xai_error TEXT,
    xai_generated_at TEXT,
    kolb_price_p50 REAL,
    kolb_price_low REAL,
    kolb_price_high REAL,
    kolb_horizon_days INTEGER,
    kolb_band_level REAL,
    UNIQUE(run_id, symbol),
    FOREIGN KEY(run_id) REFERENCES global_model_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_peer_scores_symbol ON peer_scores(symbol);
CREATE INDEX IF NOT EXISTS idx_peer_scores_run ON peer_scores(run_id);
"""

_PEER_COLS = [
    "symbol", "as_of_date", "peer_score", "peer_percentile", "peer_label",
    "raw_pred", "universe_size", "segment_liq", "segment_vol", "segment_sector",
    "segment_icir", "confidence_label", "confidence_reasons", "confidence_warnings",
    "trend_label", "trend_prob_up", "trend_expected_return", "xai_top_features",
    "xai_method", "xai_approximate", "xai_error", "xai_generated_at",
    "kolb_price_p50", "kolb_price_low", "kolb_price_high", "kolb_horizon_days",
    "kolb_band_level",
]

# Eski DB'lere (run_id<3) eklenen kolonlar: ALTER TABLE ile geriye-uyumlu migrasyon.
_PEER_MIGRATIONS = {
    "trend_label": "TEXT",
    "trend_prob_up": "REAL",
    "trend_expected_return": "REAL",
    "xai_top_features": "TEXT",  # E2 Kol-B XAI (JSON: top_positive/top_negative)
    "xai_method": "TEXT",
    "xai_approximate": "INTEGER",
    "xai_error": "TEXT",
    "xai_generated_at": "TEXT",
    "kolb_price_p50": "REAL",
    "kolb_price_low": "REAL",
    "kolb_price_high": "REAL",
    "kolb_horizon_days": "INTEGER",
    "kolb_band_level": "REAL",
}


@dataclass(frozen=True)
class GlobalRunMeta:
    model_name: str
    as_of_date: str
    data_snapshot_hash: str = ""
    n_symbols: int = 0
    n_rows: int = 0
    horizon: int = 5
    ic_mean: float = float("nan")
    icir: float = float("nan")
    pct_ic_positive: float = float("nan")
    config: Optional[dict] = None


class PeerStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Eski peer_scores tablolarina eksik kolonlari ekle (idempotent)."""
        have = {r["name"] for r in conn.execute("PRAGMA table_info(peer_scores)")}
        for col, sqltype in _PEER_MIGRATIONS.items():
            if col not in have:
                conn.execute(f"ALTER TABLE peer_scores ADD COLUMN {col} {sqltype}")

    # ------------------------------------------------------------------ write
    def insert_run(self, meta: GlobalRunMeta, created_at: Optional[str] = None) -> int:
        import datetime as _dt

        created_at = created_at or _dt.datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO global_model_runs
                   (created_at, as_of_date, model_name, data_snapshot_hash,
                    n_symbols, n_rows, horizon, ic_mean, icir, pct_ic_positive, config_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (created_at, meta.as_of_date, meta.model_name, meta.data_snapshot_hash,
                 int(meta.n_symbols), int(meta.n_rows), int(meta.horizon),
                 _f(meta.ic_mean), _f(meta.icir), _f(meta.pct_ic_positive),
                 json.dumps(meta.config or {}, ensure_ascii=False)),
            )
            return int(cur.lastrowid)

    def insert_peer_scores(self, run_id: int, scored: pd.DataFrame) -> int:
        """scored: peer_scoring + segment + confidence birlesmis DataFrame.

        Eksik kolonlar None ile doldurulur; listeler (reasons/warnings) JSON'a
        cevrilir. Returns yazilan satir sayisi."""
        df = scored.copy()
        for col in _PEER_COLS:
            if col not in df.columns:
                df[col] = None
        for col in ("confidence_reasons", "confidence_warnings"):
            df[col] = df[col].map(_to_json_list)
        df["xai_top_features"] = df["xai_top_features"].map(_to_json_obj)
        rows = [
            (run_id, *(_cell(r[c]) for c in _PEER_COLS))
            for _, r in df.iterrows()
        ]
        with self._connect() as conn:
            conn.executemany(
                f"""INSERT OR REPLACE INTO peer_scores
                    (run_id, {", ".join(_PEER_COLS)})
                    VALUES ({", ".join("?" * (len(_PEER_COLS) + 1))})""",
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------------- read
    def latest_run(self) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM global_model_runs ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def get_peer_score(self, symbol: str, run_id: Optional[int] = None) -> Optional[dict]:
        with self._connect() as conn:
            if run_id is None:
                run = conn.execute(
                    "SELECT run_id FROM global_model_runs ORDER BY run_id DESC LIMIT 1"
                ).fetchone()
                if not run:
                    return None
                run_id = run["run_id"]
            row = conn.execute(
                "SELECT * FROM peer_scores WHERE run_id=? AND symbol=?",
                (run_id, str(symbol)),
            ).fetchone()
            return dict(row) if row else None

    def get_run_peer_scores(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM peer_scores WHERE run_id=? ORDER BY peer_percentile DESC",
                conn, params=(run_id,),
            )


def _f(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def _cell(v: Any) -> Any:
    if isinstance(v, float) and v != v:
        return None
    return v


def _to_json_list(v: Any) -> str:
    if v is None:
        return "[]"
    if isinstance(v, str):
        return v if v.startswith("[") else json.dumps([v], ensure_ascii=False)
    try:
        return json.dumps(list(v), ensure_ascii=False)
    except TypeError:
        return json.dumps([str(v)], ensure_ascii=False)


def _to_json_obj(v: Any) -> Optional[str]:
    """XAI dict -> JSON string. None / NaN / bos -> None (kolon NULL kalir)."""
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, str):
        return v if v.strip() else None
    try:
        return json.dumps(v, ensure_ascii=False)
    except TypeError:
        return None
