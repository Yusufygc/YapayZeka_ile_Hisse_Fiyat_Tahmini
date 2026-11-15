# -*- coding: utf-8 -*-
"""SQLite şema DDL sabitleri (stock_model_db.py bunları yeniden dışa aktarır)."""

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
    rmse_vs_benchmark REAL,
    net_return       REAL,
    buyhold_return   REAL,
    max_drawdown     REAL,
    trade_count      INTEGER,
    signal_diagnosis TEXT,
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
    rmse_vs_benchmark REAL,
    net_return      REAL,
    buyhold_return  REAL,
    max_drawdown    REAL,
    trade_count     INTEGER,
    signal_diagnosis TEXT,
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
    p10_close                REAL,
    p50_close                REAL,
    p90_close                REAL,
    predicted_return_p10     REAL,
    predicted_return_p50     REAL,
    predicted_return_p90     REAL,
    interval_method          TEXT,
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
    interval_coverage        REAL,
    interval_avg_width       REAL,
    nominal_coverage         REAL,
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

_CREATE_ANALYSIS_REFRESH_JOBS = """
CREATE TABLE IF NOT EXISTS analysis_refresh_jobs (
    job_id       TEXT PRIMARY KEY,
    symbol       TEXT NOT NULL,
    job_type     TEXT NOT NULL,
    status       TEXT NOT NULL,
    reason       TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    error        TEXT,
    payload_json TEXT
);
"""

_CREATE_IDX_REFRESH_JOBS_SYMBOL = """
CREATE INDEX IF NOT EXISTS idx_analysis_refresh_jobs_symbol
    ON analysis_refresh_jobs (symbol, status, started_at DESC);
"""

REQUIRED_SCHEMA_TABLES = (
    "experiments",
    "best_models",
    "forecast_runs",
    "forecast_points",
    "forecast_accuracy_summary",
    "analysis_refresh_jobs",
)
