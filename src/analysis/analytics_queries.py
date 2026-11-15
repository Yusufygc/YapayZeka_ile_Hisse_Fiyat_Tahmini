# -*- coding: utf-8 -*-
"""Adlandırılmış DuckDB analiz sorguları (salt SELECT).

Her sorgu: açıklama + (gerekli view'lar, SQL) varyantları; ilk uygun varyant
çalışır. `has_symbol=True` sorgular çıktıda `symbol` kolonu taşır ve
`AnalyticsDB.query(name, symbol=...)` ile süzülür (parametreli, enjeksiyon yok).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

# run_leaderboard.py ile aynı eşikler / sıralama kuralı (`_rank_leaders`, `_reliability_class`).
_TRADE_BAND_MIN = 8
_TRADE_BAND_MAX = 20
_CLONE_TOL = 0.001
_UNSTABLE_GAP = 0.25

_LEADERBOARD_SQL = f"""
WITH latest AS (
    SELECT symbol, max(run_id) AS run_id FROM runs WHERE status = 'completed' GROUP BY symbol
), final AS (
    SELECT m.symbol, m.run_id, m.model,
           m."Net_Return" AS final_net_return, m."BuyHold_Return" AS final_buyhold_return,
           m."Trade_Count" AS final_trade_count, m."Sharpe" AS final_sharpe,
           m."Signal_Diagnosis" AS final_signal_diagnosis
    FROM run_model_metrics m
    JOIN latest l ON m.symbol = l.symbol AND m.run_id = l.run_id
    JOIN runs r ON m.symbol = r.symbol AND m.run_id = r.run_id
    WHERE m.split = 'final_holdout' AND (r.n_models = 0 OR list_contains(r.model_list, m.model))
    QUALIFY row_number() OVER (PARTITION BY m.symbol, m.run_id, m.model) = 1
), wf AS (
    SELECT m.symbol, m.run_id, m.model, m."Net_Return" AS wf_net_return
    FROM run_model_metrics m
    JOIN latest l ON m.symbol = l.symbol AND m.run_id = l.run_id
    WHERE m.split = 'wf'
    QUALIFY row_number() OVER (PARTITION BY m.symbol, m.run_id, m.model) = 1
), joined AS (
    SELECT f.*, w.wf_net_return,
           f.final_net_return - f.final_buyhold_return AS final_excess_vs_buyhold,
           w.wf_net_return - f.final_net_return AS wf_final_net_gap
    FROM final f LEFT JOIN wf w USING (symbol, run_id, model)
), flagged AS (
    SELECT *,
           COALESCE(final_trade_count BETWEEN {_TRADE_BAND_MIN} AND {_TRADE_BAND_MAX}, false)
               AS trade_sufficiency_flag,
           COALESCE(final_trade_count <= 1
                    AND abs(final_net_return - final_buyhold_return) <= {_CLONE_TOL}, false)
               AS benchmark_clone_flag
    FROM joined
), classed AS (
    SELECT *,
           CASE
             WHEN benchmark_clone_flag OR final_trade_count IS NULL OR final_trade_count < 2
                 THEN 'invalid'
             WHEN trade_sufficiency_flag AND final_excess_vs_buyhold > 0 AND wf_net_return > 0
                 THEN 'stable'
             WHEN wf_net_return > 0 AND final_net_return <= 0 THEN 'unstable'
             WHEN wf_net_return > 0 AND wf_final_net_gap >= {_UNSTABLE_GAP} THEN 'unstable'
             WHEN trade_sufficiency_flag AND final_net_return >= 0 AND final_sharpe > 0
                  AND final_excess_vs_buyhold <= 0 THEN 'defensive'
             ELSE 'research_candidate'
           END AS leader_reliability_class,
           COALESCE(final_excess_vs_buyhold, -1e9) * 1000 + COALESCE(final_net_return, -1e9)
               AS trading_score
    FROM flagged
), ranked AS (
    SELECT *,
           row_number() OVER (PARTITION BY symbol ORDER BY trading_score DESC, model)
               AS trading_rank,
           row_number() OVER (PARTITION BY symbol ORDER BY wf_net_return DESC NULLS LAST, model)
               AS prediction_rank,
           count(*) OVER (PARTITION BY symbol) AS n_models_ranked
    FROM classed
), pred AS (
    SELECT symbol, model AS prediction_leader FROM ranked WHERE prediction_rank = 1
)
SELECT l.symbol, l.run_id, k.model, k.trading_score, k.wf_net_return, k.final_net_return,
       k.final_buyhold_return, k.wf_final_net_gap, k.final_excess_vs_buyhold,
       k.final_trade_count, k.final_sharpe, k.final_signal_diagnosis,
       k.trade_sufficiency_flag, k.benchmark_clone_flag, k.leader_reliability_class,
       p.prediction_leader, COALESCE(k.n_models_ranked, 0) AS n_models_ranked
FROM latest l
LEFT JOIN ranked k ON k.symbol = l.symbol AND k.trading_rank = 1
LEFT JOIN pred p ON p.symbol = l.symbol
ORDER BY CASE k.leader_reliability_class
           WHEN 'stable' THEN 0 WHEN 'defensive' THEN 1 WHEN 'research_candidate' THEN 2
           WHEN 'unstable' THEN 3 WHEN 'invalid' THEN 4 ELSE 5 END,
         k.final_excess_vs_buyhold DESC NULLS LAST, k.final_net_return DESC NULLS LAST, l.symbol
"""

_FORECAST_ACCURACY_SQL = """
SELECT r.stock_symbol AS symbol, r.model_name, p.horizon_index AS horizon_step,
       count(*) AS n,
       avg(p.direction_correct) AS direction_accuracy,
       avg(abs(p.predicted_return - p.actual_return)) AS mae_return,
       count(*) FILTER (WHERE p.p10_close IS NOT NULL AND p.p90_close IS NOT NULL) AS n_interval,
       avg(CASE WHEN p.p10_close IS NULL OR p.p90_close IS NULL THEN NULL
                WHEN p.actual_close BETWEEN least(p.p10_close, p.p90_close)
                                        AND greatest(p.p10_close, p.p90_close) THEN 1.0
                ELSE 0.0 END) AS interval_coverage,
       avg(abs(p.p90_close - p.p10_close)) AS mean_interval_width
FROM forecast_points p
JOIN forecast_runs r ON r.id = p.run_id
WHERE p.actual_close IS NOT NULL
GROUP BY r.stock_symbol, r.model_name, p.horizon_index
ORDER BY symbol, r.model_name, horizon_step
"""

_IC_HISTORY_SQL = """
SELECT run_id, created_at, as_of_date, model_name, horizon, n_symbols, n_rows,
       ic_mean, icir, pct_ic_positive,
       icir - lag(icir) OVER (ORDER BY TRY_CAST(created_at AS TIMESTAMP), run_id) AS icir_change,
       CASE WHEN config_json IS NOT NULL AND json_valid(config_json)
            THEN TRY_CAST(json_extract_string(config_json, '$.boost') AS BIGINT) END AS boost
FROM peer_runs
ORDER BY TRY_CAST(created_at AS TIMESTAMP), run_id
"""

_COVERAGE_MANIFEST_SQL = """
SELECT symbol, CAST(first_date AS DATE) AS first_date, CAST(last_date AS DATE) AS last_date,
       n_rows, rejected_rows,
       date_diff('day', CAST(last_date AS DATE), max(CAST(last_date AS DATE)) OVER ()) AS lag_days,
       cleaned_at
FROM panel_manifest
ORDER BY symbol
"""

_COVERAGE_PANEL_SQL = """
SELECT symbol, CAST(min("Date") AS DATE) AS first_date, CAST(max("Date") AS DATE) AS last_date,
       count(*) AS n_rows, CAST(NULL AS BIGINT) AS rejected_rows,
       date_diff('day', CAST(max("Date") AS DATE),
                 max(CAST(max("Date") AS DATE)) OVER ()) AS lag_days,
       CAST(NULL AS VARCHAR) AS cleaned_at
FROM panel
GROUP BY symbol
ORDER BY symbol
"""

_RUN_STATUS_SQL = """
SELECT symbol, count(*) AS n_runs,
       count(*) FILTER (WHERE status = 'completed') AS completed,
       count(*) FILTER (WHERE status = 'incomplete') AS incomplete,
       count(*) FILTER (WHERE status = 'running') AS running,
       count(*) FILTER (WHERE status = 'failed') AS failed,
       count(*) FILTER (WHERE status NOT IN ('completed', 'incomplete', 'running', 'failed'))
           AS other,
       max(run_id) AS latest_run_id,
       max(run_id) FILTER (WHERE status = 'completed') AS latest_completed_run_id
FROM runs
GROUP BY symbol
ORDER BY symbol
"""

_MODEL_STABILITY_SQL = """
SELECT symbol, "Model" AS model, "Validation_Mode" AS validation_mode,
       count(DISTINCT run_id) AS n_runs,
       avg("Dir_Acc") AS dir_acc_mean, stddev_samp("Dir_Acc") AS dir_acc_std,
       count("RMSE_vs_benchmark") AS n_rmse_obs,
       avg("RMSE_vs_benchmark") AS rmse_vs_benchmark_mean,
       stddev_samp("RMSE_vs_benchmark") AS rmse_vs_benchmark_std,
       avg("Sharpe") AS sharpe_mean, avg("Net_Return") AS net_return_mean,
       avg("Composite_Score") AS composite_score_mean
FROM run_experiments
GROUP BY symbol, "Model", "Validation_Mode"
ORDER BY symbol, model, validation_mode
"""


@dataclass(frozen=True)
class NamedQuery:
    description: str
    variants: Tuple[Tuple[Tuple[str, ...], str], ...]
    has_symbol: bool = True


QUERIES: Dict[str, NamedQuery] = {
    "leaderboard": NamedQuery(
        "Sembol başına son tamamlanan koşunun ticaret lideri (run_leaderboard._rank_leaders "
        "kuralı: excess*1000 + net) ve güvenilirlik sınıfı.",
        ((("runs", "run_model_metrics"), _LEADERBOARD_SQL),),
    ),
    "forecast_accuracy": NamedQuery(
        "Çözümlenmiş forecast noktaları: model/sembol/ufuk adımı başına n, yön doğruluğu, "
        "getiri MAE, aralık kapsaması ve ortalama genişlik.",
        ((("forecast_points", "forecast_runs"), _FORECAST_ACCURACY_SQL),),
    ),
    "ic_history": NamedQuery(
        "Havuzlu (peer) model koşularında IC / ICIR zaman serisi.",
        ((("peer_runs",), _IC_HISTORY_SQL),),
        has_symbol=False,
    ),
    "data_coverage": NamedQuery(
        "Sembol başına ilk/son tarih, satır, reddedilen satır ve en güncel tarihe göre gecikme.",
        ((("panel_manifest",), _COVERAGE_MANIFEST_SQL), (("panel",), _COVERAGE_PANEL_SQL)),
    ),
    "run_status": NamedQuery(
        "Sembol başına koşu durumu sayıları (completed/incomplete/running/failed).",
        ((("runs",), _RUN_STATUS_SQL),),
    ),
    "model_stability": NamedQuery(
        "Sembol+model(+doğrulama modu) için koşular arası Dir_Acc / RMSE_vs_benchmark "
        "sayı, ortalama ve std.",
        ((("run_experiments",), _MODEL_STABILITY_SQL),),
    ),
}


def resolve_sql(spec: NamedQuery, views: Set[str]) -> Optional[str]:
    """Mevcut view'larla çalışabilen ilk varyantın SQL'i; yoksa None."""
    for requires, sql in spec.variants:
        if set(requires) <= views:
            return sql
    return None
