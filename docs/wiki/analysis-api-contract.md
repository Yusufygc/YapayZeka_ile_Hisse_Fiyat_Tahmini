---
title: Analysis API Contract
type: concept
status: active
last_updated: 2026-06-14
owner: llm
source_count: 7
---

# Analysis API Contract

Documents the `GET /analysis/{symbol}` endpoint added in Faz 1. This is the
product-facing endpoint; existing registry endpoints (`/best-model`,
`/experiments`, `/metrics`, `/leaderboard`) are preserved unchanged.

## Endpoint

```http
GET /analysis/{symbol}
GET /v1/analysis/{symbol}   # Sprint 8 A8.6 — versiyonlu alias, ayni davranis
```

`symbol` is case-insensitive (e.g. `TUPRS`, `tuprs`).
The `/v1/` alias is intended for clients that want explicit API versioning;
its response body is byte-identical to `/analysis/{symbol}` (sans
`generated_at` timestamp). A future breaking change will introduce `/v2/`
without altering `/v1/`.

For Faz 2+, if user-specific scenario parameters are required a
`POST /assistant/stock-analysis` variant can be added.

## Response Schema

```json
{
  "symbol": "ASELS",
  "analysis_status": "ok",
  "generated_at": "2026-05-18T13:00:00+03:00",
  "data": {
    "last_observed_date": "2026-05-17",
    "last_close": 123.45,
    "data_freshness": "fresh",
    "staleness_days": 1
  },
  "model": {
    "model_name": "XGBoost",
    "model_family": "tree",
    "selection_reason": "Geçmiş doğrulama sonuçlarına göre en iyi eligible model",
    "source_experiment_id": 123,
    "run_id": "20260518_ASELS_walk_forward_xgboost",
    "validation_mode": "walk_forward",
    "trained_at": "2026-05-18T02:30:00+03:00",
    "is_trainable_model": true,
    "is_baseline": false,
    "is_ensemble": false
  },
  "forecast": {
    "horizon_days": 5,
    "trend_label": "up",
    "weekly_expected_return": 0.032,
    "trend_threshold": 0.012,
    "ensemble_agreement": 0.71,
    "trend_context": {
      "market_regime": "bull",
      "relative_strength": "outperforming",
      "alignment_with_forecast": true
    },
    "points": [
      {
        "target_date": "2026-05-26",
        "horizon_index": 1,
        "bounded_predicted_close": 102.5,
        "predicted_return": 0.025,
        "p10_close": 100.8,
        "p50_close": 102.5,
        "p90_close": 104.5,
        "predicted_return_p10": 0.008,
        "predicted_return_p50": 0.025,
        "predicted_return_p90": 0.045,
        "lower_band": 99.0,
        "upper_band": 105.0,
        "price_tick": 0.05
      }
    ]
  },
  "performance": {
    "rmse": 1.23,
    "mae": 0.89,
    "directional_accuracy": 56.4,
    "hit_rate": 52.1,
    "sharpe": 0.42,
    "rmse_vs_benchmark": 0.92,
    "composite_score": 67.5,
    "stability_score": 0.31
  },
  "confidence": {
    "label": "medium",
    "reasons": [],
    "warnings": []
  },
  "xai": {
    "available": true,
    "status": "available",
    "method": "SHAP TreeExplainer",
    "method_detail": {
      "shap_tree": 12,
      "permutation_fallback": 0
    },
    "approximate_ratio": 0.0,
    "feature_stability_top": [
      {
        "feature": "RSI_14",
        "stability": 0.82
      }
    ],
    "generated_at": "2026-05-18T03:05:00+03:00",
    "run_id": "20260518_ASELS_walk_forward_xgboost",
    "background_scope": "train_slice",
    "dictionary_coverage": {
      "coverage_ratio": 0.97,
      "covered": 29,
      "total": 30
    },
    "top_positive_reasons": [
      {
        "feature_name": "RSI_14",
        "human_label": "RSI 14: hissenin aşırı alım veya aşırı satım bölgesine yaklaşması",
        "importance": 0.42,
        "direction": "positive",
        "feature_group": "technical",
        "reason": "RSI 14 momentum tarafındaki kısa vadeli güçlenmeyi gösterdi.",
        "method": "sequence",
        "contribution": 0.08,
        "approximate": true
      }
    ],
    "top_negative_reasons": [],
    "group_summaries": [
      {
        "feature_group": "macro",
        "group_label": "Makro ekonomik sinyaller",
        "total_importance": 0.31,
        "net_contribution": -0.04,
        "direction": "asagi",
        "top_features": ["USDTRY_Return", "Rate_Level"],
        "reason": "Makro ekonomik sinyaller model tahminini asagi yonde etkileyen faktorler arasinda.",
        "approximate_ratio": 0.5
      }
    ],
    "model_family_caveat": "Tree modeller için SHAP TreeExplainer güvenilirdir.",
    "caveat": "XAI, modelin tahmininde öne çıkan değişkenleri gösterir; nedensellik kanıtı değildir."
  },
  "refresh_status": "none",
  "refresh_reason": null,
  "refresh_job_id": null,
  "forecast_source": {
    "type": "model",
    "model_name": "XGBoost",
    "source_experiment_id": 123,
    "run_at": "2026-05-18T03:00:00+03:00",
    "last_observed_date": "2026-05-17",
    "method": null,
    "members": [],
    "weights": {},
    "source_experiment_ids": [],
    "forecast_strategy": "recursive_direct_target",
    "artifact_mode": "artifact_loaded",
    "warnings": ["projected_exogenous_features"]
  },
  "disclaimer": "Bu çıktı kişisel yatırım tavsiyesi değildir. Model geçmiş verilerden üretilmiş analitik bir tahmin sunar; nihai karar kullanıcıya aittir.",
  "data_quality": {
    "psi_30d": 0.07,
    "psi_status": "stable",
    "stale_warning": false,
    "reason": null
  }
}
```

`forecast.ensemble_agreement`, `forecast.trend_context`, and
`xai.model_family_caveat` are added in Faz 2. XAI factor fields
`feature_group`, `reason`, `method`, `contribution`, and `approximate` are
optional detail fields; older consumers can keep using `feature_name`,
`human_label`, `importance`, and `direction`. The 2026-06-14 XAI audit adds
`xai.status`, `method_detail`, `approximate_ratio`, `feature_stability_top`,
`generated_at`, `run_id`, `background_scope`, and `dictionary_coverage`
additively from run-level XAI manifests. The macro/indicator XAI extension adds
`xai.group_summaries`, an additive list of group-level explanations derived
from visible XAI artifact rows. It summarizes macro, technical,
market-relative, volume, volatility, lag, cross-sectional, meta, signal, and
other groups without changing the factor arrays.

`refresh_status`, `refresh_reason`, `refresh_job_id`, and `forecast_source` were
added in the 2026-05-20 serving hardening phase. `forecast_source.type` is
`"model"` for a single model forecast and `"ensemble"` when the persisted
forecast run has ensemble metadata or an ensemble model name. Ensemble sources
surface member names, weights, source experiment ids, forecast strategy,
artifact mode, and warnings.

### Forecast Point Quantile Fields (Sprint 4 — 2026-05-25)

`forecast.points[*]` now exposes optional probabilistic forecasting fields:

| Field | Type | Source |
|---|---|---|
| `p10_close`, `p50_close`, `p90_close` | float \| null | Quantile-aware model (LightGBM Quantile, LSTM Lite MC Dropout) |
| `predicted_return_p10/p50/p90` | float \| null | Same quantiles, returns space |
| `lower_band`, `upper_band`, `price_tick` | float \| null | BIST band rules applied to bounded close |

Fields are populated only when the production model implements
`predict_quantiles()`. Scalar-only models leave them as `null` and clients
must fall back to `bounded_predicted_close` + `predicted_return`. The
quantile path is recursive — `ForecastPointGenerator.roll_forward_recursive`
calls `predict_quantiles` on every horizon step and applies the same BIST
band bounding as the median forecast.

## Analysis Status Codes

| Status | Meaning |
|---|---|
| `ok` | Analysis valid and fresh |
| `stale_data` | Data is older than staleness threshold; interpret with caution |
| `no_model` | No eligible model registered for this symbol |
| `no_forecast` | Model exists but no forecast has been run |
| `low_confidence` | Result exists but confidence label is `low` |
| `error` | Unexpected server error |

**Status priority (most severe first):**
`no_model > no_forecast > stale_data > low_confidence > ok`

XAI availability is no longer part of `analysis_status`. Forecast/model health
stays in the top-level status, while explanation health is reported under
`xai.status` (`available`, `fallback`, `missing_artifact`, or `error` when a
future reader can distinguish a malformed artifact). Clients that need to warn
about missing explanations should use `xai.status`, not `analysis_status`.

When `no_model` or `no_forecast`, the endpoint still returns HTTP 200 with the
appropriate status field — it does not return 404. This allows the desktop app
to display a helpful user message rather than a raw error.

When the best model exists but no matching forecast is available, the service
queues a refresh job and returns `analysis_status="no_forecast"` plus
`refresh_status` and `refresh_job_id`. Forecast matching requires the forecast
model name, `source_experiment_id`, and latest observed date to match the
current best model and current CSV data. Stale forecasts queue refresh work with
reason `stale_market_data`.

Refresh status enum:

| Status | Meaning |
|---|---|
| `none` | No refresh was needed or queued |
| `queued` | Refresh job is waiting |
| `running` | Refresh job is in progress |
| `completed` | Refresh finished and the response may have been rebuilt from fresh state |
| `failed` | Refresh failed; `refresh_reason` contains the failure reason when known |

## Confidence Label

Returned as a string enum, **not a percentage**:

| Label | Meaning |
|---|---|
| `low` | Do not use as primary decision input; major caveats apply |
| `medium` | Moderate analytical value; interpret with stated caveats |
| `high` | Strong historical signal; still not a guarantee |

See [Confidence and Risk Policy](confidence-and-risk-policy.md) for the
derivation rules.

## Data Freshness

`data_freshness` is `"fresh"` when `staleness_days` is within the BIST
trading-day threshold (default: ≤ 1 trading day). Values beyond the threshold
produce `"stale_data"`.

## XAI Caveat

Every XAI block must include the caveat string:
```
XAI, modelin tahmininde öne çıkan değişkenleri gösterir; nedensellik kanıtı değildir.
```

## Confidence Reasons (Sprint 8 — 2026-05-25)

`confidence.reasons` is populated for both `medium` and `high` labels with
positive signals so consumers can render rationale strings. The current
generator (`analysis_service._build_positive_reasons`) emits a subset of:

| Trigger | Example |
|---|---|
| Always when present | `Walk-forward yonsel dogruluk: %62.0` |
| `hit_rate` present | `Hit rate: %58.0` |
| `rmse_vs_benchmark < 1.0` | `RMSE benchmark altinda (rmse_vs_benchmark=0.850)` |
| `stability_score >= 0.5` | `Fold istikrari yuksek (stability_score=0.65)` |
| `composite_score` present | `Composite score: 78.0/100` |
| `ensemble_direction_agreement >= 5/7` | `Ensemble yon uzlasisi yuksek (0.83)` |
| `data_quality.psi_status == "stable"` | `Veri dagilimi stabil (PSI 30g < 0.10)` |

Negative reasons come from `compute_confidence` directly (e.g. signal
diagnosis flags, low stability, low ensemble agreement). The two lists
share the same `reasons` array.

`confidence.warnings` accumulates:

- Stale data: `Veri guncel degil; tahmin yorumlanirken dikkatli olunmali.`
- `data_drift_moderate:psi_30d=<value>` (Sprint 7)
- `data_drift_major:psi_30d=<value>_>=0.25` (Sprint 7; also forces `low`)
- Other compute_confidence warnings (PSI hard block, corporate-action,
  degraded model status, etc.)

## Forecast Ensemble Agreement (Sprint 8 A8.4)

`forecast.ensemble_agreement` is populated for ensemble forecast rows from
the persisted `ensemble_direction_agreement` column. Value range: `0.0..1.0`
(fraction of ensemble members that match the head model's direction).
Single-model forecasts return `null`. Used by `compute_confidence` as a
soft signal — `< 0.5` degrades to medium, `>= 5/7` allows high.

## Data Quality Block (Sprint 7 — 2026-05-25)

`data_quality` carries on-the-fly PSI drift between the last 30 trading days
of stationary OHLCV-derived features (log_return, range_pct, volume_log_change)
and the prior 252-day window. Bins=3 (chosen to keep the noise floor low at
30-sample holdout).

| Field | Type | Description |
|---|---|---|
| `psi_30d` | float \| null | Max per-feature PSI; null when unavailable |
| `psi_status` | enum | `stable` (<0.10) / `moderate_drift` (0.10..0.25) / `major_drift` (>=0.25) / `unavailable` |
| `stale_warning` | bool | True when CSV is missing |
| `reason` | string \| null | Diagnostic when status is `unavailable` |

Confidence interactions (`src/api/services/analysis_service.py`):

- `major_drift` → confidence label forced to `low`; warning
  `data_drift_major:psi_30d=<value>_>=0.25` appended.
- `moderate_drift` → confidence `high` is downgraded to `medium`; warning
  `data_drift_moderate:psi_30d=<value>` appended.
- `stable` / `unavailable` → no confidence change.

## Peer Block (E2 Faz 5–8, additive)

`AnalysisResponse.peer` (`PeerBlock`) is an **optional, additive** block from the
pooled global model's nightly cross-sectional scoring (`PeerStore`). It is `null`
when no peer data exists for the symbol; existing absolute `forecast`/
`confidence` fields are never changed by it.

| Field | Type | Meaning |
| --- | --- | --- |
| `available` | bool | true when peer data attached |
| `as_of_date` | string | scoring snapshot date |
| `peer_score` | float | −1..1 centered cross-sectional rank |
| `peer_percentile` | float | 0..100 |
| `peer_label` | string | outperform / inline / underperform / unknown |
| `universe_size` | int | names ranked that date |
| `segment_liq/vol/sector` | string | liquidity/volatility/sector bucket |
| `segment_icir` | float | composite segment ICIR (tradability-aware) |
| `confidence_label` | string | low / medium / high (ICIR × tradability gates) |
| `confidence_reasons` / `confidence_warnings` | string[] | human-readable |
| `model_run_id` | int | `global_model_runs.run_id` |
| `icir_overall` | float | run-level ICIR |
| `trend_label` | string | **yukarı / yatay / aşağı / belirsiz** (Faz 7b) |
| `trend_prob_up` | float \| null | calibrated P(h-day return > 0) |
| `trend_expected_return` | float \| null | calibrated mean h-day log return |
| `xai_available` | bool | true when peer XAI rows are attached |
| `xai_method` | string \| null | peer XAI method, e.g. `ensemble_permutation` |
| `xai_approximate` | bool \| null | true when peer XAI is approximate/sensitivity-based |
| `xai_error` | string \| null | structured nightly XAI failure reason, if any |
| `xai_generated_at` | string \| null | peer XAI generation timestamp |
| `xai_group_summaries` | object[] | group-level peer-rank XAI summaries |

Trend is a **probabilistic tilt, not a guarantee**, derived from the peer
percentile via Faz 7 quintile calibration (Q1..Q5 P(up) 0.435→0.541). Reliability
is governed separately by `confidence_label`. See
[Persistence and API](persistence-and-api.md) and
[E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md).

## Related Pages

- [Product Decision Support Design](product-decision-support-design.md)
- [Confidence and Risk Policy](confidence-and-risk-policy.md)
- [Persistence and API](persistence-and-api.md)
- [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md)
