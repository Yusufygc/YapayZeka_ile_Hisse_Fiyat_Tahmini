---
title: Analysis API Contract
type: concept
status: active
last_updated: 2026-05-25
owner: llm
source_count: 6
---

# Analysis API Contract

Documents the `GET /analysis/{symbol}` endpoint added in Faz 1. This is the
product-facing endpoint; existing registry endpoints (`/best-model`,
`/experiments`, `/metrics`, `/leaderboard`) are preserved unchanged.

## Endpoint

```http
GET /analysis/{symbol}
```

`symbol` is case-insensitive (e.g. `TUPRS`, `tuprs`).

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
    "method": "SHAP TreeExplainer",
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
`human_label`, `importance`, and `direction`.

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
| `xai_unavailable` | Forecast available but XAI output missing |
| `error` | Unexpected server error |

**Status priority (most severe first):**
`no_model > no_forecast > stale_data > xai_unavailable > low_confidence > ok`

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

## Related Pages

- [Product Decision Support Design](product-decision-support-design.md)
- [Confidence and Risk Policy](confidence-and-risk-policy.md)
- [Persistence and API](persistence-and-api.md)
