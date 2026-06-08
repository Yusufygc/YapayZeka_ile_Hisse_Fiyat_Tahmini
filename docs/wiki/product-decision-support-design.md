---
title: Product Decision Support Design
type: feature-plan
status: active
last_updated: 2026-06-04
owner: llm
source_count: 10
---

# Product Decision Support Design

The planned desktop AI experience treats `ts_forecasting_lab` as the model,
forecast, registry, and XAI engine behind a decision-support product. Detailed
working notes live under the local `yeniTasarim/` directory (not version
controlled). This wiki page is the maintained summary of those decisions.

## Product Position

The system is an **AI-assisted analytical decision-support tool**, not an
automated trading bot or personal investment advisor.

**Correct product language:**
- Model view / direction expectation / analytical evaluation
- Historical validation performance / past accuracy
- Risk warnings / uncertainty
- Factors the model highlighted
- "This is not personal investment advice"

**Forbidden product language:**
- "Buy" / "sell" / "add to portfolio" / "recommend"
- "Definitely rising" / "correct model" / "guaranteed"
- User-specific suitability / "suitable for your risk profile"

The regulatory boundary: the system must not behave like a personal investment
advisor. No personalised portfolio, risk-profile, income, or investment-goal
guidance. First product phase is general analytical model output only.

Every analysis response must include:
```
Bu çıktı kişisel yatırım tavsiyesi değildir. Model geçmiş verilerden
üretilmiş analitik bir tahmin sunar; nihai karar kullanıcıya aittir.
```

## Target Architecture Flow

```text
User
  -> Desktop App AI page
      -> FastAPI GET /analysis/{symbol}
          -> StockModelDB (best_models, forecast_runs, forecast_points)
          -> XAI outputs (outputs/{SYMBOL}/latest/xai/)
          -> ConfidenceCalculator
          -> RegimeContext (Faz 2)
      -> LLM explanation layer
          -> converts structured JSON to plain-language analysis
      -> User receives analytical view and makes their own decision
```

The engine layer (`ForecastingPipeline`, `ForecastRunner`, `XAI`) runs on a
**periodic batch schedule**. User queries read the latest persisted analysis
artefacts; they do not trigger training.

## Training Policy

- Per-query retraining is forbidden (slow, expensive, non-deterministic).
- Default strategy: periodic full retrain on rolling window, walk-forward
  validation, validated champion/challenger promotion.
- Weight-level continuation training and fine-tuning are reserved for a later
  experimental phase (see `yeniTasarim/08`).

## MVP Scope (Faz 1)

1. Periodic batch forecast pipeline.
2. `GET /analysis/{symbol}` endpoint.
3. Best trainable model result from registry.
4. Latest forecast points.
5. Model performance summary.
6. Simple trend label.
7. XAI top factors (positive and negative separated).
8. Data freshness check.
9. Confidence label (`low` / `medium` / `high`).
10. Controlled LLM explanation (prompt template documented in
    [LLM Explanation Policy](llm-explanation-policy.md)).
11. Disclaimer text on every response.

## Out of Scope for MVP

- Personalised portfolio advice or user-suitability assessment.
- Automated buy/sell orders.
- Per-query training.
- Continuation training / fine-tuning.
- Complex ML regime model.
- Portfolio optimisation, long-short, market-neutral strategies.
- Transaction-cost model as default decision input.
- Post-deployment operational layers (monitoring, Slack/email, Docker, Redis,
  auth, rate limiting).

## Faz 2 Additions (after MVP)

- Ensemble directional agreement as supporting confidence input.
- Regime-aware confidence label.
- Forecast accuracy resolution (live rolling dir_acc).
- Model stability score.
- Per-symbol data quality score.
- Better XAI fold-stability summaries.
- Cross-run leaderboard.
- Naive-leader rejection.
- Signal-diagnosis-based confidence capping.

## E2 Peer / Trend Tendency Output (additive, 2026-06-04)

The E2 pooled global model adds a **cross-sectional peer view** to the same
product boundary — still decision-support, never a buy/sell bot. Surfaced as the
additive `peer` block on `GET /analysis/{symbol}`:

- **Peer position**: peer_score / percentile / label (outperform/inline/
  underperform) — where the symbol ranks *relative to the BIST universe today*.
- **Trend tendency (Faz 7b)**: `yukarı / yatay / aşağı / belirsiz` + calibrated
  `prob_up` + `expected_return`. This satisfies the original product goal of an
  up/sideways/down lean — but framed honestly per the rules above:
  - It is a **probabilistic tilt, not a guarantee** ("Definitely rising" stays
    forbidden). Faz 7 evidence: the tilt is modest but real and monotone (top
    quintile ~54% up vs bottom ~43%, base ~50%).
  - It is **relative/market-neutral**, not an absolute price promise.
  - Reliability is governed separately by the segment-ICIR `confidence_label`;
    `high` confidence = tradeable *and* most accurate (Faz 7), `low` includes the
    untradeable strongest-signal tail (flagged honestly).
- **No personalisation**: no portfolio, suitability, or order generation — same
  regulatory boundary as the rest of the product.

Served via the nightly batch (no per-query training); see
[Analysis API Contract](analysis-api-contract.md) and
[E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md).

## Terminology Simplification for Non-Technical Users (2026-06-08)

To ensure that the analytical decision-support tool remains accessible to laypersons with no statistical or quantitative background, a sweeping simplification of UI-facing terminology has been implemented across the AI left panel cards:

- **AI Success Status (`PerformanceCard`):** Technical metrics have been translated into plain descriptive Turkish:
  - *Composite Score* -> "Genel Başarı Puanı"
  - *Directional Accuracy* -> "Yönü Doğru Tahmin"
  - *Hit Rate* -> "Fiyat Yakınlığı"
  - *Sharpe Ratio* -> "Kazanç Güvenilirliği" (risk-adjusted return simplified)
  - *RMSE* -> "Ortalama Sapma (Geniş)" (punishes large errors)
  - *MAE* -> "Net Fark Hata Payı" (average raw difference)
  - *Stability Score* -> "Performans Tutarlılığı"
- **AI Prediction Card (`PredictionCard`):** Interval estimation methods simplified:
  - *residual_b2* -> "Geçmiş Sapmalara Göre (%80 Olasılık)"
  - *conformal* -> "Hata Analizine Göre (%90 Olasılık)"
- **Sector Peer Position (`PeerCard`):** Segment quantiles (Q1-Q5) dynamically mapped to descriptive labels ("Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek") and sector names translated dynamically (e.g., "Basic Materials" -> "Hammadde / Temel Malzemeler").
- **AI Decision Drivers (`XAICard`):** Cleaned up SHAP attribution detail cards to hide raw log-return/probability contribution floats (e.g. `katkı: +0.02125` / `yöntem: sequence`) and show only the clean natural language reason, mapping group/method categories to clear concepts.

## Faz 3 (Research Phase)

- Regime-based model performance tracking.
- Regime-based ensemble weights.
- Champion/challenger deployment with canary activation.
- Model and data drift detection.
- Forecast confidence intervals.
- Portfolio-level correlation and risk view.
- CPCV, bootstrap p-value / reality check.

## Related Pages

- [Analysis API Contract](analysis-api-contract.md)
- [Confidence and Risk Policy](confidence-and-risk-policy.md)
- [LLM Explanation Policy](llm-explanation-policy.md)
- [Persistence and API](persistence-and-api.md)
- [Model Catalog](model-catalog.md)
- [Backtest Signal Improvement Plan](backtest-signal-improvement-plan.md)
- [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md)
