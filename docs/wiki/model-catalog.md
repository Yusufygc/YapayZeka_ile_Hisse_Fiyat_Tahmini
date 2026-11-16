---
title: Model Catalog
type: concept
status: active
last_updated: 2026-11-16
owner: llm
---

# Model Catalog

## Kalan modeller

| Model | Dosya | Rol |
|---|---|---|
| `GlobalPooledModel` (LightGBM) | `src/models/global_pooled_model.py` | Kol-B: tek koşullu model, çapraz-kesit sıra hedefi (`target_cs`) |
| `TorchMLPModel` | `src/models/torch_mlp_model.py` | Kol-B ensemble MLP bacağı (3 tohum) |
| `EnsemblePooledModel` | `src/models/ensemble_pooled_model.py` | LGB + MLP (`--model ensemble`; varsayılan `lgb`) |
| `PooledVolModel` (Ridge HAR-X) | `src/risk/model.py` | Risk: 5g oynaklık + %80 bant ([Risk Modeli](risk-model.md)) |

Çıktılar: peer skoru/yüzdeliği/etiketi, segment ICIR'e bağlı güven (`src/serving/confidence.py`),
kalibre trend eğilimi (`trend_tendency.py`, Faz 7 quintile tablosu), SHAP XAI (`peer_xai.py`),
risk seviyesi ve fiyat bandı. Beklenen getiri Kol-B quintile tablosundan; bant genişliği yalnız risk modelinden.

## Emekli: Kol-A (karar "C")

Hisse başına XGBoost / Ridge Return / LightGBM Return, LSTM, Prophet, ARIMA, DLinear, naive
benchmark'lar, `Ensemble Inverse RMSE`, model registry, `model_scope.py`, `stock_model_db.py`,
walk-forward/backtest/sinyal kalibrasyonu, `src.cli.*`: 815 walk-forward deneyin hiçbiri naive tahmini
geçemedi; ürün kapsamından çıkarıldı ve silindi. Tarihsel notlar "historical" bannerlı sayfalarda
(ör. [Validation and Backtesting](validation-and-backtesting.md), [Backtest Signal Improvement Plan](backtest-signal-improvement-plan.md)).

## Kapsam kuralları

Kol-B skoru göreli ve olasılıksaldır (karar desteği, al/sat botu değil). Risk kapısı geçmezse
`naive_fallback` servis edilir ve işaretlenir. Bkz. [E2 Pooled Global Model Epic](e2-pooled-global-model-epic.md).
