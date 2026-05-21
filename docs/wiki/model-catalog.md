---
title: Model Catalog
type: concept
status: active
last_updated: 2026-05-20
owner: llm
source_count: 8
---

# Model Catalog

All model implementations follow the `BaseModel` interface in
`src/models/base_model.py`: `train()`, `predict()`, `save()`, and `load()`.

## Scope Rules

`src/pipeline/model_scope.py` is the source of truth for benchmark and candidate
model grouping.

## Benchmarks

Benchmarks are cheap reference models. They can appear in reports and selection
pressure, but they are not production candidates.

| Model | File | Role |
|---|---|---|
| `Naive Last Value` | `src/models/naive_model.py` | Repeat latest value |
| `Naive Zero Return` | `src/models/naive_model.py` | Predict no return movement |
| `Naive Drift` | `src/models/naive_model.py` | Simple drift extrapolation |

## Production Candidate Defaults

If no selected models are passed, the default candidate set is:

| Model | File | Notes |
|---|---|---|
| `DLinear` | `src/models/linear_sequence_model.py` | Lightweight sequence baseline |
| `NLinear` | `src/models/linear_sequence_model.py` | Normalized lightweight sequence baseline |
| `XGBoost` | `src/models/xgboost_model.py` | Tree boosting with Optuna HPO |
| `LSTM` | `src/models/lstm_model.py` | Attention LSTM, Keras/TensorFlow |

`AttentionLSTM v2` is intentionally not in the default candidate set.

`TFT` is not an active source-code model. `src/models/tft_v2/` is absent and
the registry has no `TFT` `ModelSpec`; any remaining TFT names in old
`outputs/` directories are historical run artifacts.

## Full Candidate Set

`CANDIDATE_MODELS` includes:

- `Prophet`
- `ARIMA`
- `Ridge Return`
- `ElasticNet Return`
- `LightGBM Return`
- `DLinear`
- `NLinear`
- `XGBoost`
- `Random Forest`
- `LSTM`
- `LSTM Lite`
- `AttentionLSTM v2`
- `Prophet-ML/DL Hybrid`

Not all candidates are equally recommended for production. Some are legacy or
comparison-oriented.

## Model Families

### Statistical and Legacy

| Model | File | Status |
|---|---|---|
| `Prophet` | `src/models/prophet_model.py` | Legacy/comparison; optional dependency |
| `ARIMA` | `src/models/arima_model.py` | Legacy/comparison; mostly y-only |
| `Prophet-ML/DL Hybrid` | `src/models/prophet_hybrid_model.py` | Candidate; Prophet trend combined with ML/DL base models |

`ModelTrainer` treats Prophet and ARIMA as legacy due to walk-forward support and
literature/practical limitations documented in code comments.

`Prophet-ML/DL Hybrid` combines Prophet's trend modeling with base model return predictions. It supports two modes:
- `trend_gate`: Uses Prophet's trend slope as a binary filter (flats base prediction if slope <= 0).
- `residual_decomp`: Decomposes scaled prices using Prophet's trend, trains the base model on residual returns, and adds base residual predictions to the trend's returns.

### Linear Return Models

| Model | File | Notes |
|---|---|---|
| `Ridge Return` | `src/models/linear_model.py` | L2-regularized return model |
| `ElasticNet Return` | `src/models/linear_model.py` | L1/L2 sparse return model |

### Tree and Boosting Models

| Model | File | Notes |
|---|---|---|
| `XGBoost` | `src/models/xgboost_model.py` | Main tabular nonlinear model |
| `Random Forest` | `src/models/random_forest_model.py` | Optional/comparison tree ensemble |
| `LightGBM Return` | `src/models/gradient_boosting_model.py` | Optional modern boosting baseline |

### Deep Sequence Models

| Model | File | Notes |
|---|---|---|
| `LSTM` | `src/models/lstm_model.py` | Attention LSTM sequence model |
| `LSTM Lite` | `src/models/lstm_lite_model.py` | Small unidirectional LSTM; selected-only candidate |
| `AttentionLSTM v2` | `src/models/attention_lstm_v2_model.py` | Regularized two-layer BiLSTM with temporal attention XAI export; selected-only candidate |

Deep models require enough sequence samples. `ModelTrainer` skips them when fold
or split sequence counts fall below configured minimums.

`LSTM Lite` is a selected-only candidate (`default_candidate=False`) added to
test whether a smaller deep sequence model behaves more realistically on
single-symbol BIST data. It uses a one-layer unidirectional LSTM, dropout, a
small dense head, Adam with gradient clipping, and Huber loss. Its initial
minimum sequence threshold is separate from the generic deep-learning threshold:
`lstm_lite_min_sequence_samples=252`.

The model can run with optional train-only Optuna HPO through the
`deep_learning.lstm_lite` config section. HPO searches units, dense units,
dropout, learning rate, and batch size, and does not use final-holdout data.

`AttentionLSTM v2` is a separate opt-in candidate so the existing `LSTM` and
`LSTM Lite` contracts stay unchanged. It uses a smaller regularized
bidirectional LSTM stack (`64 -> 32` by default), Huber loss, gradient clipping,
dropout, chronological validation, ReduceLROnPlateau, optional train-slice HPO,
and `TemporalAttentionV2` weights. Evaluation exports attention-based XAI rows
to `xai/csv/xai_top_reasons_attention_*.csv` when the model supports the
exporter. Its initial minimum sequence threshold is
`attention_lstm_v2_min_sequence_samples=252`.

`AttentionLSTM v2` and `LSTM Lite` support enhanced regularization options to improve deep learning model quality on small datasets:
- **Cell Type**: Supports both standard `LSTM` and lightweight `GRU` cells to prevent overfitting.
- **L2 Regularization**: Adds `kernel_regularizer` and `recurrent_regularizer` options using `l2_rate`.
- **AdamW Optimizer**: Uses Adam with Weight Decay (`AdamW`) for better generalization behavior.
- **Expanded Optuna HPO**: Automates searching for `cell_type`, `l2_rate`, and `optimizer_type` when `tune_on_fit=True`.

#### LSTM Audit Notes

Current LSTM behavior is structurally correct on the main leakage boundaries:
sequence tensors use `X[t] -> y[t+1]`, scalers are fit on train slices only, and
walk-forward/final-holdout prediction alignment uses the trailing common length.
No clear off-by-one or direct train/test scaler leak was found in the inspected
pipeline.

The main weakness is statistical fit for the available BIST single-symbol data.
The implemented model is a two-layer bidirectional LSTM with attention
(`128 -> 64` units plus dense head), while recent walk-forward folds use roughly
`504-756` train rows before a `30` day sequence window and only `21` test rows
per fold. With 53 input features and feature pruning disabled in observed runs,
the LSTM has far more capacity than the fold-level sample size justifies.

Observed LSTM runs also show trading-output fragility: walk-forward backtests
typically place `0-2` trades and often fail buy-hold comparison or signal
diagnostics. Final-holdout wins can be misleading when the model simply avoids
trading during a negative buy-hold period. Treat LSTM as a research/conditional
candidate until it passes rolling holdout, trade-count, directional accuracy,
and stability requirements against naive, linear, and boosting baselines.

Recommended improvement path before promoting LSTM:

- Raise the minimum sequence requirement for LSTM beyond the generic
  `min_sequence_samples=64`; this threshold only prevents crashes, not
  overfitting.
- Shrink the default architecture for single-symbol daily data before trying
  larger networks: one unidirectional LSTM/GRU layer, smaller units
  (`16-64`), optional L2/recurrent regularization, and a smaller dense head.
- Add train-only HPO for LSTM-specific settings: `time_steps`, units, layers,
  dropout, learning rate, batch size, and patience.
- Prefer stationary/relative sequence inputs or enable feature pruning before
  feeding long feature histories to the network.
- Evaluate with trading-aware and directional metrics, not MSE alone, because
  scaled log-return MSE can look acceptable while signal quality remains poor.
- Consider panel or cross-symbol training only after the single-symbol
  evaluation protocol is stable; deep sequence models need broader data to have
  a fair chance against regularized tabular models.

Implemented first response to this audit:

- Keep existing `LSTM` unchanged and still default-selectable.
- Add `LSTM Lite` as a selected-only candidate for controlled comparison.
- Expose `LSTM Lite` in the interactive CLI manual model list and deep-learning
  preset so it can be selected from the desktop/manual workflow.
- Treat success as multi-metric: error, direction, Sharpe/excess return,
  buy-hold comparison, trade count, and signal diagnosis must be inspected
  together.

### Experimental Sequence Baselines

| Model | File | Notes |
|---|---|---|
| `DLinear` | `src/models/linear_sequence_model.py` | Lightweight linear sequence baseline |
| `NLinear` | `src/models/linear_sequence_model.py` | Last-value normalized sequence baseline |

## Ensemble

`src/models/ensemble.py` provides `EnsembleModel`. Evaluation can create ensemble
predictions when `ModelConfig.ensemble_enabled` is true.

Runtime behavior:

- Ensembles are not trained as independent model objects. They are synthesized
  after base-model predictions exist.
- `PredictionService` filters out existing `Ensemble *` rows, keeps only non-empty
  base predictions, and if `dataset_metadata.candidate_models` is populated it
  restricts ensemble members to those production candidates.
- Ensemble creation is skipped when fewer than two base models are available or
  when aligned truth arrays are missing.
- All component prediction arrays are aligned by taking the last `min_len`
  observations before weighted averaging.

Single-split and walk-forward evaluation both add the same reportable ensemble
families:

- `Ensemble Equal Weight`: equal model weights.
- `Ensemble Inverse RMSE`: weights proportional to inverse price-space RMSE.
- `Ensemble Sharpe-Weighted`: weights from directional target-space PnL Sharpe,
  with non-positive scores excluded and equal fallback when all scores fail.
- `Ensemble Risk-Parity`: weights proportional to inverse volatility of
  directional target-space PnL.
- `Ensemble Hierarchical`: equal weight by model category, then equal weight
  inside each category.
- `Ensemble Meta-Stacker`: Ridge regression over target-space base predictions;
  negative coefficients are clipped by default and weights are normalized.
- `Ensemble Cash-Gated`: starts from the Sharpe-weighted target prediction and
  zeroes low-consensus signals using a directional-agreement gate of `0.6`.
- `Ensemble Seq-Attention Equal`: equal model weights over sequence models only (`LSTM`, `LSTM Lite`, `AttentionLSTM v2`).
- `Ensemble Seq-Attention Inverse RMSE`: weights proportional to inverse price-space RMSE over sequence models only.

Important production rule:

- Ensembles can improve reports and backtests. Only `Ensemble Inverse RMSE`,
  `Ensemble Cash-Gated`, and `Ensemble Seq-Attention Inverse RMSE` are eligible as
  production ensemble leaders. Other ensemble variants remain report-only.
- A production ensemble forecast is generated by loading member artifacts,
  recursively producing member horizons, then combining bounded member prices
  by stored weights. Cash-gated ensembles neutralize a horizon when directional
  agreement is below `0.6`.
- Baseline leaders are still replaced by the best trainable experiment before
  forward forecasting.

## Training Paths

`ModelTrainer` has three major paths:

- `train_single_split()`
- `train_walk_forward()`
- `train_final_holdout_model()`

Factory methods instantiate model families consistently for single, walk-forward,
and final stages.

## XAI Model-Family Policy

Phase 4 keeps `XAIExplainer` as the public facade and routes explanations by
model family:

- Tree models (`XGBoost`, `Random Forest`, `LightGBM Return`) use
  `shap.TreeExplainer` first, with permutation fallback.
- Linear tabular models (`Ridge Return`, `ElasticNet Return`) use
  `shap.LinearExplainer` or coefficient-based contributions, with permutation
  fallback.
- Sequence models (`LSTM`, `DLinear`, `NLinear`) keep sequence permutation as
  the default explanation method.
- LIME is optional and used for local explanation helpers when the `lime`
  package is importable; missing LIME must not break XAI report generation.

## Related Pages

- [Validation and Backtesting](validation-and-backtesting.md)
- [Persistence and API](persistence-and-api.md)
- [Testing and Quality](testing-and-quality.md)
