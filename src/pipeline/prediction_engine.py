# -*- coding: utf-8 -*-
"""
prediction_engine.py - Tahmin uretimi ve ensemble koordinasyonu (Faz 2.1 Mixin).

Sorumluluklar:
  - generate_predictions(): tek split tahmin uretimi
  - _predict_single_model(): model tipine gore ham tahmin
  - _target_to_price(): log_return / return / price inverse transform
  - Ensemble tahmin birlestirme (equal weight, inverse RMSE)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.models.ensemble import EnsembleModel
from src.evaluation.evaluator import compute_metrics, plot_comparison
from src.utils.reporting_utils import route_output_path
try:
    from src.data.preprocessor import reconstruct_prices_from_logret, reconstruct_prices_from_return
except ImportError:  # pragma: no cover
    def reconstruct_prices_from_logret(log_returns: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        return np.asarray(prev_close, dtype=float).ravel() * np.exp(np.asarray(log_returns, dtype=float).ravel())

    def reconstruct_prices_from_return(returns: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        return np.asarray(prev_close, dtype=float).ravel() * (1.0 + np.asarray(returns, dtype=float).ravel())


_SEQ_MODELS = {"LSTM", "LSTM Lite", "AttentionLSTM", "DLinear", "NLinear"}
_TREE_MODELS = {"XGBoost", "Random Forest", "Ridge Return", "ElasticNet Return", "LightGBM Return"}


class _PredictionEngineMixin:
    """Mixin: tahmin uretimi ve ensemble koordinasyonu."""

    # ------------------------------------------------------------------ #
    #  Inverse transform: target -> price                                  #
    # ------------------------------------------------------------------ #

    def _target_to_price(self, preds_target: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        target_mode = self.dataset_metadata.get("target_mode", "log_return")
        preds_target = np.asarray(preds_target).ravel()
        prev_close = np.asarray(prev_close).ravel()

        if target_mode == "log_return":
            return reconstruct_prices_from_logret(preds_target, prev_close)
        if target_mode == "return":
            return reconstruct_prices_from_return(preds_target, prev_close)
        if target_mode == "price":
            return preds_target
        raise ValueError(f"Desteklenmeyen target_mode: {target_mode}")

    # ------------------------------------------------------------------ #
    #  Ensemble helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _weighted_average(predictions: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        names = list(predictions)
        arrays = [np.asarray(predictions[name], dtype=float).ravel() for name in names]
        min_len = min(len(arr) for arr in arrays)
        stacked = np.stack([arr[-min_len:] for arr in arrays], axis=0)
        weight_array = np.asarray([weights.get(name, 0.0) for name in names], dtype=float)
        if weight_array.sum() <= 0:
            weight_array = np.ones(len(names), dtype=float) / len(names)
        else:
            weight_array = weight_array / weight_array.sum()
        return np.average(stacked, axis=0, weights=weight_array)

    @staticmethod
    def _resolve_categories(names) -> Dict[str, str]:
        """Registry'den her modelin kategori adını çek; eksik → 'unknown'."""
        try:
            from src.pipeline.model_registry import ensure_loaded, has_spec, get_spec
            ensure_loaded()
        except Exception:
            return {n: "unknown" for n in names}
        out: Dict[str, str] = {}
        for n in names:
            try:
                out[n] = get_spec(n).category if has_spec(n) else "unknown"
            except Exception:
                out[n] = "unknown"
        return out

    @staticmethod
    def _base_predictions_for_ensemble(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            name: np.asarray(preds, dtype=float).ravel()
            for name, preds in predictions.items()
            if not name.startswith("Ensemble ") and len(np.asarray(preds).ravel()) > 0
        }

    def _add_single_split_ensembles(self) -> None:
        if not self.ensemble_enabled:
            return
        metadata = getattr(self, "dataset_metadata", {}) or {}
        candidates = set(metadata.get("candidate_models") or [])
        base_preds = {
            name: preds
            for name, preds in self._base_predictions_for_ensemble(self.predictions).items()
            if not candidates or name in candidates
        }
        if len(base_preds) < 2 or self.y_true_aligned is None:
            return

        equal_name = "Ensemble Equal Weight"
        inv_name = "Ensemble Inverse RMSE"
        sharpe_name = "Ensemble Sharpe-Weighted"
        rp_name = "Ensemble Risk-Parity"
        hier_name = "Ensemble Hierarchical"
        stk_name = "Ensemble Meta-Stacker"
        cg_name = "Ensemble Cash-Gated"
        equal_preds = EnsembleModel().combine(base_preds)
        inverse_weights = EnsembleModel.optimize_inverse_rmse(np.asarray(self.y_true_aligned), base_preds)
        inverse_preds = EnsembleModel(inverse_weights).combine(base_preds)
        # Faz 5 Katman 1: Sharpe-weighted blend.
        base_targets = {name: self.prediction_targets[name] for name in base_preds if name in self.prediction_targets}
        y_true_target = np.asarray(getattr(self, "y_true_target_aligned", []), dtype=float).ravel()
        sharpe_weights = (
            EnsembleModel.optimize_by_sharpe(y_true_target, base_targets)
            if len(y_true_target) and base_targets
            else {name: 1.0 / len(base_preds) for name in base_preds}
        )
        sharpe_preds = EnsembleModel(sharpe_weights).combine(base_preds)
        # Faz 5 Katman 2: Risk-parity (inverse-volatility) blend.
        if len(y_true_target) and base_targets:
            pnl_vols = EnsembleModel.compute_pnl_volatilities(y_true_target, base_targets)
            rp_weights = EnsembleModel.optimize_by_risk_parity(pnl_vols)
        else:
            rp_weights = {name: 1.0 / len(base_preds) for name in base_preds}
        rp_preds = EnsembleModel(rp_weights).combine(base_preds)
        # Faz 5 Katman 3: Kategori-gated hierarchical blend.
        categories = self._resolve_categories(list(base_preds))
        hier_weights = EnsembleModel.optimize_hierarchical_by_category(base_preds, categories)
        hier_preds = EnsembleModel(hier_weights).combine(base_preds)
        # Faz 5 Katman 4: Ridge meta-stacker (target-space).
        if len(y_true_target) and base_targets:
            stk_weights = EnsembleModel.optimize_by_ridge_stacker(y_true_target, base_targets, alpha=1.0)
        else:
            stk_weights = {name: 1.0 / len(base_preds) for name in base_preds}
        stk_preds = EnsembleModel(stk_weights).combine(base_preds)
        self.ensemble_weights[equal_name] = {name: round(1.0 / len(base_preds), 6) for name in base_preds}
        self.ensemble_weights[inv_name] = inverse_weights
        self.ensemble_weights[sharpe_name] = sharpe_weights
        self.ensemble_weights[rp_name] = rp_weights
        self.ensemble_weights[hier_name] = hier_weights
        self.ensemble_weights[stk_name] = stk_weights
        # Cash-Gated, Sharpe-Weighted ağırlıklarını miras alır (gate transformasyondur, blend değil).
        self.ensemble_weights[cg_name] = dict(sharpe_weights)

        equal_target = EnsembleModel().combine(base_targets) if len(base_targets) >= 2 else None
        inverse_target = self._weighted_average(base_targets, inverse_weights) if len(base_targets) >= 2 else None
        sharpe_target = self._weighted_average(base_targets, sharpe_weights) if len(base_targets) >= 2 else None
        rp_target = self._weighted_average(base_targets, rp_weights) if len(base_targets) >= 2 else None
        hier_target = self._weighted_average(base_targets, hier_weights) if len(base_targets) >= 2 else None
        stk_target = self._weighted_average(base_targets, stk_weights) if len(base_targets) >= 2 else None
        # Faz 5 Katman 5: Cash gate (Sharpe-Weighted hedef üzerinde, agreement >= 0.6).
        cg_target = None
        cg_preds = sharpe_preds
        if sharpe_target is not None and base_targets:
            try:
                cg_target = EnsembleModel.apply_cash_gate(
                    sharpe_target, base_targets, magnitude_threshold=0.0, agreement_threshold=0.6
                )
                cg_preds = self._target_to_price(cg_target, self.prev_close_aligned[-len(cg_target):])
            except AttributeError:
                cg_target = None
                cg_preds = sharpe_preds

        for name, pred_price, pred_target in [
            (equal_name, equal_preds, equal_target),
            (inv_name, inverse_preds, inverse_target),
            (sharpe_name, sharpe_preds, sharpe_target),
            (rp_name, rp_preds, rp_target),
            (hier_name, hier_preds, hier_target),
            (stk_name, stk_preds, stk_target),
            (cg_name, cg_preds, cg_target),
        ]:
            k = min(len(pred_price), len(self.y_true_aligned), len(self.prev_close_aligned))
            self.predictions[name] = np.asarray(pred_price)[-k:]
            if pred_target is not None:
                self.prediction_targets[name] = np.asarray(pred_target)[-k:]
            template = next(iter(self.single_backtest_inputs.values()), None)
            if template:
                payload = {}
                for key, value in template.items():
                    arr = np.asarray(value)
                    payload[key] = arr[-k:] if arr.ndim > 0 and len(arr) >= k else value
                payload["pred_price"] = self.predictions[name]
                payload["pred_target"] = self.prediction_targets.get(name)
                self.single_backtest_inputs[name] = payload
        print("  [OK] Ensemble tahminleri eklendi: Equal Weight, Inverse RMSE, Sharpe-Weighted, Risk-Parity, Hierarchical, Meta-Stacker, Cash-Gated")

    def _add_walk_forward_ensembles(
        self,
        wf_results: Dict[str, Dict[str, Any]],
        wf_predictions: Dict[str, np.ndarray],
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not self.ensemble_enabled or wf_y_true is None:
            return
        metadata = getattr(self, "dataset_metadata", {}) or {}
        candidates = set(metadata.get("candidate_models") or [])
        base_preds = {
            name: preds
            for name, preds in self._base_predictions_for_ensemble(wf_predictions).items()
            if not candidates or name in candidates
        }
        if len(base_preds) < 2:
            return

        equal_name = "Ensemble Equal Weight"
        inv_name = "Ensemble Inverse RMSE"
        sharpe_name = "Ensemble Sharpe-Weighted"
        rp_name = "Ensemble Risk-Parity"
        hier_name = "Ensemble Hierarchical"
        stk_name = "Ensemble Meta-Stacker"
        cg_name = "Ensemble Cash-Gated"
        equal_preds = EnsembleModel().combine(base_preds)
        inverse_weights = EnsembleModel.optimize_inverse_rmse(np.asarray(wf_y_true), base_preds)
        inverse_preds = EnsembleModel(inverse_weights).combine(base_preds)

        bt_inputs = wf_backtest_inputs or {}
        template = next(iter(bt_inputs.values()), None)
        base_targets = {
            name: np.asarray(bt_inputs[name]["pred_target"], dtype=float).ravel()
            for name in base_preds
            if name in bt_inputs and "pred_target" in bt_inputs[name]
        }
        # Faz 5 Katman 1: Sharpe-weighted blend (target-space).
        y_true_target = (
            np.asarray(template.get("y_true_target", []), dtype=float).ravel()
            if template else np.asarray([])
        )
        sharpe_weights = (
            EnsembleModel.optimize_by_sharpe(y_true_target, base_targets)
            if len(y_true_target) and base_targets
            else {name: 1.0 / len(base_preds) for name in base_preds}
        )
        sharpe_preds = EnsembleModel(sharpe_weights).combine(base_preds)
        # Faz 5 Katman 2: Risk-parity blend (target-space PnL).
        if len(y_true_target) and base_targets:
            pnl_vols = EnsembleModel.compute_pnl_volatilities(y_true_target, base_targets)
            rp_weights = EnsembleModel.optimize_by_risk_parity(pnl_vols)
        else:
            rp_weights = {name: 1.0 / len(base_preds) for name in base_preds}
        rp_preds = EnsembleModel(rp_weights).combine(base_preds)
        # Faz 5 Katman 3: Kategori-gated hierarchical blend.
        categories = self._resolve_categories(list(base_preds))
        hier_weights = EnsembleModel.optimize_hierarchical_by_category(base_preds, categories)
        hier_preds = EnsembleModel(hier_weights).combine(base_preds)
        # Faz 5 Katman 4: Ridge meta-stacker (OOF — WF y_true zaten fold birleşimi).
        if len(y_true_target) and base_targets:
            stk_weights = EnsembleModel.optimize_by_ridge_stacker(y_true_target, base_targets, alpha=1.0)
        else:
            stk_weights = {name: 1.0 / len(base_preds) for name in base_preds}
        stk_preds = EnsembleModel(stk_weights).combine(base_preds)
        self.ensemble_weights[equal_name] = {name: round(1.0 / len(base_preds), 6) for name in base_preds}
        self.ensemble_weights[inv_name] = inverse_weights
        self.ensemble_weights[sharpe_name] = sharpe_weights
        self.ensemble_weights[rp_name] = rp_weights
        self.ensemble_weights[hier_name] = hier_weights
        self.ensemble_weights[stk_name] = stk_weights
        self.ensemble_weights[cg_name] = dict(sharpe_weights)

        equal_target = EnsembleModel().combine(base_targets) if len(base_targets) >= 2 else None
        inverse_target = self._weighted_average(base_targets, inverse_weights) if len(base_targets) >= 2 else None
        sharpe_target = self._weighted_average(base_targets, sharpe_weights) if len(base_targets) >= 2 else None
        rp_target = self._weighted_average(base_targets, rp_weights) if len(base_targets) >= 2 else None
        hier_target = self._weighted_average(base_targets, hier_weights) if len(base_targets) >= 2 else None
        stk_target = self._weighted_average(base_targets, stk_weights) if len(base_targets) >= 2 else None
        # Faz 5 Katman 5: Cash gate (Sharpe-Weighted hedef üzerinde).
        cg_target = None
        cg_preds = sharpe_preds
        if sharpe_target is not None and base_targets:
            try:
                cg_target = EnsembleModel.apply_cash_gate(
                    sharpe_target, base_targets, magnitude_threshold=0.0, agreement_threshold=0.6
                )
                prev_close_tpl = (
                    np.asarray(template.get("prev_close", []), dtype=float).ravel()
                    if template else np.array([])
                )
                if len(prev_close_tpl) >= len(cg_target):
                    cg_preds = self._target_to_price(cg_target, prev_close_tpl[-len(cg_target):])
                else:
                    cg_target = None
                    cg_preds = sharpe_preds
            except AttributeError:
                cg_target = None
                cg_preds = sharpe_preds

        for name, pred_price, pred_target in [
            (equal_name, equal_preds, equal_target),
            (inv_name, inverse_preds, inverse_target),
            (sharpe_name, sharpe_preds, sharpe_target),
            (rp_name, rp_preds, rp_target),
            (hier_name, hier_preds, hier_target),
            (stk_name, stk_preds, stk_target),
            (cg_name, cg_preds, cg_target),
        ]:
            k = min(len(pred_price), len(np.asarray(wf_y_true).ravel()))
            wf_predictions[name] = np.asarray(pred_price)[-k:]
            y_true_price = np.asarray(wf_y_true).ravel()[-k:]
            if template:
                y_true_target = np.asarray(template.get("y_true_target", []), dtype=float).ravel()[-k:]
                prev_close = np.asarray(template.get("prev_close", []), dtype=float).ravel()[-k:]
            else:
                y_true_target = None
                prev_close = None
            wf_results[name] = compute_metrics(
                y_true_price,
                wf_predictions[name],
                y_true_target=y_true_target,
                y_pred_target=np.asarray(pred_target)[-k:] if pred_target is not None else None,
                prev_close=prev_close,
                target_mode=self.dataset_metadata.get("target_mode", "log_return"),
            )
            if template:
                payload = {}
                for key, value in template.items():
                    arr = np.asarray(value)
                    payload[key] = arr[-k:] if arr.ndim > 0 and len(arr) >= k else value
                payload["y_true_price"] = y_true_price
                payload["pred_price"] = wf_predictions[name]
                payload["pred_target"] = np.asarray(pred_target)[-k:] if pred_target is not None else None
                bt_inputs[name] = payload
        print("  [OK] Walk-forward ensemble tahminleri eklendi: Equal, Inverse RMSE, Sharpe-Weighted, Risk-Parity, Hierarchical, Meta-Stacker, Cash-Gated.")

    # ------------------------------------------------------------------ #
    #  Ensemble directional agreement (Adim 2.2)                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_ensemble_direction_agreement(
        component_predictions: Dict[str, np.ndarray],
        main_model_name: str,
    ) -> Optional[float]:
        main_preds = component_predictions.get(main_model_name)
        if main_preds is None or len(main_preds) == 0:
            return None
        main_direction = float(np.sign(main_preds[-1]))
        if main_direction == 0:
            return None
        others = [
            v for k, v in component_predictions.items()
            if k != main_model_name and not k.startswith("Ensemble ") and len(v) > 0
        ]
        if not others:
            return None
        agreements = [float(np.sign(v[-1])) == main_direction for v in others]
        return float(sum(agreements)) / len(agreements)

    # ------------------------------------------------------------------ #
    #  Plot helper                                                         #
    # ------------------------------------------------------------------ #

    def _save_selected_models_plot(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        save_path: str,
        title: str,
    ) -> None:
        if not self.selected_models:
            return
        selected_predictions = {name: preds for name, preds in predictions.items() if name in self.selected_models}
        if not selected_predictions:
            return
        plot_comparison(y_true, selected_predictions, save_path=save_path, title=title)
        print(f"[OK] Secilen modeller grafigi kaydedildi -> {route_output_path(save_path)}")

    # ------------------------------------------------------------------ #
    #  Single model prediction                                            #
    # ------------------------------------------------------------------ #

    def _predict_single_model(
        self,
        model_name: str,
        model: Any,
        tensors: dict,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ]:
        quantile_target = None

        prev_close_test = np.asarray(tensors["prev_close_test"]).ravel()
        dates_test = np.asarray(tensors["dates_test"])
        prediction_dates_test = np.asarray(tensors.get("dates_prediction", tensors["dates_test"]))
        market_regime_test = np.asarray(tensors.get("market_regime_test", np.zeros(len(prev_close_test))), dtype=float).ravel()
        y_test_price = np.asarray(tensors["original_y_test_aligned"]).ravel()
        y_test_target = np.asarray(tensors["y_test"]).ravel()

        if model_name == "Prophet":
            preds_target = model.predict(tensors["X_test"], dates_test=tensors["dates_test"])
            self.dataset_metadata["prophet_regressors_used"] = getattr(model, "regressors_used", [])
        elif model_name in _TREE_MODELS:
            preds_scaled = model.predict(tensors["X_test_s"])
            preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
        elif model_name in _SEQ_MODELS:
            if hasattr(model, "predict_quantiles"):
                quantile_scaled = np.asarray(model.predict_quantiles(tensors["X_test_seq"]))
                quantile_target = np.column_stack([
                    tensors["scaler_y"].inverse_transform(quantile_scaled[:, idx].reshape(-1, 1)).ravel()
                    for idx in range(quantile_scaled.shape[1])
                ])
                preds_target = quantile_target[:, quantile_scaled.shape[1] // 2]
            else:
                preds_scaled = model.predict(tensors["X_test_seq"])
                preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
        else:
            preds_target = model.predict(tensors["X_test"])

        preds_target = np.asarray(preds_target).ravel()
        k = min(
            len(preds_target),
            len(prev_close_test),
            len(y_test_price),
            len(y_test_target),
            len(dates_test),
            len(prediction_dates_test),
            len(market_regime_test),
        )
        preds_target = preds_target[-k:]
        prev_close_aligned = prev_close_test[-k:]
        market_regime_aligned = market_regime_test[-k:]
        pred_price = self._target_to_price(preds_target, prev_close_aligned)
        quantile_price = None
        if quantile_target is not None:
            quantile_target = quantile_target[-k:]
            quantile_price = np.column_stack([
                self._target_to_price(quantile_target[:, idx], prev_close_aligned)
                for idx in range(quantile_target.shape[1])
            ])
        return (
            pred_price,
            preds_target,
            y_test_price[-k:],
            y_test_target[-k:],
            prev_close_aligned,
            dates_test[-k:],
            prediction_dates_test[-k:],
            market_regime_aligned,
            quantile_price,
        )

    # ------------------------------------------------------------------ #
    #  Batch prediction (single split)                                    #
    # ------------------------------------------------------------------ #

    def generate_predictions(self, trained_models: dict, tensors: dict):
        print("\n" + "=" * 60)
        print("  ADIM 5 | Tahmin Uretimi ve Inverse Transform (EvaluationManager)")
        print("=" * 60)

        prev_close_test = np.asarray(tensors["prev_close_test"]).ravel()
        dates_test = np.asarray(tensors["dates_test"])
        prediction_dates_test = np.asarray(tensors.get("dates_prediction", tensors["dates_test"]))
        y_test_price = np.asarray(tensors["original_y_test_aligned"]).ravel()
        y_test_target = np.asarray(tensors["y_test"]).ravel()

        raw_preds: Dict[str, np.ndarray] = {}
        raw_pred_targets: Dict[str, np.ndarray] = {}
        raw_quantiles: Dict[str, np.ndarray] = {}
        self.latest_tensors = tensors

        for name, model in trained_models.items():
            try:
                if name == "Prophet":
                    preds_target = model.predict(tensors["X_test"], dates_test=tensors["dates_test"])
                elif name in _TREE_MODELS:
                    preds_scaled = model.predict(tensors["X_test_s"])
                    preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
                elif name in _SEQ_MODELS:
                    if hasattr(model, "predict_quantiles"):
                        quantile_scaled = model.predict_quantiles(tensors["X_test_seq"])
                        quantile_target = np.column_stack([
                            tensors["scaler_y"].inverse_transform(quantile_scaled[:, idx].reshape(-1, 1)).ravel()
                            for idx in range(quantile_scaled.shape[1])
                        ])
                        preds_target = quantile_target[:, quantile_scaled.shape[1] // 2]
                        raw_quantiles[name] = quantile_target
                    else:
                        preds_scaled = model.predict(tensors["X_test_seq"])
                        preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
                else:
                    try:
                        preds_scaled = model.predict(tensors["X_test_seq"])
                    except Exception:
                        preds_scaled = model.predict(tensors["X_test_s"])
                    preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()

                preds_target = np.asarray(preds_target).ravel()
                k = min(
                    len(preds_target),
                    len(prev_close_test),
                    len(y_test_price),
                    len(y_test_target),
                    len(dates_test),
                    len(prediction_dates_test),
                )
                preds_target = preds_target[-k:]
                prev_close_aligned = prev_close_test[-k:]
                y_true_price_aligned = y_test_price[-k:]
                y_true_target_aligned = y_test_target[-k:]
                dates_aligned = dates_test[-k:]
                prediction_dates_aligned = prediction_dates_test[-k:]
                market_regime_aligned = np.asarray(tensors.get("market_regime_test", np.zeros(k)), dtype=float).ravel()[-k:]

                raw_pred_targets[name] = preds_target
                raw_preds[name] = self._target_to_price(preds_target, prev_close_aligned)

                if name in raw_quantiles:
                    aligned_quantiles = raw_quantiles[name][-k:]
                    raw_quantiles[name] = np.column_stack([
                        self._target_to_price(aligned_quantiles[:, idx], prev_close_aligned)
                        for idx in range(aligned_quantiles.shape[1])
                    ])

                self.single_backtest_inputs[name] = {
                    "dates": dates_aligned,
                    "prediction_dates": prediction_dates_aligned,
                    "market_regime": market_regime_aligned,
                    "y_true_price": y_true_price_aligned,
                    "pred_price": raw_preds[name],
                    "prev_close": prev_close_aligned,
                    "pred_target": preds_target,
                    "y_true_target": y_true_target_aligned,
                }
                if name == "Prophet":
                    self.dataset_metadata["prophet_regressors_used"] = getattr(model, "regressors_used", [])
                print(f"  [OK] {name} tahmini uretildi - {len(raw_preds[name])} adim")
            except Exception as exc:
                print(f"  [WARN] {name} tahmini basarisiz, atlaniyor: {exc}")

        if not raw_preds:
            raise RuntimeError("Hicbir model tahmin uretemedi. Egitim adimini kontrol edin.")

        min_len = min(len(v) for v in raw_preds.values())
        self.predictions = {name: preds[-min_len:] for name, preds in raw_preds.items()}
        self.prediction_targets = {name: preds[-min_len:] for name, preds in raw_pred_targets.items()}
        self.quantile_predictions = {name: preds[-min_len:] for name, preds in raw_quantiles.items()}
        self.y_true_aligned = y_test_price[-min_len:]
        self.y_true_target_aligned = y_test_target[-min_len:]
        self.prev_close_aligned = prev_close_test[-min_len:]
        self._add_single_split_ensembles()
