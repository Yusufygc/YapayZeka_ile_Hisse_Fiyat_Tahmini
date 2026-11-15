# -*- coding: utf-8 -*-
"""
prediction_engine.py - Tahmin uretimi ve ensemble koordinasyonu (Faz 2.1 Mixin).

Sorumluluklar:
  - generate_predictions(): tek split tahmin uretimi
  - _predict_single_model(): model tipine gore ham tahmin
  - _target_to_price(): log_return / return / price inverse transform
  - Ensemble tahmin birlestirme (tek varyant: Inverse RMSE)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from src.models.ensemble import EnsembleModel
from src.evaluation.evaluator import compute_metrics

try:
    from src.data.preprocessor import reconstruct_prices_from_logret, reconstruct_prices_from_return
except ImportError:  # pragma: no cover

    def reconstruct_prices_from_logret(
        log_returns: np.ndarray, prev_close: np.ndarray
    ) -> np.ndarray:
        return np.asarray(prev_close, dtype=float).ravel() * np.exp(
            np.asarray(log_returns, dtype=float).ravel()
        )

    def reconstruct_prices_from_return(returns: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        return np.asarray(prev_close, dtype=float).ravel() * (
            1.0 + np.asarray(returns, dtype=float).ravel()
        )


logger = logging.getLogger(__name__)

_TREE_MODELS = {"XGBoost", "Ridge Return", "LightGBM Return"}


class _PredictionEngineMixin:
    """Mixin: tahmin uretimi ve ensemble koordinasyonu.

    Faz 3 (E1 owner-forward epigi): owner-forward kaldirildi. READ-ONLY config/
    identity ``self.ctx`` (EvaluationContext), mutable runtime cikti ``self.state``
    (EvaluationState) uzerinden okunur/yazilir. Servis ctor'u bu iki nesneyi
    enjekte eder (bkz. PredictionService.__init__).
    """

    # ------------------------------------------------------------------ #
    #  Inverse transform: target -> price                                  #
    # ------------------------------------------------------------------ #

    def _target_to_price(self, preds_target: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        target_mode = self.ctx.dataset_metadata.get("target_mode", "log_return")
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
    def _weighted_average(
        predictions: Dict[str, np.ndarray], weights: Dict[str, float]
    ) -> np.ndarray:
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
    def _slice_template_payload(template: Dict[str, Any], k: int) -> Dict[str, Any]:
        """Backtest input template'ini son ``k`` örneğe dilimleyerek kopyalar.

        Dizi olmayan / k'dan kısa değerler olduğu gibi bırakılır. Single-split ve
        walk-forward ensemble tüketim döngülerinin paylaştığı yardımcı (DRY).
        """
        payload: Dict[str, Any] = {}
        for key, value in template.items():
            arr = np.asarray(value)
            payload[key] = arr[-k:] if arr.ndim > 0 and len(arr) >= k else value
        return payload

    @staticmethod
    def _base_predictions_for_ensemble(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            name: np.asarray(preds, dtype=float).ravel()
            for name, preds in predictions.items()
            if not name.startswith("Ensemble ") and len(np.asarray(preds).ravel()) > 0
        }

    # Tek ensemble varyanti: Inverse RMSE (single-split + walk-forward ortak).
    # Diger varyantlar (Equal/Sharpe/Risk-Parity/Hierarchical/Meta-Stacker/Cash-Gated)
    # model_scope.REMOVED_MODELS altindadir; yeni kosularda uretilmez.
    _ENSEMBLE_NAME = "Ensemble Inverse RMSE"
    _ENSEMBLE_NAMES = (_ENSEMBLE_NAME,)

    def _fit_ensemble_weights(
        self,
        base_preds: Dict[str, np.ndarray],
        *,
        y_true_inverse: Any,
    ) -> Dict[str, Dict[str, float]]:
        """Ensemble adi -> agirlik sozlugu (yalnizca verilen satirlarda fit).

        Sadece ``y_true_inverse`` satirlarini gorur: cagiran, fit penceresini (tum test
        seti = research-only single-split; yalnizca onceki fold'lar = walk-forward)
        secer. ``y_true_inverse`` bos/None ise (fit verisi yok, orn. ilk fold) esit
        agirliga duser.

        Args:
            base_preds: Temel model fiyat tahminleri.
            y_true_inverse: Inverse-RMSE icin gercek fiyat serisi.
        """
        n = len(base_preds)
        y_inv = np.asarray([] if y_true_inverse is None else y_true_inverse, dtype=float).ravel()
        if len(y_inv) > 0:
            inv_w = EnsembleModel.optimize_inverse_rmse(y_inv, base_preds)
        else:
            inv_w = {name: round(1.0 / n, 6) for name in base_preds}
        return {self._ENSEMBLE_NAME: inv_w}

    def _apply_ensemble_weights(
        self,
        weights: Dict[str, Dict[str, float]],
        base_preds: Dict[str, np.ndarray],
        base_targets: Dict[str, np.ndarray],
    ) -> list:
        """Hazir agirliklari uygulayip (name, price_preds, target_preds) listesi uretir.

        Y_true kullanmaz (leakage yok).
        """
        out = []
        for name, w in weights.items():
            tgt = self._weighted_average(base_targets, w) if len(base_targets) >= 2 else None
            out.append((name, EnsembleModel(w).combine(base_preds), tgt))
        return out

    def _compute_ensemble_blends(
        self,
        base_preds: Dict[str, np.ndarray],
        base_targets: Dict[str, np.ndarray],
        *,
        y_true_inverse: Any,
    ) -> list:
        """Single-split (research-only): agirliklar verilen y_true uzerinde fit edilir (in-sample).

        Agirliklar `self.state.ensemble_weights`'e yazilir. Walk-forward metrikleri BU yolu
        kullanmaz (bkz. `_compute_walk_forward_blends`).
        """
        weights = self._fit_ensemble_weights(base_preds, y_true_inverse=y_true_inverse)
        self.state.ensemble_weights.update(weights)
        return self._apply_ensemble_weights(weights, base_preds, base_targets)

    def _compute_walk_forward_blends(
        self,
        base_preds: Dict[str, np.ndarray],
        base_targets: Dict[str, np.ndarray],
        *,
        fold_ids: Any,
        y_true_inverse: Any,
    ) -> tuple:
        """Expanding / prior-folds-only ensemble (OOS metrikler icin leakage yok).

        Fold f satirlari, yalnizca f'den ONCE baslayan satirlarda fit edilen agirliklarla
        harmanlanir; ilk fold esit agirlik. Tum diziler `[-m:]` kuyruk hizasindadir.
        `state.ensemble_weights` (production forward) TUM fold'larda fit edilir; metrikte
        kullanilmaz. fold_ids yoksa/hizasizsa esit agirlik (asla in-sample) + uyari.

        Returns:
            (tuples, scope) — scope "expanding_prior_folds" veya "equal_weight_no_fold_ids".
        """
        arrs = [*base_preds.values(), *base_targets.values(), np.asarray(y_true_inverse).ravel()]
        m = min(len(a) for a in arrs)

        def tail(a):
            a = None if a is None else np.asarray(a).ravel()
            return a[-m:] if a is not None and len(a) >= m else None

        preds = {k: np.asarray(v, dtype=float).ravel()[-m:] for k, v in base_preds.items()}
        targets = {k: np.asarray(v, dtype=float).ravel()[-m:] for k, v in base_targets.items()}
        y_inv = tail(y_true_inverse)
        folds = tail(fold_ids)

        # Production forward agirliklari: tum OOS satirlari (metrikte KULLANILMAZ).
        self.state.ensemble_weights.update(self._fit_ensemble_weights(preds, y_true_inverse=y_inv))
        if folds is None:
            logger.warning(
                "Walk-forward ensemble: fold_ids yok/hizasiz -> metrikler esit agirlikla "
                "hesaplanir (in-sample agirlik kullanilmaz)."
            )
            weights = self._fit_ensemble_weights(preds, y_true_inverse=None)
            return (
                self._apply_ensemble_weights(weights, preds, targets),
                "equal_weight_no_fold_ids",
            )

        _, first_idx = np.unique(folds, return_index=True)
        out_price: Dict[str, np.ndarray] = {}
        out_target: Dict[str, Optional[np.ndarray]] = {}
        for start in np.sort(first_idx):
            mask = folds == folds[start]
            prior = slice(0, int(start))  # yalnizca bu fold'dan ONCEKI satirlar
            weights = self._fit_ensemble_weights(
                {k: v[prior] for k, v in preds.items()},
                y_true_inverse=y_inv[prior],
            )
            fold_tuples = self._apply_ensemble_weights(
                weights,
                {k: v[mask] for k, v in preds.items()},
                {k: v[mask] for k, v in targets.items()},
            )
            for name, price, tgt in fold_tuples:
                out_price.setdefault(name, np.full(m, np.nan))[mask] = price
                if tgt is None or out_target.get(name, 0) is None:
                    out_target[name] = None
                else:
                    out_target.setdefault(name, np.full(m, np.nan))[mask] = tgt
        return [(name, out_price[name], out_target[name]) for name in out_price], "expanding_prior_folds"

    def _add_single_split_ensembles(self) -> None:
        # Sprint 0 (2026-05-25): Bu metot ensemble agirliklarini test setinin
        # `y_true_aligned` degeri uzerinde optimize ediyor -> in-sample leakage.
        # Tek-bolunmeli mod artik research-only oldugu icin tolerans gosterilir,
        # ancak metadata'ya `ensemble_weight_scope: "in_sample_test_set"` flag'i
        # eklenir ve uretim leaderboard'una sizmasi engellenir.
        # Gercek train-tail validation slice fix'i Sprint 4 (probabilistic
        # forecasting + multi-horizon target) ile gelecek; o sprintte
        # TensorPreparationService train-tail slice ayirmasi yapilacak.
        if not self.ctx.ensemble_enabled:
            return
        metadata = self.ctx.dataset_metadata or {}
        candidates = set(metadata.get("candidate_models") or [])
        base_preds = {
            name: preds
            for name, preds in self._base_predictions_for_ensemble(self.state.predictions).items()
            if not candidates or name in candidates
        }
        if len(base_preds) < 2 or self.state.y_true_aligned is None:
            return

        base_targets = {
            name: self.state.prediction_targets[name]
            for name in base_preds
            if name in self.state.prediction_targets
        }
        ensemble_tuples = self._compute_ensemble_blends(
            base_preds,
            base_targets,
            y_true_inverse=self.state.y_true_aligned,
        )

        # Sprint 0 (2026-05-25): leakage flag metadata (tek ensemble: Inverse RMSE).
        # EvaluationState.ensemble_weight_scope her zaman dict (default_factory).
        for _name in self._ENSEMBLE_NAMES:
            self.state.ensemble_weight_scope[_name] = "in_sample_test_set_research_only"

        for name, pred_price, pred_target in ensemble_tuples:
            k = min(len(pred_price), len(self.state.y_true_aligned), len(self.state.prev_close_aligned))
            self.state.predictions[name] = np.asarray(pred_price)[-k:]
            if pred_target is not None:
                self.state.prediction_targets[name] = np.asarray(pred_target)[-k:]
            template = next(iter(self.state.single_backtest_inputs.values()), None)
            if template:
                payload = self._slice_template_payload(template, k)
                payload["pred_price"] = self.state.predictions[name]
                payload["pred_target"] = self.state.prediction_targets.get(name)
                self.state.single_backtest_inputs[name] = payload
        print(
            "  [OK] Ensemble tahminleri eklendi: Inverse RMSE"
        )

    @staticmethod
    def _shared_fold_ids(base_preds, bt_inputs) -> Optional[np.ndarray]:
        """Temel modellerin walk-forward ``fold_ids``'i (tail hizali); uyumsuzsa/yoksa None."""
        arrays = [
            np.asarray(bt_inputs[n]["fold_ids"]).ravel()
            for n in base_preds
            if n in bt_inputs and bt_inputs[n].get("fold_ids") is not None
        ]
        if not arrays:
            return None
        m = min(len(a) for a in arrays)
        if m == 0 or any(not np.array_equal(a[-m:], arrays[0][-m:]) for a in arrays[1:]):
            logger.warning("Walk-forward ensemble: temel modellerin fold_ids hizasi tutarsiz.")
            return None
        return arrays[0]

    def _add_walk_forward_ensembles(
        self,
        wf_results: Dict[str, Dict[str, Any]],
        wf_predictions: Dict[str, np.ndarray],
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not self.ctx.ensemble_enabled or wf_y_true is None:
            return
        metadata = self.ctx.dataset_metadata or {}
        candidates = set(metadata.get("candidate_models") or [])
        base_preds = {
            name: preds
            for name, preds in self._base_predictions_for_ensemble(wf_predictions).items()
            if not candidates or name in candidates
        }
        if len(base_preds) < 2:
            return

        bt_inputs = wf_backtest_inputs or {}
        template = next(iter(bt_inputs.values()), None)
        base_targets = {
            name: np.asarray(bt_inputs[name]["pred_target"], dtype=float).ravel()
            for name in base_preds
            if name in bt_inputs and "pred_target" in bt_inputs[name]
        }
        # P0-2: agirliklar yalnizca ONCEKI fold'lardan fit edilir (expanding); aksi halde
        # ayni OOS satirlarinda metrik hesaplamak in-sample olur.
        ensemble_tuples, weight_scope = self._compute_walk_forward_blends(
            base_preds,
            base_targets,
            fold_ids=self._shared_fold_ids(base_preds, bt_inputs),
            y_true_inverse=wf_y_true,
        )
        for _name in self._ENSEMBLE_NAMES:
            if _name in self.state.ensemble_weights:
                self.state.ensemble_weight_scope[_name] = weight_scope
        if self.ctx.dataset_metadata is not None:
            self.ctx.dataset_metadata["wf_ensemble_weight_scope"] = weight_scope

        for name, pred_price, pred_target in ensemble_tuples:
            k = min(len(pred_price), len(np.asarray(wf_y_true).ravel()))
            wf_predictions[name] = np.asarray(pred_price)[-k:]
            y_true_price = np.asarray(wf_y_true).ravel()[-k:]
            if template:
                y_true_target = np.asarray(template.get("y_true_target", []), dtype=float).ravel()[
                    -k:
                ]
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
                target_mode=self.ctx.dataset_metadata.get("target_mode", "log_return"),
            )
            if template:
                payload = self._slice_template_payload(template, k)
                payload["y_true_price"] = y_true_price
                payload["pred_price"] = wf_predictions[name]
                payload["pred_target"] = (
                    np.asarray(pred_target)[-k:] if pred_target is not None else None
                )
                bt_inputs[name] = payload
        print(
            "  [OK] Walk-forward ensemble tahmini eklendi: Inverse RMSE."
        )

    # ------------------------------------------------------------------ #
    #  Single model prediction                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _predict_target_space(model_name: str, model: Any, tensors: dict) -> np.ndarray:
        """Modelin hedef-uzayi (orijinal olcek) tahmini.

        Tree/linear modeller olcekli hedefle egitilir (``X_train_s``/``y_train_s``);
        ciktilari ``scaler_y`` ile geri cevrilir. Naive modeller ham hedefle egitilir
        (``X_train``/``y_train``) ve zaten orijinal olcekte tahmin uretir; bunlari
        ``scaler_y`` ile tekrar donusturmek yanlistir (cift donusum).
        """
        if model_name in _TREE_MODELS:
            preds_scaled = model.predict(tensors["X_test_s"])
            return (
                tensors["scaler_y"]
                .inverse_transform(np.asarray(preds_scaled).reshape(-1, 1))
                .ravel()
            )
        return np.asarray(model.predict(tensors["X_test"])).ravel()

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
    ]:
        prev_close_test = np.asarray(tensors["prev_close_test"]).ravel()
        dates_test = np.asarray(tensors["dates_test"])
        prediction_dates_test = np.asarray(tensors.get("dates_prediction", tensors["dates_test"]))
        market_regime_test = np.asarray(
            tensors.get("market_regime_test", np.zeros(len(prev_close_test))), dtype=float
        ).ravel()
        y_test_price = np.asarray(tensors["original_y_test_aligned"]).ravel()
        y_test_target = np.asarray(tensors["y_test"]).ravel()

        preds_target = self._predict_target_space(model_name, model, tensors)
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
        return (
            pred_price,
            preds_target,
            y_test_price[-k:],
            y_test_target[-k:],
            prev_close_aligned,
            dates_test[-k:],
            prediction_dates_test[-k:],
            market_regime_aligned,
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
        self.state.latest_tensors = tensors

        for name, model in trained_models.items():
            try:
                preds_target = self._predict_target_space(name, model, tensors)
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
                market_regime_aligned = np.asarray(
                    tensors.get("market_regime_test", np.zeros(k)), dtype=float
                ).ravel()[-k:]

                raw_pred_targets[name] = preds_target
                raw_preds[name] = self._target_to_price(preds_target, prev_close_aligned)

                self.state.single_backtest_inputs[name] = {
                    "dates": dates_aligned,
                    "prediction_dates": prediction_dates_aligned,
                    "market_regime": market_regime_aligned,
                    "y_true_price": y_true_price_aligned,
                    "pred_price": raw_preds[name],
                    "prev_close": prev_close_aligned,
                    "pred_target": preds_target,
                    "y_true_target": y_true_target_aligned,
                }
                print(f"  [OK] {name} tahmini uretildi - {len(raw_preds[name])} adim")
            except Exception as exc:
                print(f"  [WARN] {name} tahmini basarisiz, atlaniyor: {exc}")

        if not raw_preds:
            raise RuntimeError("Hicbir model tahmin uretemedi. Egitim adimini kontrol edin.")

        min_len = min(len(v) for v in raw_preds.values())
        self.state.predictions = {name: preds[-min_len:] for name, preds in raw_preds.items()}
        self.state.prediction_targets = {
            name: preds[-min_len:] for name, preds in raw_pred_targets.items()
        }
        self.state.y_true_aligned = y_test_price[-min_len:]
        self.state.y_true_target_aligned = y_test_target[-min_len:]
        self.state.prev_close_aligned = prev_close_test[-min_len:]
        self._add_single_split_ensembles()
