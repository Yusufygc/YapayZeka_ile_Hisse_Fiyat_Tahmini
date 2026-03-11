# -*- coding: utf-8 -*-
"""
ensemble.py — Topluluk (Ensemble) Modeli + Ağırlık Optimizasyonu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Birden fazla modelin tahminlerini ağırlıklı ortalama ile birleştirir.
Ağırlık belirleme stratejileri:
  • Manuel ağırlıklar
  • Inverse RMSE ağırlıklandırma
  • Grid Search (en düşük RMSE'yi veren ağırlık kombinasyonu)
"""

import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple


class EnsembleModel:
    """Ağırlıklı ortalama tabanlı topluluk modeli — ağırlık optimizasyonu destekli."""

    def __init__(self, weights: Dict[str, float] | None = None):
        """
        Parameters
        ----------
        weights : dict | None
            Model adı → ağırlık eşleştirmesi.
            None ise tüm modellere eşit ağırlık verilir.
            Örnek: {"Prophet": 0.2, "XGBoost": 0.4, "LSTM": 0.4}
        """
        self.weights = weights

    def combine(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Tahmin dizilerini birleştirir.
        Tüm diziler en kısa dizinin uzunluğuna kırpılır (son N eleman alınır).

        Parameters
        ----------
        predictions : dict
            Model adı → tahmin dizisi (np.ndarray).

        Returns
        -------
        np.ndarray  Birleştirilmiş ensemble tahmin dizisi.
        """
        names: List[str] = list(predictions.keys())
        arrays: List[np.ndarray] = [predictions[n].ravel() for n in names]

        # En kısa diziye hizala (sondan kırp) — LSTM genelde daha kısadır
        min_len = min(len(a) for a in arrays)
        arrays = [a[-min_len:] for a in arrays]

        # Ağırlıkları belirle
        if self.weights is None:
            w = np.ones(len(names)) / len(names)
        else:
            w = np.array([self.weights.get(n, 1.0) for n in names])
            w = w / w.sum()  # Normalleştir

        # Ağırlıklı toplam
        stacked = np.stack(arrays, axis=0)  # (n_models, min_len)
        ensemble_preds = np.average(stacked, axis=0, weights=w)

        return ensemble_preds

    @staticmethod
    def optimize_inverse_rmse(
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Inverse RMSE ağırlıklandırma.
        Her modelin RMSE'sinin tersi ile orantılı ağırlık hesaplar.

        w_i = (1 / RMSE_i) / Σ(1 / RMSE_j)

        Parameters
        ----------
        y_true : np.ndarray       Gerçek değerler (orijinal ölçekte).
        predictions : dict        Model adı → tahmin dizisi.

        Returns
        -------
        dict  Model adı → optimized ağırlık.
        """
        y_true = y_true.ravel()
        inv_rmse = {}

        for name, preds in predictions.items():
            preds = preds.ravel()
            min_len = min(len(y_true), len(preds))
            rmse = float(np.sqrt(mean_squared_error(y_true[-min_len:], preds[-min_len:])))
            inv_rmse[name] = 1.0 / rmse if rmse > 0 else 0.0

        total = sum(inv_rmse.values())
        if total == 0:
            n = len(predictions)
            return {name: 1.0 / n for name in predictions}

        weights = {name: round(val / total, 4) for name, val in inv_rmse.items()}

        print("  [Inverse RMSE] Optimized ağırlıklar:")
        for name, w in weights.items():
            print(f"    • {name}: {w:.4f}")

        return weights

    @staticmethod
    def optimize_grid_search(
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        step: float = 0.05,
    ) -> Tuple[Dict[str, float], float]:
        """
        Grid Search ile en düşük RMSE'yi veren ağırlık kombinasyonunu bulur.
        Σw = 1 kısıtı ile 0.0–1.0 aralığında aranır.

        Parameters
        ----------
        y_true : np.ndarray       Gerçek değerler (orijinal ölçekte).
        predictions : dict        Model adı → tahmin dizisi.
        step : float              Grid arama adımı (varsayılan: 0.05).

        Returns
        -------
        (dict, float)  (En iyi ağırlıklar, en düşük RMSE).
        """
        names = list(predictions.keys())
        n_models = len(names)

        # Tüm dizileri hizala
        arrays = [predictions[n].ravel() for n in names]
        y_true = y_true.ravel()
        min_len = min(len(y_true), *[len(a) for a in arrays])
        arrays = [a[-min_len:] for a in arrays]
        y_true = y_true[-min_len:]

        stacked = np.stack(arrays, axis=0)  # (n_models, min_len)

        # Ağırlık adayları
        candidates = np.arange(0.0, 1.0 + step / 2, step)

        best_rmse = float("inf")
        best_weights = None

        # Grid search — tüm kombinasyonları dene (Σw ≈ 1.0)
        for combo in product(candidates, repeat=n_models):
            total = sum(combo)
            if abs(total - 1.0) > 1e-6:
                continue

            w = np.array(combo)
            preds_ensemble = np.average(stacked, axis=0, weights=w)
            rmse = float(np.sqrt(mean_squared_error(y_true, preds_ensemble)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = combo

        if best_weights is None:
            # Eşit ağırlık döndür
            equal_w = 1.0 / n_models
            best_weights = tuple([equal_w] * n_models)
            best_rmse = float(np.sqrt(mean_squared_error(
                y_true, np.average(stacked, axis=0, weights=np.array(best_weights))
            )))

        result = {names[i]: round(best_weights[i], 4) for i in range(n_models)}

        print(f"  [Grid Search] En iyi RMSE: {best_rmse:.4f}")
        print(f"  [Grid Search] En iyi ağırlıklar:")
        for name, w in result.items():
            print(f"    • {name}: {w:.4f}")

        return result, best_rmse
