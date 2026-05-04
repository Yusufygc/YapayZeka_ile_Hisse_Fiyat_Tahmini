# -*- coding: utf-8 -*-
"""
ensemble.py — Topluluk (Ensemble) Modeli + Ağırlık Optimizasyonu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Birden fazla modelin tahminlerini ağırlıklı ortalama ile birleştirir.
Ağırlık belirleme stratejileri:
  • Manuel ağırlıklar
  • Inverse RMSE ağırlıklandırma  ← production-ready, önerilen
  • Grid Search                   ← DEVRE DIŞI (exponential complexity)
"""

import numpy as np
from typing import Dict, List, Tuple

try:
    from sklearn.metrics import mean_squared_error
except ImportError:  # pragma: no cover - minimal runtimes
    def mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        return float(np.mean((y_true - y_pred) ** 2))


class EnsembleModel:
    """Ağırlıklı ortalama tabanlı topluluk modeli — ağırlık optimizasyonu destekli."""

    def __init__(self, weights: Dict[str, float] | None = None):
        """
        Parameters
        ----------
        weights : dict | None
            Model adı -> ağırlık eşleştirmesi.
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
            Model adı -> tahmin dizisi (np.ndarray).

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
        predictions : dict        Model adı -> tahmin dizisi.

        Returns
        -------
        dict  Model adı -> optimized ağırlık.
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
        [DEVRE DIŞI] Grid Search ağırlık optimizasyonu.

        Bu metod kasıtlı olarak devre dışı bırakılmıştır.

        Neden: step=0.05 ve N model için arama uzayı (1/step)^N büyüklüğünde.
        N=10 model → 20^10 ≈ 10 trilyon kombinasyon → asla bitmez.

        Alternatifler:
          - optimize_inverse_rmse()  : O(N), production-ready, genellikle yeterli.
          - scipy.optimize.minimize() ile Dirichlet kısıtlı L-BFGS-B (Faz 3'te eklenecek).

        Raises
        ------
        NotImplementedError
            Her zaman fırlatılır.
        """
        n_models = len(predictions)
        raise NotImplementedError(
            f"optimize_grid_search() {n_models} model için güvenli değil "
            f"(arama uzayı: {int(round((1.0 / step) + 1)) ** n_models:,} kombinasyon). "
            "Bunun yerine optimize_inverse_rmse() kullanın."
        )
