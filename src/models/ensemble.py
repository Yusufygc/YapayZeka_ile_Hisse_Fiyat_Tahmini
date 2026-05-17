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

    # ─────────────────────────────────────────────────────────────────
    #  Faz 5 Katman 1 — Performans metriği tabanlı ağırlık optimizerları
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _positive_normalize(scores: Dict[str, float], floor: float = 0.0) -> Dict[str, float]:
        """`floor` üzerindeki skorları normalize ederek ağırlık döner.

        Tüm skorlar floor altıysa eşit ağırlık döner (corner case).
        """
        positive = {name: max(float(val) - floor, 0.0) for name, val in scores.items()}
        total = sum(positive.values())
        if total <= 0.0:
            n = len(scores) or 1
            return {name: round(1.0 / n, 6) for name in scores}
        return {name: round(val / total, 6) for name, val in positive.items()}

    @staticmethod
    def optimize_by_sharpe(
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Direksiyonel Sharpe tabanlı ağırlıklandırma.

        Model başına basit PnL serisi: ``sign(pred) * y_true``. Bunun ham
        Sharpe oranı pozitifse ağırlık katkısı sağlar; sıfır veya negatif
        Sharpe = ağırlık 0.

        Notes
        -----
        Bu, üretim-grade Sharpe ölçümü değildir (transaction cost / position
        sizing yok). Hedef: çoklu modeli ağırlıklamak için hızlı proxy.
        """
        y = np.asarray(y_true, dtype=float).ravel()
        sharpes: Dict[str, float] = {}
        for name, preds in predictions.items():
            arr = np.asarray(preds, dtype=float).ravel()
            min_len = min(len(arr), len(y))
            if min_len < 2:
                sharpes[name] = 0.0
                continue
            sig = np.sign(arr[-min_len:])
            pnl = sig * y[-min_len:]
            std = float(pnl.std(ddof=1))
            sharpes[name] = float(pnl.mean() / std) if std > 0 else 0.0
        return EnsembleModel._positive_normalize(sharpes, floor=0.0)

    @staticmethod
    def optimize_by_dsr(deflated_sharpes: Dict[str, float]) -> Dict[str, float]:
        """Deflated Sharpe Ratio tabanlı ağırlıklandırma.

        DSR ≤ 0 modeller dışlanır (istatistiksel anlamsızlık). Tümü ≤ 0 ise
        eşit ağırlık fallback.
        """
        return EnsembleModel._positive_normalize(deflated_sharpes, floor=0.0)

    @staticmethod
    def optimize_by_profit_factor(profit_factors: Dict[str, float]) -> Dict[str, float]:
        """Profit Factor tabanlı ağırlıklandırma.

        ``PF > 1`` (kazançlı) modeller ağırlık alır; ``PF ≤ 1`` dışlanır.
        Tümü ≤ 1 ise eşit ağırlık fallback.
        """
        return EnsembleModel._positive_normalize(profit_factors, floor=1.0)

    # ─────────────────────────────────────────────────────────────────
    #  Faz 5 Katman 2 — Risk-parity (inverse-volatility) ağırlıklandırma
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_pnl_volatilities(
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Her modelin direksiyonel PnL serisinin std'sini döner.

        PnL_i = ``sign(pred_i) * y_true``. Çok kısa seri (< 2 nokta) → σ=0.
        """
        y = np.asarray(y_true, dtype=float).ravel()
        vols: Dict[str, float] = {}
        for name, preds in predictions.items():
            arr = np.asarray(preds, dtype=float).ravel()
            min_len = min(len(arr), len(y))
            if min_len < 2:
                vols[name] = 0.0
                continue
            pnl = np.sign(arr[-min_len:]) * y[-min_len:]
            vols[name] = float(pnl.std(ddof=1))
        return vols

    @staticmethod
    def optimize_by_risk_parity(volatilities: Dict[str, float]) -> Dict[str, float]:
        """Inverse-volatility ağırlıklandırma: ``w_i ∝ 1/σ_i``.

        σ ≤ 0 olan model dışlanır (ağırlık 0). Tümü ≤ 0 ise eşit fallback.

        Notes
        -----
        Modeller birbirine yakın korele olduğunda klasik risk-parity'den
        sapma olur; gerçek risk-parity covariance matrisini çözer. Bu basit
        sürüm uncorrelated-assets yaklaşımı.
        """
        inv = {
            name: (1.0 / sigma if sigma and sigma > 0.0 else 0.0)
            for name, sigma in volatilities.items()
        }
        total = sum(inv.values())
        if total <= 0.0:
            n = len(volatilities) or 1
            return {name: round(1.0 / n, 6) for name in volatilities}
        return {name: round(val / total, 6) for name, val in inv.items()}

    # ─────────────────────────────────────────────────────────────────
    #  Faz 5 Katman 5 — Cash signal / veto gate
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_directional_agreement(predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Her timestep için majority-sign oranı (n_majority / n_total).

        ``predictions`` boşsa boş array döner. Diziler en kısa uzunluğa kırpılır.
        Sıfır tahminler 'agreement'a katkı vermez (sign=0).
        """
        arrays = [np.asarray(p, dtype=float).ravel() for p in predictions.values()]
        if not arrays:
            return np.array([])
        min_len = min(len(a) for a in arrays)
        if min_len == 0:
            return np.array([])
        stacked = np.stack([a[-min_len:] for a in arrays], axis=0)
        signs = np.sign(stacked)
        n_pos = (signs > 0).sum(axis=0)
        n_neg = (signs < 0).sum(axis=0)
        n_total = signs.shape[0]
        majority = np.maximum(n_pos, n_neg)
        return majority / float(n_total)

    @staticmethod
    def apply_cash_gate(
        ensemble_target: np.ndarray,
        base_predictions: Dict[str, np.ndarray] | None = None,
        magnitude_threshold: float = 0.0,
        agreement_threshold: float = 0.0,
    ) -> np.ndarray:
        """Düşük güvenli tahminleri sıfırla (cash position).

        Gate'ler:
          • Magnitude: ``|ensemble_target| < magnitude_threshold`` → 0
          • Agreement: ``directional_agreement < agreement_threshold`` → 0

        Her iki threshold 0 ise no-op. Target=0 → backtest cash position.
        """
        target = np.asarray(ensemble_target, dtype=float).ravel().copy()
        if magnitude_threshold and magnitude_threshold > 0.0:
            target[np.abs(target) < magnitude_threshold] = 0.0
        if agreement_threshold and agreement_threshold > 0.0 and base_predictions:
            agreement = EnsembleModel.compute_directional_agreement(base_predictions)
            if len(agreement):
                min_len = min(len(target), len(agreement))
                target = target[-min_len:]
                agreement = agreement[-min_len:]
                target = np.where(agreement >= agreement_threshold, target, 0.0)
        return target

    # ─────────────────────────────────────────────────────────────────
    #  Faz 5 Katman 4 — Ridge meta-stacker (OOF time-varying blend)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def optimize_by_ridge_stacker(
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        alpha: float = 1.0,
        non_negative: bool = True,
    ) -> Dict[str, float]:
        """Ridge regresyon meta-stacker.

        Base tahmin matrisi ``X (n, k)`` üstüne ``Ridge(alpha, fit_intercept=False)``
        fit edilir. Coefficient'lar long-only blend için negatif kırpılır ve toplama
        normalize edilir. WF bağlamında ``y_true`` doğal olarak OOF (fold birleşimi)
        — bu yöntem time-varying blend katsayısı verir.

        Parameters
        ----------
        y_true : np.ndarray            Hedef (target-space; pred_target ile aynı uzay).
        predictions : dict             Model adı -> base pred dizisi.
        alpha : float                  Ridge L2 katsayısı. Yüksek alpha → eşit ağırlığa yaklaşır.
        non_negative : bool            True ise negatif coef'ler 0'a kırpılır (long-only).

        Returns
        -------
        dict  Model adı -> ağırlık (toplam 1).
        """
        y = np.asarray(y_true, dtype=float).ravel()
        names = list(predictions)
        if not names:
            return {}
        arrays = [np.asarray(predictions[n], dtype=float).ravel() for n in names]
        min_len = min([len(y)] + [len(a) for a in arrays])
        # Yeterli örnek yoksa eşit ağırlık fallback.
        if min_len < max(2, len(names) + 1):
            n = len(names)
            return {nm: round(1.0 / n, 6) for nm in names}

        X = np.column_stack([a[-min_len:] for a in arrays])
        y_aligned = y[-min_len:]

        try:
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=alpha, fit_intercept=False)
            reg.fit(X, y_aligned)
            coefs = np.asarray(reg.coef_, dtype=float).ravel()
        except Exception:
            # sklearn yoksa → analytic ridge: (X^T X + alpha I)^-1 X^T y
            XtX = X.T @ X + alpha * np.eye(X.shape[1])
            coefs = np.linalg.solve(XtX, X.T @ y_aligned)

        if non_negative:
            coefs = np.clip(coefs, 0.0, None)
        total = float(coefs.sum())
        if total <= 0.0:
            n = len(names)
            return {nm: round(1.0 / n, 6) for nm in names}
        weights = coefs / total
        return {nm: round(float(w), 6) for nm, w in zip(names, weights)}

    # ─────────────────────────────────────────────────────────────────
    #  Faz 5 Katman 3 — Kategori-gated hierarchical blend
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def optimize_hierarchical_by_category(
        predictions: Dict[str, np.ndarray],
        categories: Dict[str, str],
    ) -> Dict[str, float]:
        """İki seviyeli ağırlık: kategori-içi equal, kategoriler-arası equal.

        ``w_i = (1 / n_categories) * (1 / n_in_cat_i)``.

        Parameters
        ----------
        predictions : dict          Model adı -> tahmin dizisi (uzunluk kullanılmaz, sadece set).
        categories  : dict          Model adı -> kategori adı. Eksik model "unknown" kategorisine düşer.
        """
        if not predictions:
            return {}
        cat_to_models: Dict[str, List[str]] = {}
        for name in predictions:
            cat = categories.get(name, "unknown")
            cat_to_models.setdefault(cat, []).append(name)

        n_cat = len(cat_to_models)
        inter_w = 1.0 / n_cat
        weights: Dict[str, float] = {}
        for cat, members in cat_to_models.items():
            intra_w = 1.0 / len(members)
            for name in members:
                weights[name] = round(inter_w * intra_w, 6)
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
