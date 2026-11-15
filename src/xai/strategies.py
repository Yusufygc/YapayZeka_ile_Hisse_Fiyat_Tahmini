"""Kol-B serving XAI stratejisi (SHAP tree / permutation fallback)."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from src.xai.background import XAIBackgroundProvider


class TabularContributionStrategy:
    def __init__(
        self,
        feature_names: list[str],
        *,
        max_rows: int = 80,
        background_provider: XAIBackgroundProvider | None = None,
    ) -> None:
        self.feature_names = feature_names
        self.max_rows = max_rows
        self.background_provider = background_provider or XAIBackgroundProvider.unavailable()

    def tree_contributions(self, estimator: Any, X: np.ndarray) -> Tuple[np.ndarray, str, bool]:
        try:
            import shap  # type: ignore

            background = self.background_provider.tabular_background
            explainer = shap.TreeExplainer(estimator, data=background) if background is not None else shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            return np.asarray(shap_values, dtype=float), "shap_tree", False
        except Exception:
            contribs = self.permutation_contributions(estimator, X)
            return contribs, "permutation_fallback", True

    def permutation_contributions(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        baseline = np.asarray(estimator.predict(X), dtype=float).ravel()
        contribs = np.zeros((len(X), len(self.feature_names)), dtype=float)
        for feature_idx in range(min(X.shape[1], len(self.feature_names))):
            X_perm = X.copy()
            X_perm[:, feature_idx] = self.background_provider.tabular_mask_value(X_perm, feature_idx)
            perm_pred = np.asarray(estimator.predict(X_perm), dtype=float).ravel()
            contribs[:, feature_idx] = baseline - perm_pred
        return contribs
