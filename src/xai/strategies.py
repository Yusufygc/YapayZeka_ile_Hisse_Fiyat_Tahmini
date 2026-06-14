"""Model-family XAI strategies used by ``XAIExplainer``."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.xai.background import XAIBackgroundProvider


def compute_feature_stability_scores(
    fold_importances: List[Dict[str, float]],
    top_k: int = 5,
) -> Dict[str, float]:
    """Her fold'daki top-K özelliği sayarak stabilite skoru hesapla.

    Parameters
    ----------
    fold_importances:
        Her eleman bir fold için ``{feature_name: importance}`` dicts.
    top_k:
        Her fold'da kaç özelliğin "top" sayılacağı.

    Returns
    -------
    dict
        ``{feature_name: fold_ratio}`` — özelliğin kaç fold'da top-K'ya
        girdiğini normalize eder (0..1).
    """
    if not fold_importances:
        return {}
    n_folds = len(fold_importances)
    top_counts: Dict[str, int] = {}
    for fold_imp in fold_importances:
        if not fold_imp:
            continue
        sorted_items = sorted(fold_imp.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, _ in sorted_items[:top_k]:
            top_counts[feat] = top_counts.get(feat, 0) + 1
    return {feat: count / n_folds for feat, count in top_counts.items()}


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

    def linear_contributions(self, estimator: Any, X: np.ndarray) -> Tuple[np.ndarray, str, bool]:
        try:
            import shap  # type: ignore

            background = self.background_provider.tabular_background
            explainer = shap.LinearExplainer(estimator, background if background is not None else X)
            shap_values = explainer.shap_values(X)
            return np.asarray(shap_values, dtype=float), "shap_linear", False
        except Exception:
            coef = np.asarray(getattr(estimator, "coef_", []), dtype=float).ravel()
            if coef.size == X.shape[1]:
                baseline = self.background_provider.tabular_median
                if baseline is None or len(baseline) != X.shape[1]:
                    baseline = np.nanmedian(X, axis=0)
                centered = X - np.asarray(baseline, dtype=float).reshape(1, -1)
                return centered * coef.reshape(1, -1), "linear_coefficients", True
            return self.permutation_contributions(estimator, X), "permutation_fallback", True

    def permutation_contributions(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        baseline = np.asarray(estimator.predict(X), dtype=float).ravel()
        contribs = np.zeros((len(X), len(self.feature_names)), dtype=float)
        for feature_idx in range(min(X.shape[1], len(self.feature_names))):
            X_perm = X.copy()
            X_perm[:, feature_idx] = self.background_provider.tabular_mask_value(X_perm, feature_idx)
            perm_pred = np.asarray(estimator.predict(X_perm), dtype=float).ravel()
            contribs[:, feature_idx] = baseline - perm_pred
        return contribs

    def lime_tabular_contributions(self, estimator: Any, X: np.ndarray) -> Tuple[np.ndarray, str, bool]:
        try:
            from lime.lime_tabular import LimeTabularExplainer  # type: ignore
        except Exception:
            return np.zeros((0, len(self.feature_names)), dtype=float), "lime_unavailable", True

        sample_count = min(len(X), self.max_rows)
        if sample_count == 0:
            return np.zeros((0, len(self.feature_names)), dtype=float), "lime_tabular", True

        X_sample = np.asarray(X[-sample_count:], dtype=float)
        explainer = LimeTabularExplainer(
            training_data=(
                self.background_provider.tabular_background
                if self.background_provider.tabular_background is not None
                else np.asarray(X, dtype=float)
            ),
            feature_names=self.feature_names[: X.shape[1]],
            mode="regression",
            discretize_continuous=True,
        )
        contribs = np.zeros((sample_count, len(self.feature_names)), dtype=float)

        def predict_fn(values):
            return np.asarray(estimator.predict(values), dtype=float).ravel()

        for row_idx, row in enumerate(X_sample):
            explanation = explainer.explain_instance(
                row,
                predict_fn,
                num_features=min(len(self.feature_names), X.shape[1]),
            )
            mapped = next(iter(explanation.as_map().values()), [])
            for feature_idx, weight in mapped:
                if int(feature_idx) < contribs.shape[1]:
                    contribs[row_idx, int(feature_idx)] = float(weight)
        return contribs, "lime_tabular", True


class SequenceContributionStrategy:
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

    def permutation_contributions(self, model: Any, X_seq: np.ndarray) -> Tuple[np.ndarray, str, bool]:
        sample_count = min(len(X_seq), self.max_rows)
        X = X_seq[-sample_count:]
        baseline = np.asarray(model.predict(X), dtype=float).ravel()
        importances = np.zeros(len(self.feature_names), dtype=float)
        signs = np.zeros(len(self.feature_names), dtype=float)
        for feature_idx in range(min(X.shape[2], len(self.feature_names))):
            X_masked = X.copy()
            for lag_idx in range(X.shape[1]):
                X_masked[:, lag_idx, feature_idx] = self.background_provider.sequence_mask_value(
                    X_masked, lag_idx, feature_idx
                )
            masked_pred = np.asarray(model.predict(X_masked), dtype=float).ravel()
            delta = baseline - masked_pred
            importances[feature_idx] = float(np.mean(np.abs(delta)))
            signs[feature_idx] = float(np.mean(delta))
        contribs = (np.sign(signs) * importances).reshape(1, -1)
        return contribs, "sequence_permutation", True

    def feature_lag_contributions(
        self,
        model: Any,
        X_seq: np.ndarray,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], str, bool]:
        sample_count = min(len(X_seq), self.max_rows)
        X = np.asarray(X_seq[-sample_count:], dtype=float)
        if X.ndim != 3 or sample_count == 0:
            return np.zeros((0, len(self.feature_names)), dtype=float), [], "sequence_feature_lag_permutation", True
        baseline = np.asarray(model.predict(X), dtype=float).ravel()
        feature_signed = np.zeros(len(self.feature_names), dtype=float)
        rows: List[Dict[str, Any]] = []
        for lag_idx in range(X.shape[1]):
            lag_from_now = int(X.shape[1] - lag_idx)
            for feature_idx in range(min(X.shape[2], len(self.feature_names))):
                X_masked = X.copy()
                X_masked[:, lag_idx, feature_idx] = self.background_provider.sequence_mask_value(
                    X_masked, lag_idx, feature_idx
                )
                masked_pred = np.asarray(model.predict(X_masked), dtype=float).ravel()
                delta = baseline - masked_pred
                contribution = float(np.mean(delta))
                importance = float(np.mean(np.abs(delta)))
                feature_signed[feature_idx] += contribution
                rows.append(
                    {
                        "Feature": self.feature_names[feature_idx],
                        "Lag": lag_from_now,
                        "Contribution": contribution,
                        "Importance": importance,
                        "Method": "sequence_feature_lag_permutation",
                        "Approximate": True,
                    }
                )
        return feature_signed.reshape(1, -1), rows, "sequence_feature_lag_permutation", True

    def lime_sequence_local_contributions(self, model: Any, X_seq: np.ndarray) -> Tuple[np.ndarray, str, bool]:
        try:
            from lime.lime_tabular import LimeTabularExplainer  # type: ignore
        except Exception:
            return np.zeros((0, len(self.feature_names)), dtype=float), "lime_unavailable", True

        sample_count = min(len(X_seq), self.max_rows)
        if sample_count == 0:
            return np.zeros((0, len(self.feature_names)), dtype=float), "lime_sequence_local", True
        X = np.asarray(X_seq[-sample_count:], dtype=float)
        flattened = X.reshape(X.shape[0], -1)
        feature_labels = [
            f"{feature}_t{step}"
            for step in range(X.shape[1])
            for feature in self.feature_names[: X.shape[2]]
        ]
        explainer = LimeTabularExplainer(flattened, feature_names=feature_labels, mode="regression")

        def predict_fn(values):
            shaped = np.asarray(values, dtype=float).reshape((-1, X.shape[1], X.shape[2]))
            return np.asarray(model.predict(shaped), dtype=float).ravel()

        explanation = explainer.explain_instance(flattened[-1], predict_fn, num_features=len(feature_labels))
        contribs = np.zeros((1, len(self.feature_names)), dtype=float)
        for flat_idx, weight in next(iter(explanation.as_map().values()), []):
            feature_idx = int(flat_idx) % len(self.feature_names)
            contribs[0, feature_idx] += float(weight)
        return contribs, "lime_sequence_local", True
