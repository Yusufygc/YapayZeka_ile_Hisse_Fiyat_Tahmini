"""Model-family XAI strategies used by ``XAIExplainer``."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


class TabularContributionStrategy:
    def __init__(self, feature_names: list[str], *, max_rows: int = 80) -> None:
        self.feature_names = feature_names
        self.max_rows = max_rows

    def tree_contributions(self, estimator: Any, X: np.ndarray) -> Tuple[np.ndarray, str, bool]:
        try:
            import shap  # type: ignore

            explainer = shap.TreeExplainer(estimator)
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

            explainer = shap.LinearExplainer(estimator, X)
            shap_values = explainer.shap_values(X)
            return np.asarray(shap_values, dtype=float), "shap_linear", False
        except Exception:
            coef = np.asarray(getattr(estimator, "coef_", []), dtype=float).ravel()
            if coef.size == X.shape[1]:
                centered = X - np.nanmean(X, axis=0, keepdims=True)
                return centered * coef.reshape(1, -1), "linear_coefficients", True
            return self.permutation_contributions(estimator, X), "permutation_fallback", True

    def permutation_contributions(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        baseline = np.asarray(estimator.predict(X), dtype=float).ravel()
        contribs = np.zeros((len(X), len(self.feature_names)), dtype=float)
        for feature_idx in range(min(X.shape[1], len(self.feature_names))):
            X_perm = X.copy()
            X_perm[:, feature_idx] = np.mean(X_perm[:, feature_idx])
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
            training_data=np.asarray(X, dtype=float),
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
    def __init__(self, feature_names: list[str], *, max_rows: int = 80) -> None:
        self.feature_names = feature_names
        self.max_rows = max_rows

    def permutation_contributions(self, model: Any, X_seq: np.ndarray) -> Tuple[np.ndarray, str, bool]:
        sample_count = min(len(X_seq), self.max_rows)
        X = X_seq[-sample_count:]
        baseline = np.asarray(model.predict(X), dtype=float).ravel()
        importances = np.zeros(len(self.feature_names), dtype=float)
        signs = np.zeros(len(self.feature_names), dtype=float)
        for feature_idx in range(min(X.shape[2], len(self.feature_names))):
            X_masked = X.copy()
            X_masked[:, :, feature_idx] = np.mean(X_masked[:, :, feature_idx])
            masked_pred = np.asarray(model.predict(X_masked), dtype=float).ravel()
            delta = baseline - masked_pred
            importances[feature_idx] = float(np.mean(np.abs(delta)))
            signs[feature_idx] = float(np.mean(delta))
        contribs = (np.sign(signs) * importances).reshape(1, -1)
        return contribs, "sequence_permutation", True

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
