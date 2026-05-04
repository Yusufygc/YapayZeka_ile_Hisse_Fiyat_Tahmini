# -*- coding: utf-8 -*-
"""
permutation_test.py - Feature Importance Permutation Testi (Faz 3.5).

Motivasyon:
  Model "ogrendi" mi yoksa sadece overfit mi yapti?
  SHAP veya model-native importance'lar feature selection yaparken
  overfitting'i gizleyebilir. Permutation importance bu riske karsi
  daha guclu bir testtir.

Yontem:
  Her feature icin:
    1. O featureın degerlerini rastgele karistir (permute et).
    2. Permuted veri ile tahmin al.
    3. RMSE farki = permuted_RMSE - original_RMSE hesapla.
  Pozitif fark: feature kaldirinca model kotulesiyor → feature onemli.
  Negatife yakin: feature gurultu veya overfit.

Kullanim:
    from src.evaluation.permutation_test import permutation_importance_vs_baseline
    df = permutation_importance_vs_baseline(model, X_test, y_test, feature_names)
    df.to_csv("outputs/feature_importance_permutation_v1.csv", index=False)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def permutation_importance_vs_baseline(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 20,
    seed: int = 42,
    metric: str = "rmse",
) -> pd.DataFrame:
    """
    Her feature icin permutation importance hesapla.

    Parameters
    ----------
    model : Any
        predict(X) metoduna sahip herhangi bir model nesnesi.
        Pipeline icindeki tum modeller bu arayuzu destekler.
    X_test : np.ndarray
        Test ozellikleri. Sekil: (samples, features) veya (samples, timesteps, features).
        3D giriste son eksen feature olarak kabul edilir.
    y_test : np.ndarray
        Gercek hedef degerleri. Sekil: (samples,) veya (samples, 1).
    feature_names : list of str
        Ozelliklerin isimleri (X_test'in son ekseniyle eslesir).
    n_permutations : int
        Her feature icin kac kez karistirma yapilacagi (varsayilan: 20).
        Daha yuksek deger daha guvenilir ama daha yavastir.
    seed : int
        Rastgele tohum (tekrarlanabilirlik icin).
    metric : str
        "rmse" (varsayilan) veya "mae".

    Returns
    -------
    pd.DataFrame with columns:
        Feature             : str   — ozellik adi
        Importance          : float — ortalama (permuted_metric - original_metric)
        Importance_Std      : float — permutation turlar arasi standart sapma
        Importance_Norm     : float — normalized importance (toplam 1.0)
        Original_RMSE       : float — permute edilmeden once metrik
        Mean_Permuted_RMSE  : float — permutation sonrasi ortalama metrik
        Rank                : int   — onem sirasi (1 = en onemli)
    Satirlar Importance'a gore azalan siraya gore siralanir.
    """
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float).ravel()
    is_3d = X_test.ndim == 3

    if is_3d:
        n_features = X_test.shape[2]
    else:
        n_features = X_test.shape[1]

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names uzunlugu ({len(feature_names)}) X_test feature sayisi "
            f"({n_features}) ile eslesmiyor."
        )

    # Orijinal metric
    original_score = _score(model, X_test, y_test, metric)

    rng = np.random.default_rng(seed)
    rows = []

    for feat_idx, feat_name in enumerate(feature_names):
        perm_scores = np.empty(n_permutations, dtype=float)
        for perm_i in range(n_permutations):
            X_permuted = X_test.copy()
            if is_3d:
                perm_order = rng.permutation(X_permuted.shape[0])
                X_permuted[:, :, feat_idx] = X_permuted[perm_order, :, feat_idx]
            else:
                perm_order = rng.permutation(X_permuted.shape[0])
                X_permuted[:, feat_idx] = X_permuted[perm_order, feat_idx]
            perm_scores[perm_i] = _score(model, X_permuted, y_test, metric)

        importance = float(np.mean(perm_scores)) - original_score
        importance_std = float(np.std(perm_scores))
        rows.append({
            "Feature": feat_name,
            "Importance": round(importance, 8),
            "Importance_Std": round(importance_std, 8),
            "Original_Metric": round(original_score, 8),
            "Mean_Permuted_Metric": round(float(np.mean(perm_scores)), 8),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    # Normalized importance (sadece pozitif degerler normalize edilir)
    pos_sum = df["Importance"].clip(lower=0.0).sum()
    if pos_sum > 0:
        df["Importance_Norm"] = (df["Importance"].clip(lower=0.0) / pos_sum).round(6)
    else:
        df["Importance_Norm"] = 0.0

    return df[["Rank", "Feature", "Importance", "Importance_Std", "Importance_Norm",
               "Original_Metric", "Mean_Permuted_Metric"]]


def _score(model: Any, X: np.ndarray, y: np.ndarray, metric: str) -> float:
    """Model uzerinde tek bir skor hesapla."""
    try:
        preds = np.asarray(model.predict(X), dtype=float).ravel()
    except Exception:
        return float("nan")

    y = y[:len(preds)]
    if metric == "mae":
        return float(np.mean(np.abs(y - preds)))
    # default: rmse
    return float(np.sqrt(np.mean((y - preds) ** 2)))


def run_permutation_importance_for_pipeline(
    trained_models: Dict[str, Any],
    X_test: np.ndarray,
    X_test_seq: Optional[np.ndarray],
    y_test: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 20,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Pipeline'daki tum modeller icin permutation importance hesapla ve kaydet.

    Parameters
    ----------
    trained_models : dict
        {model_name: model} sozlugu (ModelTrainer.trained_models).
    X_test : np.ndarray         2D test matrisi (tree/linear modeller icin)
    X_test_seq : np.ndarray     3D sekans matrisi (LSTM/TFT icin), None olabilir
    y_test : np.ndarray         Gercek hedef degerleri
    feature_names : list        Ozellik isimleri
    n_permutations : int        Her feature icin permutasyon sayisi
    output_dir : str, optional  Sonuclarin kaydedilecegi dizin

    Returns
    -------
    dict: {model_name: DataFrame}
    """
    import os

    _SEQ_MODELS = {"LSTM", "TFT", "DLinear", "NLinear"}
    results = {}

    for model_name, model in trained_models.items():
        if model_name.startswith("Ensemble"):
            continue  # Ensemble'lar feature-level analiz icin uygun degil
        try:
            X = X_test_seq if (model_name in _SEQ_MODELS and X_test_seq is not None) else X_test
            if X is None:
                continue
            df = permutation_importance_vs_baseline(
                model, X, y_test, feature_names,
                n_permutations=n_permutations,
            )
            results[model_name] = df

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                safe_name = model_name.replace(" ", "_").lower()
                csv_path = os.path.join(output_dir, f"feature_importance_permutation_{safe_name}.csv")
                df.to_csv(csv_path, index=False)
                print(f"  [Permutation] {model_name}: {csv_path}")
        except Exception as exc:
            print(f"  [WARN] Permutation importance {model_name} icin basarisiz: {exc}")

    return results
