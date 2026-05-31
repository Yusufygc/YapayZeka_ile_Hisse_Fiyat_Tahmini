# -*- coding: utf-8 -*-
"""Tree-model Optuna tuning için paylaşılan yardımcılar (DRY).

XGBoost / Random Forest (ve diğer tree modeller) `tune_and_train` metotları aynı
iskeleti paylaşır: TimeSeriesSplit üzerinde sign-of-pred Sharpe ile skorlama ve
SQLite warm-start'lı Optuna study. Bu modül o ortak çekirdeği tek kaynakta toplar;
modele özgü olan yalnızca arama uzayı (params) ve fit yoludur.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Tuple

import numpy as np


def _optuna_storage_uri() -> str:
    """`data/optuna/optuna_studies.db` için SQLite storage URI'si (dizini oluşturur)."""
    optuna_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "optuna",
    )
    os.makedirs(optuna_dir, exist_ok=True)
    optuna_path = os.path.join(optuna_dir, "optuna_studies.db")
    return f"sqlite:///{optuna_path.replace(os.sep, '/')}"


def stability_adjusted_cv_objective(
    X_train: np.ndarray,
    y_flat: np.ndarray,
    tscv,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> float:
    """TimeSeriesSplit fold'larında stability-adjusted Sharpe objective.

    Her fold'da `fit_predict_fn(X_tr, y_tr, X_val, y_val)` tahmin üretir;
    sign(pred) * y_val ile günlük strateji getirisi ve yıllık Sharpe hesaplanır.
    Dönen değer `-(mean_sharpe - 0.5 * std_sharpe)` (minimize edilir). 3 tree
    modelinin paylaştığı skorlama; modele özgü tek kısım `fit_predict_fn`.
    """
    sharpe_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_flat[train_idx], y_flat[val_idx]
        preds = fit_predict_fn(X_tr, y_tr, X_val, y_val)
        signals = np.sign(preds)
        fold_ret = signals * y_val
        denom = float(np.std(fold_ret, ddof=0))
        fold_sharpe = float(np.mean(fold_ret) / denom * np.sqrt(252)) if denom > 1e-9 else -1.0
        sharpe_scores.append(fold_sharpe)
    mean_s = float(np.mean(sharpe_scores))
    std_s = float(np.std(sharpe_scores, ddof=0)) if len(sharpe_scores) > 1 else 0.0
    return -(mean_s - 0.5 * std_s)


def run_optuna_study(
    objective: Callable[[Any], float],
    *,
    n_trials: int,
    n_splits: int,
    random_state: int,
    study_name: str,
    log_prefix: str,
    study_storage: str | None = None,
) -> Tuple[Dict[str, Any], float]:
    """Ortak Optuna study: SQLite warm-start, hata durumunda bellek-içi fallback.

    Returns:
        (best_params, best_value).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"  [{log_prefix}] {n_trials} deneme başlatılıyor ({n_splits}-fold TSCV)...")
    if study_storage is None:
        study_storage = _optuna_storage_uri()
    sampler = optuna.samplers.TPESampler(seed=random_state)
    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            storage=study_storage,
            study_name=study_name,
            load_if_exists=True,
        )
    except Exception:
        # Storage hatasi durumunda hafiza ici fallback
        study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value
