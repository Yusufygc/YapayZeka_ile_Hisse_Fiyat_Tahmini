# -*- coding: utf-8 -*-
"""Tree-model Optuna tuning için paylaşılan yardımcılar (DRY).

XGBoost / Random Forest (ve diğer tree modeller) `tune_and_train` metotları aynı
iskeleti paylaşır: TimeSeriesSplit üzerinde sign-of-pred Sharpe ile skorlama ve
Optuna study (varsayılan: bellek-içi, çağrıya izole). Bu modül o ortak çekirdeği tek kaynakta toplar;
modele özgü olan yalnızca arama uzayı (params) ve fit yoludur.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np


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
    log_prefix: str,
    study_name: str | None = None,
    study_storage: str | None = None,
) -> Tuple[Dict[str, Any], float]:
    """Ortak Optuna study.

    Varsayılan (study_storage=None) bellek-içi, çağrıya izole bir study'dir:
    her çağrı sıfırdan başlar. Walk-forward fold'ları aynı study'yi paylaşırsa
    `best_params` başka (gelecek dahil) pencerelerde skorlanmış trial'lardan
    seçilir -> look-ahead sızıntısı. Kalıcı storage yalnızca çağıran hem
    `study_storage` hem `study_name` verirse kullanılır (kayıt amaçlı).

    Returns:
        (best_params, best_value).
    """
    if study_storage is not None and not study_name:
        raise ValueError("study_storage verildiğinde study_name zorunludur.")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"  [{log_prefix}] {n_trials} deneme başlatılıyor ({n_splits}-fold TSCV)...")
    sampler = optuna.samplers.TPESampler(seed=random_state)
    if study_storage is None:
        study = optuna.create_study(direction="minimize", sampler=sampler)
    else:
        try:
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                storage=study_storage,
                study_name=study_name,
                load_if_exists=True,
            )
        except Exception as exc:
            # Storage hatasi durumunda hafiza ici fallback
            print(f"  [{log_prefix}] [WARN] Optuna storage açılamadı, bellek-içi study: {exc}")
            study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value
