# -*- coding: utf-8 -*-
"""Model seçim koruma katmanı.

`best_models` tablosuna yazılacak modelin gerçek anlamda production adayı
olup olmadığını denetler. Naive benchmark model lider ise ve trade sayısı
yetersizse sembol için production candidate yoktur.
"""
from __future__ import annotations

from typing import Any, Dict, Literal

EligibilityStatus = Literal["eligible", "naive_low_trades", "insufficient_trades", "no_candidate"]

MIN_TRADES_DEFAULT = 6


def compute_eligibility(
    *,
    model_name: str,
    is_production_candidate: bool,
    is_baseline: bool,
    total_trade_count: int,
    min_trades: int = MIN_TRADES_DEFAULT,
) -> tuple[EligibilityStatus, str]:
    """Experiment satırı için eligibility durumu hesapla.

    Returns
    -------
    (status, reason)
        status: EligibilityStatus string
        reason: Kullanıcıya gösterilebilecek kısa açıklama
    """
    if not is_production_candidate:
        return ("no_candidate", "Model production candidate değil (benchmark veya ensemble).")
    if is_baseline and total_trade_count < min_trades:
        return (
            "naive_low_trades",
            f"Naive model lider, ancak trade sayısı ({total_trade_count}) minimum eşiğin ({min_trades}) altında.",
        )
    if total_trade_count < min_trades:
        return (
            "insufficient_trades",
            f"Trade sayısı ({total_trade_count}) minimum eşiğin ({min_trades}) altında.",
        )
    return ("eligible", "")


def evaluate_best_model_eligibility(
    experiment_row: Dict[str, Any],
    *,
    min_trades: int = MIN_TRADES_DEFAULT,
) -> tuple[EligibilityStatus, str]:
    """Experiment satırından eligibility hesapla.

    experiment_row beklenen alanlar:
      - model_name (str)
      - is_production_candidate (int / bool)
      - is_baseline (int / bool)  — yoksa model_scope ile kontrol edilir
      - Signal_Diagnosis (str)    — virgülle ayrılmış etiketler
      - Trade_Count (int / float) — opsiyonel, yoksa 0 sayılır
    """
    from src.pipeline.model_scope import is_benchmark_model

    model_name = str(experiment_row.get("model_name", ""))
    is_prod = bool(experiment_row.get("is_production_candidate", 0))

    # is_baseline experiment_row'da yoksa model_scope'dan türet
    is_base = bool(experiment_row.get("is_baseline", is_benchmark_model(model_name)))

    try:
        trade_count = int(float(experiment_row.get("Trade_Count", 0) or 0))
    except (TypeError, ValueError):
        trade_count = 0

    return compute_eligibility(
        model_name=model_name,
        is_production_candidate=is_prod,
        is_baseline=is_base,
        total_trade_count=trade_count,
        min_trades=min_trades,
    )
