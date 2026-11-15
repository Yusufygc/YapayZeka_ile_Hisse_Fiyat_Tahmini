# -*- coding: utf-8 -*-
"""Shared model-scope rules for training, reporting and production selection.

Faz 2 (devir): `BENCHMARK_MODELS`, `CANDIDATE_MODELS`, `DEFAULT_CANDIDATE_MODELS`
sabitleri artık `model_registry` üzerinden türetilir. Modül-seviyesi
`__getattr__` ile eski isimler korunur — `from src.pipeline.model_scope import
CANDIDATE_MODELS` çalışmaya devam eder.

Tuple sıralaması: orijinal sırayı korumak için kanonik bir öncelik listesi
(`_CANONICAL_ORDER`) kullanılır. Listede olmayan kayıtlı modeller alfabetik
olarak sona eklenir — yeni eklenen plug-in modelleri tuple'a otomatik girer.
"""
from __future__ import annotations

from typing import Iterable


# Tarihi tuple sıralaması — yeni model eklendiğinde bu listeye de eklemek
# isteğe bağlı (eklenmezse alfabetik fallback uygulanır).
_CANONICAL_ORDER: tuple[str, ...] = (
    # Benchmarks
    "Naive Last Value",
    "Naive Zero Return",
    "Naive Drift",
    # Candidates (eski CANDIDATE_MODELS sırası)
    "Ridge Return",
    "LightGBM Return",
    "XGBoost",
)

# Yeni kosularda uretilen tek ensemble varyanti.
PRODUCTION_ENSEMBLE_NAME = "Ensemble Inverse RMSE"

# Silinen (artik desteklenmeyen) per-symbol modeller. Eski DB satirlari bu
# isimleri tasiyabilir; okuma yollari cokmemeli, tahmin yollari "retrain" demeli.
REMOVED_MODELS: frozenset[str] = frozenset(
    {
        "LSTM",
        "LSTM Lite",
        "AttentionLSTM v2",
        "DLinear",
        "NLinear",
        "Prophet",
        "ARIMA",
        "Ensemble Seq-Attention Equal",
        "Ensemble Seq-Attention Inverse RMSE",
        # Audit (815 walk-forward deney): hicbir per-symbol model naive'i gecemedi.
        # ElasticNet ~ Ridge, Random Forest ~ XGBoost/LightGBM kopyasi; zayif modelleri
        # ortalayan ensemble varyantlari da naive'i gecemedi. Tek ensemble:
        # "Ensemble Inverse RMSE" (REMOVED_MODELS disinda).
        "ElasticNet Return",
        "Random Forest",
        "Ensemble Equal Weight",
        "Ensemble Sharpe-Weighted",
        "Ensemble Risk-Parity",
        "Ensemble Hierarchical",
        "Ensemble Meta-Stacker",
        "Ensemble Cash-Gated",
    }
)


def is_removed_model(model_name: object) -> bool:
    """True for model names deleted from the per-symbol path (legacy DB rows)."""
    return str(model_name or "").strip() in REMOVED_MODELS


def _sort_by_canonical(names: Iterable[str]) -> tuple[str, ...]:
    seen = set(names)
    ordered = [n for n in _CANONICAL_ORDER if n in seen]
    leftover = sorted(n for n in seen if n not in _CANONICAL_ORDER)
    return tuple(ordered + leftover)


def benchmark_models() -> tuple[str, ...]:
    from src.pipeline.model_registry import all_specs

    return _sort_by_canonical(s.name for s in all_specs(role="benchmark"))


def candidate_models() -> tuple[str, ...]:
    from src.pipeline.model_registry import all_specs

    return _sort_by_canonical(s.name for s in all_specs(role="candidate"))


def default_candidate_models() -> tuple[str, ...]:
    from src.pipeline.model_registry import all_specs, ensure_loaded

    ensure_loaded()
    return _sort_by_canonical(
        s.name for s in all_specs(role="candidate") if s.default_candidate
    )


def validate_selected_models(selected_models: Iterable[str]) -> None:
    """Kaldirilmis veya bilinmeyen model adlarinda ``ValueError`` firlatir.

    Benchmark (naive) adlari ve uretim ensemble adi bilinen adlardir (``--role
    benchmark`` gibi filtreler bunlari secime katar); aday kumesine alinmazlar ama hata
    degildir. Eski bir ``--models LSTM`` / API job / kayitli config sessizce bos aday
    kumesine (yalniz naive benchmark) dusmesin diye acik hata verilir.
    """
    known = set(candidate_models()) | set(benchmark_models()) | {PRODUCTION_ENSEMBLE_NAME}
    removed: list[str] = []
    unknown: list[str] = []
    for raw in selected_models:
        name = str(raw)
        if is_removed_model(name):
            removed.append(name)
        elif name not in known:
            unknown.append(name)
    if not removed and not unknown:
        return
    parts: list[str] = []
    if removed:
        parts.append(
            f"kaldirilmis model(ler): {sorted(set(removed))} (artik desteklenmiyor)"
        )
    if unknown:
        parts.append(f"bilinmeyen model(ler): {sorted(set(unknown))}")
    raise ValueError(
        "Gecersiz model secimi - "
        + "; ".join(parts)
        + f". Gecerli adaylar: {list(candidate_models())}"
    )


def normalize_candidate_models(selected_models: Iterable[str] | None) -> set[str]:
    """Return the production candidate model set for a run.

    Raises:
        ValueError: ``selected_models`` kaldirilmis veya bilinmeyen bir model icerirse.
    """
    candidates = set(candidate_models())
    if selected_models:
        selected = [str(model) for model in selected_models]
        validate_selected_models(selected)
        return {model for model in selected if model in candidates}
    return set(default_candidate_models())


def resolve_candidates(
    *,
    selected: Iterable[str] | None = None,
    disabled: Iterable[str] | None = None,
    require_available: bool = False,
) -> set[str]:
    """Faz 4 — nihai candidate kümesi.

    Sırayla uygulanan filtreler:
      1. ``selected`` boşsa default candidate kümesinden başla; doluysa
         registry'deki candidate isimlerinden geçerli olanları al.
      2. ``disabled`` listesindekileri çıkar.
      3. ``require_available=True`` ise optional dep'i eksik olan modelleri çıkar.

    Returns
    -------
    set[str]
        Eğitim ve raporlamaya katılacak model isimleri.
    """
    from src.pipeline.model_registry import get_spec, has_spec, is_available

    base = normalize_candidate_models(selected)
    disabled_set = {str(m) for m in (disabled or [])}
    result = {m for m in base if m not in disabled_set}

    if require_available:
        filtered: set[str] = set()
        for name in result:
            if has_spec(name):
                ok, _ = is_available(get_spec(name))
                if ok:
                    filtered.add(name)
            # has_spec=False durumu zaten normalize_candidate_models tarafından
            # filtrelenmiş; buraya düşmemeli.
        result = filtered
    return result


def is_benchmark_model(model_name: str) -> bool:
    return str(model_name) in set(benchmark_models())


def is_selection_candidate(model_name: str, candidate_models_iter: Iterable[str] | None) -> bool:
    name = str(model_name)
    if name == PRODUCTION_ENSEMBLE_NAME:
        return True
    return name in set(candidate_models_iter or [])


def report_group(model_name: str, candidate_models_iter: Iterable[str] | None) -> str:
    name = str(model_name)
    if is_selection_candidate(name, candidate_models_iter):
        return "candidate"
    if is_benchmark_model(name):
        return "benchmark"
    if name.startswith("Ensemble "):
        return "ensemble"
    return "comparison"


def reportable_model_names(model_names: Iterable[str], candidate_models_iter: Iterable[str] | None) -> set[str]:
    """Main reports include only selected candidates and cheap naive benchmarks."""
    candidates = set(candidate_models_iter or [])
    return {
        str(name)
        for name in model_names
        if (
            str(name) in candidates
            or is_benchmark_model(str(name))
            or str(name) == PRODUCTION_ENSEMBLE_NAME
        )
    }


# --- Geriye uyumluluk: eski tuple sabitleri --------------------------------
# `from src.pipeline.model_scope import CANDIDATE_MODELS` çağrıları çalışsın
# diye modül-seviyesi `__getattr__` ile dinamik tuple döndürülür.

_LEGACY_NAMES = {
    "BENCHMARK_MODELS": benchmark_models,
    "CANDIDATE_MODELS": candidate_models,
    "DEFAULT_CANDIDATE_MODELS": default_candidate_models,
}


def __getattr__(name: str):
    fn = _LEGACY_NAMES.get(name)
    if fn is not None:
        return fn()
    raise AttributeError(f"module 'src.pipeline.model_scope' has no attribute {name!r}")
