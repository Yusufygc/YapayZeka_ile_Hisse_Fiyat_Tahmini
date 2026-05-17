# -*- coding: utf-8 -*-
"""Faz 4 — paylaşılan CLI yardımcıları: model tak-çıkar bayrak çözümü.

Kullanım örnekleri:

    python -m src.cli.forecast --list-models
    python -m src.cli.batch --enable "Random Forest,DLinear" --strict-deps
    python -m src.cli.batch --disable XGBoost,NLinear
    python -m src.cli.batch --category tree,linear_decomp --role candidate

Bayraklar bağımsızdır, kombinasyon serbesttir:
    --enable      → selected_models'e satır eklenir
    --disable     → disabled_models'e satır eklenir
    --category    → kategoride olan tüm aktif modeller enable listesine eklenir
    --role        → rolde olan tüm aktif modeller enable listesine eklenir
    --strict-deps → require_available=True (eksik dep'i olan model fail)
"""
from __future__ import annotations

import argparse
from typing import Iterable


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def add_model_filter_args(parser: argparse.ArgumentParser) -> None:
    """`argparse` parser'a ortak tak-çıkar bayraklarını ekler."""
    g = parser.add_argument_group("model selection (registry)")
    g.add_argument(
        "--list-models",
        action="store_true",
        help="Registry'deki tüm modelleri listele ve çık.",
    )
    g.add_argument(
        "--enable",
        type=str,
        default=None,
        help="Virgülle ayrılmış model isimleri (selected_models'e eklenir).",
    )
    g.add_argument(
        "--disable",
        type=str,
        default=None,
        help="Virgülle ayrılmış model isimleri (disabled_models'e eklenir).",
    )
    g.add_argument(
        "--category",
        type=str,
        default=None,
        help="Virgülle ayrılmış kategoriler: tree, linear_shrinkage, linear_decomp, seq, stat, benchmark.",
    )
    g.add_argument(
        "--role",
        type=str,
        default=None,
        help="Rol filtresi: candidate veya benchmark.",
    )
    g.add_argument(
        "--strict-deps",
        action="store_true",
        help="Optional dep'i eksik modeller koşuda hata verir (varsayılan: atla).",
    )


def expand_filters(
    *,
    category: str | None = None,
    role: str | None = None,
) -> set[str]:
    """`--category` ve `--role` bayraklarını registry üzerinde model isimlerine çevirir."""
    from src.pipeline.model_registry import all_specs, ensure_loaded

    ensure_loaded()
    categories = set(_split_csv(category))
    roles = set(_split_csv(role))
    out: set[str] = set()
    for spec in all_specs():
        if categories and spec.category not in categories:
            continue
        if roles and spec.role not in roles:
            continue
        if categories or roles:
            out.add(spec.name)
    return out


def resolve_selected(
    *,
    explicit_models: Iterable[str] | None = None,
    enable: str | None = None,
    category: str | None = None,
    role: str | None = None,
) -> list[str] | None:
    """`--enable` + `--category` + `--role` + eski `--models` birleşimi.

    Hiçbir bayrak verilmemişse `None` döner → orchestrator default candidate
    kümesini kullanır.
    """
    selected: set[str] = set(explicit_models or [])
    selected.update(_split_csv(enable))
    selected.update(expand_filters(category=category, role=role))
    return sorted(selected) if selected else None


def resolve_disabled(*, disable: str | None = None) -> list[str]:
    return _split_csv(disable)


def list_models_table() -> str:
    """`--list-models` çıktısı — sabit genişlikli tablo (utf-8)."""
    from src.pipeline.model_registry import all_specs, ensure_loaded, is_available

    ensure_loaded()
    rows = [
        (
            spec.name,
            spec.category,
            spec.role,
            "Y" if spec.ensemble_eligible else "N",
            "Y" if spec.default_candidate else "N",
            "Y" if is_available(spec)[0] else "N",
            ",".join(spec.requires) if spec.requires else "-",
            spec.description or "-",
        )
        for spec in all_specs()
    ]
    headers = ("Name", "Category", "Role", "Ens", "Def", "Avail", "Requires", "Description")
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)
    ]
    # Description sütunu çok uzun olabilir; 60 karaktere kıs.
    widths[-1] = min(widths[-1], 60)

    def _fmt(values):
        seq = list(values)
        out = []
        last_idx = len(seq) - 1
        for i, v in enumerate(seq):
            s = str(v)
            if i == last_idx:
                s = s[: widths[-1]]
            out.append(s.ljust(widths[i]))
        return "  ".join(out)

    lines = [_fmt(headers), _fmt(["-" * w for w in widths])]
    lines.extend(_fmt(r) for r in rows)
    return "\n".join(lines)
