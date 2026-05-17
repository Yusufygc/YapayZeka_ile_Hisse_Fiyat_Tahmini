# -*- coding: utf-8 -*-
"""Plug-in model registry.

Faz 1 (additive): bu modül `model_factory.py` ve `model_scope.py` ile yan-yana
çalışır; mevcut akışın davranışını değiştirmez. Yeni eklenen modeller decorator
yolundan kayıt olur, eski hardcode mapping aynen korunur.

Kullanım:

    from src.pipeline.model_registry import register_model, ModelSpec

    register_model(ModelSpec(
        name="MyNewModel",
        factory=lambda **kw: MyNewModel(**kw),
        category="tree",
        role="candidate",
    ))
"""
from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ModelSpec:
    """Tek modelin tüm meta verisi.

    Attributes
    ----------
    name : str
        Pipeline-wide kanonik model adı (örn. "Random Forest").
    factory : Callable
        ``factory(**kwargs) -> BaseModel`` örneği üreten çağrılabilir.
    category : str
        ``"tree" | "linear_decomp" | "linear_shrinkage" | "seq" | "stat" | "benchmark"``.
    role : str
        ``"candidate" | "benchmark" | "legacy"``.
    ensemble_eligible : bool
        Varsayılan ensemble'a girip girmeyeceği.
    requires : tuple[str, ...]
        Modelin gerek duyduğu opsiyonel pip paketleri (importlib ile sınanır).
    target_modes : tuple[str, ...]
        Desteklenen target semantiği.
    needs_config_keys : tuple[str, ...]
        ``model_config`` dict'inden okunacak alt-sözlük adları (örn. ``("lstm",)``).
    description : str
        Kısa açıklama; ``--list-models`` çıktısında kullanılır.
    """

    name: str
    factory: Callable[..., Any]
    category: str
    role: str = "candidate"
    ensemble_eligible: bool = True
    requires: tuple[str, ...] = ()
    target_modes: tuple[str, ...] = ("return", "log_return", "price")
    needs_config_keys: tuple[str, ...] = ()
    description: str = ""
    default_candidate: bool = False


_REGISTRY: dict[str, ModelSpec] = {}
_DISCOVERY_DONE: bool = False


def register_model(spec: ModelSpec) -> ModelSpec:
    """Modeli registry'e ekle.

    Aynı isimle ikinci kez kayıt denemesi `ValueError` fırlatır — sessiz
    override engellenir.
    """
    if spec.name in _REGISTRY:
        raise ValueError(
            f"Model '{spec.name}' zaten kayıtlı (var olan: "
            f"{_REGISTRY[spec.name].factory!r})"
        )
    _REGISTRY[spec.name] = spec
    return spec


def unregister(name: str) -> None:
    """Test fixture cleanup için. Üretim kodunda kullanılmamalı."""
    _REGISTRY.pop(name, None)


def reset_registry() -> None:
    """Test izolasyonu için tam temizlik."""
    global _DISCOVERY_DONE
    _REGISTRY.clear()
    _DISCOVERY_DONE = False


def get_spec(name: str) -> ModelSpec:
    ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(
            f"Bilinmeyen model: '{name}'. Kayıtlı modeller: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def has_spec(name: str) -> bool:
    ensure_loaded()
    return name in _REGISTRY


def all_specs(
    *,
    role: str | None = None,
    category: str | None = None,
    ensemble_only: bool = False,
    target_mode: str | None = None,
) -> list[ModelSpec]:
    """Registry'i filtreleyerek listele.

    Filtre yokken tüm kayıtlı modeller döner. Sıralama: model adına göre.
    """
    ensure_loaded()
    out: Iterable[ModelSpec] = _REGISTRY.values()
    if role is not None:
        out = [s for s in out if s.role == role]
    if category is not None:
        out = [s for s in out if s.category == category]
    if ensemble_only:
        out = [s for s in out if s.ensemble_eligible]
    if target_mode is not None:
        out = [s for s in out if target_mode in s.target_modes]
    return sorted(out, key=lambda s: s.name)


def is_available(spec: ModelSpec) -> tuple[bool, str]:
    """Optional dep'leri import edilebilir mi diye sına.

    Returns
    -------
    (ok, reason) :
        ``ok=True`` → tüm `requires` import edilebildi.
        ``ok=False`` → eksik olan ilk paketin adı ve hata mesajı.
    """
    for pkg in spec.requires:
        try:
            importlib.import_module(pkg)
        except ImportError as exc:
            return False, f"{pkg} eksik ({exc})"
    return True, ""


def ensure_loaded() -> None:
    """`src.models` paketindeki tüm model modüllerini import et.

    Idempotent: ilk çağrıda keşif yapar, sonraki çağrılarda no-op.
    Bir modül `ImportError` atarsa diğerleri yine yüklenir; sessiz başarısızlık
    yerine `warnings.warn` ile bildirilir.
    """
    global _DISCOVERY_DONE
    if _DISCOVERY_DONE:
        return
    _DISCOVERY_DONE = True

    try:
        models_pkg = importlib.import_module("src.models")
    except ImportError as exc:  # pragma: no cover - paket olmadan zaten çökerdi
        warnings.warn(f"src.models import edilemedi: {exc}")
        return

    discover = getattr(models_pkg, "_discover_models", None)
    if discover is not None:
        discover()
