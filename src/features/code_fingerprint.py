# -*- coding: utf-8 -*-
"""Ozellik uretim kodunun parmak izi (onbellek otomatik gecersiz kilma).

Ozellik onbelleklerinin (Kol-A `feature_cache.py`, Kol-B `pooled_feature_cache.py`)
anahtarina eklenir: ozellik ciktisini belirleyen kaynak dosyalardan biri degisirse
anahtar degisir -> eski girdiler kullanilmaz. Elle surum artirmaya gerek kalmaz.

Dosya listesi AST ile bulunur: kok modullerden baslayip `src.*` importlari
(fonksiyon-ici importlar dahil) transitif olarak izlenir. Paket `__init__.py`
dosyalari dahil degildir (yalniz yeniden-export). Kaynak baytlari CRLF->LF
normallestirilir (checkout satir-sonu farki sahte gecersiz kilma uretmez).
Ayrica pandas/numpy/ta surumleri dahildir (hesaplama davranisi surume bagli).
"""
from __future__ import annotations

import ast
import hashlib
import logging
from functools import lru_cache
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Kol-B (sembol basi): FeaturePipeline cikti belirler.
POOLED_ROOTS: tuple[str, ...] = ("src/features/feature_pipeline.py",)
# Kol-A: FeaturePipeline + MacroPipeline.
PIPELINE_ROOTS: tuple[str, ...] = (
    "src/features/feature_pipeline.py", "src/features/macro_pipeline.py")
LIBRARIES: tuple[str, ...] = ("pandas", "numpy", "ta")


def _module_file(dotted: str) -> Path | None:
    """`src.a.b` -> proje icindeki .py dosyasi (paket __init__ haric) veya None."""
    if not dotted.startswith("src"):
        return None
    path = _PROJECT_ROOT.joinpath(*dotted.split(".")).with_suffix(".py")
    return path if path.is_file() else None


def _imported_modules(path: Path) -> set[str]:
    """Dosyadaki tum `src.*` import hedefleri (noktali ad)."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    found: set[str] = set()
    pkg = path.relative_to(_PROJECT_ROOT).with_suffix("").parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(pkg[: len(pkg) - node.level + 1])
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            found.add(mod)
            # `from src.features import macro_transforms` -> alt modul de aday
            found.update(f"{mod}.{a.name}" for a in node.names)
    return found


@lru_cache(maxsize=None)
def module_closure(roots: tuple[str, ...] = POOLED_ROOTS) -> tuple[str, ...]:
    """Kok dosyalardan erisilen proje kaynak dosyalari (goreli, posix, sirali)."""
    seen: dict[Path, None] = {}
    stack = [_PROJECT_ROOT / r for r in roots]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"parmak izi kok/bagimlilik dosyasi yok: {path}")
        seen[path] = None
        for dotted in _imported_modules(path):
            dep = _module_file(dotted)
            if dep is not None and dep not in seen:
                stack.append(dep)
    return tuple(sorted(p.relative_to(_PROJECT_ROOT).as_posix() for p in seen))


def _library_versions(libs: tuple[str, ...]) -> list[str]:
    out = []
    for lib in libs:
        try:
            out.append(f"{lib}=={metadata.version(lib)}")
        except metadata.PackageNotFoundError:
            logger.warning("code_fingerprint: %s kurulu degil; surum 'missing' sayildi", lib)
            out.append(f"{lib}==missing")
    return out


@lru_cache(maxsize=None)
def code_fingerprint(
    roots: tuple[str, ...] = POOLED_ROOTS, libs: tuple[str, ...] = LIBRARIES,
) -> str:
    """sha256 hex: kaynak icerikleri + kutuphane surumleri. Surec basina bir kez hesaplanir."""
    h = hashlib.sha256()
    for rel in module_closure(roots):
        data = (_PROJECT_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        h.update(rel.encode("utf-8") + b"\0" + hashlib.sha256(data).digest())
    h.update("|".join(_library_versions(libs)).encode("utf-8"))
    return h.hexdigest()
