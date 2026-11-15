# -*- coding: utf-8 -*-
"""Taşınabilir model yolu yardımcıları (tek kaynak).

`best_models.model_path` / `experiments.model_path` eskiden başka makineden
kalan mutlak yollar (``D:\\...\\outputs\\TUPRS\\runs\\...``) içeriyordu. Yazarken
proje köküne göre GÖRECELİ (``outputs/...``, `/` ayraçlı) saklanır; okurken
projeye göre çözülür. Başka makineden kalan mutlak yollar `outputs` segmenti
üzerinden yerel outputs köküne yeniden eşlenir.
"""
from __future__ import annotations

import os
import re
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIRNAME = "outputs"

_SEP_RE = re.compile(r"[\\/]+")
_WIN_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_absolute_any(path: str) -> bool:
    """POSIX, Windows sürücü (`D:\\`) veya UNC mutlak yolu mu (host'tan bağımsız)."""
    return os.path.isabs(path) or bool(_WIN_ABS_RE.match(path)) or path.startswith(("\\\\", "//"))


def _outputs_tail(path: str) -> Optional[str]:
    """Yoldaki SON `outputs` segmentinden sonrasını `/` ile döner (yoksa None)."""
    parts = [p for p in _SEP_RE.split(path) if p]
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == OUTPUTS_DIRNAME:
            tail = parts[idx + 1:]
            return "/".join([OUTPUTS_DIRNAME, *tail])
    return None


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:  # farklı sürücüler
        return False


def to_storable_path(path: object, project_root: Optional[str] = None) -> str:
    """Yazma tarafı: proje kökü altındaki mutlak yolu göreceli (`/` ayraçlı) yapar.

    Boş değer -> "". Proje dışındaki mutlak yollar olduğu gibi kalır (yerel
    olarak yeniden eşlenemez). Göreceli yollarda yalnız ayraçlar normalize edilir.
    İdempotenttir.
    """
    text = str(path or "").strip()
    if not text:
        return ""
    root = project_root or PROJECT_ROOT
    if not _is_absolute_any(text):
        return text.replace("\\", "/")
    if os.path.isabs(text) and _inside(text, root):
        return os.path.relpath(text, root).replace("\\", "/")
    return text


def resolve_model_path(
    stored: object,
    project_root: Optional[str] = None,
    outputs_root: Optional[str] = None,
) -> str:
    """Okuma tarafı: saklanan yolu yerel dosya sistemi yoluna çözer.

    - Göreceli -> `project_root` ile birleştirilir (`outputs/...` ve
      `outputs_root` verilmişse onun altına).
    - Mutlak ve mevcut -> aynen.
    - Mutlak ve mevcut değil + `outputs` segmenti var -> yerel outputs köküne
      yeniden eşlenir (mevcutsa); yoksa orijinal döner.
    """
    text = str(stored or "").strip()
    if not text:
        return ""
    root = project_root or PROJECT_ROOT
    out_root = outputs_root or os.path.join(root, OUTPUTS_DIRNAME)
    if not _is_absolute_any(text):
        rel = text.replace("\\", "/")
        if outputs_root and rel.split("/", 1)[0].lower() == OUTPUTS_DIRNAME:
            rel = rel.split("/", 1)[1] if "/" in rel else ""
            return os.path.normpath(os.path.join(out_root, *rel.split("/"))) if rel else out_root
        return os.path.normpath(os.path.join(root, *rel.split("/")))
    if os.path.exists(text):
        return text
    tail = _outputs_tail(text)
    if tail is None:
        return text
    rest = tail.split("/")[1:]
    candidate = os.path.normpath(os.path.join(out_root, *rest)) if rest else out_root
    return candidate if os.path.exists(candidate) else text


def migrate_stored_path(path: object, project_root: Optional[str] = None) -> str:
    """Migrasyon: `outputs` segmenti olan mutlak yolu `outputs/...` göreceliye çevirir.

    Mutlak yol bu makinede VAR ve proje kökü DIŞINDAysa dokunulmaz (kullanıcının
    harici outputs'u). Diğer tüm durumlarda değişmez (idempotent).
    """
    text = str(path or "").strip()
    if not text or not _is_absolute_any(text):
        return text
    tail = _outputs_tail(text)
    if tail is None:
        return text
    root = project_root or PROJECT_ROOT
    if os.path.exists(text) and not _inside(text, root):
        return text
    return tail
