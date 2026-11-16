# -*- coding: utf-8 -*-
"""Evrende olup birincil veri klasorunde CSV'si olmayan sembolleri yedek klasorden tohumlar.

Gece tazelemesi yalniz `data_dir/*.csv` dosyalarini gunceller. Evrendeki (Active) bir sembolun
birincil CSV'si yoksa ve yalniz `--fallback-dir` icinde duruyorsa, o sembol hic guncellenmez ve
bayat kalir (2026-11-16: KCHOL, SASA, TUPRS, SAHOL, ENKAI, LOGO, ARDYZ). Bu modul yedek dosyayi
birincil klasore KOPYALAR (yedek yerinde kalir); ardindan normal artimli tazeleme isler.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_EXCLUDE_STEMS = {"bist_universe", "advisory_history"}


def csv_symbols(data_dir: str) -> list[str]:
    """Birincil klasordeki sembol CSV'leri (evren/kayit dosyalari haric), sirali."""
    stems = (os.path.basename(p)[:-4] for p in glob.glob(os.path.join(data_dir, "*.csv")))
    return sorted(s for s in stems if s not in _EXCLUDE_STEMS)


def seed_missing_from_fallback(
    data_dir: str, universe_file: str, fallback_dirs: Sequence[str]
) -> list[str]:
    """Kopyalanan sembolleri dondurur. Evren dosyasi okunamazsa hicbir sey yapmaz (loglar)."""
    try:
        uni = pd.read_csv(universe_file)
    except (OSError, ValueError) as exc:
        logger.warning("evren dosyasi okunamadi (%s): %s; tohumlama atlandi", universe_file, exc)
        return []
    status = uni.get("Status", pd.Series("active", index=uni.index)).astype(str).str.lower()
    symbols = uni.loc[status.eq("active"), "Symbol"].astype(str).str.strip().str.upper()
    seeded = []
    for sym in symbols:
        target = os.path.join(data_dir, f"{sym}.csv")
        if os.path.exists(target):
            continue
        for fb in fallback_dirs:
            src = os.path.join(fb, f"{sym}.csv")
            if os.path.exists(src):
                shutil.copy2(src, target)
                seeded.append(sym)
                break
    return seeded
