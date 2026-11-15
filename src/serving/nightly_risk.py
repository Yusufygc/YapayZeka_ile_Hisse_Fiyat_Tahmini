# -*- coding: utf-8 -*-
"""Gecelik kosunun risk adimi: hesaplama + kapi-degisimi tespiti + durum dosyasi.

`nightly.py` 500 satir sinirinda kaldigi icin risk ile ilgili yardimcilar burada.
Risk modelinin kendisi `src/risk/`; kapi mantigi degismez.
"""
from __future__ import annotations

import argparse
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from src.risk.features import RISK_HORIZON
from src.risk.serving import (
    compute_risk_safe, gate_flip_warning, previous_risk_method, risk_run_status)


def horizon_arg(value: str) -> int:
    """`--horizon` argparse tipi: risk modeli ve Kol-B ceyreklik tablosu 5 gunluktur.

    Farkli deger sessizce yanlis (5g) risk bandi uretirdi; bu yuzden acik CLI hatasi.
    """
    try:
        h = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"gecersiz tamsayi: {value!r}") from exc
    if h != RISK_HORIZON:
        raise argparse.ArgumentTypeError(
            f"--horizon yalniz {RISK_HORIZON} desteklenir (risk modeli ve Kol-B ceyreklik "
            f"tablosu {RISK_HORIZON} gunluktur); verilen: {h}")
    return h


def run_risk_step(
    panel: pd.DataFrame, as_of: pd.Timestamp, prev_run: Optional[dict[str, Any]],
    *, log: Callable[[str], None], status: Any = None,
    quarantined: Optional[Mapping[str, int]] = None,
) -> tuple[Optional[pd.DataFrame], dict[str, Any], list[str]]:
    """Risk tablosu + config ozeti + uyarilar.

    - `status` (NightlyRunStatus) verilirse `risk={status, method, error}` yazilir
      (`ok` | `fallback` | `failed`) -> `/health` ve operator gorur.
    - `quarantined` ({sembol: atilan satir}) risk tahmini olmayan sembollerin nedenini
      siniflar (summary['missing']).
    - `prev_run` (onceki kosunun basligi) ile karsilastirilip kapi degisimi
      (har_x_ridge <-> naive_fallback) loglanir ve config uyarisina eklenir.
    """
    table, summary = compute_risk_safe(panel, as_of, log=log, quarantined=quarantined)
    info = risk_run_status(summary)
    prev = previous_risk_method(prev_run)
    flip = gate_flip_warning(prev, info["method"])
    summary = {**summary, "previous_method": prev, "gate_flip": flip is not None}
    if flip:
        log(f"UYARI {flip}")
    if info["status"] == "failed":
        log(f"UYARI risk adimi basarisiz: {info['error']}; risk_*/fiyat bandi bos")
    if status is not None:
        status.update(risk=info)
    return table, summary, ([flip] if flip else [])
