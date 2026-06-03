# -*- coding: utf-8 -*-
"""Nightly universe scoring orchestration — E2 Faz 5.

Akis (kilitli karar): per-query egitim YOK. Gecelik batch:
  1. pooled panel + cross-sectional hedef + cs-features.
  2. GlobalPooledModel'i TUM gecmise egit (serving = en guncel modele kadar).
  3. en guncel tarihin evren satirlarini skorla -> peer_score/percentile/label.
  4. her sembole segment (likidite/vol/sektor) + segment ICIR ata.
  5. confidence = segment_ICIR x tradability kapilari.
  6. peer_scores satirlari PeerStore'a yazilir; API okur.

Segment ICIR referansi Faz 6 OOS calismasindan gelir (likidite kovasi birincil
ayirici). Burada `segment_icir_map` (liq_bucket -> ICIR) DISARIDAN verilir;
boylece bu fonksiyon saf/testable kalir, agir OOS hesabi CLI'de yapilir.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

from src.serving.confidence import ConfidenceThresholds, peer_confidence
from src.serving.peer_scoring import PeerScoringConfig, score_latest_universe


def assemble_peer_table(
    model,
    panel_latest: pd.DataFrame,
    feature_cols: list[str],
    seg_table: pd.DataFrame,
    segment_icir_map: Mapping[str, float],
    *,
    tradable_for: Optional[Callable[[str], bool]] = None,
    stale_for: Optional[Callable[[str], bool]] = None,
    icir_segment_col: str = "liq_bucket",
    scoring_cfg: Optional[PeerScoringConfig] = None,
    thr: Optional[ConfidenceThresholds] = None,
) -> pd.DataFrame:
    """En guncel evreni skorla + segment + confidence -> PeerStore'a hazir tablo.

    segment_icir_map : `icir_segment_col` kovasi -> ICIR (Faz 6 referansi).
    tradable_for/stale_for : symbol -> bool (varsayilan: hepsi tradable, taze).
    Returns DataFrame[symbol, as_of_date, peer_*, raw_pred, universe_size,
                      segment_liq/vol/sector, segment_icir,
                      confidence_label/reasons/warnings].
    """
    scoring_cfg = scoring_cfg or PeerScoringConfig()
    thr = thr or ConfidenceThresholds()
    tradable_for = tradable_for or (lambda s: True)
    stale_for = stale_for or (lambda s: False)

    peer = score_latest_universe(model, panel_latest, feature_cols, scoring_cfg)
    if peer.empty:
        return peer

    seg = seg_table.rename(columns={
        "liq_bucket": "segment_liq", "vol_bucket": "segment_vol",
        "sector": "segment_sector"})
    merged = peer.merge(seg, on="symbol", how="left")

    seg_key_col = {"liq_bucket": "segment_liq", "vol_bucket": "segment_vol",
                   "sector": "segment_sector"}.get(icir_segment_col, "segment_liq")
    merged["segment_icir"] = merged[seg_key_col].map(
        lambda b: float(segment_icir_map.get(b, float("nan"))) if b is not None else float("nan"))

    universe_ok = int(merged["universe_size"].iloc[0]) >= scoring_cfg.min_names
    labels, reasons, warns = [], [], []
    for _, r in merged.iterrows():
        c = peer_confidence(
            r["segment_icir"], r["peer_label"],
            tradable=bool(tradable_for(r["symbol"])),
            stale=bool(stale_for(r["symbol"])),
            universe_ok=universe_ok, thr=thr)
        labels.append(c.label)
        reasons.append(c.reasons)
        warns.append(c.warnings)
    merged["confidence_label"] = labels
    merged["confidence_reasons"] = reasons
    merged["confidence_warnings"] = warns
    return merged


def liqlog_floor_from_turnover(min_turnover_tl: float) -> float:
    """TL/gun ciro tabanini loader liq_log birimine cevir (liq_log=log1p(ciro)).

    Tradability kapisi icin: sembolun medyan liq_log'u bu esigin altinda ise
    'islem yapilamaz' -> confidence low (Faz 6 gerginligi: en az likit = en guclu
    sinyal ama islem zor). 0/negatif -> 0 (kapi etkisiz)."""
    return float(np.log1p(max(0.0, float(min_turnover_tl))))


def segment_icir_from_table(seg_ic_table: pd.DataFrame) -> dict[str, float]:
    """Faz 6 segment_cross_sectional_ic ciktisindan {segment: icir} sozlugu."""
    out = {}
    for _, r in seg_ic_table.iterrows():
        v = r.get("icir")
        out[str(r["segment"])] = float(v) if v is not None and np.isfinite(v) else float("nan")
    return out
