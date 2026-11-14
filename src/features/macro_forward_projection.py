# -*- coding: utf-8 -*-
"""
macro_forward_projection.py - Recursive forecast icin macro seriler ileri projection.

Sprint 5 (2026-05-25) Plan A5.2 -> P1 deep-audit (2026-11) revizyonu:
  Eski surum her macro seriye ARIMA(1,1,1) fit ediyordu; ancak *_Return/*_Change
  serileri zaten durgun (stationary) oldugu icin bu asiri farklama (over-differencing)
  anlamsiz projeksiyon uretiyordu. Ayrica adim basina ARIMA refit yavasti.

  Simdi (martingale baseline, tahmin iddiasi yok):
    * ``*_Return`` / ``*_Change`` kolonlari (getiri/degisim) -> 0.0 (degisim yok)
    * diger (seviye) kolonlar -> son gozlenen deger (random walk / ffill)

Kullanim:
    projector = MacroForwardProjector()
    new_frame = projector.project_last_row(frame, target_date=ts)

Donus:
    frame'in son satirinda macro sutunlari projection sonuclariyla
    guncellenmis. Diger satirlar dokunulmaz.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


# Bilinen macro seviye/getiri kolonlari (data/wiki uyumu). Sonek kurali
# (_Return/_Change) bu listeye ek olarak genel olarak da uygulanir.
_KNOWN_MACRO_COLUMNS = {
    "USDTRY", "USDTRY_Return",
    "BIST100", "BIST100_Return",
    "VIX",
    "INTEREST_RATE",
    "CPI",
    "BRENT",
    "GOLD",
    "EURTRY",
    "BIST_BANK_Return", "BIST_HOLD_Return", "BIST_SANAYI_Return",
    "BIST_INSAAT_Return", "BIST_GIDAI_Return", "BIST_KIMYA_Return",
    "BIST_METALU_Return", "BIST_TICRT_Return", "BIST_ULASTRMA_Return",
    "BIST_TURIZM_Return", "BIST_BILMEMURENGEL_Return",
}

# Getiri/degisim sonekleri -> projeksiyon 0.0.
_FLOW_SUFFIXES = ("_Return", "_Change")

# Hissenin KENDI getiri kolonlari (sonek eslesse de macro degildir; dokunulmaz).
_OWN_RETURN_COLUMNS = {"Return", "Log_Return", "Relative_Strength", "Sector_Relative_Strength"}


def _is_flow_column(col: str) -> bool:
    """Getiri/degisim kolonu mu (``*_Return`` / ``*_Change``)?"""
    return col.endswith(_FLOW_SUFFIXES) and col not in _OWN_RETURN_COLUMNS


class MacroForwardProjector:
    """Recursive forecast row'una macro projection uygular.

    Kolon secimi: ``columns`` verilirse o liste; aksi halde frame icinde
    ``_KNOWN_MACRO_COLUMNS`` uyelari + sonek ile eslesen (``*_Return``/``*_Change``,
    hissenin kendi getirileri haric) kolonlar.
    """

    def __init__(self, columns: Optional[Iterable[str]] = None) -> None:
        self.columns = set(columns) if columns is not None else None

    def _resolve_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in frame.columns]
        return [
            c
            for c in frame.columns
            if c in _KNOWN_MACRO_COLUMNS or _is_flow_column(str(c))
        ]

    def project_last_row(
        self,
        frame: pd.DataFrame,
        *,
        target_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Son satirin macro feature'lerini bir adim ileri projeksiyona alir."""
        if frame.empty:
            return frame
        out = frame.copy()
        cols = self._resolve_columns(out)
        if not cols:
            return out
        last_idx = out.index[-1]
        # Historic series: son satir HARIC (projection input).
        hist = out.iloc[:-1]
        for col in cols:
            if _is_flow_column(str(col)):
                out.at[last_idx, col] = 0.0
                continue
            series = hist[col].dropna() if col in hist.columns else None
            if series is None or series.empty:
                continue
            out.at[last_idx, col] = float(series.iloc[-1])
        return out
