# -*- coding: utf-8 -*-
"""Dynamic stock-to-sector index resolution from the BIST universe file."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

DEFAULT_SECTOR_INDEX = "BIST100"
SUPPORTED_SECTOR_INDEXES = {
    "BIST100",
    "XBANK",
    "XUSIN",
    "XHOLD",
    "XULAS",
    "XTCRT",
    "XTEK",
}


@dataclass(frozen=True)
class SectorMapping:
    symbol: str
    sector_index: str
    sector: Optional[str]
    status: str
    reason: str
    source: str
    universe_file: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_symbol(symbol: str | None) -> str:
    clean = str(symbol or "").strip().upper()
    if clean.endswith(".IS"):
        clean = clean[:-3]
    return clean


def resolve_sector_mapping(
    symbol: str | None,
    universe_file: str | Path | None,
) -> SectorMapping:
    normalized_symbol = normalize_symbol(symbol)
    universe_path = Path(universe_file) if universe_file else None
    if universe_path is None:
        return _fallback(normalized_symbol, None, "missing_universe_file")
    if not universe_path.exists():
        return _fallback(normalized_symbol, str(universe_path), "missing_universe_file")

    frame = _read_universe(universe_path)
    if frame is None:
        return _fallback(normalized_symbol, str(universe_path), "universe_read_failed")

    symbol_column = _first_existing_column(
        frame, ("Symbol", "symbol", "SYMBOL", "Ticker", "ticker")
    )
    if symbol_column is None:
        return _fallback(normalized_symbol, str(universe_path), "missing_symbol_column")

    frame = frame.copy()
    frame["_normalized_symbol"] = frame[symbol_column].map(normalize_symbol)
    matches = frame[frame["_normalized_symbol"] == normalized_symbol]
    if matches.empty:
        return _fallback(normalized_symbol, str(universe_path), "symbol_not_found")

    row = matches.iloc[0]
    sector = _optional_value(
        row, _first_existing_column(frame, ("Sector", "sector", "Sektor", "sektor"))
    )
    sector_index_column = _first_existing_column(
        frame,
        ("Sector_Index", "sector_index", "SECTOR_INDEX", "SectorIndex", "sectorIndex"),
    )
    sector_index = _normalize_sector_index(_optional_value(row, sector_index_column))
    if sector_index not in SUPPORTED_SECTOR_INDEXES:
        return SectorMapping(
            symbol=normalized_symbol,
            sector_index=DEFAULT_SECTOR_INDEX,
            sector=sector,
            status="fallback",
            reason="missing_or_unsupported_sector_index",
            source="universe_csv",
            universe_file=str(universe_path),
        )

    return SectorMapping(
        symbol=normalized_symbol,
        sector_index=sector_index,
        sector=sector,
        status="matched",
        reason="matched_universe_sector_index",
        source="universe_csv",
        universe_file=str(universe_path),
    )


def sector_return_column(sector_index: str | None) -> str:
    normalized = _normalize_sector_index(sector_index)
    if normalized == DEFAULT_SECTOR_INDEX:
        return "BIST100_Return"
    return f"{normalized}_Return"


def _fallback(symbol: str, universe_file: str | None, reason: str) -> SectorMapping:
    return SectorMapping(
        symbol=symbol,
        sector_index=DEFAULT_SECTOR_INDEX,
        sector=None,
        status="fallback",
        reason=reason,
        source="fallback",
        universe_file=universe_file,
    )


def _normalize_sector_index(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized.endswith(".IS"):
        normalized = normalized[:-3]
    return normalized


def _optional_value(row: pd.Series, column: str | None) -> Optional[str]:
    if column is None:
        return None
    value = row.get(column)
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    columns = {str(column).lstrip("\ufeff").strip(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _read_universe(path: Path) -> Optional[pd.DataFrame]:
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin5"):
        try:
            frame = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
            frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
            return frame
        except Exception:
            continue
    return None
