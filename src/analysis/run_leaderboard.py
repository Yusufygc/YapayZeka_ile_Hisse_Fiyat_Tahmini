# -*- coding: utf-8 -*-
"""Run-level leaderboard and durability diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

REQUIRED_RUN_FILES = (
    "run_manifest.json",
    "csv/backtest_report_wf.csv",
    "csv/backtest_report_final_holdout.csv",
    "csv/validation_protocol_report.csv",
)
TRADE_BAND_MIN = 8
TRADE_BAND_MAX = 20
BENCHMARK_CLONE_TOLERANCE = 0.001
UNSTABLE_GAP_THRESHOLD = 0.25
DEFAULT_LONG_HISTORY_YEARS = 10.0
DEFAULT_SHORT_HISTORY_YEARS = 5.0
RELIABILITY_SORT_ORDER = {
    "stable": 0,
    "defensive": 1,
    "research_candidate": 2,
    "unstable": 3,
    "invalid": 4,
    "incomplete": 5,
}


def build_run_leaderboard(
    *,
    outputs_base: str | Path,
    symbol: str,
    data_dir: str | Path | None = None,
    sector_file: str | Path | None = None,
    min_history_years: float | None = DEFAULT_LONG_HISTORY_YEARS,
    long_history_years: float | None = None,
    short_history_years: float = DEFAULT_SHORT_HISTORY_YEARS,
) -> pd.DataFrame:
    """Build a model/run decision table from run-scoped output directories."""
    outputs_base = Path(outputs_base)
    symbol = str(symbol).upper()
    metadata = _symbol_metadata(
        symbol=symbol,
        data_dir=data_dir,
        sector_file=sector_file,
        long_history_years=_resolve_long_history_years(
            min_history_years=min_history_years,
            long_history_years=long_history_years,
        ),
        short_history_years=short_history_years,
    )
    runs_dir = outputs_base / symbol / "runs"
    rows: List[Dict[str, Any]] = []
    if not runs_dir.exists():
        return pd.DataFrame(columns=_leaderboard_columns())

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        rows.extend(_rows_for_run(symbol=symbol, run_dir=run_dir, metadata=metadata))

    frame = _rank_leaders(pd.DataFrame(rows, columns=_leaderboard_columns()))
    if not frame.empty:
        frame["_reliability_sort"] = (
            frame["leader_reliability_class"].map(RELIABILITY_SORT_ORDER).fillna(99)
        )
        frame.sort_values(
            by=["_reliability_sort", "final_excess_vs_buyhold", "final_net_return"],
            ascending=[True, False, False],
            inplace=True,
            kind="stable",
        )
        frame.drop(columns=["_reliability_sort"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return frame


def build_multi_symbol_leaderboard(
    *,
    outputs_base: str | Path,
    symbols: Sequence[str],
    data_dir: str | Path | None = None,
    sector_file: str | Path | None = None,
    min_history_years: float | None = DEFAULT_LONG_HISTORY_YEARS,
    long_history_years: float | None = None,
    short_history_years: float = DEFAULT_SHORT_HISTORY_YEARS,
) -> pd.DataFrame:
    """Build a combined run-level leaderboard for multiple symbols."""
    frames = [
        build_run_leaderboard(
            outputs_base=outputs_base,
            symbol=symbol,
            data_dir=data_dir,
            sector_file=sector_file,
            min_history_years=min_history_years,
            long_history_years=long_history_years,
            short_history_years=short_history_years,
        )
        for symbol in symbols
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=_leaderboard_columns())
    normalized = [frame.dropna(axis=1, how="all") for frame in frames]
    result = pd.concat(normalized, ignore_index=True)
    for column in _leaderboard_columns():
        if column not in result:
            result[column] = None
    return result[_leaderboard_columns()]


def list_symbols_with_runs(outputs_base: str | Path) -> List[str]:
    """Return output symbols that contain a runs directory."""
    base = Path(outputs_base)
    if not base.exists():
        return []
    return sorted(
        path.name.upper() for path in base.iterdir() if path.is_dir() and (path / "runs").is_dir()
    )


def build_sector_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize reliability and repeated model-family patterns by sector."""
    columns = [
        "sector",
        "symbol_count",
        "run_row_count",
        "history_bucket_breakdown",
        "stable_count",
        "defensive_count",
        "research_candidate_count",
        "unstable_count",
        "invalid_count",
        "incomplete_count",
        "incomplete_invalid_rate",
        "avg_wf_final_net_gap",
        "repeated_model_families",
        "top_model_family",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    working = frame.copy()
    working["sector"] = working["sector"].fillna("unknown")
    working["history_bucket"] = working.get("history_bucket", "unknown")
    working["model_family"] = working["model"].map(_model_family)
    rows: List[Dict[str, Any]] = []
    for sector, group in working.groupby("sector", dropna=False):
        reliability_counts = group["leader_reliability_class"].value_counts().to_dict()
        history_counts = group["history_bucket"].fillna("unknown").value_counts().to_dict()
        family_counts = group["model_family"].value_counts().to_dict()
        incomplete_invalid = int(reliability_counts.get("incomplete", 0)) + int(
            reliability_counts.get("invalid", 0)
        )
        repeated = {
            family: int(count)
            for family, count in family_counts.items()
            if family and int(count) > 1
        }
        top_family = None
        if family_counts:
            top_family = sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        rows.append(
            {
                "sector": sector,
                "symbol_count": int(group["symbol"].nunique()),
                "run_row_count": int(len(group)),
                "history_bucket_breakdown": "|".join(
                    f"{bucket}:{int(count)}" for bucket, count in sorted(history_counts.items())
                ),
                "stable_count": int(reliability_counts.get("stable", 0)),
                "defensive_count": int(reliability_counts.get("defensive", 0)),
                "research_candidate_count": int(reliability_counts.get("research_candidate", 0)),
                "unstable_count": int(reliability_counts.get("unstable", 0)),
                "invalid_count": int(reliability_counts.get("invalid", 0)),
                "incomplete_count": int(reliability_counts.get("incomplete", 0)),
                "incomplete_invalid_rate": round(incomplete_invalid / len(group), 6),
                "avg_wf_final_net_gap": _mean_or_none(group["wf_final_net_gap"]),
                "repeated_model_families": "|".join(
                    f"{family}:{count}" for family, count in sorted(repeated.items())
                ),
                "top_model_family": top_family,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("sector").reset_index(drop=True)


def build_history_effect_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize reliability, model-family, and WF/final gap by history bucket."""
    columns = [
        "history_bucket",
        "symbol_count",
        "run_row_count",
        "stable_count",
        "defensive_count",
        "research_candidate_count",
        "unstable_count",
        "invalid_count",
        "incomplete_count",
        "incomplete_invalid_rate",
        "avg_wf_final_net_gap",
        "top_model_family",
        "model_family_breakdown",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    working = frame.copy()
    working["history_bucket"] = working.get("history_bucket", "unknown")
    working["history_bucket"] = working["history_bucket"].fillna("unknown")
    working["model_family"] = working["model"].map(_model_family)
    rows: List[Dict[str, Any]] = []
    for bucket, group in working.groupby("history_bucket", dropna=False):
        reliability_counts = group["leader_reliability_class"].value_counts().to_dict()
        family_counts = group["model_family"].value_counts().to_dict()
        incomplete_invalid = int(reliability_counts.get("incomplete", 0)) + int(
            reliability_counts.get("invalid", 0)
        )
        top_family = None
        if family_counts:
            top_family = sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        rows.append(
            {
                "history_bucket": bucket,
                "symbol_count": int(group["symbol"].nunique()),
                "run_row_count": int(len(group)),
                "stable_count": int(reliability_counts.get("stable", 0)),
                "defensive_count": int(reliability_counts.get("defensive", 0)),
                "research_candidate_count": int(reliability_counts.get("research_candidate", 0)),
                "unstable_count": int(reliability_counts.get("unstable", 0)),
                "invalid_count": int(reliability_counts.get("invalid", 0)),
                "incomplete_count": int(reliability_counts.get("incomplete", 0)),
                "incomplete_invalid_rate": round(incomplete_invalid / len(group), 6),
                "avg_wf_final_net_gap": _mean_or_none(group["wf_final_net_gap"]),
                "top_model_family": top_family,
                "model_family_breakdown": "|".join(
                    f"{family}:{int(count)}" for family, count in sorted(family_counts.items())
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("history_bucket").reset_index(drop=True)


def leaderboard_to_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return JSON-safe records from a leaderboard frame."""
    records: List[Dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        records.append({key: None if _is_null(value) else value for key, value in raw.items()})
    return records


def _rows_for_run(
    *,
    symbol: str,
    run_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    missing = _missing_required_files(run_dir)
    manifest = _read_manifest(run_dir / "run_manifest.json")
    wf_report = _read_report(run_dir / "csv" / "backtest_report_wf.csv")
    final_report = _read_report(run_dir / "csv" / "backtest_report_final_holdout.csv")
    models = _models_for_run(
        manifest=manifest,
        reports=(wf_report, final_report),
        run_dir=run_dir,
    )

    rows = []
    for model in models:
        wf_row = _report_row_for_model(wf_report, model)
        final_row = _report_row_for_model(final_report, model)
        row_missing = list(missing)
        if final_report is not None and final_row is None:
            row_missing.append("final_holdout_model_row")
        rows.append(
            _build_row(symbol, run_dir, model, wf_row, final_row, row_missing, metadata or {})
        )
    return rows


def _build_row(
    symbol: str,
    run_dir: Path,
    model: str,
    wf_row: Optional[pd.Series],
    final_row: Optional[pd.Series],
    missing: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    wf_net = _to_float(_value(wf_row, "Net_Return"))
    final_net = _to_float(_value(final_row, "Net_Return"))
    final_buyhold = _to_float(_value(final_row, "BuyHold_Return"))
    final_trade_count = _to_float(_value(final_row, "Trade_Count"))
    final_sharpe = _to_float(_value(final_row, "Sharpe"))
    wf_final_gap = _diff(wf_net, final_net)
    final_excess = _diff(final_net, final_buyhold)
    holdout_complete = not missing and final_row is not None
    trade_sufficient = _in_trade_band(final_trade_count)
    benchmark_clone = _is_benchmark_clone(final_net, final_buyhold, final_trade_count)
    reliability = _reliability_class(
        holdout_complete=holdout_complete,
        benchmark_clone=benchmark_clone,
        wf_net=wf_net,
        final_net=final_net,
        final_excess=final_excess,
        final_trade_count=final_trade_count,
        final_sharpe=final_sharpe,
        wf_final_gap=wf_final_gap,
    )
    return {
        "symbol": symbol,
        "run_id": run_dir.name,
        "model": model,
        "history_years": metadata.get("history_years"),
        "history_class": metadata.get("history_class"),
        "history_bucket": metadata.get("history_bucket"),
        "meets_10y_reference": metadata.get("meets_10y_reference"),
        "data_history_warning": metadata.get("data_history_warning"),
        "sector": metadata.get("sector"),
        "prediction_leader_rank": None,
        "trading_leader_rank": None,
        "wf_net_return": wf_net,
        "final_net_return": final_net,
        "final_buyhold_return": final_buyhold,
        "wf_final_net_gap": wf_final_gap,
        "final_excess_vs_buyhold": final_excess,
        "final_trade_count": final_trade_count,
        "final_sharpe": final_sharpe,
        "final_signal_diagnosis": _value(final_row, "Signal_Diagnosis"),
        "holdout_complete_flag": bool(holdout_complete),
        "trade_sufficiency_flag": bool(trade_sufficient),
        "benchmark_clone_flag": bool(benchmark_clone),
        "leader_reliability_class": reliability,
        "missing_required_files": "|".join(missing),
    }


def _reliability_class(
    *,
    holdout_complete: bool,
    benchmark_clone: bool,
    wf_net: Optional[float],
    final_net: Optional[float],
    final_excess: Optional[float],
    final_trade_count: Optional[float],
    final_sharpe: Optional[float],
    wf_final_gap: Optional[float],
) -> str:
    if not holdout_complete:
        return "incomplete"
    if benchmark_clone or final_trade_count is None or final_trade_count < 2:
        return "invalid"
    if _in_trade_band(final_trade_count):
        if _positive(final_excess) and _positive(wf_net):
            return "stable"
    if _positive(wf_net) and (final_net is not None and final_net <= 0):
        return "unstable"
    if _positive(wf_net) and wf_final_gap is not None and wf_final_gap >= UNSTABLE_GAP_THRESHOLD:
        return "unstable"
    if _in_trade_band(final_trade_count) and (
        final_net is not None
        and final_net >= 0
        and final_sharpe is not None
        and final_sharpe > 0
        and final_excess is not None
        and final_excess <= 0
    ):
        return "defensive"
    return "research_candidate"


def _read_report(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    attempts = (
        {"sep": None, "engine": "python"},
        {"sep": ";"},
        {},
    )
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            frame = pd.read_csv(path, **kwargs)
            if len(frame.columns) == 1 and ";" in str(frame.columns[0]):
                continue
            frame = frame.copy()
            frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
            return frame
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return None


def _read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _symbol_metadata(
    *,
    symbol: str,
    data_dir: str | Path | None,
    sector_file: str | Path | None,
    long_history_years: float,
    short_history_years: float,
) -> Dict[str, Any]:
    history_years, missing_history = _history_years(symbol=symbol, data_dir=data_dir)
    history_bucket = classify_history_bucket(
        history_years,
        data_dir_provided=data_dir is not None,
        missing_history=missing_history,
        long_history_years=long_history_years,
        short_history_years=short_history_years,
    )
    return {
        "history_years": history_years,
        "history_class": history_bucket,
        "history_bucket": history_bucket,
        "meets_10y_reference": bool(
            history_years is not None and history_years >= long_history_years
        ),
        "data_history_warning": data_history_warning(history_bucket),
        "sector": _sector_for_symbol(symbol=symbol, sector_file=sector_file),
    }


def _history_years(*, symbol: str, data_dir: str | Path | None) -> tuple[Optional[float], bool]:
    if data_dir is None:
        return None, False
    path = Path(data_dir) / f"{symbol}.csv"
    if not path.exists():
        return None, True
    frame = _read_table(path)
    if frame is None:
        return None, True
    date_column = _first_existing_column(frame, ("Date", "date", "Tarih", "tarih"))
    if date_column is None:
        return None, True
    dates = parse_history_dates(frame[date_column]).dropna()
    if dates.empty:
        return None, True
    return round(float((dates.max() - dates.min()).days / 365.25), 4), False


def parse_history_dates(values: Any) -> pd.Series:
    """Parse mixed BIST CSV dates without shifting ISO YYYY-MM-DD values."""
    raw = pd.Series(values)
    text = raw.astype(str).str.strip()
    iso_mask = text.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$", na=False)
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(
            raw.loc[iso_mask],
            errors="coerce",
            yearfirst=True,
        )
    if (~iso_mask).any():
        parsed.loc[~iso_mask] = pd.to_datetime(
            raw.loc[~iso_mask],
            errors="coerce",
            dayfirst=True,
        )
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw.loc[missing], errors="coerce")
    return parsed


def _resolve_long_history_years(
    *,
    min_history_years: float | None,
    long_history_years: float | None,
) -> float:
    if long_history_years is not None:
        return float(long_history_years)
    if min_history_years is not None:
        return float(min_history_years)
    return DEFAULT_LONG_HISTORY_YEARS


def classify_history_bucket(
    history_years: Optional[float],
    *,
    data_dir_provided: bool = True,
    missing_history: bool = False,
    long_history_years: float = DEFAULT_LONG_HISTORY_YEARS,
    short_history_years: float = DEFAULT_SHORT_HISTORY_YEARS,
) -> str:
    """Classify history length for diagnostics, never for automatic exclusion."""
    if history_years is None:
        return "missing_data" if data_dir_provided and missing_history else "unknown"
    if history_years >= long_history_years:
        return "long_history"
    if history_years >= short_history_years:
        return "mid_history"
    return "short_history"


def data_history_warning(history_bucket: str) -> str:
    warnings = {
        "long_history": "",
        "mid_history": "5-10 years of data; compare against long_history cohorts before blaming model quality.",
        "short_history": "Less than 5 years of data; higher regime and holdout uncertainty.",
        "missing_data": "CSV missing or date history is unreadable; skipped unless data is restored.",
        "unknown": "History was not evaluated because data_dir was not provided.",
    }
    return warnings.get(str(history_bucket), "")


def _sector_for_symbol(*, symbol: str, sector_file: str | Path | None) -> Optional[str]:
    if sector_file is None:
        return None
    path = Path(sector_file)
    if not path.exists():
        return None
    frame = _read_table(path)
    if frame is None:
        return None
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    symbol_column = _first_existing_column(
        frame, ("Symbol", "symbol", "SYMBOL", "Ticker", "ticker")
    )
    sector_column = _first_existing_column(
        frame, ("Sector", "sector", "Sektor", "sektor", "SECTOR")
    )
    sector_index_column = _first_existing_column(
        frame,
        ("Sector_Index", "sector_index", "SECTOR_INDEX", "SectorIndex", "sectorIndex"),
    )
    if symbol_column is None or (sector_column is None and sector_index_column is None):
        return None
    matches = frame[frame[symbol_column].astype(str).str.upper() == symbol.upper()]
    if matches.empty:
        return None
    value = matches.iloc[0].get(sector_column) if sector_column else None
    if pd.isna(value) or str(value).strip() == "":
        value = matches.iloc[0].get(sector_index_column) if sector_index_column else None
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    columns = {str(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin5"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception:
            continue
    return None


def _rank_leaders(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    complete_mask = frame["holdout_complete_flag"].fillna(False).astype(bool)
    if "wf_net_return" in frame:
        prediction_ranks = (
            frame.loc[complete_mask, "wf_net_return"]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype("Int64")
        )
        frame.loc[prediction_ranks.index, "prediction_leader_rank"] = prediction_ranks
    if {"final_excess_vs_buyhold", "final_net_return"}.issubset(frame.columns):
        eligible = frame.loc[complete_mask].copy()
        eligible["_trading_score"] = eligible["final_excess_vs_buyhold"].fillna(
            -1e9
        ) * 1000 + eligible["final_net_return"].fillna(-1e9)
        trading_ranks = (
            eligible["_trading_score"]
            .rank(
                method="min",
                ascending=False,
                na_option="bottom",
            )
            .astype("Int64")
        )
        frame.loc[trading_ranks.index, "trading_leader_rank"] = trading_ranks
    return frame


def _mean_or_none(values: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.mean()), 6)


def _model_family(model: Any) -> str:
    name = str(model or "").strip()
    if not name:
        return "unknown"
    if name.startswith("Ensemble "):
        return "Ensemble"
    for suffix in (" Return", " v2", " Lite"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if "Hybrid" in name:
        return "Prophet Hybrid"
    return name


def _models_for_run(
    *,
    manifest: Dict[str, Any],
    reports: Iterable[Optional[pd.DataFrame]],
    run_dir: Path,
) -> List[str]:
    models = [str(model) for model in manifest.get("model_list", []) if str(model).strip()]
    if models:
        return models
    for report in reports:
        if report is not None and "Model" in report.columns:
            models.extend(str(model) for model in report["Model"].dropna().tolist())
    if not models:
        models.append(_model_from_run_id(run_dir.name))
    seen = set()
    unique = []
    for model in models:
        if model not in seen:
            seen.add(model)
            unique.append(model)
    return unique


def _report_row_for_model(report: Optional[pd.DataFrame], model: str) -> Optional[pd.Series]:
    if report is None or "Model" not in report.columns:
        return None
    matches = report[report["Model"].astype(str) == str(model)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _missing_required_files(run_dir: Path) -> List[str]:
    return [relative for relative in REQUIRED_RUN_FILES if not (run_dir / relative).exists()]


def _model_from_run_id(run_id: str) -> str:
    marker = "_model-"
    if marker in run_id:
        return run_id.split(marker, 1)[1]
    return "unknown"


def _value(row: Optional[pd.Series], column: str) -> Any:
    if row is None or column not in row:
        return None
    value = row.get(column)
    if pd.isna(value):
        return None
    return value


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _is_null(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _diff(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def _positive(value: Optional[float]) -> bool:
    return value is not None and value > 0


def _in_trade_band(value: Optional[float]) -> bool:
    return value is not None and TRADE_BAND_MIN <= value <= TRADE_BAND_MAX


def _is_benchmark_clone(
    final_net: Optional[float],
    final_buyhold: Optional[float],
    final_trade_count: Optional[float],
) -> bool:
    if final_net is None or final_buyhold is None or final_trade_count is None:
        return False
    return final_trade_count <= 1 and abs(final_net - final_buyhold) <= BENCHMARK_CLONE_TOLERANCE


def _leaderboard_columns() -> List[str]:
    return [
        "symbol",
        "run_id",
        "model",
        "history_years",
        "history_class",
        "history_bucket",
        "meets_10y_reference",
        "data_history_warning",
        "sector",
        "prediction_leader_rank",
        "trading_leader_rank",
        "wf_net_return",
        "final_net_return",
        "final_buyhold_return",
        "wf_final_net_gap",
        "final_excess_vs_buyhold",
        "final_trade_count",
        "final_sharpe",
        "final_signal_diagnosis",
        "holdout_complete_flag",
        "trade_sufficiency_flag",
        "benchmark_clone_flag",
        "leader_reliability_class",
        "missing_required_files",
    ]
