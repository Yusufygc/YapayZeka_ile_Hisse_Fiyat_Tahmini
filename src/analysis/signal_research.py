# -*- coding: utf-8 -*-
"""Dry-run utilities for Plan 1 signal research automation."""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from src.analysis.run_leaderboard import (
    DEFAULT_LONG_HISTORY_YEARS,
    DEFAULT_SHORT_HISTORY_YEARS,
    build_history_effect_summary,
    build_multi_symbol_leaderboard,
    build_sector_summary,
    classify_history_bucket,
    data_history_warning,
    parse_history_dates,
)
from src.backtesting.signals import SignalConfig
from src.pipeline.config import (
    DataConfig,
    ExecutionConfig,
    ModelConfig,
    PipelineConfig,
    ValidationConfig,
)

DEFAULT_MODELS = (
    "Ridge Return",
    "ElasticNet Return",
    "LightGBM Return",
    "XGBoost",
    "Random Forest",
    "DLinear",
    "NLinear",
    "LSTM Lite",
    "AttentionLSTM v2",
    "Prophet",
    "ARIMA",
    "Prophet-ML/DL Hybrid",
)

PLAN1_PRIMARY_SYMBOLS = (
    "KCHOL",
    "SAHOL",
    "ENKAI",
    "EREGL",
    "TUPRS",
    "SASA",
    "ASELS",
    "LOGO",
    "ARDYZ",
)

SIGNAL_POLICY_PROFILES: Dict[str, Dict[str, Any]] = {
    "V0": {
        "description": "Current baseline pipeline profile.",
        "signal_mode": "simple",
        "quality_gate_mode": "current",
        "uses_final_holdout_for_selection": False,
    },
    "V1": {
        "description": "Professional signal mode with soft quality gate.",
        "signal_mode": "professional",
        "quality_gate_mode": "soft",
        "uses_final_holdout_for_selection": False,
    },
    "V2": {
        "description": "Percentile entry trial parameters on walk-forward calibration data.",
        "signal_mode": "professional",
        "quality_gate_mode": "soft",
        "percentile_entry_enabled": True,
        "uses_final_holdout_for_selection": False,
    },
    "V3": {
        "description": "Trade-band calibration target.",
        "signal_mode": "professional",
        "quality_gate_mode": "soft",
        "trade_count_min": 8,
        "trade_count_max": 20,
        "exposure_min": 25,
        "exposure_max": 70,
        "uses_final_holdout_for_selection": False,
    },
    "V4": {
        "description": "Sector-relative confirmation as an opt-in entry filter.",
        "signal_mode": "professional",
        "quality_gate_mode": "soft",
        "sector_relative_confirmation": True,
        "required_feature": "Sector_Relative_Strength",
        "uses_final_holdout_for_selection": False,
    },
}


def check_universe(
    *,
    symbols: Sequence[str],
    data_dir: str | Path,
    universe_file: str | Path | None = None,
    min_history_years: float | None = DEFAULT_LONG_HISTORY_YEARS,
    long_history_years: float | None = None,
    short_history_years: float = DEFAULT_SHORT_HISTORY_YEARS,
) -> pd.DataFrame:
    """Report CSV availability and history bucket for a research symbol set."""
    sector_lookup = _load_sector_lookup(universe_file)
    long_threshold = _resolve_long_history_years(
        min_history_years=min_history_years,
        long_history_years=long_history_years,
    )
    rows = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        data_path = Path(data_dir) / f"{symbol}.csv"
        history = _history_from_csv(data_path)
        history_years = history["history_years"]
        history_bucket = classify_history_bucket(
            history_years,
            data_dir_provided=True,
            missing_history=history_years is None,
            long_history_years=long_threshold,
            short_history_years=short_history_years,
        )
        rows.append(
            {
                "symbol": symbol,
                "data_file_exists": data_path.exists(),
                "first_date": history["first_date"],
                "last_date": history["last_date"],
                "history_years": history_years,
                "history_class": history_bucket,
                "history_bucket": history_bucket,
                "meets_10y_reference": bool(
                    history_years is not None and history_years >= long_threshold
                ),
                "data_history_warning": data_history_warning(history_bucket),
                "sector": sector_lookup.get(symbol),
                "eligible_10y": bool(history_years is not None and history_years >= long_threshold),
            }
        )
    return pd.DataFrame(rows)


def plan_runs(
    *,
    symbols: Sequence[str],
    models: Sequence[str] | None = None,
    policies: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the symbol/model/policy matrix without starting training."""
    models = tuple(models or DEFAULT_MODELS)
    policies = tuple(policies or SIGNAL_POLICY_PROFILES)
    rows: List[Dict[str, Any]] = []
    for symbol in [str(item).strip().upper() for item in symbols if str(item).strip()]:
        for model in models:
            for policy in policies:
                profile = dict(SIGNAL_POLICY_PROFILES[str(policy).upper()])
                rows.append(
                    {
                        "symbol": symbol,
                        "model": model,
                        "policy": str(policy).upper(),
                        "dry_run_only": True,
                        **profile,
                    }
                )
    return pd.DataFrame(rows)


def run_research_matrix(
    *,
    symbols: Sequence[str],
    data_dir: str | Path,
    outputs_base: str | Path,
    universe_file: str | Path | None = None,
    models: Sequence[str] | None = None,
    policies: Sequence[str] | None = None,
    mode: str = "walk_forward",
    workers: int = 1,
    resume: bool = False,
    dry_run: bool = False,
    old_data_dir: str | Path | None = None,
    phase: str = "plan1",
) -> pd.DataFrame:
    """Run or dry-run the Plan 1 symbol/model/policy matrix sequentially."""
    data_dir = Path(data_dir)
    outputs_base = Path(outputs_base)
    old_data_dir = Path(old_data_dir) if old_data_dir is not None else data_dir / "old"
    models = _resolve_models(models)
    policies = tuple(str(policy).strip().upper() for policy in (policies or SIGNAL_POLICY_PROFILES))
    normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    data_preflight = {
        symbol: ensure_symbol_data(
            symbol=symbol,
            data_dir=data_dir,
            old_data_dir=old_data_dir,
            restore_from_old=not dry_run,
        )
        for symbol in normalized_symbols
    }
    universe = check_universe(
        symbols=normalized_symbols,
        data_dir=data_dir,
        universe_file=universe_file,
    )
    universe_rows = {row["symbol"]: row for row in universe.to_dict(orient="records")}
    matrix = plan_runs(symbols=normalized_symbols, models=models, policies=policies)
    rows: List[Dict[str, Any]] = []
    if int(workers or 1) != 1:
        rows.append(
            {
                "symbol": "",
                "model": "",
                "policy": "",
                "status": "warning",
                "message": "signal_research run executes sequentially; workers is recorded only.",
                "workers": int(workers or 1),
            }
        )

    for item in matrix.to_dict(orient="records"):
        symbol = str(item["symbol"]).upper()
        model = str(item["model"])
        policy = str(item["policy"]).upper()
        profile = dict(SIGNAL_POLICY_PROFILES[policy])
        symbol_row = dict(universe_rows.get(symbol, {}))
        data_path = data_dir / f"{symbol}.csv"
        ensure_result = data_preflight.get(symbol) or ensure_symbol_data(
            symbol=symbol,
            data_dir=data_dir,
            old_data_dir=old_data_dir,
            restore_from_old=not dry_run,
        )
        base_row = {
            "symbol": symbol,
            "model": model,
            "policy": policy,
            "mode": mode,
            "workers": int(workers or 1),
            "history_years": symbol_row.get("history_years"),
            "history_bucket": symbol_row.get("history_bucket"),
            "meets_10y_reference": symbol_row.get("meets_10y_reference"),
            "sector": symbol_row.get("sector"),
            "data_status": ensure_result["status"],
            "data_source": ensure_result.get("source"),
            "uses_final_holdout_for_selection": False,
            **profile,
        }
        if not data_path.exists():
            rows.append(
                {
                    **base_row,
                    "status": "skipped_missing_data",
                    "message": ensure_result.get("message", "data file missing"),
                }
            )
            continue
        if resume:
            existing = _find_complete_policy_run(
                outputs_base=outputs_base,
                symbol=symbol,
                model=model,
                policy=policy,
            )
            if existing is not None:
                rows.append(
                    {
                        **base_row,
                        "status": "skipped_resume",
                        "run_id": existing.get("run_id"),
                        "output_dir": existing.get("output_dir"),
                    }
                )
                continue
        if dry_run:
            rows.append({**base_row, "status": "planned", "dry_run_only": True})
            continue

        started = time.time()
        try:
            pipeline_cfg = build_research_pipeline_config(
                data_file=data_path,
                universe_file=universe_file,
                mode=mode,
                model=model,
                policy=policy,
                profile=profile,
                history_bucket=symbol_row.get("history_bucket"),
                sector=symbol_row.get("sector"),
                phase=phase,
            )
            from src.pipeline.orchestrator import ForecastingPipeline

            pipeline = ForecastingPipeline(cfg=pipeline_cfg)
            pipeline.run_all()
            rows.append(
                {
                    **base_row,
                    "status": "ok",
                    "run_id": pipeline.run_id,
                    "output_dir": pipeline.outputs_dir,
                    "duration_sec": round(time.time() - started, 1),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base_row,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "duration_sec": round(time.time() - started, 1),
                }
            )
    return pd.DataFrame(rows)


def ensure_symbol_data(
    *,
    symbol: str,
    data_dir: str | Path,
    old_data_dir: str | Path | None = None,
    restore_from_old: bool = True,
) -> Dict[str, Any]:
    """Ensure a symbol CSV exists, restoring from data/old when available."""
    symbol = str(symbol).strip().upper()
    data_dir = Path(data_dir)
    path = data_dir / f"{symbol}.csv"
    if path.exists():
        return {"status": "exists", "path": str(path), "source": str(path)}
    old_data_dir = Path(old_data_dir) if old_data_dir is not None else data_dir / "old"
    fallback = old_data_dir / f"{symbol}.csv"
    if fallback.exists() and restore_from_old:
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback, path)
        return {
            "status": "restored_from_old",
            "path": str(path),
            "source": str(fallback),
        }
    if fallback.exists():
        return {
            "status": "available_in_old",
            "path": str(path),
            "source": str(fallback),
            "message": f"CSV data/old altinda bulundu: {fallback}",
        }
    return {
        "status": "missing_data",
        "path": str(path),
        "source": None,
        "message": f"CSV bulunamadi: {path}",
    }


def ensure_data_for_symbols(
    *,
    symbols: Sequence[str],
    data_dir: str | Path,
    old_data_dir: str | Path | None = None,
    restore_from_old: bool = True,
) -> pd.DataFrame:
    """Ensure CSV files exist for a symbol set and report the source used."""
    rows = []
    for symbol in [str(item).strip().upper() for item in symbols if str(item).strip()]:
        result = ensure_symbol_data(
            symbol=symbol,
            data_dir=data_dir,
            old_data_dir=old_data_dir,
            restore_from_old=restore_from_old,
        )
        rows.append({"symbol": symbol, **result})
    return pd.DataFrame(rows)


def build_research_pipeline_config(
    *,
    data_file: str | Path,
    universe_file: str | Path | None,
    mode: str,
    model: str,
    policy: str,
    profile: Dict[str, Any],
    history_bucket: Any,
    sector: Any,
    phase: str,
) -> PipelineConfig:
    """Build a pipeline config for one research policy run."""
    policy = str(policy).upper()
    signal_cfg = _signal_config_for_policy(profile)
    execution = ExecutionConfig(
        signal_mode=str(profile.get("signal_mode", "simple")),
        signal_config=signal_cfg,
        enable_signal_execution_calibration=policy in {"V2", "V3"},
        signal_calibration_profile="research" if policy in {"V2", "V3"} else "production",
        signal_calibration_min_trades=int(profile.get("trade_count_min", 6) or 6),
        signal_calibration_objective=("trade_band" if policy == "V3" else "risk_adjusted"),
        report_detail_level="research",
        research_policy=policy,
        research_phase=phase,
        research_metadata={
            "policy_profile": profile,
            "history_bucket": history_bucket,
            "sector": sector,
            "uses_final_holdout_for_selection": False,
        },
    )
    return PipelineConfig(
        data=DataConfig(
            data_file=str(data_file),
            auto_update_data=True,
            auto_update_interactive=False,
            universe_file=(
                str(universe_file) if universe_file is not None else "data/bist_universe.csv"
            ),
        ),
        validation=ValidationConfig(validation_mode=mode),
        models=ModelConfig(selected_models=[model], require_available=False),
        execution=execution,
    )


def summarize_completed_runs(
    *,
    outputs_base: str | Path,
    symbols: Sequence[str],
    data_dir: str | Path | None = None,
    sector_file: str | Path | None = None,
    min_history_years: float | None = DEFAULT_LONG_HISTORY_YEARS,
    long_history_years: float | None = None,
    short_history_years: float = DEFAULT_SHORT_HISTORY_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Plan 1 summary tables from completed run outputs only."""
    leaderboard = build_multi_symbol_leaderboard(
        outputs_base=outputs_base,
        symbols=symbols,
        data_dir=data_dir,
        sector_file=sector_file,
        min_history_years=min_history_years,
        long_history_years=long_history_years,
        short_history_years=short_history_years,
    )
    return (
        leaderboard,
        build_sector_summary(leaderboard),
        build_history_effect_summary(leaderboard),
    )


def build_decision_report(
    *,
    leaderboard: pd.DataFrame,
    sector_summary: pd.DataFrame,
    history_summary: pd.DataFrame,
) -> str:
    """Build a compact Markdown decision report from completed run outputs."""
    lines = [
        "# Plan 1 History Effect Decision Report",
        "",
        "Final holdout is reported as confirmation only; policy selection must come from walk-forward data.",
        "",
        "## Overview",
        "",
        f"- Run rows: {len(leaderboard)}",
        f"- Symbols: {0 if leaderboard.empty else int(leaderboard['symbol'].nunique())}",
        f"- History buckets: {_joined_values(leaderboard, 'history_bucket')}",
        "",
        "## Reliability by History",
        "",
        _markdown_table(history_summary),
        "",
        "## Sector Summary",
        "",
        _markdown_table(sector_summary),
        "",
        "## Current Decision Notes",
        "",
    ]
    if leaderboard.empty:
        lines.append("- No run-level rows were available.")
    else:
        complete = leaderboard[leaderboard["holdout_complete_flag"].fillna(False).astype(bool)]
        if complete.empty:
            lines.append("- No complete final-holdout rows were available.")
        else:
            first = complete.iloc[0]
            lines.append(
                "- First ranked complete row: "
                f"{first.get('symbol')} / {first.get('model')} / "
                f"{first.get('leader_reliability_class')} / "
                f"{first.get('history_bucket')}."
            )
        incomplete_invalid = leaderboard[
            leaderboard["leader_reliability_class"].isin(["incomplete", "invalid"])
        ]
        lines.append(f"- Incomplete or invalid rows: {len(incomplete_invalid)}")
    lines.append("")
    return "\n".join(lines)


def parse_csv_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_models(models: Sequence[str] | None) -> tuple[str, ...]:
    if not models:
        return DEFAULT_MODELS
    normalized = tuple(str(model).strip() for model in models if str(model).strip())
    if not normalized or any(model.lower() == "all" for model in normalized):
        return DEFAULT_MODELS
    return normalized


def _signal_config_for_policy(profile: Dict[str, Any]) -> SignalConfig:
    quality_gate_mode = str(profile.get("quality_gate_mode") or "soft")
    if quality_gate_mode == "current":
        quality_gate_mode = "soft"
    cfg = replace(SignalConfig(), quality_gate_mode=quality_gate_mode)
    if profile.get("percentile_entry_enabled"):
        cfg = replace(cfg, min_entry_threshold=0.001, entry_threshold_multiplier=0.85)
    if profile.get("sector_relative_confirmation"):
        cfg = replace(cfg, min_entry_threshold=max(float(cfg.min_entry_threshold), 0.001))
    if profile.get("trade_count_min"):
        cfg = replace(cfg, max_holding_bars=15, cooldown_bars=1)
    return cfg


def _find_complete_policy_run(
    *,
    outputs_base: str | Path,
    symbol: str,
    model: str,
    policy: str,
) -> Dict[str, Any] | None:
    runs_dir = Path(outputs_base) / str(symbol).upper() / "runs"
    if not runs_dir.exists():
        return None
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "run_manifest.json"
        final_report = run_dir / "csv" / "backtest_report_final_holdout.csv"
        if not manifest_path.exists() or not final_report.exists():
            continue
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(manifest.get("research_policy", "")).upper() != policy:
            continue
        models = {str(item) for item in manifest.get("model_list", [])}
        if model not in models:
            continue
        status = manifest.get("final_holdout_status") or {}
        if isinstance(status, dict) and status.get("status") not in {None, "success"}:
            continue
        return {"run_id": run_dir.name, "output_dir": str(run_dir)}
    return None


def _history_from_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"first_date": None, "last_date": None, "history_years": None}
    frame = _read_table(path)
    if frame is None:
        return {"first_date": None, "last_date": None, "history_years": None}
    date_column = _find_column(frame, ("Date", "date", "Tarih", "tarih"))
    if date_column is None:
        return {"first_date": None, "last_date": None, "history_years": None}
    dates = parse_history_dates(frame[date_column]).dropna()
    if dates.empty:
        return {"first_date": None, "last_date": None, "history_years": None}
    first = dates.min().normalize()
    last = dates.max().normalize()
    return {
        "first_date": first.strftime("%Y-%m-%d"),
        "last_date": last.strftime("%Y-%m-%d"),
        "history_years": round(float((last - first).days / 365.25), 4),
    }


def _load_sector_lookup(universe_file: str | Path | None) -> Dict[str, str]:
    if universe_file is None:
        return {}
    path = Path(universe_file)
    if not path.exists():
        return {}
    frame = _read_table(path)
    if frame is None:
        return {}
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    symbol_column = _find_column(frame, ("Symbol", "symbol", "SYMBOL", "Ticker", "ticker"))
    sector_column = _find_column(frame, ("Sector", "sector", "Sektor", "sektor", "SECTOR"))
    sector_index_column = _find_column(
        frame,
        ("Sector_Index", "sector_index", "SECTOR_INDEX", "SectorIndex", "sectorIndex"),
    )
    if symbol_column is None or (sector_column is None and sector_index_column is None):
        return {}
    lookup: Dict[str, str] = {}
    for _, row in frame.iterrows():
        symbol = str(row.get(symbol_column, "")).strip().upper()
        sector = row.get(sector_column) if sector_column else None
        if pd.isna(sector) or not str(sector).strip():
            sector = row.get(sector_index_column) if sector_index_column else None
        if symbol and not pd.isna(sector) and str(sector).strip():
            lookup[symbol] = str(sector).strip()
    return lookup


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    columns = {str(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _read_table(path: Path) -> pd.DataFrame | None:
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin5"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception:
            continue
    return None


def _joined_values(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = sorted(str(value) for value in frame[column].dropna().unique())
    return ", ".join(values)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.fillna("")
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row.get(column, "")).replace("|", "\\|") for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


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
