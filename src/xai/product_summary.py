# -*- coding: utf-8 -*-
"""User-facing XAI product summary."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.api.constants import XAI_CAVEAT
from src.xai.feature_dictionary import describe_feature, feature_group

_TREE_MODELS = {"XGBoost", "Random Forest", "LightGBM Return", "Random Forest Return"}
_LINEAR_MODELS = {"Ridge Return", "ElasticNet Return"}
_SEQ_MODELS = {"LSTM", "LSTM Lite", "DLinear", "NLinear", "AttentionLSTM", "AttentionLSTM v2"}


def _model_family_caveat(model_name: str) -> str:
    if model_name in _TREE_MODELS:
        return "Tree modellerde SHAP TreeExplainer kullanilir; ozellik katkilari guvenilirdir."
    if model_name in _LINEAR_MODELS:
        return "Lineer modellerde katsayi bazli katkilar hesaplanir; yorumlama nispeten dogrudur."
    if model_name in _SEQ_MODELS:
        return (
            "Derin ogrenme modellerinde ozellik katkilari yaklasiktir; "
            "daha temkinli yorumlanmalidir."
        )
    return "Model ailesi icin aciklanabilirlik kalitesi bilinmiyor."


@dataclass
class XaiFeatureFactor:
    feature_name: str
    human_label: str
    importance: float
    direction: str
    feature_group: Optional[str] = None
    reason: Optional[str] = None
    method: Optional[str] = None
    contribution: Optional[float] = None
    approximate: Optional[bool] = None


@dataclass
class XaiProductSummary:
    available: bool
    method: str = ""
    top_positive_reasons: List[XaiFeatureFactor] = field(default_factory=list)
    top_negative_reasons: List[XaiFeatureFactor] = field(default_factory=list)
    feature_stability_top: Dict[str, float] = field(default_factory=dict)
    model_family_caveat: str = ""
    caveat: str = XAI_CAVEAT


def _unavailable(reason: str = "") -> XaiProductSummary:
    caveat = XAI_CAVEAT if not reason else f"{XAI_CAVEAT} ({reason})"
    return XaiProductSummary(available=False, caveat=caveat)


def build_xai_product_summary(
    symbol: str,
    model_name: str,
    outputs_base: Optional[str] = None,
    run_id: Optional[str] = None,
    model_path: Optional[str] = None,
    top_k: int = 5,
) -> XaiProductSummary:
    symbol = symbol.upper()
    if outputs_base is None:
        here = os.path.dirname(os.path.abspath(__file__))
        outputs_base = os.path.join(here, "..", "..", "outputs")

    xai_dirs = _candidate_xai_dirs(
        outputs_base=outputs_base,
        symbol=symbol,
        run_id=run_id,
        model_path=model_path,
    )
    existing_dirs = [xai_dir for xai_dir in xai_dirs if os.path.isdir(xai_dir)]
    if not existing_dirs:
        return _unavailable("xai dizini bulunamadi")

    last_unavailable: Optional[XaiProductSummary] = None
    for xai_dir in existing_dirs:
        standard_table = _find_standard_xai_table(xai_dir)
        if standard_table is not None:
            summary = _summary_from_standard_table(
                table_path=standard_table,
                model_name=model_name,
                top_k=top_k,
            )
            if summary.available:
                return summary
            last_unavailable = summary

        legacy_table = _find_legacy_importance_table(xai_dir, model_name)
        if legacy_table is not None:
            summary = _summary_from_legacy_importance(
                table_path=legacy_table,
                model_name=model_name,
                top_k=top_k,
            )
            if summary.available:
                return summary
            last_unavailable = summary

    return last_unavailable or _unavailable("xai tablosu bulunamadi")


def _candidate_xai_dirs(
    *,
    outputs_base: str,
    symbol: str,
    run_id: Optional[str] = None,
    model_path: Optional[str] = None,
) -> List[str]:
    candidates: List[str] = []
    if run_id:
        candidates.append(os.path.join(outputs_base, symbol, "runs", str(run_id), "xai"))
    model_xai_dir = _xai_dir_from_model_path(model_path)
    if model_xai_dir:
        candidates.append(model_xai_dir)
    candidates.append(os.path.join(outputs_base, symbol, "latest", "xai"))

    seen: set[str] = set()
    unique: List[str] = []
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def _xai_dir_from_model_path(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None
    try:
        path = Path(model_path).expanduser()
        start = path if path.is_dir() else path.parent
        for candidate in (start, *start.parents):
            if candidate.parent.name == "runs":
                return str(candidate / "xai")
    except Exception:
        return None
    return None


def _read_csv_table(table_path: str) -> pd.DataFrame:
    attempts = (
        {"sep": None, "engine": "python"},
        {"sep": ";"},
        {},
    )
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(table_path, **kwargs)
            if _looks_like_misparsed_semicolon_csv(df):
                continue
            return _normalize_columns(df)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(table_path)


def _looks_like_misparsed_semicolon_csv(df: pd.DataFrame) -> bool:
    return len(df.columns) == 1 and ";" in str(df.columns[0])


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).lstrip("\ufeff").strip() for column in df.columns]
    return df


def _find_standard_xai_table(latest_dir: str) -> Optional[str]:
    patterns = [
        os.path.join(latest_dir, "csv", "xai_top_reasons_*.csv"),
        os.path.join(latest_dir, "xai_top_reasons_*.csv"),
        os.path.join(latest_dir, "**", "xai_top_reasons_*.csv"),
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    return sorted(set(matches))[-1] if matches else None


def _find_legacy_importance_table(latest_dir: str, model_name: str) -> Optional[str]:
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    candidates = [
        os.path.join(latest_dir, f"feature_importance_{safe_name}_wf.csv"),
        os.path.join(latest_dir, f"feature_importance_{safe_name}_final_holdout.csv"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    matches = glob.glob(os.path.join(latest_dir, f"feature_importance_{safe_name}_*.csv"))
    return sorted(matches)[-1] if matches else None


def _summary_from_standard_table(*, table_path: str, model_name: str, top_k: int) -> XaiProductSummary:
    try:
        df = _read_csv_table(table_path)
    except Exception as exc:
        return _unavailable(f"xai top reasons tablosu okunamadi: {type(exc).__name__}")

    required = {"Model", "Feature", "Readable_Feature", "Importance"}
    if df.empty or not required.issubset(df.columns):
        return _unavailable("xai top reasons tablosu beklenen kolonlari icermiyor")

    model_df = df[df["Model"].astype(str) == str(model_name)].copy()
    if model_df.empty:
        return _unavailable("en iyi model icin xai satiri bulunamadi")

    model_df["Importance"] = pd.to_numeric(model_df["Importance"], errors="coerce")
    model_df.dropna(subset=["Importance"], inplace=True)
    if model_df.empty:
        return _unavailable("xai onem degerleri okunamadi")

    model_df.sort_values("Importance", ascending=False, inplace=True)
    return _summary_from_feature_frame(
        df=model_df,
        model_name=model_name,
        feature_col="Feature",
        label_col="Readable_Feature",
        importance_col="Importance",
        top_k=top_k,
    )


def _summary_from_legacy_importance(*, table_path: str, model_name: str, top_k: int) -> XaiProductSummary:
    try:
        df = _read_csv_table(table_path)
    except Exception as exc:
        return _unavailable(f"ozellik onem dosyasi okunamadi: {type(exc).__name__}")

    importance_col = next((c for c in df.columns if "importance" in c.lower() or "mean" in c.lower()), None)
    feature_col = next((c for c in df.columns if "feature" in c.lower()), None)
    if importance_col is None or feature_col is None:
        return _unavailable("xai dosyasinda beklenen kolonlar bulunamadi")
    df = df[[feature_col, importance_col]].copy()
    df["Readable_Feature"] = df[feature_col].map(lambda v: describe_feature(str(v)))
    df[importance_col] = pd.to_numeric(df[importance_col], errors="coerce")
    df.dropna(subset=[importance_col], inplace=True)
    return _summary_from_feature_frame(
        df=df,
        model_name=model_name,
        feature_col=feature_col,
        label_col="Readable_Feature",
        importance_col=importance_col,
        top_k=top_k,
    )


def _summary_from_feature_frame(
    *,
    df: pd.DataFrame,
    model_name: str,
    feature_col: str,
    label_col: str,
    importance_col: str,
    top_k: int,
) -> XaiProductSummary:
    if df.empty:
        return _unavailable("xai tablosu bos")
    work = df.copy()
    work[importance_col] = pd.to_numeric(work[importance_col], errors="coerce")
    work.dropna(subset=[importance_col], inplace=True)
    if work.empty:
        return _unavailable("xai onem degerleri okunamadi")

    contribution_col = "Contribution" if "Contribution" in work.columns else None
    if contribution_col is not None:
        work["_xai_contribution"] = pd.to_numeric(work[contribution_col], errors="coerce")
    else:
        work["_xai_contribution"] = pd.NA

    direction_col = "Direction" if "Direction" in work.columns else None
    if direction_col is not None:
        direction_text = work[direction_col].astype(str).str.strip().str.lower()
    else:
        direction_text = pd.Series([""] * len(work), index=work.index)

    positive_direction = direction_text.isin({"positive", "up", "buy", "hold", "long"})
    negative_direction = direction_text.isin({"negative", "down", "sell", "exit", "short"})
    has_direction_signal = bool((positive_direction | negative_direction).any())
    has_contribution_signal = bool(work["_xai_contribution"].notna().any())

    if has_contribution_signal:
        positive_mask = work["_xai_contribution"].fillna(0) > 0
        negative_mask = work["_xai_contribution"].fillna(0) < 0
    elif has_direction_signal:
        positive_mask = positive_direction
        negative_mask = negative_direction
    else:
        positive_mask = work[importance_col] > 0
        negative_mask = work[importance_col] < 0

    work["_xai_sort_importance"] = work[importance_col].abs()
    work.sort_values("_xai_sort_importance", ascending=False, inplace=True)

    def _factor(row: Any, direction: str) -> XaiFeatureFactor:
        feature = str(row[feature_col])
        label = _resolved_label(feature, row.get(label_col))
        contribution = _optional_float(row.get("Contribution"))
        method = _optional_text(row.get("Method"))
        reason = _optional_text(row.get("Reason"))
        group = _resolved_group(feature, row.get("Feature_Group"))
        approximate = _optional_bool(row.get("Approximate"))
        return XaiFeatureFactor(
            feature_name=feature,
            human_label=label,
            importance=abs(float(row[importance_col])),
            direction=direction,
            feature_group=group,
            reason=reason,
            method=method,
            contribution=contribution,
            approximate=approximate,
        )

    positives = work[positive_mask].head(top_k)
    negatives = work[negative_mask].head(top_k)
    if negatives.empty and not positives.empty:
        top_positive = [_factor(row, "positive") for _, row in positives.head(top_k).iterrows()]
        top_negative = [_factor(row, "negative") for _, row in work.tail(top_k).iloc[::-1].iterrows()]
    else:
        top_positive = [_factor(row, "positive") for _, row in positives.iterrows()]
        top_negative = [_factor(row, "negative") for _, row in negatives.iterrows()]

    method = "SHAP TreeExplainer" if model_name in _TREE_MODELS else "Feature Importance"
    return XaiProductSummary(
        available=True,
        method=method,
        top_positive_reasons=top_positive,
        top_negative_reasons=top_negative,
        model_family_caveat=_model_family_caveat(model_name),
        caveat=XAI_CAVEAT,
    )


def _optional_text(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _resolved_label(feature: str, raw_label: Any) -> str:
    label = _optional_text(raw_label)
    if label and "teknik veya makro sinyal" not in label.lower():
        return label
    return describe_feature(feature)


def _resolved_group(feature: str, raw_group: Any) -> Optional[str]:
    group = _optional_text(raw_group)
    if not group or group.lower() == "other":
        return feature_group(feature)
    return group


def _optional_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "evet"}:
        return True
    if text in {"false", "0", "no", "hayir", "hayır"}:
        return False
    return None
